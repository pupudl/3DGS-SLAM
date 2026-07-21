import csv
import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
import torch


def _to_hw_bool(mask: Optional[torch.Tensor], height: int, width: int) -> torch.Tensor:
    if mask is None:
        return torch.ones((height, width), dtype=torch.bool)
    mask = mask.detach().cpu()
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    if mask.shape != (height, width):
        raise ValueError(f"Expected mask shape {(height, width)}, got {tuple(mask.shape)}")
    return mask.bool()


def _extract_anomaly_components(
    anomaly_map: torch.Tensor,
    valid_mask: torch.Tensor,
    anomaly_threshold: float,
    min_component_pixels: int,
    component_dilation: int,
) -> tuple[torch.Tensor, list[Dict[str, object]]]:
    masked_anomaly = anomaly_map.clone()
    masked_anomaly[~valid_mask] = 0.0
    anomaly_binary = masked_anomaly >= anomaly_threshold
    anomaly_binary_np = anomaly_binary.numpy().astype(np.uint8)
    if component_dilation > 0:
        kernel_size = component_dilation * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        anomaly_binary_np = cv2.dilate(anomaly_binary_np, kernel, iterations=1)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        anomaly_binary_np,
        connectivity=8,
    )

    support_mask = torch.zeros_like(anomaly_binary)
    components = []
    for label_idx in range(1, component_count):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        left = int(stats[label_idx, cv2.CC_STAT_LEFT])
        top = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        component_mask = torch.from_numpy(labels == label_idx)
        component_mask = component_mask & valid_mask
        if int(component_mask.sum().item()) < min_component_pixels:
            continue
        support_mask |= component_mask
        component_values = masked_anomaly[component_mask]
        components.append(
            {
                "label": int(label_idx),
                "bbox": (left, top, left + width, top + height),
                "area": int(component_mask.sum().item()),
                "peak": float(component_values.max().item()),
                "mean": float(component_values.mean().item()),
            }
        )

    return support_mask, components


def _resolve_anomaly_threshold(
    anomaly_map: torch.Tensor,
    valid_mask: torch.Tensor,
    max_threshold: float,
    threshold_floor: float,
    primary_quantile: float,
    fallback_quantile: float,
) -> tuple[float, float, float]:
    values = anomaly_map[valid_mask]
    values = values[values > 0]
    if values.numel() == 0:
        return max_threshold, max_threshold, max_threshold

    primary_threshold = max(
        threshold_floor,
        min(max_threshold, float(torch.quantile(values, primary_quantile).item())),
    )
    fallback_threshold = max(
        threshold_floor,
        min(max_threshold, float(torch.quantile(values, fallback_quantile).item())),
    )
    return primary_threshold, fallback_threshold, float(values.max().item())


def compute_gaussian_anomaly_scores(
    anomaly_map: torch.Tensor,
    gaussian_data: Dict[str, torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
    topk: int = 64,
    radius_scale: float = 1.5,
    min_radius: float = 1.0,
    min_valid_pixels: int = 4,
    use_distance_weight: bool = True,
    anomaly_threshold: float = 0.35,
    min_component_pixels: int = 16,
    component_dilation: int = 2,
    use_adaptive_threshold: bool = True,
    enable_threshold_fallback: bool = True,
    anomaly_threshold_floor: float = 0.05,
    anomaly_threshold_primary_quantile: float = 0.95,
    anomaly_threshold_fallback_quantile: float = 0.85,
) -> Dict[str, object]:
    anomaly_map = anomaly_map.detach().cpu().float()
    if anomaly_map.dim() != 2:
        raise ValueError(f"Expected 2D anomaly map, got shape {tuple(anomaly_map.shape)}")
    height, width = anomaly_map.shape
    valid_mask_hw = _to_hw_bool(valid_mask, height, width)

    centers = gaussian_data["projected_means2D"].detach().cpu().float()
    radius = gaussian_data["radius"].detach().cpu().float()
    seen = gaussian_data["seen"].detach().cpu().bool()
    gaussian_ids = gaussian_data.get("gaussian_ids")
    if gaussian_ids is None:
        gaussian_ids = torch.arange(radius.shape[0], dtype=torch.long)
    else:
        gaussian_ids = gaussian_ids.detach().cpu().long()
    timestep = gaussian_data.get("timestep")
    if timestep is not None:
        timestep = timestep.detach().cpu().float()
    depth = gaussian_data.get("depth")
    if depth is not None:
        depth = depth.detach().cpu().float()

    records = []
    scored_ids = []
    rank_scores = []
    mean_scores = []
    mass_scores = []
    peak_scores = []
    valid_pixels_list = []
    support_pixels_list = []
    footprint_pixels_list = []
    valid_ratios = []
    center_x_list = []
    center_y_list = []
    raw_radius_list = []
    eff_radius_list = []
    depth_list = []
    timestep_list = []

    candidate_mask = seen & torch.isfinite(radius) & (radius > 0)
    if centers.shape[0] != radius.shape[0]:
        raise ValueError(
            "Projected centers and radius lengths must match, "
            f"got {centers.shape[0]} and {radius.shape[0]}"
        )

    if use_adaptive_threshold:
        primary_threshold, fallback_threshold, max_anomaly_value = _resolve_anomaly_threshold(
            anomaly_map=anomaly_map,
            valid_mask=valid_mask_hw,
            max_threshold=anomaly_threshold,
            threshold_floor=anomaly_threshold_floor,
            primary_quantile=anomaly_threshold_primary_quantile,
            fallback_quantile=anomaly_threshold_fallback_quantile,
        )
    else:
        fixed_threshold = float(anomaly_threshold)
        primary_threshold = fixed_threshold
        fallback_threshold = fixed_threshold
        values = anomaly_map[valid_mask_hw]
        values = values[values > 0]
        max_anomaly_value = fixed_threshold if values.numel() == 0 else float(values.max().item())

    support_mask, components = _extract_anomaly_components(
        anomaly_map=anomaly_map,
        valid_mask=valid_mask_hw,
        anomaly_threshold=primary_threshold,
        min_component_pixels=min_component_pixels,
        component_dilation=component_dilation,
    )
    threshold_mode = "primary"
    if enable_threshold_fallback and not components and fallback_threshold < primary_threshold:
        support_mask, components = _extract_anomaly_components(
            anomaly_map=anomaly_map,
            valid_mask=valid_mask_hw,
            anomaly_threshold=fallback_threshold,
            min_component_pixels=min_component_pixels,
            component_dilation=component_dilation,
        )
        threshold_mode = "fallback"

    candidate_indices = set()
    if components:
        visible_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
        visible_centers = centers[visible_indices]
        visible_radii = torch.clamp(radius[visible_indices] * radius_scale, min=min_radius)
        for component in components:
            bbox_left, bbox_top, bbox_right, bbox_bottom = component["bbox"]
            overlap_mask = (
                (visible_centers[:, 0] + visible_radii) >= bbox_left
            ) & (
                (visible_centers[:, 0] - visible_radii) <= bbox_right
            ) & (
                (visible_centers[:, 1] + visible_radii) >= bbox_top
            ) & (
                (visible_centers[:, 1] - visible_radii) <= bbox_bottom
            )
            candidate_indices.update(visible_indices[overlap_mask].tolist())
    else:
        candidate_indices = set()

    for local_idx in sorted(candidate_indices):
        center_x = float(centers[local_idx, 0].item())
        center_y = float(centers[local_idx, 1].item())
        if not np.isfinite(center_x) or not np.isfinite(center_y):
            continue

        raw_radius = float(radius[local_idx].item())
        eff_radius = max(raw_radius * radius_scale, min_radius)
        left = max(int(np.floor(center_x - eff_radius)), 0)
        right = min(int(np.ceil(center_x + eff_radius)) + 1, width)
        top = max(int(np.floor(center_y - eff_radius)), 0)
        bottom = min(int(np.ceil(center_y + eff_radius)) + 1, height)
        if left >= right or top >= bottom:
            continue

        xs = torch.arange(left, right, dtype=torch.float32)
        ys = torch.arange(top, bottom, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2
        footprint = dist_sq <= (eff_radius ** 2)
        footprint_pixels = int(footprint.sum().item())
        if footprint_pixels <= 0:
            continue

        local_support_mask = support_mask[top:bottom, left:right] & footprint
        support_pixels = int(local_support_mask.sum().item())
        if support_pixels < min_valid_pixels:
            continue

        local_valid_mask = valid_mask_hw[top:bottom, left:right] & footprint
        valid_pixels = int(local_valid_mask.sum().item())

        if use_distance_weight:
            sigma = max(eff_radius * 0.5, 1e-6)
            weights = torch.exp(-0.5 * dist_sq / (sigma ** 2))
            weights = weights * local_support_mask.float()
        else:
            weights = local_support_mask.float()

        weight_sum = float(weights.sum().item())
        if weight_sum <= 0.0:
            continue

        local_anomaly = anomaly_map[top:bottom, left:right]
        mass_score = float((local_anomaly * weights).sum().item())
        mean_score = mass_score / weight_sum
        peak_score = float(local_anomaly[local_support_mask].max().item())
        valid_ratio = support_pixels / max(footprint_pixels, 1)
        rank_score = mass_score

        gaussian_id = int(gaussian_ids[local_idx].item())
        timestep_value = None if timestep is None else float(timestep[local_idx].item())
        depth_value = None if depth is None else float(depth[local_idx].item())
        record = {
            "gaussian_id": gaussian_id,
            "rank_score": rank_score,
            "mean_score": mean_score,
            "mass_score": mass_score,
            "peak_score": peak_score,
            "valid_pixels": valid_pixels,
            "support_pixels": support_pixels,
            "footprint_pixels": footprint_pixels,
            "valid_ratio": valid_ratio,
            "center_x": center_x,
            "center_y": center_y,
            "raw_radius": raw_radius,
            "effective_radius": eff_radius,
            "depth": depth_value,
            "timestep": timestep_value,
        }
        records.append(record)
        scored_ids.append(gaussian_id)
        rank_scores.append(rank_score)
        mean_scores.append(mean_score)
        mass_scores.append(mass_score)
        peak_scores.append(peak_score)
        valid_pixels_list.append(valid_pixels)
        support_pixels_list.append(support_pixels)
        footprint_pixels_list.append(footprint_pixels)
        valid_ratios.append(valid_ratio)
        center_x_list.append(center_x)
        center_y_list.append(center_y)
        raw_radius_list.append(raw_radius)
        eff_radius_list.append(eff_radius)
        depth_list.append(np.nan if depth_value is None else depth_value)
        timestep_list.append(np.nan if timestep_value is None else timestep_value)

    records.sort(key=lambda item: item["rank_score"], reverse=True)
    top_records = records[:topk]

    payload = {
        "records": records,
        "top_records": top_records,
        "components": components,
        "support_mask": support_mask,
        "candidate_gaussian_count": len(candidate_indices),
        "used_threshold": fallback_threshold if threshold_mode == "fallback" else primary_threshold,
        "primary_threshold": primary_threshold,
        "fallback_threshold": fallback_threshold,
        "threshold_mode": threshold_mode,
        "max_anomaly_value": max_anomaly_value,
        "gaussian_ids": torch.tensor(scored_ids, dtype=torch.long),
        "rank_score": torch.tensor(rank_scores, dtype=torch.float32),
        "mean_score": torch.tensor(mean_scores, dtype=torch.float32),
        "mass_score": torch.tensor(mass_scores, dtype=torch.float32),
        "peak_score": torch.tensor(peak_scores, dtype=torch.float32),
        "valid_pixels": torch.tensor(valid_pixels_list, dtype=torch.int32),
        "support_pixels": torch.tensor(support_pixels_list, dtype=torch.int32),
        "footprint_pixels": torch.tensor(footprint_pixels_list, dtype=torch.int32),
        "valid_ratio": torch.tensor(valid_ratios, dtype=torch.float32),
        "center_x": torch.tensor(center_x_list, dtype=torch.float32),
        "center_y": torch.tensor(center_y_list, dtype=torch.float32),
        "raw_radius": torch.tensor(raw_radius_list, dtype=torch.float32),
        "effective_radius": torch.tensor(eff_radius_list, dtype=torch.float32),
        "depth": torch.tensor(depth_list, dtype=torch.float32),
        "timestep": torch.tensor(timestep_list, dtype=torch.float32),
    }
    return payload


def save_gaussian_anomaly_artifacts(
    frame_dir: Path,
    gt_rgb: np.ndarray,
    anomaly_map: torch.Tensor,
    gaussian_data: Dict[str, torch.Tensor],
    valid_mask: Optional[torch.Tensor] = None,
    topk: int = 64,
    radius_scale: float = 1.5,
    min_valid_pixels: int = 4,
    use_distance_weight: bool = True,
    anomaly_threshold: float = 0.35,
    min_component_pixels: int = 16,
    component_dilation: int = 2,
    use_adaptive_threshold: bool = True,
    enable_threshold_fallback: bool = True,
    anomaly_threshold_floor: float = 0.05,
    anomaly_threshold_primary_quantile: float = 0.95,
    anomaly_threshold_fallback_quantile: float = 0.85,
) -> Dict[str, object]:
    frame_dir = Path(frame_dir)
    scores_dir = frame_dir / "gaussian_anomaly"
    scores_dir.mkdir(parents=True, exist_ok=True)

    anomaly_map = anomaly_map.detach().cpu().float().clamp(0.0, 1.0)
    anomaly_uint8 = (anomaly_map.numpy() * 255.0).astype(np.uint8)
    valid_mask_hw = _to_hw_bool(valid_mask, anomaly_map.shape[0], anomaly_map.shape[1])
    masked_anomaly_uint8 = anomaly_uint8.copy()
    masked_anomaly_uint8[~valid_mask_hw.numpy()] = 0
    Image.fromarray(masked_anomaly_uint8).save(scores_dir / "anomaly_masked.png")

    payload = compute_gaussian_anomaly_scores(
        anomaly_map=anomaly_map,
        gaussian_data=gaussian_data,
        valid_mask=valid_mask_hw,
        topk=topk,
        radius_scale=radius_scale,
        min_valid_pixels=min_valid_pixels,
        use_distance_weight=use_distance_weight,
        anomaly_threshold=anomaly_threshold,
        min_component_pixels=min_component_pixels,
        component_dilation=component_dilation,
        use_adaptive_threshold=use_adaptive_threshold,
        enable_threshold_fallback=enable_threshold_fallback,
        anomaly_threshold_floor=anomaly_threshold_floor,
        anomaly_threshold_primary_quantile=anomaly_threshold_primary_quantile,
        anomaly_threshold_fallback_quantile=anomaly_threshold_fallback_quantile,
    )

    torch.save(
        {
            "gaussian_ids": payload["gaussian_ids"],
            "rank_score": payload["rank_score"],
            "mean_score": payload["mean_score"],
            "mass_score": payload["mass_score"],
            "peak_score": payload["peak_score"],
            "valid_pixels": payload["valid_pixels"],
            "support_pixels": payload["support_pixels"],
            "footprint_pixels": payload["footprint_pixels"],
            "valid_ratio": payload["valid_ratio"],
            "center_x": payload["center_x"],
            "center_y": payload["center_y"],
            "raw_radius": payload["raw_radius"],
            "effective_radius": payload["effective_radius"],
            "depth": payload["depth"],
            "timestep": payload["timestep"],
        },
        scores_dir / "gaussian_anomaly_scores.pt",
    )

    csv_path = scores_dir / "gaussian_anomaly_scores.csv"
    with open(csv_path, "w", newline="", encoding="ascii") as csv_file:
        fieldnames = [
            "gaussian_id",
            "rank_score",
            "mean_score",
            "mass_score",
            "peak_score",
            "valid_pixels",
            "support_pixels",
            "footprint_pixels",
            "valid_ratio",
            "center_x",
            "center_y",
            "raw_radius",
            "effective_radius",
            "depth",
            "timestep",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["records"])

    summary = {
        "num_scored_gaussians": len(payload["records"]),
        "candidate_gaussian_count": payload["candidate_gaussian_count"],
        "num_anomaly_components": len(payload["components"]),
        "topk": topk,
        "radius_scale": radius_scale,
        "min_valid_pixels": min_valid_pixels,
        "use_distance_weight": use_distance_weight,
        "anomaly_threshold": anomaly_threshold,
        "use_adaptive_threshold": use_adaptive_threshold,
        "enable_threshold_fallback": enable_threshold_fallback,
        "used_threshold": payload["used_threshold"],
        "primary_threshold": payload["primary_threshold"],
        "fallback_threshold": payload["fallback_threshold"],
        "threshold_mode": payload["threshold_mode"],
        "max_anomaly_value": payload["max_anomaly_value"],
        "min_component_pixels": min_component_pixels,
        "component_dilation": component_dilation,
        "components": payload["components"],
        "top_records": payload["top_records"],
    }
    with open(scores_dir / "gaussian_anomaly_summary.json", "w", encoding="ascii") as json_file:
        json.dump(summary, json_file, indent=2)
    summary["scored_gaussian_ids"] = [record["gaussian_id"] for record in payload["records"]]
    summary["top_gaussian_ids"] = [record["gaussian_id"] for record in payload["top_records"]]
    return summary

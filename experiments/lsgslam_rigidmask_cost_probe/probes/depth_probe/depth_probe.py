import json
import os
from pathlib import Path
from typing import Dict, Optional

import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch

from utils.pnp_fused_icp_utils import (
    lidar_file_for_frame,
    load_velo_to_cam,
    read_velodyne_bin,
)


SUPPORTED_LIDAR_DATASETS = {"kitti", "kitti360"}


def _ensure_chw_depth(depth: torch.Tensor) -> torch.Tensor:
    tensor = depth.detach().float().cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3 or tensor.shape[0] != 1:
        raise ValueError(f"Expected depth tensor with shape (1, H, W), got {tuple(tensor.shape)}")
    return tensor


def _ensure_bool_mask(mask: Optional[torch.Tensor], height: int, width: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    tensor = mask.detach().cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.shape != (1, height, width):
        raise ValueError(f"Expected mask shape (1, {height}, {width}), got {tuple(tensor.shape)}")
    return tensor.bool()


def _intrinsics_to_numpy(intrinsics: torch.Tensor) -> np.ndarray:
    tensor = intrinsics.detach().float().cpu()
    if tensor.shape == (4, 4):
        tensor = tensor[:3, :3]
    if tensor.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3) or (4, 4), got {tuple(tensor.shape)}")
    return tensor.numpy()


def _project_lidar_to_depth_map(
    dataset,
    frame_id: str,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
    project_root: str,
) -> Dict[str, object]:
    dataset_name = str(getattr(dataset, "name", "")).lower()
    if dataset_name not in SUPPORTED_LIDAR_DATASETS:
        raise ValueError(f"LiDAR depth probe is unsupported for dataset '{dataset_name}'")

    lidar_path = lidar_file_for_frame(dataset, frame_id)
    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(f"Missing LiDAR file for frame {frame_id}: {lidar_path}")

    xyz_velo = read_velodyne_bin(lidar_path)
    velo_to_cam = load_velo_to_cam(dataset, project_root)

    xyz_h = np.concatenate(
        [xyz_velo.astype(np.float64), np.ones((xyz_velo.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    xyz_cam = (velo_to_cam @ xyz_h.T).T[:, :3]

    z = xyz_cam[:, 2]
    positive_mask = z > 1e-6
    xyz_cam = xyz_cam[positive_mask]
    z = z[positive_mask]

    if xyz_cam.shape[0] == 0:
        depth_map = np.zeros((image_height, image_width), dtype=np.float32)
        return {
            "lidar_depth": torch.from_numpy(depth_map).unsqueeze(0),
            "lidar_path": lidar_path,
            "num_projected_points": 0,
            "row_min": None,
            "row_max": None,
        }

    intrinsics_np = _intrinsics_to_numpy(intrinsics)
    fx = float(intrinsics_np[0, 0])
    fy = float(intrinsics_np[1, 1])
    cx = float(intrinsics_np[0, 2])
    cy = float(intrinsics_np[1, 2])

    u = np.rint((fx * xyz_cam[:, 0] / z) + cx).astype(np.int64)
    v = np.rint((fy * xyz_cam[:, 1] / z) + cy).astype(np.int64)
    inside = (
        (u >= 0)
        & (u < image_width)
        & (v >= 0)
        & (v < image_height)
    )
    u = u[inside]
    v = v[inside]
    z = z[inside].astype(np.float32)

    depth_map = np.full((image_height, image_width), np.inf, dtype=np.float32)
    if z.size > 0:
        np.minimum.at(depth_map, (v, u), z)
    depth_map[~np.isfinite(depth_map)] = 0.0

    return {
        "lidar_depth": torch.from_numpy(depth_map).unsqueeze(0),
        "lidar_path": lidar_path,
        "num_projected_points": int(z.size),
        "row_min": int(v.min()) if z.size > 0 else None,
        "row_max": int(v.max()) if z.size > 0 else None,
    }


def _build_lidar_row_band_mask(
    image_height: int,
    image_width: int,
    row_min: Optional[int],
    row_max: Optional[int],
    margin: int,
) -> torch.Tensor:
    mask = torch.zeros((1, image_height, image_width), dtype=torch.bool)
    if row_min is None or row_max is None:
        return mask
    start = max(0, int(row_min) - int(margin))
    end = min(image_height - 1, int(row_max) + int(margin))
    if end < start:
        return mask
    mask[:, start : end + 1, :] = True
    return mask


def _compute_auto_vis_max(values: torch.Tensor, percentile: float, min_value: float) -> float:
    valid = values[torch.isfinite(values)]
    if valid.numel() == 0:
        return min_value
    vis_max = float(torch.quantile(valid, percentile).item())
    return max(vis_max, min_value)


def _colorize_depth(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    vis_max: float,
    cmap_name: str,
) -> np.ndarray:
    array = values.squeeze(0).detach().cpu().numpy()
    mask = valid_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    normalized = np.clip(array / max(vis_max, 1e-6), 0.0, 1.0)
    rgb = cm.get_cmap(cmap_name)(normalized)[..., :3]
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def _colorize_signed_diff(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    vis_max: float,
    cmap_name: str,
) -> np.ndarray:
    array = values.squeeze(0).detach().cpu().numpy()
    mask = valid_mask.squeeze(0).detach().cpu().numpy().astype(bool)
    normalized = np.clip((array / max(vis_max, 1e-6) + 1.0) * 0.5, 0.0, 1.0)
    rgb = cm.get_cmap(cmap_name)(normalized)[..., :3]
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path)


def save_depth_probe_artifacts(
    output_root: Optional[str],
    frame_name: str,
    dataset,
    frame_id: str,
    intrinsics: torch.Tensor,
    render_depth: torch.Tensor,
    fallback_depth: torch.Tensor,
    project_root: str,
    sky_mask: Optional[torch.Tensor] = None,
    enabled: bool = True,
    save_visualizations: bool = True,
    save_raw_tensors: bool = True,
    mask_sky: bool = True,
    fallback_only_within_lidar_rows: bool = True,
    lidar_row_band_margin: int = 0,
    max_depth_m: Optional[float] = None,
    depth_vis_max: Optional[float] = None,
    abs_diff_vis_max: Optional[float] = 5.0,
    signed_diff_vis_max: Optional[float] = 5.0,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "enabled": bool(enabled),
        "frame_name": frame_name,
        "frame_id": frame_id,
        "dataset": str(getattr(dataset, "name", "")),
    }
    if not enabled:
        summary["status"] = "disabled"
        return summary

    if output_root is None:
        raise ValueError("depth_probe is enabled but output_root is None.")

    frame_dir = Path(output_root) / frame_name
    frame_dir.mkdir(parents=True, exist_ok=True)

    render_depth_t = _ensure_chw_depth(render_depth)
    fallback_depth_t = _ensure_chw_depth(fallback_depth)
    height, width = render_depth_t.shape[1:]
    if fallback_depth_t.shape[1:] != (height, width):
        raise ValueError(
            "Render depth and fallback depth must share the same spatial size, "
            f"got {(height, width)} and {tuple(fallback_depth_t.shape[1:])}"
        )

    sky_mask_t = _ensure_bool_mask(sky_mask, height, width)

    try:
        lidar_payload = _project_lidar_to_depth_map(
            dataset=dataset,
            frame_id=frame_id,
            intrinsics=intrinsics,
            image_height=height,
            image_width=width,
            project_root=project_root,
        )
    except Exception as exc:
        summary["status"] = "skipped"
        summary["reason"] = str(exc)
        with open(frame_dir / "depth_probe_summary.json", "w", encoding="ascii") as f:
            json.dump(summary, f, indent=2)
        return summary

    lidar_depth_t = _ensure_chw_depth(lidar_payload["lidar_depth"])
    lidar_coverage_mask = lidar_depth_t > 0
    lidar_row_band_mask = _build_lidar_row_band_mask(
        image_height=height,
        image_width=width,
        row_min=lidar_payload.get("row_min"),
        row_max=lidar_payload.get("row_max"),
        margin=lidar_row_band_margin,
    )
    if fallback_only_within_lidar_rows:
        fallback_allowed_mask = lidar_row_band_mask
    else:
        fallback_allowed_mask = torch.ones_like(lidar_coverage_mask, dtype=torch.bool)
    fallback_depth_limited_t = torch.where(
        fallback_allowed_mask,
        fallback_depth_t,
        torch.zeros_like(fallback_depth_t),
    )
    fused_depth_t = torch.where(lidar_coverage_mask, lidar_depth_t, fallback_depth_limited_t)
    valid_mask = (fused_depth_t > 0) & torch.isfinite(fused_depth_t) & torch.isfinite(render_depth_t)
    depth_range_mask = torch.ones_like(valid_mask, dtype=torch.bool)
    if max_depth_m is not None:
        depth_range_mask = fused_depth_t <= float(max_depth_m)
        valid_mask = valid_mask & depth_range_mask
    if mask_sky and sky_mask_t is not None:
        valid_mask = valid_mask & (~sky_mask_t)

    abs_diff_t = torch.abs(render_depth_t - fused_depth_t)
    signed_diff_t = render_depth_t - fused_depth_t

    valid_abs_diff = abs_diff_t[valid_mask]
    valid_signed_diff = signed_diff_t[valid_mask]
    sky_kept_ratio = None
    if sky_mask_t is not None:
        sky_kept_ratio = float((~sky_mask_t).float().mean().item())

    summary.update(
        {
            "status": "ok",
            "lidar_path": str(lidar_payload["lidar_path"]),
            "num_projected_points": int(lidar_payload["num_projected_points"]),
            "lidar_coverage_ratio": float(lidar_coverage_mask.float().mean().item()),
            "lidar_row_band_ratio": float(lidar_row_band_mask.float().mean().item()),
            "valid_ratio": float(valid_mask.float().mean().item()),
            "mask_sky": bool(mask_sky),
            "fallback_only_within_lidar_rows": bool(fallback_only_within_lidar_rows),
            "lidar_row_band_margin": int(lidar_row_band_margin),
            "max_depth_m": None if max_depth_m is None else float(max_depth_m),
            "lidar_row_min": lidar_payload.get("row_min"),
            "lidar_row_max": lidar_payload.get("row_max"),
            "sky_keep_ratio": sky_kept_ratio,
            "abs_diff_mean": float(valid_abs_diff.mean().item()) if valid_abs_diff.numel() > 0 else None,
            "abs_diff_median": float(valid_abs_diff.median().item()) if valid_abs_diff.numel() > 0 else None,
            "abs_diff_max": float(valid_abs_diff.max().item()) if valid_abs_diff.numel() > 0 else None,
            "signed_diff_mean": float(valid_signed_diff.mean().item()) if valid_signed_diff.numel() > 0 else None,
        }
    )

    if save_raw_tensors:
        torch.save(
            {
                "render_depth": render_depth_t,
                "lidar_depth": lidar_depth_t,
                "fallback_depth_limited": fallback_depth_limited_t,
                "fused_depth": fused_depth_t,
                "valid_mask": valid_mask,
                "depth_range_mask": depth_range_mask,
                "lidar_coverage_mask": lidar_coverage_mask,
                "lidar_row_band_mask": lidar_row_band_mask,
                "abs_diff": abs_diff_t,
                "signed_diff": signed_diff_t,
            },
            frame_dir / "depth_probe_tensors.pt",
        )

    if save_visualizations:
        lidar_valid_mask = lidar_coverage_mask.clone()
        if max_depth_m is not None:
            lidar_valid_mask = lidar_valid_mask & (lidar_depth_t <= float(max_depth_m))
        if mask_sky and sky_mask_t is not None:
            lidar_valid_mask = lidar_valid_mask & (~sky_mask_t)

        resolved_depth_vis_max = depth_vis_max
        if resolved_depth_vis_max is None:
            resolved_depth_vis_max = _compute_auto_vis_max(fused_depth_t[valid_mask], 0.99, 1.0)
        resolved_abs_vis_max = abs_diff_vis_max
        if resolved_abs_vis_max is None:
            resolved_abs_vis_max = _compute_auto_vis_max(valid_abs_diff, 0.99, 0.5)
        resolved_signed_vis_max = signed_diff_vis_max
        if resolved_signed_vis_max is None:
            resolved_signed_vis_max = _compute_auto_vis_max(valid_abs_diff, 0.99, 0.5)

        summary["depth_vis_max"] = float(resolved_depth_vis_max)
        summary["abs_diff_vis_max"] = float(resolved_abs_vis_max)
        summary["signed_diff_vis_max"] = float(resolved_signed_vis_max)

        lidar_vis = _colorize_depth(
            lidar_depth_t,
            lidar_valid_mask,
            float(resolved_depth_vis_max),
            "turbo",
        )
        fused_vis = _colorize_depth(
            fused_depth_t,
            valid_mask,
            float(resolved_depth_vis_max),
            "turbo",
        )
        render_vis = _colorize_depth(
            render_depth_t,
            valid_mask,
            float(resolved_depth_vis_max),
            "turbo",
        )
        abs_diff_vis = _colorize_depth(
            abs_diff_t,
            valid_mask,
            float(resolved_abs_vis_max),
            "inferno",
        )
        signed_diff_vis = _colorize_signed_diff(
            signed_diff_t,
            valid_mask,
            float(resolved_signed_vis_max),
            "bwr",
        )

        _save_image(frame_dir / "lidar_depth.png", lidar_vis)
        _save_image(frame_dir / "fused_gt_depth.png", fused_vis)
        _save_image(frame_dir / "render_depth.png", render_vis)
        _save_image(frame_dir / "depth_diff_abs.png", abs_diff_vis)
        _save_image(frame_dir / "depth_diff_signed.png", signed_diff_vis)
        _save_image(
            frame_dir / "lidar_coverage_mask.png",
            (lidar_valid_mask.squeeze(0).numpy().astype(np.uint8) * 255),
        )
        _save_image(
            frame_dir / "lidar_row_band_mask.png",
            (lidar_row_band_mask.squeeze(0).numpy().astype(np.uint8) * 255),
        )
        _save_image(
            frame_dir / "depth_valid_mask.png",
            (valid_mask.squeeze(0).numpy().astype(np.uint8) * 255),
        )
        _save_image(
            frame_dir / "depth_range_mask.png",
            (depth_range_mask.squeeze(0).numpy().astype(np.uint8) * 255),
        )

    with open(frame_dir / "depth_probe_summary.json", "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2)
    return summary

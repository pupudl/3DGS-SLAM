from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Sequence

import cv2
import numpy as np
import open3d as o3d


@dataclass
class DynamicStepInfo:
    frame_idx: int
    frame_id: str
    pose_source: str
    points_total: int
    matched_points: int
    unmatched_points: int
    dynamic_count: int
    dynamic_ratio: float
    mean_residual_m: float
    median_residual_m: float


def _ensure_numpy_depth(depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 3:
        return depth[0]
    return depth


def build_dense_depth_observations(
    depth: np.ndarray,
    k_mat: np.ndarray,
    depth_near: float,
    depth_far: float,
    pixel_stride: int = 1,
) -> Dict[str, np.ndarray]:
    depth = _ensure_numpy_depth(depth).astype(np.float64)
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    valid = (depth > max(0.1, depth_near)) & (depth < max(30.0, depth_far))
    if pixel_stride > 1:
        valid &= ((xs % pixel_stride) == 0) & ((ys % pixel_stride) == 0)

    xs_valid = xs[valid].astype(np.float64)
    ys_valid = ys[valid].astype(np.float64)
    depths_valid = depth[valid]
    uv1 = np.stack([xs_valid, ys_valid, np.ones_like(xs_valid)], axis=0)
    points_cam = (np.linalg.inv(k_mat) @ uv1) * depths_valid[None, :]

    return {
        "points_cam": points_cam.T.astype(np.float64),
        "uv": np.stack([xs_valid, ys_valid], axis=1),
        "depth": depths_valid,
        "image_shape": np.array([h, w], dtype=np.int32),
    }


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ pts_h.T).T
    return transformed[:, :3]


def project_points(points_cam: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-8, None)
    uv = (k_mat @ points_cam.T).T
    return uv[:, :2] / z[:, None]


def _make_pointcloud(points_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    if points_xyz.shape[0] >= 10:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
    return pcd


def _window_has_valid_depth(depth_map: np.ndarray, uv: np.ndarray, search_radius_px: int) -> np.ndarray:
    h, w = depth_map.shape
    x0 = np.rint(uv[:, 0]).astype(np.int32)
    y0 = np.rint(uv[:, 1]).astype(np.int32)
    found = np.zeros((uv.shape[0],), dtype=bool)
    for dy in range(-search_radius_px, search_radius_px + 1):
        ys = y0 + dy
        y_valid = (ys >= 0) & (ys < h)
        if not np.any(y_valid):
            continue
        for dx in range(-search_radius_px, search_radius_px + 1):
            xs = x0 + dx
            valid = y_valid & (xs >= 0) & (xs < w)
            if not np.any(valid):
                continue
            sampled = np.zeros((uv.shape[0],), dtype=np.float64)
            sampled[valid] = depth_map[ys[valid], xs[valid]]
            found |= valid & np.isfinite(sampled) & (sampled > 0.1)
    return found


def run_icp_dynamic_residuals(
    source_obs: Dict[str, np.ndarray],
    target_obs: Dict[str, np.ndarray],
    init_transform: np.ndarray,
    k_mat: np.ndarray,
    curr_depth: np.ndarray,
    corr_threshold: float,
    residual_thresh_m: float,
    unmatched_penalty: float,
    search_radius_px: int,
) -> Dict[str, np.ndarray | float]:
    source_points = source_obs["points_cam"]
    target_points = target_obs["points_cam"]
    curr_depth = _ensure_numpy_depth(curr_depth).astype(np.float64)

    source_pcd = _make_pointcloud(source_points)
    target_pcd = _make_pointcloud(target_points)

    reg = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        float(corr_threshold),
        init_transform.astype(np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-4,
            max_iteration=100,
        ),
    )

    corr = np.asarray(reg.correspondence_set, dtype=np.int64)
    residual_transform = reg.transformation
    transformed_by_pose = transform_points(source_points, residual_transform)
    transformed_uv = project_points(transformed_by_pose, k_mat)

    matched_source_idx = corr[:, 0] if corr.size > 0 else np.zeros((0,), dtype=np.int64)
    matched_target_idx = corr[:, 1] if corr.size > 0 else np.zeros((0,), dtype=np.int64)

    matched_mask_full = np.zeros((source_points.shape[0],), dtype=bool)
    matched_mask_full[matched_source_idx] = True

    matched_last_uv = source_obs["uv"][matched_source_idx]
    matched_curr_uv = target_obs["uv"][matched_target_idx]
    matched_residuals = (
        np.linalg.norm(
            transformed_by_pose[matched_source_idx] - target_points[matched_target_idx],
            axis=1,
        )
        if matched_source_idx.size > 0
        else np.zeros((0,), dtype=np.float64)
    )
    matched_dynamic = matched_residuals > residual_thresh_m
    matched_scores = matched_residuals / max(residual_thresh_m, 1e-6)

    unmatched_idx = np.where(~matched_mask_full)[0]
    unmatched_in_frame = np.zeros((unmatched_idx.shape[0],), dtype=bool)
    unmatched_curr_uv = np.zeros((unmatched_idx.shape[0], 2), dtype=np.float64)
    if unmatched_idx.size > 0:
        unmatched_points = transformed_by_pose[unmatched_idx]
        unmatched_curr_uv = transformed_uv[unmatched_idx]
        in_frame = (
            (unmatched_curr_uv[:, 0] >= 0.0)
            & (unmatched_curr_uv[:, 0] < curr_depth.shape[1] - 1)
            & (unmatched_curr_uv[:, 1] >= 0.0)
            & (unmatched_curr_uv[:, 1] < curr_depth.shape[0] - 1)
            & (unmatched_points[:, 2] > 0.1)
        )
        if np.any(in_frame):
            valid_overlap = _window_has_valid_depth(curr_depth, unmatched_curr_uv[in_frame], search_radius_px)
            unmatched_in_frame[in_frame] = valid_overlap

    unmatched_keep_idx = unmatched_idx[unmatched_in_frame]
    unmatched_last_uv = source_obs["uv"][unmatched_keep_idx]
    unmatched_curr_uv = transformed_uv[unmatched_keep_idx]
    unmatched_residuals = np.full((unmatched_keep_idx.shape[0],), np.nan, dtype=np.float64)
    unmatched_dynamic = np.ones((unmatched_keep_idx.shape[0],), dtype=bool)
    unmatched_scores = np.full((unmatched_keep_idx.shape[0],), float(unmatched_penalty), dtype=np.float64)

    last_uv = np.concatenate([matched_last_uv, unmatched_last_uv], axis=0)
    curr_uv = np.concatenate([matched_curr_uv, unmatched_curr_uv], axis=0)
    residuals = np.concatenate([matched_residuals, unmatched_residuals], axis=0)
    dynamic_mask = np.concatenate([matched_dynamic, unmatched_dynamic], axis=0)
    static_mask = ~dynamic_mask
    scores = np.concatenate([matched_scores, unmatched_scores], axis=0)
    is_matched = np.concatenate(
        [
            np.ones((matched_last_uv.shape[0],), dtype=bool),
            np.zeros((unmatched_last_uv.shape[0],), dtype=bool),
        ],
        axis=0,
    )

    return {
        "last_uv": last_uv,
        "curr_uv": curr_uv,
        "residuals": residuals,
        "dynamic_mask": dynamic_mask,
        "static_mask": static_mask,
        "scores": scores,
        "is_matched": is_matched,
        "matched_points": int(matched_last_uv.shape[0]),
        "unmatched_points": int(unmatched_last_uv.shape[0]),
        "icp_fitness": float(reg.fitness),
        "icp_rmse": float(reg.inlier_rmse),
        "icp_transform": reg.transformation,
    }


def filter_points_in_image(
    points_cam: np.ndarray,
    uv: np.ndarray,
    image_shape: np.ndarray,
    min_depth: float = 0.1,
) -> np.ndarray:
    h = int(image_shape[0])
    w = int(image_shape[1])
    return (
        np.isfinite(points_cam[:, 2])
        & (points_cam[:, 2] > min_depth)
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < w - 1)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < h - 1)
    )


def build_lidar_observations(
    points_cam: np.ndarray,
    k_mat: np.ndarray,
    image_shape: np.ndarray,
) -> Dict[str, np.ndarray]:
    uv = project_points(points_cam, k_mat)
    valid = filter_points_in_image(points_cam, uv, image_shape)
    return {
        "points_cam": points_cam[valid].astype(np.float64),
        "uv": uv[valid].astype(np.float64),
        "image_shape": np.asarray(image_shape, dtype=np.int32),
    }


def run_lidar_icp_dynamic_residuals(
    source_obs: Dict[str, np.ndarray],
    target_obs: Dict[str, np.ndarray],
    init_transform: np.ndarray,
    k_mat: np.ndarray,
    curr_image_shape: np.ndarray,
    corr_threshold: float,
    residual_thresh_m: float,
    unmatched_penalty: float,
) -> Dict[str, np.ndarray | float]:
    source_points = source_obs["points_cam"]
    target_points = target_obs["points_cam"]

    source_pcd = _make_pointcloud(source_points)
    target_pcd = _make_pointcloud(target_points)

    reg = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        float(corr_threshold),
        init_transform.astype(np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-4,
            max_iteration=100,
        ),
    )

    corr = np.asarray(reg.correspondence_set, dtype=np.int64)
    residual_transform = reg.transformation
    transformed_by_pose = transform_points(source_points, residual_transform)
    transformed_uv = project_points(transformed_by_pose, k_mat)

    matched_source_idx = corr[:, 0] if corr.size > 0 else np.zeros((0,), dtype=np.int64)
    matched_target_idx = corr[:, 1] if corr.size > 0 else np.zeros((0,), dtype=np.int64)

    matched_mask_full = np.zeros((source_points.shape[0],), dtype=bool)
    matched_mask_full[matched_source_idx] = True

    matched_last_uv = source_obs["uv"][matched_source_idx]
    matched_curr_uv = target_obs["uv"][matched_target_idx]
    matched_residuals = (
        np.linalg.norm(
            transformed_by_pose[matched_source_idx] - target_points[matched_target_idx],
            axis=1,
        )
        if matched_source_idx.size > 0
        else np.zeros((0,), dtype=np.float64)
    )
    matched_dynamic = matched_residuals > residual_thresh_m
    matched_scores = matched_residuals / max(residual_thresh_m, 1e-6)

    unmatched_idx = np.where(~matched_mask_full)[0]
    unmatched_last_uv = np.zeros((0, 2), dtype=np.float64)
    unmatched_curr_uv = np.zeros((0, 2), dtype=np.float64)
    unmatched_residuals = np.zeros((0,), dtype=np.float64)
    unmatched_dynamic = np.zeros((0,), dtype=bool)
    unmatched_scores = np.zeros((0,), dtype=np.float64)
    if unmatched_idx.size > 0:
        unmatched_points = transformed_by_pose[unmatched_idx]
        unmatched_uv_proj = transformed_uv[unmatched_idx]
        visible = filter_points_in_image(unmatched_points, unmatched_uv_proj, curr_image_shape)
        keep_idx = unmatched_idx[visible]
        if keep_idx.size > 0:
            unmatched_last_uv = source_obs["uv"][keep_idx]
            unmatched_curr_uv = transformed_uv[keep_idx]
            unmatched_residuals = np.full((keep_idx.shape[0],), np.nan, dtype=np.float64)
            unmatched_dynamic = np.ones((keep_idx.shape[0],), dtype=bool)
            unmatched_scores = np.full((keep_idx.shape[0],), float(unmatched_penalty), dtype=np.float64)

    last_uv = np.concatenate([matched_last_uv, unmatched_last_uv], axis=0)
    curr_uv = np.concatenate([matched_curr_uv, unmatched_curr_uv], axis=0)
    residuals = np.concatenate([matched_residuals, unmatched_residuals], axis=0)
    dynamic_mask = np.concatenate([matched_dynamic, unmatched_dynamic], axis=0)
    static_mask = ~dynamic_mask
    scores = np.concatenate([matched_scores, unmatched_scores], axis=0)
    is_matched = np.concatenate(
        [
            np.ones((matched_last_uv.shape[0],), dtype=bool),
            np.zeros((unmatched_last_uv.shape[0],), dtype=bool),
        ],
        axis=0,
    )

    return {
        "last_uv": last_uv,
        "curr_uv": curr_uv,
        "residuals": residuals,
        "dynamic_mask": dynamic_mask,
        "static_mask": static_mask,
        "scores": scores,
        "is_matched": is_matched,
        "matched_points": int(matched_last_uv.shape[0]),
        "unmatched_points": int(unmatched_last_uv.shape[0]),
        "icp_fitness": float(reg.fitness),
        "icp_rmse": float(reg.inlier_rmse),
        "icp_transform": reg.transformation,
    }

def write_step_csv(path: str, steps: Sequence[DynamicStepInfo]) -> None:
    import csv

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DynamicStepInfo.__dataclass_fields__.keys()))
        writer.writeheader()
        for step in steps:
            writer.writerow(step.__dict__)


def _paint_mask(mask: np.ndarray, points: np.ndarray, radius: int) -> np.ndarray:
    painted = mask.copy()
    for uv in points:
        x = int(round(uv[0]))
        y = int(round(uv[1]))
        if x < 0 or y < 0 or x >= painted.shape[1] or y >= painted.shape[0]:
            continue
        cv2.circle(painted, (x, y), radius, 255, -1, lineType=cv2.LINE_AA)
    return painted


def draw_dynamic_overlay(
    image: np.ndarray,
    dynamic_points: np.ndarray,
    static_points: np.ndarray,
    output_path: str,
    alpha: float = 0.35,
    radius: int = 2,
) -> None:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    dynamic_mask = np.zeros((h, w), dtype=np.uint8)
    static_mask = np.zeros((h, w), dtype=np.uint8)
    static_mask = _paint_mask(static_mask, static_points, radius)
    dynamic_mask = _paint_mask(dynamic_mask, dynamic_points, radius)
    static_mask[dynamic_mask > 0] = 0

    overlay = canvas.copy()
    overlay[static_mask > 0] = (0, 255, 0)
    overlay[dynamic_mask > 0] = (255, 0, 0)
    blended = cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0)
    mask = (static_mask > 0) | (dynamic_mask > 0)
    canvas[mask] = blended[mask]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _normalize_depth_for_vis(
    depth: np.ndarray,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    depth = _ensure_numpy_depth(depth).astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0.1)
    norm = np.zeros(depth.shape, dtype=np.uint8)
    if max_depth <= min_depth:
        max_depth = min_depth + 1e-3
    clipped = np.clip((depth - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    norm[valid] = np.round(clipped[valid] * 255.0).astype(np.uint8)
    return norm


def draw_paired_depth_visualization(
    prev_depth: np.ndarray,
    curr_depth: np.ndarray,
    output_path: str,
    prev_label: str = "Prev Depth",
    curr_label: str = "Curr Depth",
) -> None:
    prev_depth = _ensure_numpy_depth(prev_depth)
    curr_depth = _ensure_numpy_depth(curr_depth)

    valid_prev = np.isfinite(prev_depth) & (prev_depth > 0.1)
    valid_curr = np.isfinite(curr_depth) & (curr_depth > 0.1)
    all_valid_depths = []
    if np.any(valid_prev):
        all_valid_depths.append(prev_depth[valid_prev])
    if np.any(valid_curr):
        all_valid_depths.append(curr_depth[valid_curr])

    if all_valid_depths:
        merged = np.concatenate(all_valid_depths, axis=0).astype(np.float32)
        min_depth = float(np.percentile(merged, 2.0))
        max_depth = float(np.percentile(merged, 98.0))
    else:
        min_depth = 0.0
        max_depth = 1.0

    prev_norm = _normalize_depth_for_vis(prev_depth, min_depth, max_depth)
    curr_norm = _normalize_depth_for_vis(curr_depth, min_depth, max_depth)
    prev_color = cv2.applyColorMap(prev_norm, cv2.COLORMAP_TURBO)
    curr_color = cv2.applyColorMap(curr_norm, cv2.COLORMAP_TURBO)
    prev_color[~valid_prev] = 0
    curr_color[~valid_curr] = 0

    panel_h, panel_w = prev_color.shape[:2]
    title_band_h = 34
    separator_w = 12
    canvas = np.zeros((panel_h + title_band_h, panel_w * 2 + separator_w, 3), dtype=np.uint8)
    canvas[title_band_h:, :panel_w] = prev_color
    canvas[title_band_h:, panel_w + separator_w:] = curr_color

    text_color = (240, 240, 240)
    meta_color = (180, 180, 180)
    cv2.putText(canvas, prev_label, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        curr_label,
        (panel_w + separator_w + 12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        2,
        cv2.LINE_AA,
    )
    meta_text = f"shared range: {min_depth:.2f}m - {max_depth:.2f}m"
    cv2.putText(canvas, meta_text, (12, panel_h + title_band_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, meta_color, 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)

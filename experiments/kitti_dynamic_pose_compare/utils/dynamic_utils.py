from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class DynamicStepInfo:
    frame_idx: int
    frame_id: str
    pose_source: str
    matches: int
    pnp_inliers: int
    pose_inlier_count: int
    dynamic_count: int
    dynamic_ratio: float
    mean_reproj_px: float
    mean_depth_err_m: float


def project_points(points_cam: np.ndarray, k_mat: np.ndarray) -> np.ndarray:
    z = np.clip(points_cam[:, 2], 1e-8, None)
    uv = (k_mat @ points_cam.T).T
    uv = uv[:, :2] / z[:, None]
    return uv


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ pts_h.T).T
    return transformed[:, :3]


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    xs = np.rint(uv[:, 0]).astype(np.int32)
    ys = np.rint(uv[:, 1]).astype(np.int32)
    values = np.full((uv.shape[0],), np.nan, dtype=np.float64)
    valid = (
        (xs >= 0)
        & (xs < depth.shape[1])
        & (ys >= 0)
        & (ys < depth.shape[0])
    )
    values[valid] = depth[ys[valid], xs[valid]]
    return values


def compute_dynamic_residuals(
    points_last_cam: np.ndarray,
    uv_curr: np.ndarray,
    curr_depth: np.ndarray,
    k_mat: np.ndarray,
    transform: np.ndarray,
    reproj_thresh_px: float,
    depth_thresh_m: float,
    depth_weight: float,
) -> Dict[str, np.ndarray]:
    points_curr_cam = transform_points(points_last_cam, transform)
    proj_uv = project_points(points_curr_cam, k_mat)
    reproj_err = np.linalg.norm(proj_uv - uv_curr, axis=1)

    sampled_depth = sample_depth(curr_depth, uv_curr)
    valid_depth = np.isfinite(sampled_depth) & (sampled_depth > 0.1)
    depth_err = np.full_like(reproj_err, np.nan, dtype=np.float64)
    depth_err[valid_depth] = np.abs(sampled_depth[valid_depth] - points_curr_cam[valid_depth, 2])

    dynamic_mask = reproj_err > reproj_thresh_px
    if np.any(valid_depth):
        dynamic_mask = dynamic_mask | (valid_depth & (depth_err > depth_thresh_m))

    score = reproj_err / max(reproj_thresh_px, 1e-6)
    if np.any(valid_depth):
        score[valid_depth] += depth_weight * (depth_err[valid_depth] / max(depth_thresh_m, 1e-6))

    in_frame = (
        (proj_uv[:, 0] >= 0.0)
        & (proj_uv[:, 0] < curr_depth.shape[1] - 1)
        & (proj_uv[:, 1] >= 0.0)
        & (proj_uv[:, 1] < curr_depth.shape[0] - 1)
        & (points_curr_cam[:, 2] > 0.1)
    )
    return {
        "proj_uv": proj_uv,
        "reproj_err": reproj_err,
        "depth_err": depth_err,
        "dynamic_mask": dynamic_mask & in_frame,
        "score": score,
        "in_frame": in_frame,
        "valid_depth": valid_depth,
    }


def refine_pose_from_static_points(
    points_last_cam: np.ndarray,
    uv_curr: np.ndarray,
    k_mat: np.ndarray,
    init_transform: np.ndarray,
    reproj_filter_px: float,
    min_points: int,
) -> Tuple[np.ndarray, int]:
    init_points_curr = transform_points(points_last_cam, init_transform)
    init_proj = project_points(init_points_curr, k_mat)
    reproj = np.linalg.norm(init_proj - uv_curr, axis=1)
    keep = np.where(reproj < reproj_filter_px)[0]
    if keep.shape[0] < min_points:
        return init_transform, int(keep.shape[0])

    rvec_init, _ = cv2.Rodrigues(init_transform[:3, :3])
    tvec_init = init_transform[:3, 3:4].copy()
    try:
        ok, rvec, tvec = cv2.solvePnP(
            points_last_cam[keep].astype(np.float64),
            uv_curr[keep].reshape(-1, 1, 2).astype(np.float64),
            k_mat.astype(np.float64),
            np.zeros((5, 1), dtype=np.float64),
            rvec=rvec_init,
            tvec=tvec_init,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return init_transform, int(keep.shape[0])
    if not ok:
        return init_transform, int(keep.shape[0])

    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            points_last_cam[keep].astype(np.float64),
            uv_curr[keep].reshape(-1, 1, 2).astype(np.float64),
            k_mat.astype(np.float64),
            np.zeros((5, 1), dtype=np.float64),
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    refined = np.eye(4, dtype=np.float64)
    refined[:3, :3] = cv2.Rodrigues(rvec)[0]
    refined[:3, 3:4] = tvec
    return refined, int(keep.shape[0])


class TrackAccumulator:
    def __init__(self, match_radius_px: float) -> None:
        self.match_radius_px = float(match_radius_px)
        self.next_track_id = 0
        self.active_points: List[Tuple[int, np.ndarray]] = []
        self.stats: Dict[int, Dict[str, float]] = {}

    def _new_track(self) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.stats[track_id] = {
            "observations": 0.0,
            "dynamic_hits": 0.0,
            "score_sum": 0.0,
        }
        return track_id

    def update(self, last_uv: np.ndarray, curr_uv: np.ndarray, dynamic_mask: np.ndarray, scores: np.ndarray) -> None:
        assigned_ids: List[int] = []
        used_prev = set()
        for idx, point_last in enumerate(last_uv):
            best_track = None
            best_dist = self.match_radius_px
            for prev_idx, (track_id, prev_uv) in enumerate(self.active_points):
                if prev_idx in used_prev:
                    continue
                dist = float(np.linalg.norm(prev_uv - point_last))
                if dist <= best_dist:
                    best_dist = dist
                    best_track = (prev_idx, track_id)
            if best_track is None:
                track_id = self._new_track()
            else:
                used_prev.add(best_track[0])
                track_id = best_track[1]
            assigned_ids.append(track_id)
            stat = self.stats[track_id]
            stat["observations"] += 1.0
            stat["dynamic_hits"] += float(dynamic_mask[idx])
            stat["score_sum"] += float(scores[idx])

        self.active_points = [(track_id, curr_uv[idx].copy()) for idx, track_id in enumerate(assigned_ids)]

    def export_csv(self, path: str, dynamic_ratio_thresh: float, score_thresh: float) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "observations", "dynamic_hits", "dynamic_ratio", "score_sum", "is_dynamic"])
            for track_id, stat in sorted(self.stats.items()):
                observations = max(stat["observations"], 1.0)
                dynamic_ratio = stat["dynamic_hits"] / observations
                is_dynamic = int(dynamic_ratio >= dynamic_ratio_thresh or stat["score_sum"] >= score_thresh)
                writer.writerow(
                    [
                        track_id,
                        int(stat["observations"]),
                        int(stat["dynamic_hits"]),
                        dynamic_ratio,
                        stat["score_sum"],
                        is_dynamic,
                    ]
                )


def write_step_csv(path: str, steps: Sequence[DynamicStepInfo]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(DynamicStepInfo.__dataclass_fields__.keys()))
        writer.writeheader()
        for step in steps:
            writer.writerow(step.__dict__)


def draw_dynamic_overlay(
    image: np.ndarray,
    points: np.ndarray,
    dynamic_mask: np.ndarray,
    output_path: str,
) -> None:
    canvas = image.copy()
    for idx, uv in enumerate(points):
        color = (255, 64, 64) if dynamic_mask[idx] else (64, 255, 64)
        center = (int(round(uv[0])), int(round(uv[1])))
        cv2.circle(canvas, center, 2, color, -1, lineType=cv2.LINE_AA)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

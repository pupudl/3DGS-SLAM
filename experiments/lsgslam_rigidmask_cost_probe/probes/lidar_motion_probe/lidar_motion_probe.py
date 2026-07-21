import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from utils.pnp_fused_icp_utils import (
    get_dataset_frame_id,
    lidar_file_for_frame,
    load_velo_to_cam,
    read_velodyne_bin,
)


def _transform_xyz(xyz, transform):
    if xyz.shape[0] == 0:
        return xyz.copy()
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
    return (transform @ xyz_h.T).T[:, :3]


def _uniform_downsample(xyz, max_points):
    max_points = int(max_points)
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    indices = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
    return xyz[indices]


def _pose_w2c_from_params(params, time_idx, device):
    with torch.no_grad():
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = params["cam_trans"][..., time_idx].detach()
    q = cam_rot.squeeze(0).detach().cpu().numpy().astype(np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    r, x, y, z = q
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
            [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
            [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3] = cam_tran.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return w2c


class LidarMotionProbe:
    def __init__(self, probe_cfg, project_root):
        self.cfg = probe_cfg
        self.project_root = project_root
        self.run_every = int(probe_cfg.get("run_every", 1))
        self._velo_to_cam = None

    def should_run(self, time_idx):
        return time_idx > 0 and self.run_every > 0 and time_idx % self.run_every == 0

    def save_pair(self, output_root, dataset, params, prev_time_idx, curr_time_idx, device):
        pair_dir = None
        try:
            prev_frame_id = get_dataset_frame_id(dataset, prev_time_idx)
            curr_frame_id = get_dataset_frame_id(dataset, curr_time_idx)
            pair_name = f"{prev_frame_id}_{curr_frame_id}"
            pair_dir = os.path.join(output_root, pair_name)
            os.makedirs(pair_dir, exist_ok=True)

            if self._velo_to_cam is None:
                self._velo_to_cam = load_velo_to_cam(dataset, self.project_root)

            prev_lidar_path = lidar_file_for_frame(dataset, prev_frame_id)
            curr_lidar_path = lidar_file_for_frame(dataset, curr_frame_id)
            prev_velo = self._load_lidar(prev_lidar_path)
            curr_velo = self._load_lidar(curr_lidar_path)

            prev_w2c = _pose_w2c_from_params(params, prev_time_idx, device)
            curr_w2c = _pose_w2c_from_params(params, curr_time_idx, device)
            prev_c2w = np.linalg.inv(prev_w2c)
            curr_c2w = np.linalg.inv(curr_w2c)

            prev_world = _transform_xyz(_transform_xyz(prev_velo, self._velo_to_cam), prev_c2w)
            curr_world = _transform_xyz(_transform_xyz(curr_velo, self._velo_to_cam), curr_c2w)

            bev_reference_frame = str(self.cfg.get("bev_reference_frame", "prev_camera")).lower()
            bev_reference_w2c = self._get_bev_reference_w2c(
                bev_reference_frame,
                prev_w2c,
                curr_w2c,
            )
            prev_bev = _transform_xyz(prev_world, bev_reference_w2c)
            curr_bev = _transform_xyz(curr_world, bev_reference_w2c)

            prev_vis = self._filter_for_bev(prev_bev)
            curr_vis = self._filter_for_bev(curr_bev)

            self._save_bev_overlay(pair_dir, prev_vis, curr_vis, prev_frame_id, curr_frame_id)
            residual_summary = {}
            if self.cfg.get("save_residual", True):
                residual_summary = self._save_bev_residual(pair_dir, prev_vis, curr_vis)
            nonground_summary = {}
            if self.cfg.get("save_nonground_visualizations", True):
                prev_nonground = self._filter_nonground(prev_vis)
                curr_nonground = self._filter_nonground(curr_vis)
                self._save_bev_overlay(
                    pair_dir,
                    prev_nonground,
                    curr_nonground,
                    prev_frame_id,
                    curr_frame_id,
                    filename="bev_overlay_nonground.png",
                )
                nonground_summary = {
                    "prev_nonground_points": int(prev_nonground.shape[0]),
                    "curr_nonground_points": int(curr_nonground.shape[0]),
                }
                if self.cfg.get("save_residual", True):
                    nonground_residual_summary = self._save_bev_residual(
                        pair_dir,
                        prev_nonground,
                        curr_nonground,
                        filename="bev_residual_nonground.png",
                    )
                    nonground_summary.update(
                        {f"nonground_{k}": v for k, v in nonground_residual_summary.items()}
                    )
            if self.cfg.get("save_npz", True):
                np.savez_compressed(
                    os.path.join(pair_dir, "aligned_lidar_pair.npz"),
                    prev_bev=prev_vis.astype(np.float32),
                    curr_bev=curr_vis.astype(np.float32),
                    bev_reference_w2c=bev_reference_w2c.astype(np.float32),
                    prev_w2c=prev_w2c.astype(np.float32),
                    curr_w2c=curr_w2c.astype(np.float32),
                    velo_to_cam=self._velo_to_cam.astype(np.float32),
                    prev_frame_id=prev_frame_id,
                    curr_frame_id=curr_frame_id,
                    prev_lidar_path=prev_lidar_path,
                    curr_lidar_path=curr_lidar_path,
                )

            summary = {
                "status": "ok",
                "prev_time_idx": int(prev_time_idx),
                "curr_time_idx": int(curr_time_idx),
                "prev_frame_id": prev_frame_id,
                "curr_frame_id": curr_frame_id,
                "prev_lidar_path": prev_lidar_path,
                "curr_lidar_path": curr_lidar_path,
                "prev_raw_points": int(prev_velo.shape[0]),
                "curr_raw_points": int(curr_velo.shape[0]),
                "prev_bev_points": int(prev_vis.shape[0]),
                "curr_bev_points": int(curr_vis.shape[0]),
                "bev_reference_frame": bev_reference_frame,
                "bev_reference_w2c": bev_reference_w2c.tolist(),
                "prev_w2c": prev_w2c.tolist(),
                "curr_w2c": curr_w2c.tolist(),
                **residual_summary,
                **nonground_summary,
            }
            self._write_summary(pair_dir, summary)
            return summary
        except Exception as exc:
            summary = {
                "status": "skipped",
                "reason": str(exc),
                "prev_time_idx": int(prev_time_idx),
                "curr_time_idx": int(curr_time_idx),
            }
            if pair_dir is not None:
                self._write_summary(pair_dir, summary)
            return summary

    def _load_lidar(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing LiDAR file: {path}")
        xyz = read_velodyne_bin(path)
        min_forward_m = float(self.cfg.get("min_forward_m", 0.0))
        max_forward_m = float(self.cfg.get("max_forward_m", 0.0))
        if min_forward_m > 0.0 or max_forward_m > 0.0:
            mask = np.ones(xyz.shape[0], dtype=bool)
            if min_forward_m > 0.0:
                mask &= xyz[:, 0] > min_forward_m
            if max_forward_m > 0.0:
                mask &= xyz[:, 0] < max_forward_m
            xyz = xyz[mask]
        return _uniform_downsample(xyz, self.cfg.get("max_points", 120000)).astype(np.float64)

    def _get_bev_reference_w2c(self, reference_frame, prev_w2c, curr_w2c):
        if reference_frame == "prev_camera":
            return prev_w2c
        if reference_frame == "curr_camera":
            return curr_w2c
        if reference_frame == "world":
            return np.eye(4, dtype=np.float64)
        raise ValueError(f"Unsupported bev_reference_frame: {reference_frame}")

    def _filter_for_bev(self, xyz):
        if xyz.shape[0] == 0:
            return xyz
        mask = np.isfinite(xyz).all(axis=1)
        min_height_m = self.cfg.get("min_height_m", None)
        max_height_m = self.cfg.get("max_height_m", None)
        if min_height_m is not None:
            mask &= xyz[:, 1] >= float(min_height_m)
        if max_height_m is not None:
            mask &= xyz[:, 1] <= float(max_height_m)

        bev_xlim = self.cfg.get("bev_xlim", None)
        bev_zlim = self.cfg.get("bev_zlim", None)
        if bev_xlim is not None:
            mask &= (xyz[:, 0] >= float(bev_xlim[0])) & (xyz[:, 0] <= float(bev_xlim[1]))
        if bev_zlim is not None:
            mask &= (xyz[:, 2] >= float(bev_zlim[0])) & (xyz[:, 2] <= float(bev_zlim[1]))
        return xyz[mask]

    def _filter_nonground(self, xyz):
        if xyz.shape[0] == 0:
            return xyz
        method = str(self.cfg.get("nonground_filter_method", "plane")).lower()
        if method == "plane":
            return self._filter_nonground_by_plane(xyz)
        if method != "height":
            raise ValueError(f"Unsupported nonground_filter_method: {method}")
        min_y = self.cfg.get("nonground_min_y_m", None)
        max_y = self.cfg.get("nonground_max_y_m", 1.35)
        mask = np.ones(xyz.shape[0], dtype=bool)
        if min_y is not None:
            mask &= xyz[:, 1] >= float(min_y)
        if max_y is not None:
            mask &= xyz[:, 1] <= float(max_y)
        return xyz[mask]

    def _filter_nonground_by_plane(self, xyz):
        min_points = int(self.cfg.get("ground_min_candidate_points", 1000))
        if xyz.shape[0] < min_points:
            return xyz

        candidate = xyz
        percentile = self.cfg.get("ground_candidate_y_percentile", 45.0)
        if percentile is not None:
            y_min = np.percentile(candidate[:, 1], float(percentile))
            candidate = candidate[candidate[:, 1] >= y_min]
        if candidate.shape[0] < min_points:
            candidate = xyz

        max_fit_points = int(self.cfg.get("ground_fit_max_points", 50000))
        candidate_fit = _uniform_downsample(candidate, max_fit_points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(candidate_fit.astype(np.float64))

        distance_threshold = float(self.cfg.get("ground_distance_threshold_m", 0.18))
        ransac_n = int(self.cfg.get("ground_ransac_n", 3))
        num_iterations = int(self.cfg.get("ground_num_iterations", 200))
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        if len(inliers) < min_points:
            return xyz

        normal = np.asarray(plane_model[:3], dtype=np.float64)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return xyz
        normal = normal / normal_norm
        min_normal_y = float(self.cfg.get("ground_min_abs_normal_y", 0.65))
        if abs(normal[1]) < min_normal_y:
            return xyz

        plane = np.asarray(plane_model, dtype=np.float64)
        distances = np.abs(xyz @ plane[:3] + plane[3]) / normal_norm
        return xyz[distances > distance_threshold]

    def _save_bev_overlay(
        self,
        pair_dir,
        prev_xyz,
        curr_xyz,
        prev_frame_id,
        curr_frame_id,
        filename="bev_overlay.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 9), dpi=180)
        point_size = float(self.cfg.get("point_size", 0.2))
        alpha = float(self.cfg.get("alpha", 0.45))
        if prev_xyz.shape[0] > 0:
            ax.scatter(prev_xyz[:, 0], prev_xyz[:, 2], s=point_size, c="#2f6fdb", alpha=alpha, label=prev_frame_id)
        if curr_xyz.shape[0] > 0:
            ax.scatter(curr_xyz[:, 0], curr_xyz[:, 2], s=point_size, c="#d94841", alpha=alpha, label=curr_frame_id)
        self._format_bev_axis(ax)
        ax.legend(loc="upper right", markerscale=8, frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, filename))
        plt.close(fig)

    def _save_bev_residual(self, pair_dir, prev_xyz, curr_xyz, filename="bev_residual.png"):
        if prev_xyz.shape[0] == 0 or curr_xyz.shape[0] == 0:
            return {"residual_status": "skipped_empty_pointcloud"}

        max_points = int(self.cfg.get("residual_nn_max_points", 60000))
        prev_nn = _uniform_downsample(prev_xyz, max_points)
        curr_nn = _uniform_downsample(curr_xyz, max_points)
        prev_pcd = o3d.geometry.PointCloud()
        prev_pcd.points = o3d.utility.Vector3dVector(prev_nn.astype(np.float64))
        kd_tree = o3d.geometry.KDTreeFlann(prev_pcd)

        residuals = np.empty(curr_nn.shape[0], dtype=np.float32)
        for idx, point in enumerate(curr_nn):
            _, _, dists2 = kd_tree.search_knn_vector_3d(point.astype(np.float64), 1)
            residuals[idx] = float(np.sqrt(dists2[0])) if dists2 else np.nan

        vis_max = float(self.cfg.get("residual_vis_max_m", 1.5))
        fig, ax = plt.subplots(figsize=(8, 9), dpi=180)
        ax.scatter(prev_nn[:, 0], prev_nn[:, 2], s=0.1, c="#9aa3ad", alpha=0.18)
        points = ax.scatter(
            curr_nn[:, 0],
            curr_nn[:, 2],
            s=max(float(self.cfg.get("point_size", 0.2)), 0.4),
            c=np.clip(residuals, 0.0, vis_max),
            cmap="magma",
            vmin=0.0,
            vmax=vis_max,
            alpha=0.85,
        )
        self._format_bev_axis(ax)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label="nearest distance (m)")
        fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, filename))
        plt.close(fig)

        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            return {"residual_status": "no_finite_distances"}
        return {
            "residual_status": "ok",
            "residual_points": int(finite.size),
            "residual_mean_m": float(np.mean(finite)),
            "residual_median_m": float(np.median(finite)),
            "residual_p90_m": float(np.percentile(finite, 90.0)),
        }

    def _format_bev_axis(self, ax):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("bev x (m)")
        ax.set_ylabel("bev z (m)")
        ax.grid(True, color="#d8dde3", linewidth=0.4, alpha=0.8)
        bev_xlim = self.cfg.get("bev_xlim", None)
        bev_zlim = self.cfg.get("bev_zlim", None)
        if bev_xlim is not None:
            ax.set_xlim(float(bev_xlim[0]), float(bev_xlim[1]))
        if bev_zlim is not None:
            ax.set_ylim(float(bev_zlim[0]), float(bev_zlim[1]))

    def _write_summary(self, pair_dir, summary):
        with open(os.path.join(pair_dir, "summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

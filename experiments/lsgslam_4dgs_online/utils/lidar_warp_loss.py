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
from utils.slam_external import build_rotation


def _transform_points_np(points, transform):
    if points.shape[0] == 0:
        return points.copy()
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )
    return (transform @ points_h.T).T[:, :3]


def _uniform_downsample(points, max_points):
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
    return points[idx]


def _torch_w2c_from_params(params, time_idx, device, detach=False):
    if detach:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = params["cam_trans"][..., time_idx].detach()
    else:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx])
        cam_tran = params["cam_trans"][..., time_idx]
    w2c = torch.eye(4, device=device).float()
    w2c[:3, :3] = build_rotation(cam_rot)
    w2c[:3, 3] = cam_tran
    return w2c


def _smooth_abs(residual, beta):
    beta = float(beta)
    if beta <= 0.0:
        return residual.abs()
    abs_res = residual.abs()
    return torch.where(abs_res < beta, 0.5 * abs_res * abs_res / beta, abs_res - 0.5 * beta)


class LidarWarpLoss:
    def __init__(self, dataset, project_root, cfg, device):
        self.dataset = dataset
        self.project_root = project_root
        self.cfg = dict(cfg or {})
        self.device = device
        self.enabled = bool(self.cfg.get("enabled", False))
        self.mode = str(self.cfg.get("mode", "local_map")).lower()
        self._velo_to_cam = None
        self._lidar_cam_cache = {}

    def refresh_every(self):
        return int(self.cfg.get("refresh_every", 10))

    def prepare_pair(self, params, time_idx, last_data, curr_data, keyframe_list=None):
        if not self.enabled or time_idx <= 0:
            return None

        dataset_name = getattr(self.dataset, "name", "").lower()
        if dataset_name not in ("kitti", "kitti360"):
            return None

        try:
            curr_frame_id = curr_data.get("frame_id", get_dataset_frame_id(self.dataset, time_idx))
            curr_cam = self._load_frame_lidar_cam(curr_frame_id, curr_data)
            curr_cam = _uniform_downsample(curr_cam, self.cfg.get("max_points", 15000))
            if curr_cam.shape[0] == 0:
                return None

            if self.mode == "local_map":
                pair_data = self._prepare_local_map(time_idx, keyframe_list)
                if pair_data is None:
                    pair_data = self._prepare_prev_frame(time_idx, last_data)
            else:
                pair_data = self._prepare_prev_frame(time_idx, last_data)

            if pair_data is None:
                return None

            pair_data["curr_points_all_cam"] = curr_cam.astype(np.float32)
            return self.refresh_correspondences(params, time_idx, pair_data)
        except Exception as exc:
            print(f"LiDAR warp loss skipped at frame {time_idx}: {exc}")
            return None

    def refresh_correspondences(self, params, time_idx, pair_data):
        source_points = pair_data["curr_points_all_cam"]
        source_in_ref = self._curr_points_in_ref_np(
            source_points,
            params,
            time_idx,
            pair_data["world_to_ref_torch"],
        )
        target_points, target_normals, source_corr, distances = self._build_correspondences(
            source_points,
            source_in_ref,
            pair_data["ref_points_np"],
            pair_data["ref_normals_np"],
            pair_data["ref_kd_tree"],
        )
        min_corr = int(self.cfg.get("min_correspondences", 300))
        if source_corr.shape[0] < min_corr:
            return None

        pair_data["curr_points_cam"] = torch.from_numpy(source_corr).float().to(self.device)
        pair_data["target_points_ref"] = torch.from_numpy(target_points).float().to(self.device)
        pair_data["target_normals_ref"] = torch.from_numpy(target_normals).float().to(self.device)
        pair_data["num_correspondences"] = int(source_corr.shape[0])
        pair_data["mean_init_corr_m"] = float(np.mean(distances)) if distances.size > 0 else 0.0
        return pair_data

    def compute_loss(self, params, time_idx, pair_data):
        curr_points = pair_data["curr_points_cam"]
        target_points = pair_data["target_points_ref"].detach()
        target_normals = pair_data["target_normals_ref"].detach()
        world_to_ref = pair_data["world_to_ref_torch"].detach()

        curr_w2c = _torch_w2c_from_params(params, time_idx, self.device, detach=False)
        t_ref_curr = world_to_ref @ torch.linalg.inv(curr_w2c)

        warped = curr_points @ t_ref_curr[:3, :3].T + t_ref_curr[:3, 3]
        residual = warped - target_points
        beta = float(self.cfg.get("robust_beta_m", 0.05))

        point_to_plane_weight = float(self.cfg.get("point_to_plane_weight", 1.0))
        plane_residual = torch.sum(residual * target_normals, dim=-1)
        plane_loss = _smooth_abs(plane_residual, beta).mean()

        p2p_weight = float(self.cfg.get("point_to_point_weight", 0.05))
        p2p_loss = torch.zeros((), device=self.device)
        if p2p_weight > 0.0:
            p2p_loss = _smooth_abs(torch.linalg.norm(residual, dim=-1), beta).mean()

        loss = (
            point_to_plane_weight * plane_loss
            + p2p_weight * p2p_loss
        )
        return loss, {
            "lidar_warp": loss.detach(),
            "lidar_warp_plane": plane_loss.detach(),
            "lidar_warp_p2p": p2p_loss.detach(),
        }

    def _prepare_prev_frame(self, time_idx, last_data):
        prev_frame_id = last_data.get("frame_id", get_dataset_frame_id(self.dataset, time_idx - 1))
        prev_cam = self._load_frame_lidar_cam(prev_frame_id, last_data)
        prev_cam = self._voxel_downsample(prev_cam, self.cfg.get("voxel_size", 0.2))
        prev_cam = _uniform_downsample(prev_cam, self.cfg.get("max_reference_points", 60000))
        if prev_cam.shape[0] == 0:
            return None

        ref_points, ref_normals, ref_kd_tree = self._make_reference(prev_cam)
        if ref_points is None:
            return None
        return {
            "mode": "prev_frame",
            "ref_points_np": ref_points,
            "ref_normals_np": ref_normals,
            "ref_kd_tree": ref_kd_tree,
            "world_to_ref_torch": last_data["est_w2c"].detach().float().to(self.device),
        }

    def _prepare_local_map(self, time_idx, keyframe_list):
        if not keyframe_list:
            return None

        local_map_size = int(self.cfg.get("local_map_size", 8))
        keyframes = [kf for kf in keyframe_list if int(kf.get("id", -1)) < int(time_idx)]
        keyframes = keyframes[-local_map_size:]
        if not keyframes:
            return None

        world_points = []
        for keyframe in keyframes:
            frame_id = keyframe.get("frame_id", get_dataset_frame_id(self.dataset, int(keyframe["id"])))
            points_cam = self._load_frame_lidar_cam(frame_id, keyframe)
            if points_cam.shape[0] == 0:
                continue
            kf_w2c = keyframe["est_w2c"].detach().cpu().numpy().astype(np.float64)
            points_world = _transform_points_np(points_cam, np.linalg.inv(kf_w2c))
            world_points.append(points_world)

        if not world_points:
            return None

        ref_world = np.concatenate(world_points, axis=0).astype(np.float32)
        ref_world = self._voxel_downsample(ref_world, self.cfg.get("voxel_size", 0.2))
        ref_world = _uniform_downsample(ref_world, self.cfg.get("max_reference_points", 80000))
        ref_points, ref_normals, ref_kd_tree = self._make_reference(ref_world)
        if ref_points is None:
            return None

        world_to_ref = torch.eye(4, device=self.device).float()
        return {
            "mode": "local_map",
            "ref_points_np": ref_points,
            "ref_normals_np": ref_normals,
            "ref_kd_tree": ref_kd_tree,
            "world_to_ref_torch": world_to_ref,
            "local_map_keyframes": [int(kf["id"]) for kf in keyframes],
        }

    def _make_reference(self, points):
        min_reference = int(self.cfg.get("min_reference_points", 300))
        if points.shape[0] < min_reference:
            return None, None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        normal_radius = float(self.cfg.get("normal_radius_m", 0.8))
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))

        ref_points = np.asarray(pcd.points, dtype=np.float32)
        ref_normals = np.asarray(pcd.normals, dtype=np.float32)
        if ref_points.shape[0] == 0:
            return None, None, None
        kd_tree = o3d.geometry.KDTreeFlann(pcd)
        return ref_points, ref_normals, kd_tree

    def _load_frame_lidar_cam(self, frame_id, frame_data):
        cache_key = str(frame_id)
        if cache_key in self._lidar_cam_cache:
            points_cam = self._lidar_cam_cache[cache_key].copy()
        else:
            path = lidar_file_for_frame(self.dataset, frame_id)
            points_velo = read_velodyne_bin(path)
            points_velo = self._filter_lidar_range(points_velo)
            if self._velo_to_cam is None:
                self._velo_to_cam = load_velo_to_cam(self.dataset, self.project_root)
            points_cam = _transform_points_np(points_velo, self._velo_to_cam)
            points_cam = self._filter_camera_points(points_cam).astype(np.float32)
            self._lidar_cam_cache[cache_key] = points_cam.copy()

        if self.cfg.get("use_image_masks", True):
            points_cam = self._filter_by_image_masks(points_cam, frame_data)
        return points_cam.astype(np.float32)

    def _filter_lidar_range(self, points_velo):
        if points_velo.shape[0] == 0:
            return points_velo
        mask = np.isfinite(points_velo).all(axis=1)
        min_forward = float(self.cfg.get("min_forward_m", 0.0))
        max_forward = float(self.cfg.get("max_forward_m", 0.0))
        if min_forward > 0.0:
            mask &= points_velo[:, 0] > min_forward
        if max_forward > 0.0:
            mask &= points_velo[:, 0] < max_forward
        return points_velo[mask]

    def _filter_camera_points(self, points_cam):
        if points_cam.shape[0] == 0:
            return points_cam
        mask = np.isfinite(points_cam).all(axis=1)
        min_depth = float(self.cfg.get("min_depth_m", 0.5))
        max_depth = float(self.cfg.get("max_depth_m", 80.0))
        mask &= points_cam[:, 2] > min_depth
        if max_depth > 0.0:
            mask &= points_cam[:, 2] < max_depth
        min_y = self.cfg.get("min_y_m", None)
        max_y = self.cfg.get("max_y_m", None)
        if min_y is not None:
            mask &= points_cam[:, 1] >= float(min_y)
        if max_y is not None:
            mask &= points_cam[:, 1] <= float(max_y)
        return points_cam[mask]

    def _filter_by_image_masks(self, points_cam, frame_data):
        image = frame_data.get("im", frame_data.get("color"))
        intrinsics = frame_data.get("intrinsics")
        if points_cam.shape[0] == 0 or image is None or intrinsics is None:
            return points_cam

        intrinsics = intrinsics.detach().cpu().numpy()
        z = points_cam[:, 2]
        uv = points_cam[:, :2] / z[:, None]
        uv[:, 0] = uv[:, 0] * intrinsics[0, 0] + intrinsics[0, 2]
        uv[:, 1] = uv[:, 1] * intrinsics[1, 1] + intrinsics[1, 2]

        height = int(image.shape[1])
        width = int(image.shape[2])
        xs = np.round(uv[:, 0]).astype(np.int64)
        ys = np.round(uv[:, 1]).astype(np.int64)
        keep = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)

        sky_mask = self._mask_to_numpy(frame_data.get("sky_mask"))
        if sky_mask is not None:
            keep &= ~sky_mask[ys.clip(0, height - 1), xs.clip(0, width - 1)]

        dynamic_mask = self._mask_to_numpy(frame_data.get("dynamic_mask"))
        if dynamic_mask is not None:
            keep &= ~dynamic_mask[ys.clip(0, height - 1), xs.clip(0, width - 1)]

        return points_cam[keep]

    def _mask_to_numpy(self, mask):
        if mask is None:
            return None
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[0]
        return mask.astype(bool)

    def _voxel_downsample(self, points, voxel_size):
        voxel_size = float(voxel_size)
        if voxel_size <= 0.0 or points.shape[0] == 0:
            return points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        return np.asarray(pcd.voxel_down_sample(voxel_size).points, dtype=np.float32)

    def _curr_points_in_ref_np(self, curr_points_cam, params, time_idx, world_to_ref):
        with torch.no_grad():
            curr_w2c = _torch_w2c_from_params(params, time_idx, self.device, detach=True)
            t_ref_curr = world_to_ref.detach().float().to(self.device) @ torch.linalg.inv(curr_w2c)
            points = torch.from_numpy(curr_points_cam).float().to(self.device)
            warped = points @ t_ref_curr[:3, :3].T + t_ref_curr[:3, 3]
        return warped.detach().cpu().numpy().astype(np.float32)

    def _build_correspondences(
        self,
        source_cam,
        source_init_ref,
        target_points,
        target_normals,
        kd_tree,
    ):
        max_corr = float(self.cfg.get("max_corr_m", 0.7))
        target_corr = []
        normal_corr = []
        source_corr = []
        distances = []
        for idx, point in enumerate(source_init_ref):
            count, nn_idx, nn_dist2 = kd_tree.search_knn_vector_3d(point.astype(np.float64), 1)
            if count == 0:
                continue
            dist = float(np.sqrt(nn_dist2[0]))
            if max_corr > 0.0 and dist > max_corr:
                continue
            target_idx = int(nn_idx[0])
            target_corr.append(target_points[target_idx])
            normal_corr.append(target_normals[target_idx])
            source_corr.append(source_cam[idx])
            distances.append(dist)

        if not source_corr:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, empty, empty, np.empty((0,), dtype=np.float32)

        return (
            np.asarray(target_corr, dtype=np.float32),
            np.asarray(normal_corr, dtype=np.float32),
            np.asarray(source_corr, dtype=np.float32),
            np.asarray(distances, dtype=np.float32),
        )

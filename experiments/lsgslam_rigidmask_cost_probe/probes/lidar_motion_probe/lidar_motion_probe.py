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
)

try:
    from .gndnet_adapter import GndNetGroundFilter
except ImportError:
    from gndnet_adapter import GndNetGroundFilter

#用一个 4×4 的变换矩阵，将 LiDAR 点云从 LiDAR 坐标系转换到相机坐标系，然后再将其从相机坐标系转换到世界坐标系。最后，将点云从世界坐标系转换到 BEV（鸟瞰图）参考坐标系。
def _transform_xyz(xyz, transform):
    if xyz.shape[0] == 0:
        return xyz.copy()
    xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=xyz.dtype)], axis=1)
    return (transform @ xyz_h.T).T[:, :3]

#降采样，将点云中的点均匀地采样到指定的最大点数。
def _uniform_downsample(xyz, max_points):
    max_points = int(max_points)
    if max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz
    indices = np.linspace(0, xyz.shape[0] - 1, max_points).astype(np.int64)
    return xyz[indices]


def _uniform_downsample_indices(count, max_points):
    max_points = int(max_points)
    if max_points <= 0 or count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, max_points).astype(np.int64)

#从 SLAM 参数中取出某一帧相机的旋转和平移，组成 world-to-camera 变换矩阵 w2c。
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
        self._gndnet_filter = None
        self._projection_calib = None

    def should_run(self, time_idx):
        return time_idx > 0 and self.run_every > 0 and time_idx % self.run_every == 0

    def save_pair(self, output_root, dataset, params, prev_time_idx, curr_time_idx, device):
        pair_dir = None
        try:
            #找到当前帧和前一帧的时间索引对应的帧 ID，并创建一个输出目录来保存结果。
            prev_frame_id = get_dataset_frame_id(dataset, prev_time_idx)
            curr_frame_id = get_dataset_frame_id(dataset, curr_time_idx)
            pair_name = f"{prev_frame_id}_{curr_frame_id}"
            pair_dir = os.path.join(output_root, pair_name)
            os.makedirs(pair_dir, exist_ok=True)

            if self._velo_to_cam is None:
                self._velo_to_cam = load_velo_to_cam(dataset, self.project_root)

            prev_lidar_path = lidar_file_for_frame(dataset, prev_frame_id)
            curr_lidar_path = lidar_file_for_frame(dataset, curr_frame_id)

            #加载前一帧和当前帧的 LiDAR 点云数据，此时在 LiDAR 坐标系下。
            prev_velo, prev_xyzi = self._load_lidar(prev_lidar_path, return_xyzi=True)
            curr_velo, curr_xyzi = self._load_lidar(curr_lidar_path, return_xyzi=True)

            #从 SLAM 参数中取出某一帧相机的旋转和平移，组成 world-to-camera 变换矩阵 w2c。
            prev_w2c = _pose_w2c_from_params(params, prev_time_idx, device)
            curr_w2c = _pose_w2c_from_params(params, curr_time_idx, device)
            prev_c2w = np.linalg.inv(prev_w2c)
            curr_c2w = np.linalg.inv(curr_w2c)

            #把lidar点云从 LiDAR 坐标系转换到相机坐标系，再转换到世界坐标系。
            prev_world = _transform_xyz(_transform_xyz(prev_velo, self._velo_to_cam), prev_c2w)
            curr_world = _transform_xyz(_transform_xyz(curr_velo, self._velo_to_cam), curr_c2w)

            #根据配置中的 BEV 参考坐标系选择，返回相应的 world-to-camera 变换矩阵。
            bev_reference_frame = str(self.cfg.get("bev_reference_frame", "prev_camera")).lower()
            bev_reference_w2c = self._get_bev_reference_w2c(
                bev_reference_frame,
                prev_w2c,
                curr_w2c,
            )

            #将两帧的点云从世界坐标系转换到 BEV 参考坐标系下，并进行可视化和残差计算。
            prev_bev = _transform_xyz(prev_world, bev_reference_w2c)
            curr_bev = _transform_xyz(curr_world, bev_reference_w2c)

            #过滤可视化范围
            prev_vis = self._filter_for_bev(prev_bev)
            curr_vis = self._filter_for_bev(curr_bev)

            #保存bev_overlay.png
            self._save_bev_overlay(pair_dir, prev_vis, curr_vis, prev_frame_id, curr_frame_id)

            nonground_summary = {}
            if self.cfg.get("save_nonground_visualizations", True):
                (
                    prev_nonground,
                    curr_nonground,
                    prev_nonground_xyzi,
                    curr_nonground_xyzi,
                    curr_visible_xyzi,
                    filter_summary,
                ) = self._filter_nonground_pair_with_xyzi(
                    prev_bev,
                    curr_bev,
                    prev_xyzi,
                    curr_xyzi,
                    device,
                )
                #保存bev_overlay_nonground.png
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
                    **filter_summary,
                }
                image_path = self._image_path_for_time_idx(dataset, curr_time_idx, curr_frame_id)
                if self.cfg.get("save_static_projection_masks", True):
                    static_projection_summary = self._save_lidar_static_image_masks(
                        pair_dir,
                        curr_visible_xyzi,
                        curr_nonground_xyzi,
                        image_path,
                    )
                    nonground_summary.update(static_projection_summary)
                #保存bev_residual_nonground_features.png和bev_features_nonground.png
                if self.cfg.get("save_feature_residual", False):
                    #这个函数主要是为了保存bev_residual_nonground_features.png，只是顺带保存了bev_features_nonground.png和image_residual_nonground_features.png
                    feature_residual_summary = self._save_bev_feature_residual(
                        pair_dir,
                        prev_nonground,
                        curr_nonground,
                        filename="bev_residual_nonground_features.png",#bev_residual_nonground_features.png
                        mask_filename="bev_features_nonground.png",#bev_features_nonground.png
                        curr_xyzi=curr_nonground_xyzi,
                        image_path=image_path,
                        projection_filename="image_residual_nonground_features.png",
                    )
                    nonground_summary.update(
                        {f"nonground_{k}": v for k, v in feature_residual_summary.items()}
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

    #读取lidar
    def _load_lidar(self, path, return_xyzi=False):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing LiDAR file: {path}")
        #读取 LiDAR 点云数据
        xyzi = np.fromfile(path, dtype=np.float32).reshape(-1, 4).astype(np.float64)
        #根据配置中的最小和最大前向距离进行过滤
        min_forward_m = float(self.cfg.get("min_forward_m", 0.0))
        max_forward_m = float(self.cfg.get("max_forward_m", 0.0))
        if min_forward_m > 0.0 or max_forward_m > 0.0:
            mask = np.ones(xyzi.shape[0], dtype=bool)
            if min_forward_m > 0.0:
                mask &= xyzi[:, 0] > min_forward_m
            if max_forward_m > 0.0:
                mask &= xyzi[:, 0] < max_forward_m
            xyzi = xyzi[mask]
        #对点云进行均匀降采样，确保点数不超过指定的最大值
        xyzi = _uniform_downsample(xyzi, self.cfg.get("max_points", 120000)).astype(np.float64)
        xyz = xyzi[:, :3]
        if return_xyzi:
            return xyz, xyzi
        return xyz

    #根据配置中的 BEV 参考坐标系选择，返回相应的 world-to-camera 变换矩阵。
    def _get_bev_reference_w2c(self, reference_frame, prev_w2c, curr_w2c):
        if reference_frame == "prev_camera":
            return prev_w2c
        if reference_frame == "curr_camera":
            return curr_w2c
        if reference_frame == "world":
            return np.eye(4, dtype=np.float64)
        raise ValueError(f"Unsupported bev_reference_frame: {reference_frame}")

    #过滤点云，保留在 BEV 可视化范围内的点。
    def _filter_for_bev(self, xyz):
        return xyz[self._filter_for_bev_mask(xyz)]

    #根据 BEV 可视化范围过滤点云，返回一个布尔掩码，表示哪些点在可视化范围内。
    def _filter_for_bev_mask(self, xyz):
        if xyz.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
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
        return mask

    #输入前后两帧 LiDAR 点云，使用 GndNet 判断哪些点是非地面点，然后把地面点过滤掉
    def _filter_nonground_pair_with_xyzi(
        self,
        prev_bev,
        curr_bev,
        prev_xyzi,
        curr_xyzi,
        device,
    ):
        method = str(self.cfg.get("nonground_filter_method", "gndnet")).lower()
        if method != "gndnet":
            raise ValueError("lidar_motion_probe now supports only GndNet non-ground filtering")

        gndnet_filter = self._get_gndnet_filter(device)
        #使用 GndNet 模型对前后两帧的 LiDAR 点云进行非地面点预测，返回非地面点的掩码和相关信息。
        prev_nonground_mask, prev_info = gndnet_filter.predict_nonground_mask(prev_xyzi)
        curr_nonground_mask, curr_info = gndnet_filter.predict_nonground_mask(curr_xyzi)

        #_filter_for_bev_mask 根据 BEV 可视化范围过滤点云，返回一个布尔掩码，表示哪些点在可视化范围内。然后将这个掩码与 GndNet 预测的非地面点掩码进行逻辑与运算，得到最终的保留点掩码。
        #保留的是在 BEV 可视化范围内且被 GndNet 预测为非地面点的点。
        prev_bev_visible = self._filter_for_bev_mask(prev_bev)
        curr_bev_visible = self._filter_for_bev_mask(curr_bev)
        prev_keep = prev_bev_visible & prev_nonground_mask
        curr_keep = curr_bev_visible & curr_nonground_mask
        summary = {
            "nonground_filter_method": "gndnet",
            "nonground_filter_backend": "gndnet",
            "gndnet_status": "ok",
        }
        summary.update({f"prev_gndnet_{key}": value for key, value in prev_info.items()})
        summary.update({f"curr_gndnet_{key}": value for key, value in curr_info.items()})
        return (
            prev_bev[prev_keep],
            curr_bev[curr_keep],
            prev_xyzi[prev_keep],
            curr_xyzi[curr_keep],
            curr_xyzi[curr_bev_visible],
            summary,
        )

    #加载 GndNet 模型
    def _get_gndnet_filter(self, device):
        if self._gndnet_filter is not None:
            return self._gndnet_filter

        #获取Gndnet根目录
        repo_root = self.cfg.get(
            "gndnet_repo_root",
            os.path.join(self.project_root, "third_party", "GndNet"),
        )
        #获取模型参数
        checkpoint_path = self.cfg.get(
            "gndnet_checkpoint_path",
            os.path.join(repo_root, "trained_models", "checkpoint.pth.tar"),
        )
        #获取配置文件
        config_path = self.cfg.get(
            "gndnet_config_path",
            os.path.join(repo_root, "config", "config_kittiSem.yaml"),
        )
        gndnet_device = self.cfg.get("gndnet_device", str(device))
        self._gndnet_filter = GndNetGroundFilter(
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=gndnet_device,
            threshold_m=self.cfg.get("gndnet_ground_threshold_m", 0.2),
            keep_outside_as_nonground=self.cfg.get("gndnet_keep_outside_as_nonground", True),
        )
        return self._gndnet_filter

    def _image_path_for_time_idx(self, dataset, time_idx, frame_id):
        color_paths = getattr(dataset, "color_paths", None)
        if color_paths is not None and time_idx < len(color_paths):
            return color_paths[time_idx]
        input_folder = getattr(dataset, "input_folder", "")
        name = getattr(dataset, "name", "").lower()
        if name == "kitti360" and input_folder:
            return os.path.join(input_folder, "image_00", "data_rect", f"{frame_id}.png")
        if name == "kitti" and input_folder:
            return os.path.join(input_folder, "image_2", f"{frame_id}.png")
        return None

    #获取投影矩阵和旋转矩阵，用于将 LiDAR 点云投影到图像平面上。
    def _get_projection_calib(self):
        if self._projection_calib is not None:
            return self._projection_calib
        perspective_path = os.path.join(self.project_root, "data", "kitti360", "calibration", "perspective.txt")
        if not os.path.isfile(perspective_path):
            raise FileNotFoundError(f"Missing KITTI-360 perspective calibration: {perspective_path}")
        params = {}
        with open(perspective_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                params[key.strip()] = value.strip()
        p_rect = np.asarray([float(x) for x in params["P_rect_00"].split()], dtype=np.float64).reshape(3, 4)
        r_rect = np.asarray([float(x) for x in params["R_rect_00"].split()], dtype=np.float64).reshape(3, 3)
        self._projection_calib = (p_rect, r_rect)
        return self._projection_calib

    #将 LiDAR 点云投影到图像平面上，并计算哪些点在图像中是可见的。返回投影后的像素坐标、可见性掩码和深度信息。
    def _project_velodyne_to_image(self, points_xyz, image_shape):
        if points_xyz is None or points_xyz.shape[0] == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=bool),
                np.empty((0,), dtype=np.float64),
            )
        #获取投影矩阵和旋转矩阵，用于将 LiDAR 点云投影到图像平面上。
        p_rect, r_rect = self._get_projection_calib()
        points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
        cam = (self._velo_to_cam @ points_h.T).T[:, :3]
        rect = (r_rect @ cam.T).T
        rect_h = np.concatenate([rect, np.ones((rect.shape[0], 1), dtype=np.float64)], axis=1)
        proj = (p_rect @ rect_h.T).T
        depth = proj[:, 2]
        uv = np.full((proj.shape[0], 2), np.nan, dtype=np.float64)
        positive = depth > 1e-12
        uv[positive] = proj[positive, :2] / depth[positive, None]

        height, width = image_shape[:2]
        valid = np.isfinite(uv).all(axis=1)
        valid &= depth > float(self.cfg.get("feature_residual_projection_min_depth_m", 0.1))
        max_depth = float(self.cfg.get("feature_residual_projection_max_depth_m", 80.0))
        if max_depth > 0.0:
            valid &= depth < max_depth
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        return uv, valid, depth

    def _projection_mask_from_uv(self, uv, valid, image_shape, radius, cv2):
        height, width = image_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if uv.shape[0] == 0 or not np.any(valid):
            return mask.astype(bool)
        visible_uv = uv[valid]
        xs = np.round(visible_uv[:, 0]).astype(np.int64)
        ys = np.round(visible_uv[:, 1]).astype(np.int64)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not np.any(inside):
            return mask.astype(bool)
        mask[ys[inside], xs[inside]] = 255
        radius = max(0, int(round(float(radius))))
        if radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * radius + 1, 2 * radius + 1),
            )
            mask = cv2.dilate(mask, kernel)
        return mask.astype(bool)

    def _save_lidar_static_image_masks(
        self,
        pair_dir,
        curr_visible_xyzi,
        curr_nonground_xyzi,
        image_path,
    ):
        if image_path is None or not os.path.isfile(image_path):
            return {
                "static_projection_status": "skipped_missing_image",
                "static_projection_image_path": image_path or "",
            }
        try:
            import cv2
        except ImportError as exc:
            return {
                "static_projection_status": "skipped_missing_cv2",
                "static_projection_error": str(exc),
            }

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            visible_uv, visible_valid, _visible_depth = self._project_velodyne_to_image(
                curr_visible_xyzi[:, :3],
                image.shape,
            )
            nonground_uv, nonground_valid, _nonground_depth = self._project_velodyne_to_image(
                curr_nonground_xyzi[:, :3],
                image.shape,
            )

            point_radius = int(
                self.cfg.get(
                    "static_projection_point_radius",
                    self.cfg.get("feature_residual_projection_point_radius", 3),
                )
            )
            visible_mask = self._projection_mask_from_uv(
                visible_uv,
                visible_valid,
                image.shape,
                point_radius,
                cv2,
            )
            nonground_mask = self._projection_mask_from_uv(
                nonground_uv,
                nonground_valid,
                image.shape,
                point_radius,
                cv2,
            )

            height, width = image.shape[:2]
            above_range_mask = np.zeros((height, width), dtype=bool)
            row_cutoff = 0
            top_row = None
            if np.any(visible_valid):
                rows = visible_uv[visible_valid, 1]
                row_percentile = float(self.cfg.get("static_projection_top_row_percentile", 0.1))
                row_margin = float(self.cfg.get("static_projection_top_row_margin_px", 8.0))
                top_row = float(np.percentile(rows, row_percentile))
                row_cutoff = int(np.floor(top_row - row_margin))
                row_cutoff = max(0, min(height, row_cutoff))
                above_range_mask[:row_cutoff, :] = True

            static_exclusion_mask = above_range_mask
            outputs = {
                "image_lidar_visible_mask.png": visible_mask,
                "image_lidar_nonground_mask.png": nonground_mask,
                "image_lidar_above_range_mask.png": above_range_mask,
                "image_lidar_static_exclusion_mask.png": static_exclusion_mask,
            }
            for filename, mask in outputs.items():
                cv2.imwrite(
                    os.path.join(pair_dir, filename),
                    mask.astype(np.uint8) * 255,
                )

            out_npz = os.path.join(pair_dir, "image_lidar_static_masks.npz")
            np.savez_compressed(
                out_npz,
                visible_mask=visible_mask,
                nonground_mask=nonground_mask,
                above_lidar_range_mask=above_range_mask,
                static_exclusion_mask=static_exclusion_mask,
                image_shape=np.asarray(image.shape[:2], dtype=np.int32),
                top_row=np.asarray(-1.0 if top_row is None else top_row, dtype=np.float32),
                row_cutoff=np.asarray(row_cutoff, dtype=np.int32),
                visible_points=np.asarray(int(np.count_nonzero(visible_valid)), dtype=np.int32),
                nonground_points=np.asarray(int(np.count_nonzero(nonground_valid)), dtype=np.int32),
            )
            return {
                "static_projection_status": "ok",
                "static_projection_npz": out_npz,
                "static_projection_visible_points": int(np.count_nonzero(visible_valid)),
                "static_projection_nonground_points": int(np.count_nonzero(nonground_valid)),
                "static_projection_top_row": top_row,
                "static_projection_row_cutoff": int(row_cutoff),
                "static_projection_above_range_fraction": float(above_range_mask.mean()),
            }
        except Exception as exc:
            return {
                "static_projection_status": "skipped_error",
                "static_projection_image_path": image_path,
                "static_projection_error": str(exc),
            }

    #保存bev_overlay.png和bev_overlay_nonground.png
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

    #保存bev_residual_nonground_features.png和bev_features_nonground.png
    def _save_bev_feature_residual(
        self,
        pair_dir,
        prev_xyz,
        curr_xyz,
        filename="bev_residual_features.png",
        mask_filename=None,
        curr_xyzi=None,
        image_path=None,
        projection_filename="image_residual_features.png",
    ):
        #如果前后两帧点云为空，则跳过计算特征残差。
        if prev_xyz.shape[0] == 0 or curr_xyz.shape[0] == 0:
            return {"feature_residual_status": "skipped_empty_pointcloud"}

        max_points = int(
            self.cfg.get(
                "feature_residual_max_points",
                60000,
            )
        )
        #根据最大点数限制，从前后两帧点云中均匀地选择一些点进行特征残差计算。
        prev_work_indices = _uniform_downsample_indices(prev_xyz.shape[0], max_points)
        curr_work_indices = _uniform_downsample_indices(curr_xyz.shape[0], max_points)
        prev_work = prev_xyz[prev_work_indices]
        curr_work = curr_xyz[curr_work_indices]
        curr_work_xyzi = curr_xyzi[curr_work_indices] if curr_xyzi is not None else None

        #计算前后两帧点云的局部几何特征，包括法向量和曲率等信息，并根据这些特征生成结构化特征掩码，筛选出具有明显结构特征的点。
        prev_metrics = self._compute_local_geometry_metrics(prev_work)
        curr_metrics = self._compute_local_geometry_metrics(curr_work)
        #根据局部几何特征生成 结构化特征掩码 ，筛选出具有明显结构特征的点。
        prev_feature_mask = self._structural_feature_mask(prev_metrics)
        curr_feature_mask = self._structural_feature_mask(curr_metrics)
        prev_features = prev_work[prev_feature_mask]
        curr_features = curr_work[curr_feature_mask]
        #如果提供了当前帧点云的 xyzi 信息，则根据 结构化特征掩码 筛选出对应的 xyzi 信息。
        curr_feature_xyzi = curr_work_xyzi[curr_feature_mask] if curr_work_xyzi is not None else None
        #根据前一帧点云的 结构化特征掩码 筛选出对应的法向量和平面度信息。
        prev_normals = prev_metrics["normals"][prev_feature_mask]
        prev_planarity = prev_metrics["planarity"][prev_feature_mask]

        #保存bev_features_nonground.png
        if mask_filename and self.cfg.get("save_feature_mask_visualization", True):
            self._save_bev_feature_mask(
                pair_dir,
                prev_work,
                curr_work,
                prev_feature_mask,
                curr_feature_mask,
                filename=mask_filename,
            )

        #如果前后两帧点云的结构化特征点数不足，则跳过计算特征残差。
        min_features = int(self.cfg.get("feature_residual_min_points", 200))
        if prev_features.shape[0] < min_features or curr_features.shape[0] < min_features:
            return {
                "feature_residual_status": "skipped_too_few_features",
                "prev_feature_points": int(prev_features.shape[0]),
                "curr_feature_points": int(curr_features.shape[0]),
                "prev_feature_rate": float(prev_features.shape[0] / max(prev_work.shape[0], 1)),
                "curr_feature_rate": float(curr_features.shape[0] / max(curr_work.shape[0], 1)),
            }

        prev_pcd = o3d.geometry.PointCloud()
        prev_pcd.points = o3d.utility.Vector3dVector(prev_features.astype(np.float64))
        kd_tree = o3d.geometry.KDTreeFlann(prev_pcd)

        #计算特征残差时使用的参数，包括残差计算模式、欧几里得距离权重和平面度阈值。
        residual_mode = str(self.cfg.get("feature_residual_mode", "hybrid")).lower()
        euclidean_weight = float(self.cfg.get("feature_residual_euclidean_weight", 0.35))
        plane_min_planarity = float(
            self.cfg.get(
                "feature_residual_plane_min_planarity",
                self.cfg.get("feature_min_planarity", 0.35),
            )
        )

        #计算当前帧点云的每个结构化特征点在前一帧点云中的最近邻点，并计算残差。
        residuals = np.empty(curr_features.shape[0], dtype=np.float32)
        for idx, point in enumerate(curr_features):
            _, nn_indices, dists2 = kd_tree.search_knn_vector_3d(point.astype(np.float64), 1)
            if not dists2:
                residuals[idx] = np.nan
                continue
            nn_idx = int(nn_indices[0])
            euclidean = float(np.sqrt(dists2[0]))
            delta = point - prev_features[nn_idx]
            point_to_plane = float(abs(delta @ prev_normals[nn_idx]))
            if residual_mode == "point_to_plane" and prev_planarity[nn_idx] >= plane_min_planarity:
                residuals[idx] = point_to_plane
            elif residual_mode == "nearest":
                residuals[idx] = euclidean
            else:
                residuals[idx] = max(point_to_plane, euclidean_weight * euclidean)

        #首先获取配置中设置的最大可视化残差值 vis_max，然后创建一个 Matplotlib 图形和坐标轴。接着绘制前一帧点云和当前帧点云的散点图，分别使用不同的颜色和透明度。对于当前帧的结构化特征点，根据计算得到的残差值进行颜色映射，使用 "magma" 颜色映射，并设置颜色条的范围为 [0, vis_max]。最后对坐标轴进行格式化，添加颜色条，并保存图像到指定路径。
        vis_max = float(self.cfg.get("feature_residual_vis_max_m", 1.5))
        fig, ax = plt.subplots(figsize=(8, 9), dpi=180)
        ax.scatter(prev_work[:, 0], prev_work[:, 2], s=0.08, c="#9aa3ad", alpha=0.10)
        ax.scatter(prev_features[:, 0], prev_features[:, 2], s=0.12, c="#6f7b86", alpha=0.30)
        points = ax.scatter(
            curr_features[:, 0],
            curr_features[:, 2],
            s=max(float(self.cfg.get("point_size", 0.2)), 0.45),
            c=np.clip(residuals, 0.0, vis_max),
            cmap="magma",
            vmin=0.0,
            vmax=vis_max,
            alpha=0.90,
        )
        self._format_bev_axis(ax)
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04, label="feature residual (m)")
        fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, filename))
        plt.close(fig)

        #将当前帧点云的结构化特征点投影到图像平面上，并根据残差值生成一个叠加图像，保存为 image_residual_features.png。
        projection_summary = self._save_feature_residual_image_projection(
            pair_dir,
            curr_features,
            curr_feature_xyzi,
            residuals,
            image_path,
            projection_filename,
        )

        #计算残差的统计信息，包括均值、中位数和第 90 百分位数，并返回一个包含这些信息的字典。
        finite = residuals[np.isfinite(residuals)]
        if finite.size == 0:
            return {
                "feature_residual_status": "no_finite_distances",
                "prev_feature_points": int(prev_features.shape[0]),
                "curr_feature_points": int(curr_features.shape[0]),
                **projection_summary,
            }
        return {
            "feature_residual_status": "ok",
            "feature_residual_points": int(finite.size),
            "feature_residual_mean_m": float(np.mean(finite)),
            "feature_residual_median_m": float(np.median(finite)),
            "feature_residual_p90_m": float(np.percentile(finite, 90.0)),
            "prev_feature_points": int(prev_features.shape[0]),
            "curr_feature_points": int(curr_features.shape[0]),
            "prev_feature_rate": float(prev_features.shape[0] / max(prev_work.shape[0], 1)),
            "curr_feature_rate": float(curr_features.shape[0] / max(curr_work.shape[0], 1)),
            **projection_summary,
        }

    #保存bev_residual_features.png的同时，还可以将当前帧点云的结构化特征点投影到图像平面上，并根据残差值生成一个叠加图像，保存为 image_residual_features.png。
    def _save_feature_residual_image_projection(
        self,
        pair_dir,
        curr_feature_bev,
        curr_feature_xyzi,
        residuals,
        image_path,
        filename,
    ):
        #如果配置中禁用了特征残差投影到图像的功能，则直接返回状态为 "disabled"。
        if not self.cfg.get("save_feature_residual_image_projection", True):
            return {"feature_residual_projection_status": "disabled"}
        if curr_feature_xyzi is None:
            return {"feature_residual_projection_status": "skipped_missing_curr_xyzi"}
        if image_path is None or not os.path.isfile(image_path):
            return {
                "feature_residual_projection_status": "skipped_missing_image",
                "feature_residual_projection_image_path": image_path or "",
            }
        try:
            import cv2
        except ImportError as exc:
            return {
                "feature_residual_projection_status": "skipped_missing_cv2",
                "feature_residual_projection_error": str(exc),
            }

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            #将当前帧点云的结构化特征点投影到图像平面上，并计算哪些点在图像中是可见的。然后根据残差值对可见点进行颜色映射，生成一个叠加图像，并保存到指定路径。
            uv, valid_projection, depth = self._project_velodyne_to_image(curr_feature_xyzi[:, :3], image.shape)
            visible = valid_projection & np.isfinite(residuals)
            vis_max = float(
                self.cfg.get(
                    "feature_residual_projection_vis_max_m",
                    self.cfg.get("feature_residual_vis_max_m", 1.5),
                )
            )
            #绘制特征残差投影到图像的叠加图像
            overlay = self._draw_feature_residual_projection(
                image,
                uv[visible],
                depth[visible],
                residuals[visible],
                vis_max,
                cv2,
            )
            out_image = os.path.join(pair_dir, filename)
            cv2.imwrite(out_image, overlay)

            out_npz = ""
            if self.cfg.get("save_feature_residual_projection_npz", True):
                out_npz = os.path.join(pair_dir, os.path.splitext(filename)[0] + ".npz")
                np.savez_compressed(
                    out_npz,
                    curr_feature_bev=curr_feature_bev.astype(np.float32),
                    curr_feature_xyzi=curr_feature_xyzi.astype(np.float32),
                    residuals=residuals.astype(np.float32),
                    uv=uv.astype(np.float32),
                    depth=depth.astype(np.float32),
                    valid_projection=valid_projection,
                    visible=visible,
                )

            return {
                "feature_residual_projection_status": "ok",
                "feature_residual_projection_image_path": image_path,
                "feature_residual_projection_output": out_image,
                "feature_residual_projection_npz": out_npz,
                "feature_residual_projected_points": int(np.count_nonzero(valid_projection)),
                "feature_residual_visible_points": int(np.count_nonzero(visible)),
            }
        except Exception as exc:
            return {
                "feature_residual_projection_status": "skipped_error",
                "feature_residual_projection_image_path": image_path,
                "feature_residual_projection_error": str(exc),
            }
    #绘制特征残差投影到图像的叠加图像
    def _draw_feature_residual_projection(self, image, uv, depth, residuals, vis_max, cv2):
        overlay = image.copy()
        if residuals.shape[0] == 0:
            self._draw_feature_residual_colorbar(overlay, vis_max, cv2)
            return overlay

        order = np.argsort(depth)[::-1]
        uv_ordered = uv[order]
        residuals_ordered = residuals[order]
        values = np.clip(residuals_ordered / max(float(vis_max), 1e-12), 0.0, 1.0)
        colormap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)
        colors = cv2.applyColorMap((values[:, None] * 255.0).astype(np.uint8), colormap).reshape(-1, 3)
        radius = int(self.cfg.get("feature_residual_projection_point_radius", 3))
        #绘制每个可见点的残差值在图像上的投影，使用不同的颜色表示不同的残差值。对于每个点，根据其投影坐标和颜色，在叠加图像上绘制一个圆点或单个像素点，具体取决于配置中的半径设置。然后将叠加图像与原始图像进行加权融合，生成最终的可视化图像。
        for point_uv, color in zip(uv_ordered, colors):
            point = (int(round(point_uv[0])), int(round(point_uv[1])))
            color_bgr = tuple(int(x) for x in color)
            if radius <= 1:
                overlay[point[1], point[0]] = color_bgr
            else:
                cv2.circle(overlay, point, radius, color_bgr, -1, lineType=cv2.LINE_AA)

        alpha = float(self.cfg.get("feature_residual_projection_alpha", 0.88))
        blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)
        #绘制颜色条
        self._draw_feature_residual_colorbar(blended, vis_max, cv2)
        return blended

    #绘制特征残差颜色条
    def _draw_feature_residual_colorbar(self, image, vis_max, cv2):
        height, width = image.shape[:2]
        bar_h = min(180, max(90, height - 48))
        bar_w = 16
        x0 = width - 42
        y0 = 24
        colormap = getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET)
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
        colors = cv2.applyColorMap(gradient, colormap)
        image[y0 : y0 + bar_h, x0 : x0 + bar_w] = colors
        cv2.rectangle(image, (x0 - 1, y0 - 1), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)
        cv2.putText(
            image,
            f"{vis_max:.1f}m",
            (x0 - 34, y0 + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "0",
            (x0 - 14, y0 + bar_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    #计算点云的局部几何特征，包括线性度、平面度、散射度、曲率、邻居半径和法向量等信息。
    def _compute_local_geometry_metrics(self, xyz):
        count = xyz.shape[0]
        #含义：线性度(linearity) = (λ1 - λ2) / λ1，平面度(planarity) = (λ2 - λ3) / λ1，散射度(scattering) = λ3 / λ1，曲率(curvature) = λ3 / (λ1 + λ2 + λ3)，邻居半径(neighbor_radius_m) = 最近邻点的最大距离，法向量(normals) = 协方差矩阵的最小特征值对应的特征向量。
        metrics = {
            "linearity": np.zeros(count, dtype=np.float32),
            "planarity": np.zeros(count, dtype=np.float32),
            "scattering": np.ones(count, dtype=np.float32),
            "curvature": np.ones(count, dtype=np.float32),
            "neighbor_radius_m": np.full(count, np.inf, dtype=np.float32),
            "normals": np.zeros((count, 3), dtype=np.float64),
            "valid": np.zeros(count, dtype=bool),
        }
        if count == 0:
            return metrics

        #根据配置中的参数，确定用于计算局部几何特征的最近邻点数量，确保至少有4个邻居点。
        k_neighbors = max(int(self.cfg.get("feature_knn", 20)), 4)
        k_neighbors = min(k_neighbors, count)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        kd_tree = o3d.geometry.KDTreeFlann(pcd)

        #遍历每个点，计算其局部几何特征。对于每个点，使用 KD 树查找其 k 个最近邻点，并计算这些邻居点的协方差矩阵，然后求解特征值和特征向量，从而得到线性度、平面度、散射度、曲率、邻居半径和法向量等信息。
        for idx, point in enumerate(xyz):
            #使用 KD 树查找当前点的 k 个最近邻点，并获取邻居点的索引和距离平方。
            nn_count, nn_indices, dists2 = kd_tree.search_knn_vector_3d(
                point.astype(np.float64),
                k_neighbors,
            )
            if nn_count < 4:
                continue
            #计算邻居点的协方差矩阵，并求解其特征值和特征向量。
            neighbors = xyz[np.asarray(nn_indices, dtype=np.int64)]
            centered = neighbors - np.mean(neighbors, axis=0, keepdims=True)
            covariance = centered.T @ centered / max(nn_count - 1, 1)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            #根据特征值计算线性度、平面度、散射度和曲率等局部几何特征，并将结果存储在 metrics 字典中。
            lambda1 = float(eigenvalues[2])
            lambda2 = float(eigenvalues[1])
            lambda3 = float(eigenvalues[0])
            total = lambda1 + lambda2 + lambda3
            if lambda1 <= 1e-12 or total <= 1e-12:
                continue
            metrics["linearity"][idx] = (lambda1 - lambda2) / lambda1
            metrics["planarity"][idx] = (lambda2 - lambda3) / lambda1
            metrics["scattering"][idx] = lambda3 / lambda1
            metrics["curvature"][idx] = lambda3 / total
            metrics["neighbor_radius_m"][idx] = float(np.sqrt(max(dists2))) if dists2 else np.inf
            metrics["normals"][idx] = eigenvectors[:, 0]
            metrics["valid"][idx] = True
        return metrics

    #根据局部几何特征生成结构化特征掩码，筛选出具有明显结构特征的点。
    def _structural_feature_mask(self, metrics):
        #根据配置中的阈值，筛选出具有明显结构特征的点。具体来说，线性度(linearity)和平面度(planarity)用于判断点是否具有结构特征，而散射度(scattering)、曲率(curvature)和邻居半径(neighbor_radius_m)用于进一步过滤不稳定的点。
        min_linearity = float(self.cfg.get("feature_min_linearity", 0.45))
        min_planarity = float(self.cfg.get("feature_min_planarity", 0.35))
        max_scattering = float(self.cfg.get("feature_max_scattering", 0.20))
        max_curvature = float(self.cfg.get("feature_max_curvature", 0.12))
        max_neighbor_radius = float(self.cfg.get("feature_max_neighbor_radius_m", 1.8))

        #根据线性度和平面度的阈值，筛选出具有结构特征的点。具体来说，如果一个点的线性度大于等于 min_linearity 或者平面度大于等于 min_planarity，则认为该点具有结构特征。
        structural = (metrics["linearity"] >= min_linearity) | (metrics["planarity"] >= min_planarity)
        stable = (
            metrics["valid"]
            & structural
            & (metrics["scattering"] <= max_scattering)
            & (metrics["curvature"] <= max_curvature)
            & (metrics["neighbor_radius_m"] <= max_neighbor_radius)
        )
        return stable

    #保存bev_features.png和bev_features_nonground.png
    def _save_bev_feature_mask(
        self,
        pair_dir,
        prev_xyz,
        curr_xyz,
        prev_feature_mask,
        curr_feature_mask,
        filename="bev_features.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 9), dpi=180)
        ax.scatter(prev_xyz[:, 0], prev_xyz[:, 2], s=0.08, c="#9aa3ad", alpha=0.10)
        ax.scatter(curr_xyz[:, 0], curr_xyz[:, 2], s=0.08, c="#9aa3ad", alpha=0.10)
        if np.any(prev_feature_mask):
            prev_features = prev_xyz[prev_feature_mask]
            ax.scatter(prev_features[:, 0], prev_features[:, 2], s=0.22, c="#2f6fdb", alpha=0.55, label="prev features")
        if np.any(curr_feature_mask):
            curr_features = curr_xyz[curr_feature_mask]
            ax.scatter(curr_features[:, 0], curr_features[:, 2], s=0.22, c="#d94841", alpha=0.55, label="curr features")
        self._format_bev_axis(ax)
        ax.legend(loc="upper right", markerscale=8, frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(pair_dir, filename))
        plt.close(fig)

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

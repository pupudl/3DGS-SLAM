#!/usr/bin/env python3
"""
KITTI-360 front-end odometry ablation for LSG-SLAM.

This experiment intentionally lives outside the main pipeline code. It reuses
the KITTI360 dataset definition and follows the big-flow pose convention:
- dataset poses are `c2w` and relative to the first loaded frame
- front-end estimates are accumulated as `w2c`
- plotting converts poses back to `c2w` and draws the x-z plane
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from glob import glob
from importlib.machinery import SourceFileLoader
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lsgslam")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from feature_matching import estimate_pnp, extract_feature, match_feature
from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint

try:
    import open3d as o3d
except ImportError:
    o3d = None


@dataclass
class FrameData:
    color: np.ndarray
    gray: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    gt_c2w: np.ndarray
    gt_w2c: np.ndarray
    frame_id: str
    color_t: torch.Tensor
    depth_t: torch.Tensor
    depth_original_t: torch.Tensor
    intrinsics_t: torch.Tensor
    gt_w2c_t: torch.Tensor
    feats: Optional[dict] = None
    descs: Optional[torch.Tensor] = None


@dataclass
class StepInfo:
    frame_idx: int
    frame_id: str
    pnp_success: bool
    pnp_inliers: int
    matches: int
    used_icp: bool
    icp_fitness: float
    icp_rmse: float
    fallback: str


def natural_key(path: str) -> List[object]:
    name = os.path.basename(path)
    parts = []
    current = ""
    is_digit = False
    for char in name:
        if char.isdigit() == is_digit:
            current += char
        else:
            if current:
                parts.append(int(current) if is_digit else current)
            current = char
            is_digit = char.isdigit()
    if current:
        parts.append(int(current) if is_digit else current)
    return parts


class LocalKittiDataset:
    """
    Minimal local copy of the repo's KITTI/KITTI360 dataset behavior.

    KITTI expects:
        {basedir}/{sequence}/image_2/*.png
        {basedir}/{sequence}/depth_sceneflow/*.npy
        {basedir}/{sequence}/traj.txt

    KITTI360 expects:
        {basedir}/{sequence}/image_00/data_rect/*.png
        {basedir}/{sequence}/depth_sceneflow/*.npy
        {basedir}/{sequence}/traj.txt
    """

    def __init__(
        self,
        config_dict: dict,
        basedir: str,
        sequence: str,
        start: int,
        end: int,
        stride: int,
        desired_height: int,
        desired_width: int,
        device: str,
        relative_pose: bool = True,
    ) -> None:
        self.config_dict = config_dict
        self.dataset_name = config_dict["dataset_name"].lower()
        self.basedir = basedir
        self.sequence = sequence
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        cam = config_dict["camera_params"]
        self.orig_height = cam["image_height"]
        self.orig_width = cam["image_width"]
        self.fx = cam["fx"]
        self.fy = cam["fy"]
        self.cx = cam["cx"]
        self.cy = cam["cy"]
        self.png_depth_scale = cam["png_depth_scale"]
        self.depth_filter_near = cam["depth_filter_near"]
        self.depth_filter_far = cam["depth_filter_far"]
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.height_downsample_ratio = float(desired_height) / self.orig_height
        self.width_downsample_ratio = float(desired_width) / self.orig_width
        self.device = device
        self.relative_pose = relative_pose

        depth_paths = sorted(glob(os.path.join(self.input_folder, "depth_sceneflow", "*.npy")), key=natural_key)
        frame_ids = [os.path.splitext(os.path.basename(path))[0] for path in depth_paths]
        if self.dataset_name == "kitti360":
            color_dir = os.path.join(self.input_folder, "image_00", "data_rect")
        elif self.dataset_name == "kitti":
            color_dir = os.path.join(self.input_folder, "image_2")
        else:
            raise ValueError(f"Unsupported dataset for this experiment: {self.dataset_name}")
        color_paths = [os.path.join(color_dir, f"{frame_id}.png") for frame_id in frame_ids]
        poses = self.load_poses()
        if len(poses) != len(depth_paths):
            raise ValueError(
                f"Pose/depth count mismatch: {len(poses)} poses in {self.pose_path}, "
                f"{len(depth_paths)} depth files under depth_sceneflow"
            )

        if end == -1:
            end_exclusive = len(depth_paths)
        else:
            end_exclusive = end + 1

        self.color_paths = color_paths[start:end_exclusive:stride]
        self.depth_paths = depth_paths[start:end_exclusive:stride]
        self.poses = poses[start:end_exclusive:stride]
        if len(self.color_paths) == 0:
            raise ValueError("No KITTI360 frames selected. Check --basedir/--sequence/--start/--end.")

        self.poses = np.asarray(self.poses, dtype=np.float64)
        if self.relative_pose:
            first_w2c = np.linalg.inv(self.poses[0])
            self.poses = np.asarray([first_w2c @ pose for pose in self.poses], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.color_paths)

    def load_poses(self) -> List[np.ndarray]:
        poses = []
        with open(self.pose_path, "r") as f:
            for line in f:
                values = [float(v) for v in line.strip().split()]
                if len(values) != 12:
                    raise ValueError(f"Expected 12 KITTI pose values, got {len(values)} in {self.pose_path}")
                pose = np.eye(4, dtype=np.float64)
                pose[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
                poses.append(pose)
        return poses

    def intrinsics(self) -> np.ndarray:
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = self.fx * self.width_downsample_ratio
        K[1, 1] = self.fy * self.height_downsample_ratio
        K[0, 2] = self.cx * self.width_downsample_ratio
        K[1, 2] = self.cy * self.height_downsample_ratio
        return K

    def __getitem__(self, index: int):
        color_bgr = cv2.imread(self.color_paths[index], cv2.IMREAD_COLOR)
        if color_bgr is None:
            raise FileNotFoundError(f"Could not read image: {self.color_paths[index]}")
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)
        color = cv2.resize(color, (self.desired_width, self.desired_height), interpolation=cv2.INTER_LINEAR)
        depth_original = np.load(self.depth_paths[index]).astype(np.float64)
        depth = depth_original.copy()
        valid = (depth > self.depth_filter_near) & (depth < self.depth_filter_far)
        depth[~valid] = 0.0
        depth = cv2.resize(depth, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        depth_original = cv2.resize(
            depth_original,
            (self.desired_width, self.desired_height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth / self.png_depth_scale, axis=-1)
        depth_original = np.expand_dims(depth_original / self.png_depth_scale, axis=-1)

        intrinsics = np.eye(4, dtype=np.float64)
        intrinsics[:3, :3] = self.intrinsics()
        pose = self.poses[index]
        return (
            torch.from_numpy(color).to(self.device).float(),
            torch.from_numpy(depth).to(self.device).float(),
            torch.from_numpy(intrinsics).to(self.device).float(),
            torch.from_numpy(pose).to(self.device).float(),
            torch.from_numpy(depth_original).to(self.device).float(),
            None,
        )


def load_dataset(
    yaml_path: str,
    basedir: str,
    sequence: str,
    start: int,
    end: int,
    stride: int,
    height: int,
    width: int,
    device: str,
) -> LocalKittiDataset:
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return LocalKittiDataset(
        cfg,
        basedir=basedir,
        sequence=sequence,
        start=start,
        end=end,
        stride=stride,
        desired_height=height,
        desired_width=width,
        device=device,
        relative_pose=True,
    )


def frame_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def load_frame(dataset: LocalKittiDataset, idx: int) -> FrameData:
    color_t, _depth_t, intrinsics_t, pose_t, depth_original_t, _global_feature = dataset[idx]
    color = color_t.detach().cpu().numpy()
    color_u8 = np.clip(color, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(color_u8, cv2.COLOR_RGB2GRAY)
    depth = depth_original_t.detach().cpu().numpy()
    if depth.ndim == 3:
        depth = depth[..., 0]
    K = intrinsics_t.detach().cpu().numpy()[:3, :3]
    gt_c2w = pose_t.detach().cpu().numpy()
    gt_w2c = np.linalg.inv(gt_c2w)
    color_chw = (color_t.permute(2, 0, 1) / 255.0).contiguous()
    depth_chw = _depth_t.permute(2, 0, 1).contiguous()
    depth_original_chw = depth_original_t.permute(2, 0, 1).contiguous()
    intrinsics_3x3 = intrinsics_t[:3, :3].contiguous()
    gt_w2c_t = torch.linalg.inv(pose_t).contiguous()
    return FrameData(
        color=color_u8,
        gray=gray,
        depth=depth.astype(np.float64),
        K=K.astype(np.float64),
        gt_c2w=gt_c2w.astype(np.float64),
        gt_w2c=gt_w2c.astype(np.float64),
        frame_id=frame_id_from_path(dataset.color_paths[idx]),
        color_t=color_chw,
        depth_t=depth_chw,
        depth_original_t=depth_original_chw,
        intrinsics_t=intrinsics_3x3,
        gt_w2c_t=gt_w2c_t,
    )


def extract_original_features(frame: FrameData, sp_extractor: SuperPoint, device: torch.device, depth_filter_far: float) -> None:
    color_feature = torch.clone(frame.color_t)
    depth_original = frame.depth_original_t
    mask = (depth_original < 0.1) | (depth_original > np.min([depth_filter_far, 15.0]))
    color_feature[:, mask[0]] = 0
    frame.feats, frame.descs = extract_feature(color_feature, depth_original, sp_extractor, device)


def build_pnp_data(curr: FrameData, prev: FrameData, est_w2c: Sequence[np.ndarray]) -> Tuple[dict, dict]:
    gt_list = [torch.from_numpy(pose).to(curr.color_t.device).float() for pose in [prev.gt_w2c, curr.gt_w2c]]
    curr_data = {
        "im": curr.color_t,
        "depth": curr.depth_t,
        "depth_original": curr.depth_original_t,
        "intrinsics": curr.intrinsics_t,
        "iter_gt_w2c_list": gt_list,
        "id": len(est_w2c) - 1,
        "feats": curr.feats,
        "descs": curr.descs,
    }
    last_data = {
        "im": prev.color_t,
        "depth": prev.depth_t,
        "depth_original": prev.depth_original_t,
        "intrinsics": prev.intrinsics_t,
        "iter_gt_w2c_list": gt_list,
        "est_w2c": torch.from_numpy(est_w2c[-1]).to(prev.color_t.device).float(),
        "id": len(est_w2c) - 2,
        "feats": prev.feats,
        "descs": prev.descs,
    }
    return curr_data, last_data


def depth_to_pointcloud(frame: FrameData, depth_near: float, depth_far: float, max_points: int):
    if o3d is None:
        raise RuntimeError("open3d is required for PnP+ICP mode.")
    depth = frame.depth
    mask = np.isfinite(depth) & (depth > depth_near) & (depth < depth_far)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return o3d.geometry.PointCloud()
    if len(xs) > max_points:
        choice = np.random.choice(len(xs), size=max_points, replace=False)
        xs = xs[choice]
        ys = ys[choice]
    z = depth[ys, xs]
    fx, fy = frame.K[0, 0], frame.K[1, 1]
    cx, cy = frame.K[0, 2], frame.K[1, 2]
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (ys.astype(np.float64) - cy) * z / fy
    pts = np.column_stack([x, y, z])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def read_velodyne_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts.astype(np.float64)


def load_kitti360_velo_to_cam() -> np.ndarray:
    path = os.path.join(PROJECT_ROOT, "data", "kitti360", "calibration", "calib_cam_to_velo.txt")
    with open(path, "r") as f:
        vals = [float(x) for x in f.readline().strip().split()]
    if len(vals) != 12:
        raise ValueError(f"Expected 12 values in {path}, got {len(vals)}")
    # KITTI-360 names this file `calib_cam_to_velo.txt`: it stores cam0 -> Velodyne.
    # The rest of this script wants Velodyne -> cam0, so invert it here.
    T_cam_to_velo = np.eye(4, dtype=np.float64)
    T_cam_to_velo[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return np.linalg.inv(T_cam_to_velo)


def load_kitti_velo_to_cam(dataset: LocalKittiDataset) -> np.ndarray:
    path = os.path.join(dataset.input_folder, "calib.txt")
    p2_vals = None
    tr_vals = None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("P2:"):
                p2_vals = [float(x) for x in line.split(":", 1)[1].strip().split()]
            if line.startswith("Tr:") or line.startswith("Tr_velo_to_cam:"):
                tr_vals = [float(x) for x in line.split(":", 1)[1].strip().split()]
                if len(tr_vals) != 12:
                    raise ValueError(f"Expected 12 Tr values in {path}, got {len(tr_vals)}")

    # KITTI odometry `sequences/*/calib.txt` often only stores P0..P3.
    # Fall back to the standard Velodyne -> cam0 extrinsic and convert it to cam2
    # because this experiment uses `image_2` / depth in the cam2 frame.
    if tr_vals is not None:
        T_velo_to_cam0 = np.eye(4, dtype=np.float64)
        T_velo_to_cam0[:3, :4] = np.asarray(tr_vals, dtype=np.float64).reshape(3, 4)
    else:
        T_velo_to_cam0 = np.array(
            [
                [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    if p2_vals is None or len(p2_vals) != 12:
        return T_velo_to_cam0

    P2 = np.asarray(p2_vals, dtype=np.float64).reshape(3, 4)
    fx = P2[0, 0]
    fy = P2[1, 1]
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        return T_velo_to_cam0

    T_cam0_to_cam2 = np.eye(4, dtype=np.float64)
    T_cam0_to_cam2[0, 3] = P2[0, 3] / fx
    T_cam0_to_cam2[1, 3] = P2[1, 3] / fy
    T_cam0_to_cam2[2, 3] = P2[2, 3]
    return T_cam0_to_cam2 @ T_velo_to_cam0


def lidar_file_for_frame(dataset: LocalKittiDataset, frame_id: str) -> str:
    if dataset.dataset_name == "kitti360":
        data_3d_root = dataset.basedir.replace("data_2d_raw", "data_3d_raw")
        return os.path.join(data_3d_root, dataset.sequence, "velodyne_points", "data", f"{frame_id}.bin")
    if dataset.dataset_name == "kitti":
        return os.path.join(dataset.input_folder, "velodyne", f"{frame_id}.bin")
    raise ValueError(f"Unsupported dataset for LiDAR ICP: {dataset.dataset_name}")


def lidar_to_pointcloud(
    dataset: LocalKittiDataset,
    frame: FrameData,
    max_points: int,
    min_forward_m: float,
    max_forward_m: float,
):
    if o3d is None:
        raise RuntimeError("open3d is required for LiDAR ICP mode.")
    path = lidar_file_for_frame(dataset, frame.frame_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing LiDAR file for frame {frame.frame_id}: {path}. "
            "For KITTI, place Velodyne .bin files under <sequence>/velodyne/."
        )
    xyz = read_velodyne_bin(path)
    if min_forward_m > 0.0 or max_forward_m > 0.0:
        x = xyz[:, 0]
        mask = np.ones(x.shape[0], dtype=bool)
        if min_forward_m > 0.0:
            mask &= x > min_forward_m
        if max_forward_m > 0.0:
            mask &= x < max_forward_m
        xyz = xyz[mask]
    if xyz.shape[0] > max_points:
        idx = np.random.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def transform_pointcloud(pcd, transform: np.ndarray):
    transformed = o3d.geometry.PointCloud(pcd)
    transformed.transform(transform.astype(np.float64))
    return transformed


def merge_pointclouds(pointclouds: Sequence):
    merged = o3d.geometry.PointCloud()
    for pcd in pointclouds:
        if len(pcd.points) == 0:
            continue
        merged += pcd
    return merged


def run_pointcloud_icp(
    source,
    target,
    init_transform: np.ndarray,
    voxel_size: float,
    corr_threshold: float,
) -> Tuple[np.ndarray, float, float]:
    if len(source.points) < 100 or len(target.points) < 100:
        return init_transform, 0.0, float("inf")

    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)
    source, _ = source.remove_radius_outlier(20, max(0.4, voxel_size * 4.0))
    target, _ = target.remove_radius_outlier(20, max(0.4, voxel_size * 4.0))
    if len(source.points) < 50 or len(target.points) < 50:
        return init_transform, 0.0, float("inf")

    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2 * 2.0, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2 * 2.0, max_nn=30))

    transform = init_transform.astype(np.float64)
    reg = None
    thresholds = [corr_threshold, 0.5, 0.1] if corr_threshold > 0.5 else [0.5, 0.1]
    for threshold in thresholds:
        reg = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=0.001 if threshold == 0.1 else 0.1,
                max_iteration=100,
            ),
        )
        transform = np.asarray(reg.transformation, dtype=np.float64)

    assert reg is not None
    return transform, float(reg.fitness), float(reg.inlier_rmse)


def run_icp(
    prev: FrameData,
    curr: FrameData,
    init_T_curr_prev: np.ndarray,
    depth_near: float,
    depth_far: float,
    max_points: int,
    voxel_size: float,
    corr_threshold: float,
) -> Tuple[np.ndarray, float, float]:
    source = depth_to_pointcloud(prev, depth_near, depth_far, max_points)
    target = depth_to_pointcloud(curr, depth_near, depth_far, max_points)
    return run_pointcloud_icp(source, target, init_T_curr_prev, voxel_size, corr_threshold)


def run_lidar_icp(
    dataset: LocalKittiDataset,
    prev: FrameData,
    curr: FrameData,
    init_T_curr_prev_cam: np.ndarray,
    max_points: int,
    voxel_size: float,
    corr_threshold: float,
    min_forward_m: float,
    max_forward_m: float,
) -> Tuple[np.ndarray, float, float]:
    if dataset.dataset_name == "kitti360":
        T_velo_to_cam = load_kitti360_velo_to_cam()
    else:
        T_velo_to_cam = load_kitti_velo_to_cam(dataset)
    T_cam_to_velo = np.linalg.inv(T_velo_to_cam)
    init_T_curr_prev_velo = T_cam_to_velo @ init_T_curr_prev_cam @ T_velo_to_cam

    source = lidar_to_pointcloud(dataset, prev, max_points, min_forward_m, max_forward_m)
    target = lidar_to_pointcloud(dataset, curr, max_points, min_forward_m, max_forward_m)
    T_curr_prev_velo, fitness, rmse = run_pointcloud_icp(
        source,
        target,
        init_T_curr_prev_velo,
        voxel_size,
        corr_threshold,
    )
    T_curr_prev_cam = T_velo_to_cam @ T_curr_prev_velo @ T_cam_to_velo
    return T_curr_prev_cam, fitness, rmse


def run_fused_icp(
    dataset: LocalKittiDataset,
    prev: FrameData,
    curr: FrameData,
    init_T_curr_prev_cam: np.ndarray,
    depth_near: float,
    depth_far: float,
    depth_max_points: int,
    lidar_max_points: int,
    voxel_size: float,
    corr_threshold: float,
    min_forward_m: float,
    max_forward_m: float,
) -> Tuple[np.ndarray, float, float]:
    rgbd_source = depth_to_pointcloud(prev, depth_near, depth_far, depth_max_points)
    rgbd_target = depth_to_pointcloud(curr, depth_near, depth_far, depth_max_points)

    if dataset.dataset_name == "kitti360":
        T_velo_to_cam = load_kitti360_velo_to_cam()
    else:
        T_velo_to_cam = load_kitti_velo_to_cam(dataset)

    lidar_source = lidar_to_pointcloud(dataset, prev, lidar_max_points, min_forward_m, max_forward_m)
    lidar_target = lidar_to_pointcloud(dataset, curr, lidar_max_points, min_forward_m, max_forward_m)
    lidar_source_cam = transform_pointcloud(lidar_source, T_velo_to_cam)
    lidar_target_cam = transform_pointcloud(lidar_target, T_velo_to_cam)

    source = merge_pointclouds([rgbd_source, lidar_source_cam])
    target = merge_pointclouds([rgbd_target, lidar_target_cam])
    return run_pointcloud_icp(source, target, init_T_curr_prev_cam, voxel_size, corr_threshold)


def rotation_angle_deg(R: np.ndarray) -> float:
    value = (np.trace(R) - 1.0) * 0.5
    value = float(np.clip(value, -1.0, 1.0))
    return math.degrees(math.acos(value))


def align_se3_by_positions(gt: np.ndarray, est: np.ndarray) -> np.ndarray:
    gt_pts = gt[:, :3, 3]
    est_pts = est[:, :3, 3]
    gt_mean = gt_pts.mean(axis=0)
    est_mean = est_pts.mean(axis=0)
    X = est_pts - est_mean
    Y = gt_pts - gt_mean
    H = X.T @ Y
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_mean - R @ est_mean

    aligned = est.copy()
    for i in range(len(aligned)):
        aligned[i, :3, :3] = R @ aligned[i, :3, :3]
        aligned[i, :3, 3] = R @ aligned[i, :3, 3] + t
    return aligned


def compute_metrics(gt_c2w: np.ndarray, est_c2w: np.ndarray) -> Dict[str, float]:
    n = min(len(gt_c2w), len(est_c2w))
    gt_c2w = gt_c2w[:n]
    est_c2w = est_c2w[:n]

    ate = np.linalg.norm(gt_c2w[:, :3, 3] - est_c2w[:, :3, 3], axis=1)
    aligned = align_se3_by_positions(gt_c2w, est_c2w)
    ate_aligned = np.linalg.norm(gt_c2w[:, :3, 3] - aligned[:, :3, 3], axis=1)

    rpe_t = []
    rpe_r = []
    for i in range(1, n):
        d_gt = np.linalg.inv(gt_c2w[i - 1]) @ gt_c2w[i]
        d_est = np.linalg.inv(est_c2w[i - 1]) @ est_c2w[i]
        err = np.linalg.inv(d_gt) @ d_est
        rpe_t.append(float(np.linalg.norm(err[:3, 3])))
        rpe_r.append(rotation_angle_deg(err[:3, :3]))

    return {
        "num_frames": int(n),
        "ate_rmse_m": float(np.sqrt(np.mean(ate * ate))),
        "ate_mean_m": float(np.mean(ate)),
        "ate_median_m": float(np.median(ate)),
        "ate_aligned_rmse_m": float(np.sqrt(np.mean(ate_aligned * ate_aligned))),
        "ate_aligned_mean_m": float(np.mean(ate_aligned)),
        "rpe_trans_rmse_m": float(np.sqrt(np.mean(np.square(rpe_t)))) if rpe_t else 0.0,
        "rpe_trans_mean_m": float(np.mean(rpe_t)) if rpe_t else 0.0,
        "rpe_rot_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_r)))) if rpe_r else 0.0,
        "rpe_rot_mean_deg": float(np.mean(rpe_r)) if rpe_r else 0.0,
    }


def write_pose_file(path: str, poses_c2w: Sequence[np.ndarray]) -> None:
    with open(path, "w") as f:
        for pose in poses_c2w:
            f.write(" ".join(f"{v:.9g}" for v in pose[:3, :].reshape(-1)) + "\n")


def write_step_csv(path: str, steps: Sequence[StepInfo]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(StepInfo.__dataclass_fields__.keys()))
        writer.writeheader()
        for step in steps:
            writer.writerow(step.__dict__)


def plot_outputs(
    output_dir: str,
    mode: str,
    gt_c2w: np.ndarray,
    est_c2w: np.ndarray,
    metrics: Dict[str, float],
) -> None:
    gt_xyz = gt_c2w[:, :3, 3]
    est_xyz = est_c2w[:, :3, 3]
    n = min(len(gt_xyz), len(est_xyz))

    plt.figure(figsize=(10, 8))
    plt.plot(gt_xyz[:n, 0], gt_xyz[:n, 2], color="tab:blue", label="GT")
    plt.plot(est_xyz[:n, 0], est_xyz[:n, 2], color="tab:red", label=mode)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(
        f"{mode} vs GT | ATE RMSE {metrics['ate_rmse_m']:.3f} m | "
        f"RPE {metrics['rpe_trans_rmse_m']:.3f} m"
    )
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_xz.png"), dpi=160)
    plt.close()

    ate = np.linalg.norm(gt_xyz[:n] - est_xyz[:n], axis=1)
    rel_t = [0.0]
    rel_r = [0.0]
    for i in range(1, n):
        d_gt = np.linalg.inv(gt_c2w[i - 1]) @ gt_c2w[i]
        d_est = np.linalg.inv(est_c2w[i - 1]) @ est_c2w[i]
        err = np.linalg.inv(d_gt) @ d_est
        rel_t.append(np.linalg.norm(err[:3, 3]))
        rel_r.append(rotation_angle_deg(err[:3, :3]))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(ate, color="tab:purple")
    axes[0].set_ylabel("ATE (m)")
    axes[1].plot(rel_t, color="tab:green")
    axes[1].set_ylabel("RPE trans (m)")
    axes[2].plot(rel_r, color="tab:orange")
    axes[2].set_ylabel("RPE rot (deg)")
    axes[2].set_xlabel("Frame")
    fig.suptitle(f"{mode} error curves")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "error_curves.png"), dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if args.mode in {"pnp_icp", "pnp_lidar_icp", "pnp_fused_icp"} and o3d is None:
        raise RuntimeError(f"{args.mode} mode requires open3d.")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dataset = load_dataset(
        yaml_path=args.yaml,
        basedir=args.basedir,
        sequence=args.sequence,
        start=args.start,
        end=args.end,
        stride=args.stride,
        height=args.height,
        width=args.width,
        device=args.device,
    )
    if args.max_frames > 0:
        num_frames = min(args.max_frames, len(dataset))
    else:
        num_frames = len(dataset)
    if num_frames < 2:
        raise RuntimeError("Need at least two frames for odometry.")

    sp_extractor = SuperPoint(max_num_keypoints=args.max_num_keypoints).eval().to(device)
    match_conf = {
        "width_confidence": 0.99,
        "depth_confidence": 0.95,
    }
    lg_matcher = LightGlue(pretrained="superpoint", **match_conf).eval().to(device)
    match_config = {"tracking": {"use_gt_poses": False}}

    prev = load_frame(dataset, 0)
    with torch.inference_mode():
        extract_original_features(prev, sp_extractor, device, dataset.depth_filter_far)
    gt_c2w = [prev.gt_c2w]
    est_w2c = [np.eye(4, dtype=np.float64)]
    steps: List[StepInfo] = []

    prev_step_T = np.eye(4, dtype=np.float64)
    for idx in range(1, num_frames):
        curr = load_frame(dataset, idx)
        gt_c2w.append(curr.gt_c2w)
        with torch.inference_mode():
            extract_original_features(curr, sp_extractor, device, dataset.depth_filter_far)

        curr_data, last_data = build_pnp_data(curr, prev, est_w2c)
        with torch.inference_mode():
            mkpts_cur, mkpts_last, mscores = match_feature(
                match_config,
                curr.color_t,
                curr.feats,
                curr.intrinsics_t,
                prev.color_t,
                prev.feats,
                lg_matcher,
                device,
                topk=args.match_topk,
                save_path=None,
            )

        num_matches = 0 if mkpts_cur is None else int(mkpts_cur.shape[0])
        pnp_prior_pose_w2c = None
        T_curr_prev = None
        inliers = 0
        if mkpts_cur is not None and num_matches > 10:
            pnp_result = estimate_pnp(mkpts_cur, mkpts_last, curr_data, last_data, dataset)
            if pnp_result is not None:
                pnp_prior_pose_w2c, T_curr_prev, inliers = pnp_result

        pnp_success = T_curr_prev is not None and inliers > args.min_pnp_inliers
        used_icp = False
        icp_fitness = 0.0
        icp_rmse = float("inf")
        fallback = ""

        if pnp_success and args.mode == "pnp_only":
            T_curr_prev = np.asarray(T_curr_prev, dtype=np.float64)
        elif pnp_success and args.mode == "pnp_icp":
            T_curr_prev, icp_fitness, icp_rmse = run_icp(
                prev,
                curr,
                init_T_curr_prev=T_curr_prev,
                depth_near=args.depth_near,
                depth_far=args.depth_far,
                max_points=args.icp_max_points,
                voxel_size=args.icp_voxel_size,
                corr_threshold=args.icp_corr_threshold,
            )
            used_icp = True
        elif pnp_success and args.mode == "pnp_lidar_icp":
            T_curr_prev, icp_fitness, icp_rmse = run_lidar_icp(
                dataset,
                prev,
                curr,
                init_T_curr_prev_cam=T_curr_prev,
                max_points=args.icp_max_points,
                voxel_size=args.icp_voxel_size,
                corr_threshold=args.icp_corr_threshold,
                min_forward_m=args.lidar_min_forward_m,
                max_forward_m=args.lidar_max_forward_m,
            )
            used_icp = True
        elif pnp_success and args.mode == "pnp_fused_icp":
            T_curr_prev, icp_fitness, icp_rmse = run_fused_icp(
                dataset,
                prev,
                curr,
                init_T_curr_prev_cam=T_curr_prev,
                depth_near=args.depth_near,
                depth_far=args.depth_far,
                depth_max_points=args.icp_max_points,
                lidar_max_points=args.fused_lidar_max_points,
                voxel_size=args.icp_voxel_size,
                corr_threshold=args.icp_corr_threshold,
                min_forward_m=args.lidar_min_forward_m,
                max_forward_m=args.lidar_max_forward_m,
            )
            used_icp = True
        else:
            T_curr_prev = prev_step_T.copy()
            fallback = "motion"

        est_w2c.append(T_curr_prev @ est_w2c[-1])
        prev_step_T = T_curr_prev.copy()
        steps.append(
            StepInfo(
                frame_idx=idx,
                frame_id=curr.frame_id,
                pnp_success=pnp_success,
                pnp_inliers=inliers,
                matches=num_matches,
                used_icp=used_icp,
                icp_fitness=icp_fitness,
                icp_rmse=icp_rmse,
                fallback=fallback,
            )
        )
        if idx % args.log_every == 0 or idx == num_frames - 1:
            print(
                f"[{args.mode}] {idx}/{num_frames - 1} frame={curr.frame_id} "
                f"matches={num_matches} pnp_inliers={inliers} "
                f"fallback={fallback or '-'} icp_fitness={icp_fitness:.3f}"
            )
        prev = curr
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    est_c2w = np.asarray([np.linalg.inv(pose) for pose in est_w2c], dtype=np.float64)
    gt_c2w_np = np.asarray(gt_c2w, dtype=np.float64)
    metrics = compute_metrics(gt_c2w_np, est_c2w)
    metrics.update(
        {
            "mode": args.mode,
            "sequence": args.sequence,
            "start": args.start,
            "end": args.end,
            "stride": args.stride,
            "pnp_success_rate": float(np.mean([s.pnp_success for s in steps])) if steps else 0.0,
            "pnp_mean_inliers": float(np.mean([s.pnp_inliers for s in steps])) if steps else 0.0,
        }
    )

    write_pose_file(os.path.join(args.output_dir, "estimated_c2w_kitti.txt"), est_c2w)
    write_pose_file(os.path.join(args.output_dir, "gt_c2w_kitti.txt"), gt_c2w_np)
    write_step_csv(os.path.join(args.output_dir, "per_frame_stats.csv"), steps)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_outputs(args.output_dir, args.mode, gt_c2w_np, est_c2w, metrics)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to: {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITTI-360 PnP-only / PnP+ICP front-end odometry experiment")
    parser.add_argument("config", help="Experiment config, e.g. configs/kitti360/lsgslam_pnp_only.py")
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output_dir", default=None)
    return config_to_args(parser.parse_args())


def resolve_path(path: str, base: str = PROJECT_ROOT) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base, path))


def config_to_args(cli: argparse.Namespace) -> argparse.Namespace:
    config_path = resolve_path(cli.config, os.getcwd())
    config_module = SourceFileLoader("frontend_ablation_config", config_path).load_module()
    config = config_module.config
    data = config["data"]
    frontend = config["frontend"]
    tracking = config.get("tracking", {})

    sequence = data["sequence"] if cli.sequence is None else cli.sequence
    start = data.get("start", 0) if cli.start is None else cli.start
    end = data.get("end", -1) if cli.end is None else cli.end
    stride = data.get("stride", 1) if cli.stride is None else cli.stride
    if cli.sequence is not None or cli.start is not None or cli.end is not None or cli.stride is not None:
        run_name = f"{sequence}_{start}_{end}_{stride}"
    else:
        run_name = config.get("run_name") or f"{sequence}_{start}_{end}_{stride}"
    workdir = config.get("workdir", os.path.join("experiments", "kitti360_pnp_icp_compare", "outputs"))
    output_dir = cli.output_dir or config.get("output_dir") or os.path.join(resolve_path(workdir), run_name)

    return SimpleNamespace(
        mode=frontend["mode"],
        output_dir=output_dir,
        basedir=resolve_path(data["basedir"]),
        sequence=sequence,
        yaml=resolve_path(data["gradslam_data_cfg"]),
        start=start,
        end=end,
        stride=stride,
        max_frames=data.get("num_frames", -1) if cli.max_frames is None else cli.max_frames,
        height=data["desired_image_height"],
        width=data["desired_image_width"],
        device=cli.device or config.get("primary_device", "cuda:0"),
        max_num_keypoints=frontend.get("max_num_keypoints", 1024),
        match_topk=frontend.get("match_topk", 1024),
        min_pnp_inliers=frontend.get("min_pnp_inliers", 50),
        depth_near=frontend.get("depth_near", 0.1),
        depth_far=frontend.get("depth_far", 30.0),
        icp_corr_threshold=tracking.get("icp_corr_threshold", frontend.get("icp_corr_threshold", 0.5)),
        icp_voxel_size=frontend.get("icp_voxel_size", 0.1),
        icp_max_points=frontend.get("icp_max_points", 120000),
        fused_lidar_max_points=frontend.get("fused_lidar_max_points", frontend.get("icp_max_points", 120000)),
        lidar_min_forward_m=frontend.get("lidar_min_forward_m", 0.0),
        lidar_max_forward_m=frontend.get("lidar_max_forward_m", 0.0),
        log_every=config.get("report_global_progress_every", 10),
    )


if __name__ == "__main__":
    run(parse_args())

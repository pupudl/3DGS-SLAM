import os
from typing import Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from utils.slam_external import build_rotation
from utils.slam_helpers import matrix_to_quaternion

DEFAULT_DATA_ROOT = "/home/qiuyu/data/Projects/LSG-SLAM/data"


def _data_root() -> str:
    return os.environ.get("LSGSLAM_DATA_ROOT", DEFAULT_DATA_ROOT)


def get_dataset_frame_id(dataset, time_idx: int) -> str:
    depth_path = dataset.depth_paths[time_idx]
    return os.path.splitext(os.path.basename(depth_path))[0]


def load_kitti360_velo_to_cam(project_root: str) -> np.ndarray:
    path = os.path.join(_data_root(), "kitti360", "calibration", "calib_cam_to_velo.txt")
    with open(path, "r") as f:
        vals = [float(x) for x in f.readline().strip().split()]
    if len(vals) != 12:
        raise ValueError(f"Expected 12 values in {path}, got {len(vals)}")
    t_cam_to_velo = np.eye(4, dtype=np.float64)
    t_cam_to_velo[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
    return np.linalg.inv(t_cam_to_velo)


def load_kitti_velo_to_cam(dataset) -> np.ndarray:
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

    if tr_vals is not None:
        t_velo_to_cam0 = np.eye(4, dtype=np.float64)
        t_velo_to_cam0[:3, :4] = np.asarray(tr_vals, dtype=np.float64).reshape(3, 4)
    else:
        t_velo_to_cam0 = np.array(
            [
                [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
                [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
                [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    if p2_vals is None or len(p2_vals) != 12:
        return t_velo_to_cam0

    p2 = np.asarray(p2_vals, dtype=np.float64).reshape(3, 4)
    fx = p2[0, 0]
    fy = p2[1, 1]
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        return t_velo_to_cam0

    t_cam0_to_cam2 = np.eye(4, dtype=np.float64)
    t_cam0_to_cam2[0, 3] = p2[0, 3] / fx
    t_cam0_to_cam2[1, 3] = p2[1, 3] / fy
    t_cam0_to_cam2[2, 3] = p2[2, 3]
    return t_cam0_to_cam2 @ t_velo_to_cam0


def load_velo_to_cam(dataset, project_root: str) -> np.ndarray:
    if dataset.name.lower() == "kitti360":
        return load_kitti360_velo_to_cam(project_root)
    if dataset.name.lower() == "kitti":
        return load_kitti_velo_to_cam(dataset)
    raise ValueError(f"Unsupported dataset for fused ICP: {dataset.name}")


def lidar_file_for_frame(dataset, frame_id: str) -> str:
    if dataset.name.lower() == "kitti360":
        seq_dir = os.path.abspath(dataset.input_folder)
        seq_name = os.path.basename(seq_dir)
        data_2d_root = os.path.dirname(seq_dir)
        data_root = os.path.dirname(data_2d_root)
        data_3d_root = os.path.join(data_root, "data_3d_raw")
        return os.path.join(data_3d_root, seq_name, "velodyne_points", "data", f"{frame_id}.bin")
    if dataset.name.lower() == "kitti":
        return os.path.join(dataset.input_folder, "velodyne", f"{frame_id}.bin")
    raise ValueError(f"Unsupported dataset for fused ICP: {dataset.name}")


def read_velodyne_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts.astype(np.float64)


def lidar_to_pointcloud(
    dataset,
    frame_id: str,
    max_points: int,
    min_forward_m: float,
    max_forward_m: float,
):
    path = lidar_file_for_frame(dataset, frame_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing LiDAR file for frame {frame_id}: {path}. "
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


def merge_pointclouds(pointclouds):
    merged = o3d.geometry.PointCloud()
    for pcd in pointclouds:
        if len(pcd.points) == 0:
            continue
        merged += pcd
    return merged


def update_camera_pose_from_relative(
    params,
    curr_time_idx: int,
    pre2now: np.ndarray,
    device,
):
    with torch.no_grad():
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., curr_time_idx - 1].detach())
        cam_tran = params["cam_trans"][..., curr_time_idx - 1].detach()
        w2pre = torch.eye(4).float().to(device=device)
        w2pre[:3, :3] = build_rotation(cam_rot)
        w2pre[:3, 3] = cam_tran
        pre2now_tmp = torch.tensor(pre2now, dtype=torch.float32, device=device)
        w2now = pre2now_tmp @ w2pre
        rel_w2c_rot = w2now[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
        rel_w2c_tran = w2now[:3, 3].detach()
        params["cam_unnorm_rots"][..., curr_time_idx] = rel_w2c_rot_quat
        params["cam_trans"][..., curr_time_idx] = rel_w2c_tran
    return params


def fused_icp_init_camera_pose(
    params,
    curr_time_idx: int,
    now_pc,
    pre_pc,
    dataset,
    curr_frame_id: str,
    last_frame_id: str,
    init_pose: np.ndarray,
    corr_threshold: float,
    device,
    project_root: str,
    icp_fn,
    lidar_max_points: int = 120000,
    lidar_min_forward_m: float = 0.0,
    lidar_max_forward_m: float = 0.0,
) -> Tuple[object, np.ndarray, float, float]:
    t_velo_to_cam = load_velo_to_cam(dataset, project_root)
    lidar_pre = lidar_to_pointcloud(dataset, last_frame_id, lidar_max_points, lidar_min_forward_m, lidar_max_forward_m)
    lidar_now = lidar_to_pointcloud(dataset, curr_frame_id, lidar_max_points, lidar_min_forward_m, lidar_max_forward_m)
    lidar_pre_cam = transform_pointcloud(lidar_pre, t_velo_to_cam)
    lidar_now_cam = transform_pointcloud(lidar_now, t_velo_to_cam)
    fused_pre = merge_pointclouds([pre_pc, lidar_pre_cam])
    fused_now = merge_pointclouds([now_pc, lidar_now_cam])
    pre2now, fitness, inlier_rmse = icp_fn(fused_now, fused_pre, init_pose, corr_threshold)
    params = update_camera_pose_from_relative(params, curr_time_idx, pre2now, device)
    return params, pre2now, fitness, inlier_rmse

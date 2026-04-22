import os

import numpy as np
import torch
import torch.nn.functional as F

from datasets.gradslam_datasets import Kitti360Dataset
from tools.kitti360_parser.kitti360_lidar_icp_odom import (
    icp as run_lidar_icp,
    lidar_to_open3d,
    read_velodyne_bin,
    resolve_velo_dir,
)
from tools.kitti360_parser.pose_alignment_utils import load_velo_to_cam


def get_dataset_frame_id(dataset, time_idx):
    if hasattr(dataset, "depth_paths"):
        return os.path.splitext(os.path.basename(dataset.depth_paths[time_idx]))[0]
    if hasattr(dataset, "color_paths"):
        return os.path.splitext(os.path.basename(dataset.color_paths[time_idx]))[0]
    return f"{time_idx:010d}"


def setup_kitti360_lidar_icp_state(config, dataset):
    lidar_cfg = config.get("lidar_icp", {})
    if not isinstance(dataset, Kitti360Dataset):
        raise ValueError("pose_init_method='pnp_lidar_icp' is only implemented for Kitti360Dataset.")
    if not hasattr(dataset, "input_folder"):
        raise ValueError("Kitti360Dataset is missing input_folder; cannot infer KITTI-360 root.")

    kitti360_root = lidar_cfg.get("kitti360_root")
    if kitti360_root is None:
        kitti360_root = os.path.dirname(os.path.dirname(dataset.input_folder))
    sequence = lidar_cfg.get("sequence", os.path.basename(dataset.input_folder))
    velo_dir = lidar_cfg.get("velo_dir", resolve_velo_dir(kitti360_root, sequence))
    calib_path = lidar_cfg.get(
        "calib_cam_to_velo",
        os.path.join(kitti360_root, "calibration", "calib_cam_to_velo.txt"),
    )
    if not os.path.isdir(velo_dir):
        raise FileNotFoundError(f"KITTI-360 Velodyne directory not found: {velo_dir}")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(f"KITTI-360 calib_cam_to_velo not found: {calib_path}")

    state = dict(
        velo_dir=velo_dir,
        T_velo_to_cam=load_velo_to_cam(calib_path),
        voxel_size=float(lidar_cfg.get("voxel_size", 0.1)),
        min_forward_m=float(lidar_cfg.get("min_forward_m", 0.0)),
        max_forward_m=float(lidar_cfg.get("max_forward_m", 0.0)),
        max_points=int(lidar_cfg.get("max_points", 2000000000)),
        verbose=bool(lidar_cfg.get("verbose", False)),
    )
    print(f"KITTI-360 LiDAR ICP velo_dir: {velo_dir}")
    print(f"KITTI-360 LiDAR ICP calib: {calib_path}")
    return state


def load_lidar_pointcloud(frame_id, lidar_icp_state):
    path = os.path.join(lidar_icp_state["velo_dir"], f"{frame_id}.bin")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"KITTI-360 Velodyne frame not found: {path}")
    xyz = read_velodyne_bin(path)
    return lidar_to_open3d(
        xyz,
        min_forward_m=lidar_icp_state["min_forward_m"],
        max_forward_m=lidar_icp_state["max_forward_m"],
        max_points=lidar_icp_state["max_points"],
    )


def lidar_icp_init_camera_pose(
    params,
    curr_time_idx,
    curr_frame_id,
    prev_frame_id,
    init_pose_cam,
    corr_threshold,
    lidar_icp_state,
    device,
    build_rotation_fn,
    matrix_to_quaternion_fn,
):
    with torch.no_grad():
        T_velo_to_cam = lidar_icp_state["T_velo_to_cam"]
        T_cam_to_velo = np.linalg.inv(T_velo_to_cam)
        init_pose_velo = T_cam_to_velo @ init_pose_cam @ T_velo_to_cam
        curr_pc = load_lidar_pointcloud(curr_frame_id, lidar_icp_state)
        prev_pc = load_lidar_pointcloud(prev_frame_id, lidar_icp_state)
        prev_to_curr_velo, fitness, inlier_rmse = run_lidar_icp(
            target=curr_pc,
            source=prev_pc,
            init_pose_T12=init_pose_velo,
            corr_threshold=corr_threshold,
            voxel_size=lidar_icp_state["voxel_size"],
            verbose=lidar_icp_state["verbose"],
        )
        prev_to_curr_cam = T_velo_to_cam @ prev_to_curr_velo @ T_cam_to_velo

        cam_rot = F.normalize(params["cam_unnorm_rots"][..., curr_time_idx - 1].detach())
        cam_tran = params["cam_trans"][..., curr_time_idx - 1].detach()
        w2pre = torch.eye(4).float().to(device=device)
        w2pre[:3, :3] = build_rotation_fn(cam_rot)
        w2pre[:3, 3] = cam_tran
        pre2now_tmp = torch.tensor(prev_to_curr_cam, dtype=torch.float32, device=device)
        w2now = pre2now_tmp @ w2pre
        rel_w2c_rot = w2now[:3, :3].unsqueeze(0).detach()
        rel_w2c_rot_quat = matrix_to_quaternion_fn(rel_w2c_rot)
        rel_w2c_tran = w2now[:3, 3].detach()
        params["cam_unnorm_rots"][..., curr_time_idx] = rel_w2c_rot_quat
        params["cam_trans"][..., curr_time_idx] = rel_w2c_tran

    return params, prev_to_curr_cam, fitness, inlier_rmse

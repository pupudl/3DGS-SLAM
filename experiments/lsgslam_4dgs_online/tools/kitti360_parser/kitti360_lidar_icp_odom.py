#!/usr/bin/env python3
"""
KITTI-360: run point-to-plane ICP on adjacent Velodyne scans and accumulate an odometry trajectory.

Alignment policy:
- Follow the original project pipeline pose convention as closely as possible.
- Main saved poses use `cam0 w2c` because the original pipeline mainly stores `w2c`.
- Visualization and downstream plotting should still convert to `c2w` first, then use `x-z`.

Outputs:
1. `main/`
   - `traj_lidar_icp_cam0_w2c.txt`
   - `traj_lidar_icp_cam0_local_fixed_w2c.txt`
   - `traj_lidar_icp_frame_ids.txt`
   - `traj_lidar_icp_meta.txt`
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import open3d as o3d
except ImportError as e:
    raise SystemExit("需要安装 open3d: pip install open3d") from e

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted  # type: ignore

from tools.kitti360_parser.pose_alignment_utils import (
    ORIGINAL_PIPELINE_POSE_CONVENTION,
    ORIGINAL_PIPELINE_SENSOR_FRAME,
    align_c2w_poses_to_first_frame,
    conjugate_c2w_poses_by_rotation,
    convert_pose_convention,
    convert_sensor_frame,
    load_velo_to_cam,
    read_cam0_to_world_map,
    summarize_axis_span,
    to_plot_trajectory,
    write_pose_file,
)


def _project_root() -> str:
    return PROJECT_ROOT


def read_velodyne_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    return pts


def lidar_to_open3d(
    xyz: np.ndarray,
    min_forward_m: float,
    max_forward_m: float,
    max_points: int,
) -> o3d.geometry.PointCloud:
    if xyz.size == 0:
        return o3d.geometry.PointCloud()
    # Keep LiDAR-specific cropping optional so the default behavior stays close
    # to the original splatam ICP implementation.
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
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return pcd


def _legacy_preprocess_for_icp(
    target: o3d.geometry.PointCloud,
    source: o3d.geometry.PointCloud,
    voxel_size: float,
    verbose: bool,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    source = source.voxel_down_sample(voxel_size)
    target = target.voxel_down_sample(voxel_size)

    num_points = 20
    radius = max(0.4, voxel_size * 4.0)
    source, _ = source.remove_radius_outlier(num_points, radius)
    target, _ = target.remove_radius_outlier(num_points, radius)
    if verbose:
        print("filter outliers (radius outlier)")

    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30)
    )
    return target, source


def icp(
    target: o3d.geometry.PointCloud,
    source: o3d.geometry.PointCloud,
    init_pose_T12: np.ndarray,
    corr_threshold: float,
    voxel_size: float,
    verbose: bool,
) -> Tuple[np.ndarray, float, float]:
    """
    Keep the same semantics used in the original project ICP call:
    `target=current`, `source=previous`, return transform that maps previous -> current.
    """
    if len(target.points) < 100 or len(source.points) < 100:
        return np.eye(4, dtype=np.float64), 0.0, float("inf")

    target, source = _legacy_preprocess_for_icp(target, source, voxel_size, verbose)
    if len(target.points) < 50 or len(source.points) < 50:
        return np.eye(4, dtype=np.float64), 0.0, float("inf")

    thresholds = [corr_threshold, 0.5, 0.1] if corr_threshold > 0.5 else [0.5, 0.1]
    if verbose:
        print(f"ICP thresholds: {thresholds}")

    relative_transform = init_pose_T12.astype(np.float64)
    reg_p2l = None
    for max_correspondence_distance in thresholds:
        relative_rmse = 0.001 if max_correspondence_distance == 0.1 else 0.1
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance,
            relative_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6,
                relative_rmse=relative_rmse,
                max_iteration=100,
            ),
        )
        relative_transform = reg_p2l.transformation

    assert reg_p2l is not None
    if verbose:
        print("refine Transformation:\n", reg_p2l.transformation)
        print("correspondences", np.asarray(reg_p2l.correspondence_set).shape)
    return (
        np.asarray(reg_p2l.transformation, dtype=np.float64),
        float(reg_p2l.fitness),
        float(reg_p2l.inlier_rmse),
    )


def resolve_velo_dir(k360_root: str, sequence: str) -> str:
    seq_dir = os.path.join(k360_root, "data_3d_raw", sequence, "velodyne_points", "data")
    flat_dir = os.path.join(k360_root, "data_3d_raw", "velodyne_points", "data")
    if os.path.isdir(seq_dir):
        return seq_dir
    if os.path.isdir(flat_dir):
        return flat_dir
    return seq_dir


def save_lidar_icp_outputs(
    out_dir: str,
    sequence: str,
    frame_ids: List[str],
    poses_velo_c2w: List[np.ndarray],
    T_velo_to_cam: np.ndarray,
    lines_meta: List[str],
) -> None:
    """
    Main saved result uses `cam0 w2c`, matching the original pipeline.
    We keep the main raw `cam0 w2c` output plus a local-fixed `cam0 w2c`
    output that is ready for direct x-z plotting and fusion with the main pipeline.
    """
    main_dir = os.path.join(out_dir, "main")
    os.makedirs(main_dir, exist_ok=True)

    poses_cam_c2w = convert_sensor_frame(
        poses_velo_c2w,
        src_sensor="velo",
        dst_sensor=ORIGINAL_PIPELINE_SENSOR_FRAME,
        pose_convention="c2w",
        T_velo_to_cam=T_velo_to_cam,
    )
    poses_cam_w2c = convert_pose_convention(poses_cam_c2w, "c2w", ORIGINAL_PIPELINE_POSE_CONVENTION)
    poses_velo_w2c = convert_pose_convention(poses_velo_c2w, "c2w", ORIGINAL_PIPELINE_POSE_CONVENTION)
    poses_cam_local_fixed_c2w = conjugate_c2w_poses_by_rotation(
        align_c2w_poses_to_first_frame(poses_cam_c2w),
        T_velo_to_cam[:3, :3],
    )
    poses_cam_local_fixed_w2c = convert_pose_convention(
        poses_cam_local_fixed_c2w,
        "c2w",
        ORIGINAL_PIPELINE_POSE_CONVENTION,
    )

    out_cam_w2c = os.path.join(main_dir, "traj_lidar_icp_cam0_w2c.txt")
    out_cam_local_fixed_w2c = os.path.join(main_dir, "traj_lidar_icp_cam0_local_fixed_w2c.txt")
    out_frame_ids = os.path.join(main_dir, "traj_lidar_icp_frame_ids.txt")
    meta_path = os.path.join(main_dir, "traj_lidar_icp_meta.txt")

    write_pose_file(out_cam_w2c, poses_cam_w2c)
    write_pose_file(out_cam_local_fixed_w2c, poses_cam_local_fixed_w2c)

    cam_plot = to_plot_trajectory(poses_cam_w2c, pose_convention=ORIGINAL_PIPELINE_POSE_CONVENTION)
    cam_span = summarize_axis_span(cam_plot["xyz"])
    velo_span = summarize_axis_span(np.array([pose[:3, 3] for pose in poses_velo_c2w], dtype=np.float64))

    with open(meta_path, "w") as f:
        f.write(f"sequence {sequence}\n")
        f.write(f"frames {len(frame_ids)}\n")
        f.write(f"main_pose_convention {ORIGINAL_PIPELINE_POSE_CONVENTION}\n")
        f.write(f"main_sensor_frame {ORIGINAL_PIPELINE_SENSOR_FRAME}\n")
        f.write("main_output_dir main\n")
        f.write("main_output_file main/traj_lidar_icp_cam0_w2c.txt\n")
        f.write("fusion_output_file main/traj_lidar_icp_cam0_local_fixed_w2c.txt\n")
        f.write(f"plot_plane_like_original {cam_plot['plot_axes'][0]}-{cam_plot['plot_axes'][1]}\n")
        f.write(f"plot_plane_reason {cam_plot['reason']}\n")
        f.write(
            "cam0_c2w_axis_span_m "
            f"x={cam_span['x']:.3f} y={cam_span['y']:.3f} z={cam_span['z']:.3f}\n"
        )
        f.write(
            "velo_c2w_axis_span_m "
            f"x={velo_span['x']:.3f} y={velo_span['y']:.3f} z={velo_span['z']:.3f}\n"
        )
        for line in lines_meta:
            f.write(line + "\n")

    with open(out_frame_ids, "w") as f:
        for fid in frame_ids:
            f.write(fid + "\n")

    print(f"结果目录: {out_dir}")
    print(f"写入 main/cam0 {ORIGINAL_PIPELINE_POSE_CONVENTION}: {out_cam_w2c}")
    print(f"写入 main/cam0 local-fixed {ORIGINAL_PIPELINE_POSE_CONVENTION}: {out_cam_local_fixed_w2c}")
    print(f"写入 main/frame ids: {out_frame_ids}")
    print(f"写入 main/meta: {meta_path}")


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="KITTI-360 Velodyne ICP 里程计")
    parser.add_argument(
        "--kitti360_root",
        default=os.path.join(root, "data", "kitti360"),
        help="KITTI-360 根目录（含 calibration / data_2d_raw / data_3d_raw）",
    )
    parser.add_argument("--sequence", default="2013_05_28_drive_0000_sync", help="序列名")
    parser.add_argument(
        "--calib_cam_to_velo",
        default=None,
        help="默认 <kitti360_root>/calibration/calib_cam_to_velo.txt",
    )
    parser.add_argument(
        "--frame_list",
        choices=("depth", "all_bins"),
        default="depth",
        help="帧列表：与 depth_sceneflow 对齐 | 使用目录下全部 .bin",
    )
    parser.add_argument("--max_frames", type=int, default=-1, help="仅处理前 N 帧（-1 为全部）")
    parser.add_argument("--stride", type=int, default=1, help="每隔 stride 帧取一次")
    parser.add_argument("--corr_threshold", type=float, default=1.0, help="ICP 最粗对应距离（米）")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="体素下采样边长（米），默认与 splatam ICP 一致")
    parser.add_argument("--min_forward_m", type=float, default=0.0, help="可选 LiDAR 前向裁剪；<=0 表示关闭")
    parser.add_argument("--max_forward_m", type=float, default=0.0, help="可选 LiDAR 前向裁剪；<=0 表示关闭")
    parser.add_argument("--max_points", type=int, default=2000000000, help="每帧随机保留最多点数；默认基本等价于不裁点")
    parser.add_argument(
        "--no_warm_start",
        action="store_true",
        help="关闭上一帧相对变换初值；默认行为更贴近原流程",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose_icp", action="store_true", help="每步打印 ICP 细节")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="默认 <kitti360_root>/data_3d_raw/<sequence>",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    k360_root = os.path.abspath(args.kitti360_root)
    seq = args.sequence
    seq_2d = os.path.join(k360_root, "data_2d_raw", seq)
    out_dir = args.output_dir or os.path.join(k360_root, "data_3d_raw", seq)
    velo_dir = resolve_velo_dir(k360_root, seq)
    depth_dir = os.path.join(seq_2d, "depth_sceneflow")

    #每一帧相机（cam0）的真实位姿,每一帧都有：T_w_cam0,表示：相机在世界坐标系中的位置和方向
    gt_cam0_to_world_path = os.path.join(k360_root, "data_poses", seq, "cam0_to_world.txt")
    #用来把ICP结果从 LiDAR 转到相机
    calib_path = args.calib_cam_to_velo or os.path.join(k360_root, "calibration", "calib_cam_to_velo.txt")

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(velo_dir):
        print(f"错误：找不到雷达目录 {velo_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(calib_path):
        print(f"错误：找不到标定文件 {calib_path}", file=sys.stderr)
        sys.exit(1)

    print(f"sequence: {seq}")
    print(f"kitti360_root: {k360_root}")
    print(f"velo_dir: {velo_dir}")
    print(f"frame_list: {args.frame_list}")
    print(
        "输出约定: "
        f"main={ORIGINAL_PIPELINE_SENSOR_FRAME}/{ORIGINAL_PIPELINE_POSE_CONVENTION}, "
        "viz=convert_to_c2w_then_plot_x-z"
    )
    print(
        "ICP 默认参数: "
        f"voxel_size={args.voxel_size}, corr_threshold={args.corr_threshold}, "
        f"warm_start={'off' if args.no_warm_start else 'on'}, "
        f"forward_crop={'off' if args.min_forward_m <= 0.0 and args.max_forward_m <= 0.0 else 'on'}"
    )

    T_velo_to_cam = load_velo_to_cam(calib_path)

    if args.frame_list == "depth":
        if not os.path.isdir(depth_dir):
            print(f"错误：找不到 depth_sceneflow {depth_dir}", file=sys.stderr)
            sys.exit(1)
        depth_files = natsorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
        frame_ids = [os.path.splitext(f)[0] for f in depth_files]
    else:
        frame_ids = natsorted(os.path.splitext(f)[0] for f in os.listdir(velo_dir) if f.endswith(".bin"))

    if args.stride < 1:
        print("stride 必须 >= 1", file=sys.stderr)
        sys.exit(1)
    if args.stride > 1:
        frame_ids = frame_ids[:: args.stride]
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if len(frame_ids) < 2:
        print("帧数不足 2，无法执行 ICP。", file=sys.stderr)
        sys.exit(1)

    print(f"总帧数: {len(frame_ids)}（执行 {len(frame_ids) - 1} 次 ICP）")

    for fid in frame_ids[:3]:
        path = os.path.join(velo_dir, f"{fid}.bin")
        if not os.path.isfile(path):
            print(f"错误：缺少雷达文件 {path}", file=sys.stderr)
            sys.exit(1)

    first_fid_int = int(frame_ids[0])
    T_w_cam0_first = None
    if os.path.isfile(gt_cam0_to_world_path):
        gt_map = read_cam0_to_world_map(gt_cam0_to_world_path)
        T_w_cam0_first = gt_map.get(first_fid_int, None)
    if T_w_cam0_first is None:
        print("警告：未找到首帧 GT，回退到局部坐标系（首帧 c2w=I）。", file=sys.stderr)
        T_w_cam0_first = np.eye(4, dtype=np.float64)

    poses_cam0_c2w = [T_w_cam0_first.copy()]
    poses_velo_c2w = convert_sensor_frame(
        poses_cam0_c2w,
        src_sensor="cam0",
        dst_sensor="velo",
        pose_convention="c2w",
        T_velo_to_cam=T_velo_to_cam,
    )

    T_w_velo = poses_velo_c2w[0].copy()
    T_prev_curr = np.eye(4, dtype=np.float64)
    lines_meta: List[str] = []

    for idx in range(1, len(frame_ids)):
        fid_prev, fid_curr = frame_ids[idx - 1], frame_ids[idx]
        path_prev = os.path.join(velo_dir, f"{fid_prev}.bin")
        path_curr = os.path.join(velo_dir, f"{fid_curr}.bin")
        if not os.path.isfile(path_curr):
            print(f"错误：缺少雷达文件 {path_curr}", file=sys.stderr)
            sys.exit(1)

        xyz_prev = read_velodyne_bin(path_prev)
        xyz_curr = read_velodyne_bin(path_curr)
        pcd_prev = lidar_to_open3d(xyz_prev, args.min_forward_m, args.max_forward_m, args.max_points)
        pcd_curr = lidar_to_open3d(xyz_curr, args.min_forward_m, args.max_forward_m, args.max_points)

        init = np.eye(4, dtype=np.float64) if args.no_warm_start else T_prev_curr
        T_prev_to_curr, fitness, rmse = icp(
            target=pcd_curr,
            source=pcd_prev,
            init_pose_T12=init,
            corr_threshold=args.corr_threshold,
            voxel_size=args.voxel_size,
            verbose=args.verbose_icp,
        )

        T_w_velo = T_w_velo @ np.linalg.inv(T_prev_to_curr)
        poses_velo_c2w.append(T_w_velo.copy())
        T_prev_curr = T_prev_to_curr

        lines_meta.append(f"{fid_curr} fitness={fitness:.4f} rmse={rmse:.4f}")
        if not args.verbose_icp and (idx % 10 == 0 or idx == len(frame_ids) - 1):
            print(f"[{idx}/{len(frame_ids)-1}] frame {fid_curr} fitness={fitness:.4f} rmse={rmse:.4f}")

    save_lidar_icp_outputs(
        out_dir=out_dir,
        sequence=seq,
        frame_ids=frame_ids,
        poses_velo_c2w=poses_velo_c2w,
        T_velo_to_cam=T_velo_to_cam,
        lines_meta=lines_meta,
    )


if __name__ == "__main__":
    main()

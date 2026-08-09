#!/usr/bin/env python3
"""
KITTI-360 LiDAR ICP visualization.

This script intentionally follows the original pipeline visualization logic:
- stored/main poses are treated as `w2c`
- plotting converts them to `c2w`
- trajectory is drawn on the `x-z` plane
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted  # type: ignore

from tools.kitti360_parser.pose_alignment_utils import (
    ORIGINAL_PIPELINE_POSE_CONVENTION,
    ORIGINAL_PIPELINE_PLOT_AXES,
    ORIGINAL_PIPELINE_SENSOR_FRAME,
    align_c2w_poses_to_first_frame,
    align_poses_to_first_frame,
    frame_ids_from_gt_map,
    load_lidar_icp_poses,
    read_cam0_to_world_map,
    read_frame_ids,
    to_plot_trajectory,
)


def maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def read_meta_lines(path: str) -> List[str]:
    lines: List[str] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and "fitness=" in s and "rmse=" in s:
                lines.append(s)
    return lines


def _infer_frame_ids(seq_2d_dir: str, n_poses: int) -> List[str]:
    depth_dir = os.path.join(seq_2d_dir, "depth_sceneflow")
    if os.path.isdir(depth_dir):
        depth_files = natsorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
        ids = [os.path.splitext(f)[0] for f in depth_files]
        if len(ids) >= n_poses:
            return ids[:n_poses]
    return [f"{i:010d}" for i in range(n_poses)]


def print_plot_debug(name: str, traj_info) -> None:
    spans = traj_info["axis_span"]
    plane = f"{traj_info['plot_axes'][0]}-{traj_info['plot_axes'][1]}"
    print(f"{name} 轴向跨度: x={spans['x']:.3f}m y={spans['y']:.3f}m z={spans['z']:.3f}m")
    print(f"{name} 当前画图平面: {plane}")
    print(f"{name} 选用原因: {traj_info['reason']}")


def save_compare_plot(
    plt,
    out_path: str,
    seq: str,
    est_xyz,
    axis_0: str,
    axis_1: str,
    title_suffix: str,
    gt_xyz=None,
    gt_label: str = "gt_align_absolute",
    est_label: str = "odo_lidar_icp",
) -> None:
    idx_0 = 0 if axis_0 == "x" else 1 if axis_0 == "y" else 2
    idx_1 = 0 if axis_1 == "x" else 1 if axis_1 == "y" else 2
    fig = plt.figure(figsize=(8, 6))
    if gt_xyz is not None and len(gt_xyz) == len(est_xyz):
        plt.plot(gt_xyz[:, idx_0], gt_xyz[:, idx_1], color="blue", linewidth=2, label=gt_label)
    plt.plot(est_xyz[:, idx_0], est_xyz[:, idx_1], color="red", linewidth=2, label=est_label)
    plt.axis("equal")
    plt.xlabel(axis_0)
    plt.ylabel(axis_1)
    plt.grid(linestyle="--", alpha=0.4)
    plt.legend()
    plt.title(f"{seq} trajectory compare ({title_suffix})")
    plt.savefig(out_path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def save_plots(
    out_dir: str,
    seq: str,
    frame_ids: List[str],
    est_cam_w2c,
    lines_meta: List[str],
    gt_cam0_to_world_path: Optional[str],
    est_velo_w2c=None,
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError("未安装 matplotlib，无法可视化。")
    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    est_traj_abs = to_plot_trajectory(est_cam_w2c, pose_convention=ORIGINAL_PIPELINE_POSE_CONVENTION)
    print_plot_debug("LiDAR ICP cam0 absolute-world", est_traj_abs)

    est_cam_w2c_local = align_poses_to_first_frame(est_cam_w2c, ORIGINAL_PIPELINE_POSE_CONVENTION)
    est_traj_local = to_plot_trajectory(est_cam_w2c_local, pose_convention=ORIGINAL_PIPELINE_POSE_CONVENTION)
    print_plot_debug("LiDAR ICP cam0 local-world", est_traj_local)

    if est_velo_w2c is not None:
        velo_traj = to_plot_trajectory(est_velo_w2c, pose_convention=ORIGINAL_PIPELINE_POSE_CONVENTION)
        print_plot_debug("LiDAR ICP velo absolute-world", velo_traj)

    axis_0, axis_1 = ORIGINAL_PIPELINE_PLOT_AXES
    gt_xyz_local = None

    if gt_cam0_to_world_path is not None and os.path.isfile(gt_cam0_to_world_path):
        gt_map = read_cam0_to_world_map(gt_cam0_to_world_path)
        gt_cam_c2w = frame_ids_from_gt_map(frame_ids, gt_map)
        if len(gt_cam_c2w) == len(est_cam_w2c):
            gt_cam_c2w_local = align_c2w_poses_to_first_frame(gt_cam_c2w)
            gt_traj_local = to_plot_trajectory(gt_cam_c2w_local, pose_convention="c2w")
            print_plot_debug("GT cam0 local-world", gt_traj_local)
            gt_xyz_local = gt_traj_local["xyz"]
            main_err = np.linalg.norm(est_traj_local["xyz"] - gt_xyz_local, axis=1)
            print(
                "主对比图直接绘图误差: "
                f"mean={float(main_err.mean()):.4f} m, "
                f"max={float(main_err.max()):.4f} m"
            )
        else:
            print("警告：GT 数量与 ICP 轨迹数量不一致，GT 只用于日志，不叠加绘图。")

    compare_local_png = os.path.join(debug_dir, "traj_compare.png")
    save_compare_plot(
        plt=plt,
        out_path=compare_local_png,
        seq=seq,
        est_xyz=est_traj_local["xyz"],
        gt_xyz=gt_xyz_local,
        axis_0=axis_0,
        axis_1=axis_1,
        title_suffix="local cam0 trajectory, direct x-z plot",
        gt_label="GT",
        est_label="Odo",
    )
    print(f"写入主对比图: {compare_local_png}")

    steps = []
    fitnesses = []
    rmses = []
    for i, line in enumerate(lines_meta, start=1):
        parts = line.split()
        if len(parts) >= 3 and "fitness=" in parts[1] and "rmse=" in parts[2]:
            try:
                fitnesses.append(float(parts[1].split("=")[1]))
                rmses.append(float(parts[2].split("=")[1]))
                steps.append(i)
            except Exception:
                continue

    if steps:
        fig, ax1 = plt.subplots(figsize=(9, 4.2))
        ax1.plot(steps, fitnesses, color="tab:blue")
        ax1.set_xlabel("ICP step")
        ax1.set_ylabel("fitness", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(linestyle="--", alpha=0.4)
        ax2 = ax1.twinx()
        ax2.plot(steps, rmses, color="tab:red")
        ax2.set_ylabel("inlier rmse (m)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        plt.title(f"{seq} ICP quality")
        quality_png = os.path.join(debug_dir, "traj_lidar_icp_quality.png")
        plt.savefig(quality_png, bbox_inches="tight", dpi=180)
        plt.close(fig)
        print(f"写入 ICP 质量图: {quality_png}")



def main() -> None:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description="可视化 KITTI-360 LiDAR ICP 结果")
    parser.add_argument(
        "--kitti360_root",
        default=os.path.join(root, "data", "kitti360"),
        help="KITTI-360 根目录",
    )
    parser.add_argument("--sequence", default="2013_05_28_drive_0000_sync")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="默认 <kitti360_root>/data_3d_raw/<sequence>",
    )
    parser.add_argument(
        "--traj_cam",
        default=None,
        help="默认优先 <output_dir>/traj_lidar_icp_cam0_local_fixed_w2c.txt，否则回退到 traj_lidar_icp_cam0_w2c.txt",
    )
    parser.add_argument(
        "--traj_velo",
        default=None,
        help="默认 <output_dir>/traj_lidar_icp_velo_w2c.txt",
    )
    parser.add_argument(
        "--meta",
        default=None,
        help="默认 <output_dir>/traj_lidar_icp_meta.txt",
    )
    parser.add_argument(
        "--calib_cam_to_velo",
        default=None,
        help="保留参数兼容；当前主图默认不在可视化阶段做额外旋转",
    )
    parser.add_argument(
        "--frame_ids",
        default=None,
        help="默认 <output_dir>/traj_lidar_icp_frame_ids.txt",
    )
    parser.add_argument(
        "--gt_cam0_to_world",
        default=None,
        help="默认 <kitti360_root>/data_poses/<sequence>/cam0_to_world.txt",
    )
    args = parser.parse_args()

    seq_2d_dir = os.path.join(args.kitti360_root, "data_2d_raw", args.sequence)
    out_dir = args.output_dir or os.path.join(args.kitti360_root, "data_3d_raw", args.sequence)
    main_dir = os.path.join(out_dir, "main")
    debug_dir = os.path.join(out_dir, "debug")
    default_local_fixed_traj_cam = os.path.join(main_dir, "traj_lidar_icp_cam0_local_fixed_w2c.txt")
    default_raw_traj_cam = os.path.join(main_dir, "traj_lidar_icp_cam0_w2c.txt")
    traj_cam = args.traj_cam or (
        default_local_fixed_traj_cam if os.path.isfile(default_local_fixed_traj_cam) else default_raw_traj_cam
    )
    traj_velo = args.traj_velo or os.path.join(debug_dir, "traj_lidar_icp_velo_w2c.txt")
    meta = args.meta or os.path.join(main_dir, "traj_lidar_icp_meta.txt")
    frame_ids_path = args.frame_ids or os.path.join(main_dir, "traj_lidar_icp_frame_ids.txt")
    gt_path = args.gt_cam0_to_world or os.path.join(
        args.kitti360_root, "data_poses", args.sequence, "cam0_to_world.txt"
    )
    if not os.path.isfile(traj_cam):
        raise SystemExit(f"找不到轨迹文件: {traj_cam}")
    if not os.path.isfile(meta):
        raise SystemExit(f"找不到 meta 文件: {meta}")

    est_cam_w2c = load_lidar_icp_poses(
        path=traj_cam,
        pose_convention="w2c",
        sensor_frame="cam0",
        target_pose_convention="w2c",
        target_sensor_frame="cam0",
    )
    est_velo_w2c = None
    if os.path.isfile(traj_velo):
        est_velo_w2c = load_lidar_icp_poses(
            path=traj_velo,
            pose_convention="w2c",
            sensor_frame="velo",
            target_pose_convention="w2c",
            target_sensor_frame="velo",
        )

    lines_meta = read_meta_lines(meta)
    if os.path.isfile(frame_ids_path):
        frame_ids = read_frame_ids(frame_ids_path)
        if len(frame_ids) != len(est_cam_w2c):
            print(f"警告：frame_ids 数量({len(frame_ids)})与轨迹数量({len(est_cam_w2c)})不一致，回退自动推断。")
            frame_ids = _infer_frame_ids(seq_2d_dir, len(est_cam_w2c))
    else:
        frame_ids = _infer_frame_ids(seq_2d_dir, len(est_cam_w2c))

    print(
        "当前可视化约定: "
        f"sensor={ORIGINAL_PIPELINE_SENSOR_FRAME}, pose={ORIGINAL_PIPELINE_POSE_CONVENTION}, "
        f"plot_plane={ORIGINAL_PIPELINE_PLOT_AXES[0]}-{ORIGINAL_PIPELINE_PLOT_AXES[1]}"
    )
    print(f"主轨迹文件: {traj_cam}")

    save_plots(
        out_dir=out_dir,
        seq=args.sequence,
        frame_ids=frame_ids,
        est_cam_w2c=est_cam_w2c,
        lines_meta=lines_meta,
        gt_cam0_to_world_path=gt_path if os.path.isfile(gt_path) else None,
        est_velo_w2c=est_velo_w2c,
    )


if __name__ == "__main__":
    main()

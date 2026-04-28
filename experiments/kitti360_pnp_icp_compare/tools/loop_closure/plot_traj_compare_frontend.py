#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-lsgslam")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_kitti_poses(path: str) -> np.ndarray:
    rows = np.loadtxt(path)
    rows = np.atleast_2d(rows)
    poses = []
    for row in rows:
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = row.reshape(3, 4)
        poses.append(T)
    return np.asarray(poses)


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot front-end trajectory experiment outputs")
    parser.add_argument("--pnp_dir", required=True)
    parser.add_argument("--pnp_icp_dir", required=True)
    parser.add_argument("--pnp_lidar_icp_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    gt = read_kitti_poses(os.path.join(args.pnp_dir, "gt_c2w_kitti.txt"))
    pnp = read_kitti_poses(os.path.join(args.pnp_dir, "estimated_c2w_kitti.txt"))
    pnp_icp = read_kitti_poses(os.path.join(args.pnp_icp_dir, "estimated_c2w_kitti.txt"))
    pnp_metrics = load_metrics(os.path.join(args.pnp_dir, "metrics.json"))
    pnp_icp_metrics = load_metrics(os.path.join(args.pnp_icp_dir, "metrics.json"))
    pnp_lidar_icp = None
    pnp_lidar_icp_metrics = None
    if args.pnp_lidar_icp_dir is not None:
        pnp_lidar_icp = read_kitti_poses(os.path.join(args.pnp_lidar_icp_dir, "estimated_c2w_kitti.txt"))
        pnp_lidar_icp_metrics = load_metrics(os.path.join(args.pnp_lidar_icp_dir, "metrics.json"))

    lengths = [len(gt), len(pnp), len(pnp_icp)]
    if pnp_lidar_icp is not None:
        lengths.append(len(pnp_lidar_icp))
    n = min(lengths)
    gt_xyz = gt[:n, :3, 3]
    pnp_xyz = pnp[:n, :3, 3]
    pnp_icp_xyz = pnp_icp[:n, :3, 3]
    pnp_lidar_icp_xyz = None if pnp_lidar_icp is None else pnp_lidar_icp[:n, :3, 3]

    plt.figure(figsize=(11, 8))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], color="tab:blue", label="GT")
    plt.plot(
        pnp_xyz[:, 0],
        pnp_xyz[:, 2],
        color="tab:red",
        label=f"PnP only ATE {pnp_metrics['ate_rmse_m']:.3f}m",
    )
    plt.plot(
        pnp_icp_xyz[:, 0],
        pnp_icp_xyz[:, 2],
        color="tab:green",
        label=f"PnP+ICP ATE {pnp_icp_metrics['ate_rmse_m']:.3f}m",
    )
    if pnp_lidar_icp_xyz is not None:
        plt.plot(
            pnp_lidar_icp_xyz[:, 0],
            pnp_lidar_icp_xyz[:, 2],
            color="tab:orange",
            label=f"PnP+LiDAR ICP ATE {pnp_lidar_icp_metrics['ate_rmse_m']:.3f}m",
        )
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("KITTI-360 front-end trajectory comparison")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "trajectory_compare_xz.png"), dpi=180)
    plt.close()

    labels = ["ATE RMSE", "ATE aligned RMSE", "RPE trans RMSE", "RPE rot RMSE"]
    keys = ["ate_rmse_m", "ate_aligned_rmse_m", "rpe_trans_rmse_m", "rpe_rot_rmse_deg"]
    series = [
        ("PnP only", [pnp_metrics[k] for k in keys], "tab:red"),
        ("PnP+RGB-D ICP", [pnp_icp_metrics[k] for k in keys], "tab:green"),
    ]
    if pnp_lidar_icp_metrics is not None:
        series.append(("PnP+LiDAR ICP", [pnp_lidar_icp_metrics[k] for k in keys], "tab:orange"))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    plt.figure(figsize=(11, 5))
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for offset, (name, values, color) in zip(offsets, series):
        plt.bar(x + offset, values, width, label=name, color=color)
    plt.xticks(x, labels)
    plt.ylabel("m / deg")
    plt.title("Error metrics")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "metrics_compare.png"), dpi=180)
    plt.close()

    with open(os.path.join(args.output_dir, "metrics_compare.json"), "w") as f:
        metrics = {"pnp_only": pnp_metrics, "pnp_icp": pnp_icp_metrics}
        if pnp_lidar_icp_metrics is not None:
            metrics["pnp_lidar_icp"] = pnp_lidar_icp_metrics
        json.dump(metrics, f, indent=2)

    print(f"Saved compare outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()

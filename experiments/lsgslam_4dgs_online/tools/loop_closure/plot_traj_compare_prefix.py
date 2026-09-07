import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_wxyz_to_rot(q):
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_segment_poses(params_path):
    params = np.load(params_path, allow_pickle=True)
    rots = params["cam_unnorm_rots"]
    trans = params["cam_trans"]
    gt_w2cs = params["gt_w2c_all_frames"]

    w2cs = []
    num_frames = rots.shape[-1]
    for idx in range(num_frames):
        q = rots[..., idx].reshape(-1)
        t = trans[..., idx].reshape(-1)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quat_wxyz_to_rot(q)
        w2c[:3, 3] = t[:3]
        w2cs.append(w2c)

    return w2cs, [np.asarray(p, dtype=np.float64) for p in gt_w2cs]


def parse_chunk_name(name):
    parts = name.split("_")
    try:
        start, end, stride = map(int, parts[-3:])
    except ValueError:
        return None
    return start, end, stride


def find_chunk_dirs(base_folder, scene_name):
    chunk_dirs = []
    for name in os.listdir(base_folder):
        path = os.path.join(base_folder, name)
        if not os.path.isdir(path):
            continue
        if "_loops" in name or not name.startswith(scene_name):
            continue
        chunk = parse_chunk_name(name)
        if chunk is None:
            continue
        params_path = os.path.join(path, "params.npz")
        if os.path.exists(params_path):
            chunk_dirs.append((chunk[0], chunk[1], chunk[2], path))
    return sorted(chunk_dirs, key=lambda item: item[0])


def stitch_odom_poses(base_folder, scene_name):
    all_est_w2cs = []
    all_gt_w2cs = []

    for _, _, _, chunk_dir in find_chunk_dirs(base_folder, scene_name):
        est_w2cs, gt_w2cs = load_segment_poses(os.path.join(chunk_dir, "params.npz"))
        if not all_est_w2cs:
            all_est_w2cs = est_w2cs
            all_gt_w2cs = gt_w2cs
            continue

        last_est_w2c = all_est_w2cs[-1]
        last_gt_w2c = all_gt_w2cs[-1]
        all_est_w2cs.extend([pose @ last_est_w2c for pose in est_w2cs[1:]])
        all_gt_w2cs.extend([pose @ last_gt_w2c for pose in gt_w2cs[1:]])

    if not all_est_w2cs:
        raise FileNotFoundError(
            f"No chunk params.npz found under {base_folder} for scene {scene_name}"
        )
    return np.asarray(all_est_w2cs), np.asarray(all_gt_w2cs)


def poses_w2c_to_xz(w2cs, limit):
    points = []
    for w2c in w2cs[:limit]:
        c2w = np.linalg.inv(w2c)
        points.append(c2w[:3, 3])
    points = np.asarray(points)
    return points[:, 0], points[:, 2]


def poses_w2c_to_xyz(w2cs, limit):
    points = []
    for w2c in w2cs[:limit]:
        c2w = np.linalg.inv(w2c)
        points.append(c2w[:3, 3])
    return np.asarray(points, dtype=np.float64)


def translation_errors(gt_w2cs, est_w2cs, limit, axes=None):
    gt_xyz = poses_w2c_to_xyz(gt_w2cs, limit)
    est_xyz = poses_w2c_to_xyz(est_w2cs, limit)
    if axes is not None:
        gt_xyz = gt_xyz[:, axes]
        est_xyz = est_xyz[:, axes]
    return np.linalg.norm(gt_xyz - est_xyz, axis=1)


def stats_from_errors(errors):
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "final": float(errors[-1]),
    }


def translation_error_stats(gt_w2cs, est_w2cs, limit, axes=None):
    return stats_from_errors(translation_errors(gt_w2cs, est_w2cs, limit, axes=axes))


def per_axis_abs_error_stats(gt_w2cs, est_w2cs, limit):
    gt_xyz = poses_w2c_to_xyz(gt_w2cs, limit)
    est_xyz = poses_w2c_to_xyz(est_w2cs, limit)
    abs_errors = np.abs(gt_xyz - est_xyz)
    return {
        "x_mean": float(abs_errors[:, 0].mean()),
        "x_final": float(abs_errors[-1, 0]),
        "y_mean": float(abs_errors[:, 1].mean()),
        "y_final": float(abs_errors[-1, 1]),
        "z_mean": float(abs_errors[:, 2].mean()),
        "z_final": float(abs_errors[-1, 2]),
    }


def latest_pose_graph_csv(base_folder, scene_name):
    csv_dir = os.path.join(base_folder, "PoseGraphResult", "csvs")
    if not os.path.isdir(csv_dir):
        return None

    optimized = [
        path
        for path in glob.glob(os.path.join(csv_dir, f"pose{scene_name}optimized_*.csv"))
        if "unoptimized" not in os.path.basename(path)
    ]
    candidates = optimized or glob.glob(os.path.join(csv_dir, f"pose{scene_name}unoptimized_*.csv"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_pose_graph_xz(csv_path, limit):
    poses = np.loadtxt(csv_path, delimiter=",")
    poses = np.atleast_2d(poses).reshape(-1, 4, 4)
    points = poses[:limit, :3, 3]
    return points[:, 0], points[:, 2]


def plot_compare(gt_xz, odom_xz, loop_xz, output_path, max_frames, stats_3d=None, stats_xz=None, errors_3d=None, errors_xz=None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if errors_3d is None or errors_xz is None:
        fig, ax_traj = plt.subplots(figsize=(12, 9))
        ax_err = None
    else:
        fig, (ax_traj, ax_err) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.15, 1.0]})

    ax_traj.plot(gt_xz[0], gt_xz[1], color="blue", label="GT")
    ax_traj.plot(odom_xz[0], odom_xz[1], color="red", label="Odo")
    if loop_xz is not None:
        ax_traj.plot(loop_xz[0], loop_xz[1], color="green", label="Loop")
    ax_traj.axis("equal")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("z")
    title = f"Trajectory Compare (First {max_frames} Frames)"
    if stats_3d is not None and stats_xz is not None:
        title += (
            f"\n3D rmse={stats_3d['rmse']:.4f}, mean={stats_3d['mean']:.4f}, final={stats_3d['final']:.4f}"
            f" | XZ rmse={stats_xz['rmse']:.4f}, mean={stats_xz['mean']:.4f}, final={stats_xz['final']:.4f}"
        )
    ax_traj.set_title(title)
    ax_traj.legend(loc="best")

    if ax_err is not None:
        ax_err.plot(errors_3d, color="black", label="3D error")
        ax_err.plot(errors_xz, color="purple", label="XZ error")
        ax_err.set_xlabel("frame")
        ax_err.set_ylabel("translation error (m)")
        ax_err.set_title("Per-frame Error")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_axis_curves(gt_w2cs, est_w2cs, output_path, limit):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    gt_xyz = poses_w2c_to_xyz(gt_w2cs, limit)
    est_xyz = poses_w2c_to_xyz(est_w2cs, limit)
    frame_ids = np.arange(limit)
    axis_names = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    for axis_idx, axis_name in enumerate(axis_names):
        axes[axis_idx, 0].plot(frame_ids, gt_xyz[:, axis_idx], color="blue", label="GT")
        axes[axis_idx, 0].plot(frame_ids, est_xyz[:, axis_idx], color="red", label="Odo")
        axes[axis_idx, 0].set_ylabel(f"{axis_name} (m)")
        axes[axis_idx, 0].grid(True, alpha=0.3)
        axes[axis_idx, 0].legend(loc="best")

        axis_error = est_xyz[:, axis_idx] - gt_xyz[:, axis_idx]
        axes[axis_idx, 1].plot(frame_ids, axis_error, color="black", label=f"{axis_name} error")
        axes[axis_idx, 1].axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
        axes[axis_idx, 1].set_ylabel("error (m)")
        axes[axis_idx, 1].grid(True, alpha=0.3)
        axes[axis_idx, 1].legend(loc="best")

    axes[-1, 0].set_xlabel("frame")
    axes[-1, 1].set_xlabel("frame")
    fig.suptitle(f"Axis Curves (First {limit} Frames)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_chunk_comparisons(base_folder, scene_name, output_dir, max_frames):
    os.makedirs(output_dir, exist_ok=True)
    stats_rows = []
    consumed_frames = 0

    for start, end, stride, chunk_dir in find_chunk_dirs(base_folder, scene_name):
        if consumed_frames >= max_frames:
            break

        chunk_name = os.path.basename(chunk_dir)
        est_w2cs, gt_w2cs = load_segment_poses(os.path.join(chunk_dir, "params.npz"))
        chunk_limit = min(len(est_w2cs), len(gt_w2cs), max_frames - consumed_frames)
        if chunk_limit <= 0:
            continue

        gt_xz = poses_w2c_to_xz(gt_w2cs, chunk_limit)
        odom_xz = poses_w2c_to_xz(est_w2cs, chunk_limit)
        stats = translation_error_stats(gt_w2cs, est_w2cs, chunk_limit)
        stats_xz = translation_error_stats(gt_w2cs, est_w2cs, chunk_limit, axes=[0, 2])

        fig_path = os.path.join(output_dir, f"{chunk_name}_traj_compare.png")
        plt.figure(figsize=(10, 7))
        plt.plot(gt_xz[0], gt_xz[1], color="blue", label="GT")
        plt.plot(odom_xz[0], odom_xz[1], color="red", label="Odo")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title(
            f"{chunk_name} | 3D mean={stats['mean']:.4f} final={stats['final']:.4f}"
            f" | XZ mean={stats_xz['mean']:.4f} final={stats_xz['final']:.4f}"
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        stats_rows.append(
            {
                "chunk": chunk_name,
                "start": start,
                "end": end,
                "stride": stride,
                "frames": chunk_limit,
                "rmse": stats["rmse"],
                "mean": stats["mean"],
                "median": stats["median"],
                "final": stats["final"],
                "xz_rmse": stats_xz["rmse"],
                "xz_mean": stats_xz["mean"],
                "xz_median": stats_xz["median"],
                "xz_final": stats_xz["final"],
                "figure": fig_path,
            }
        )
        consumed_frames += chunk_limit

    csv_path = os.path.join(output_dir, "chunk_stats.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "chunk", "start", "end", "stride", "frames",
            "rmse", "mean", "median", "final",
            "xz_rmse", "xz_mean", "xz_median", "xz_final",
            "figure",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats_rows)

    return stats_rows, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT/Odometry/Loop trajectory comparison for the first N frames."
    )
    parser.add_argument(
        "--base_folder",
        required=True,
        help="Result group folder, e.g. results/kitti360-0000-all",
    )
    parser.add_argument(
        "--scene_name",
        default="2013_05_28_drive_0000_sync",
        help="Scene name prefix used by chunk folders and pose graph CSV files.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=1000,
        help="Number of leading frames to plot.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: <base_folder>/PoseGraphResult/traj_compare_first<N>.png",
    )
    parser.add_argument(
        "--loop_csv",
        default=None,
        help="Optional explicit pose graph CSV. If omitted, the newest optimized CSV is used.",
    )
    parser.add_argument(
        "--no_loop",
        action="store_true",
        help="Only plot GT and odometry, even if pose graph CSVs exist.",
    )
    parser.add_argument(
        "--plot_chunks",
        action="store_true",
        help="Also save per-chunk GT/odometry comparisons before trajectory stitching.",
    )
    parser.add_argument(
        "--chunk_output_dir",
        default=None,
        help="Output folder for --plot_chunks. Default: <base_folder>/PoseGraphResult/chunk_traj_compare",
    )
    parser.add_argument(
        "--plot_axis_curves",
        action="store_true",
        help="Save x/y/z coordinate curves and per-axis signed errors for the stitched odometry.",
    )
    parser.add_argument(
        "--axis_output",
        default=None,
        help="Output PNG for --plot_axis_curves. Default: <base_folder>/PoseGraphResult/axis_curves_first<N>.png",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(
        args.base_folder,
        "PoseGraphResult",
        f"traj_compare_first{args.max_frames}.png",
    )

    est_w2cs, gt_w2cs = stitch_odom_poses(args.base_folder, args.scene_name)
    limit = min(args.max_frames, len(est_w2cs), len(gt_w2cs))
    gt_xz = poses_w2c_to_xz(gt_w2cs, limit)
    odom_xz = poses_w2c_to_xz(est_w2cs, limit)

    loop_xz = None
    loop_csv = None
    if not args.no_loop:
        loop_csv = args.loop_csv or latest_pose_graph_csv(args.base_folder, args.scene_name)
        if loop_csv is not None:
            loop_xz = load_pose_graph_xz(loop_csv, limit)

    stats = translation_error_stats(gt_w2cs, est_w2cs, limit)
    stats_xz = translation_error_stats(gt_w2cs, est_w2cs, limit, axes=[0, 2])
    axis_stats = per_axis_abs_error_stats(gt_w2cs, est_w2cs, limit)
    errors_3d = translation_errors(gt_w2cs, est_w2cs, limit)
    errors_xz = translation_errors(gt_w2cs, est_w2cs, limit, axes=[0, 2])

    plot_compare(
        gt_xz,
        odom_xz,
        loop_xz,
        output,
        limit,
        stats_3d=stats,
        stats_xz=stats_xz,
        errors_3d=errors_3d,
        errors_xz=errors_xz,
    )
    print(f"Saved: {output}")
    print(
        "Raw odometry ATE (m): "
        f"rmse={stats['rmse']:.4f}, "
        f"mean={stats['mean']:.4f}, "
        f"median={stats['median']:.4f}, "
        f"final={stats['final']:.4f}"
    )
    print(
        "Raw odometry XZ error (m): "
        f"rmse={stats_xz['rmse']:.4f}, "
        f"mean={stats_xz['mean']:.4f}, "
        f"median={stats_xz['median']:.4f}, "
        f"final={stats_xz['final']:.4f}"
    )
    print(
        "Raw odometry per-axis abs error (m): "
        f"x_mean={axis_stats['x_mean']:.4f}, "
        f"x_final={axis_stats['x_final']:.4f}, "
        f"y_mean={axis_stats['y_mean']:.4f}, "
        f"y_final={axis_stats['y_final']:.4f}, "
        f"z_mean={axis_stats['z_mean']:.4f}, "
        f"z_final={axis_stats['z_final']:.4f}"
    )
    if loop_csv is not None:
        print(f"Loop CSV: {loop_csv}")
    print(f"Frames: {limit}")

    if args.plot_axis_curves:
        axis_output = args.axis_output or os.path.join(
            args.base_folder,
            "PoseGraphResult",
            f"axis_curves_first{limit}.png",
        )
        plot_axis_curves(gt_w2cs, est_w2cs, axis_output, limit)
        print(f"Saved axis curves: {axis_output}")

    if args.plot_chunks:
        chunk_output_dir = args.chunk_output_dir or os.path.join(
            args.base_folder,
            "PoseGraphResult",
            "chunk_traj_compare",
        )
        chunk_stats, chunk_csv = plot_chunk_comparisons(
            args.base_folder,
            args.scene_name,
            chunk_output_dir,
            limit,
        )
        print(f"Saved chunk comparisons: {chunk_output_dir}")
        print(f"Saved chunk stats: {chunk_csv}")
        for row in chunk_stats:
            print(
                f"Chunk {row['chunk']}: "
                f"frames={row['frames']}, "
                f"rmse={row['rmse']:.4f}, "
                f"mean={row['mean']:.4f}, "
                f"median={row['median']:.4f}, "
                f"final={row['final']:.4f}, "
                f"xz_mean={row['xz_mean']:.4f}, "
                f"xz_final={row['xz_final']:.4f}"
            )


if __name__ == "__main__":
    main()

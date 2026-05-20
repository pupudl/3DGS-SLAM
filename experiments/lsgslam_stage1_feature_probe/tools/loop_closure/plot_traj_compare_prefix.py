import argparse
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


def plot_compare(gt_xz, odom_xz, loop_xz, output_path, max_frames):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 9))
    plt.plot(gt_xz[0], gt_xz[1], color="blue", label="GT")
    plt.plot(odom_xz[0], odom_xz[1], color="red", label="Odo")
    if loop_xz is not None:
        plt.plot(loop_xz[0], loop_xz[1], color="green", label="Loop")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.title(f"Trajectory Compare (First {max_frames} Frames)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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

    plot_compare(gt_xz, odom_xz, loop_xz, output, limit)
    print(f"Saved: {output}")
    if loop_csv is not None:
        print(f"Loop CSV: {loop_csv}")
    print(f"Frames: {limit}")


if __name__ == "__main__":
    main()

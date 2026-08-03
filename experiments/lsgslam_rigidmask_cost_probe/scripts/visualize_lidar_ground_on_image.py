import argparse
import importlib.util
import json
import os
import sys

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_ROOT))
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

from probes.lidar_motion_probe.gndnet_adapter import GndNetGroundFilter
from utils.pnp_fused_icp_utils import load_kitti360_velo_to_cam


def load_config_module(config_path):
    spec = importlib.util.spec_from_file_location("lsgslam_ground_viz_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_perspective_calib(path):
    params = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
    p_rect = np.asarray([float(x) for x in params["P_rect_00"].split()], dtype=np.float64).reshape(3, 4)
    r_rect = np.asarray([float(x) for x in params["R_rect_00"].split()], dtype=np.float64).reshape(3, 3)
    return p_rect, r_rect


def selected_frame_ids(sequence_dir, start, end, stride, all_lidar_frames=False):
    if all_lidar_frames:
        data_2d_root = os.path.dirname(sequence_dir)
        data_root = os.path.dirname(data_2d_root)
        seq_name = os.path.basename(sequence_dir)
        lidar_dir = os.path.join(data_root, "data_3d_raw", seq_name, "velodyne_points", "data")
        frame_paths = sorted(p for p in os.listdir(lidar_dir) if p.endswith(".bin"))
        return [os.path.splitext(p)[0] for p in frame_paths]

    depth_dir = os.path.join(sequence_dir, "depth_sceneflow")
    depth_paths = sorted(p for p in os.listdir(depth_dir) if p.endswith(".npy"))
    if end is None or end < 0:
        selected = depth_paths[start::stride]
    else:
        selected = depth_paths[start:end:stride]
    return [os.path.splitext(p)[0] for p in selected]


def lidar_path_for_frame(sequence_dir, frame_id):
    data_2d_root = os.path.dirname(sequence_dir)
    data_root = os.path.dirname(data_2d_root)
    seq_name = os.path.basename(sequence_dir)
    return os.path.join(
        data_root,
        "data_3d_raw",
        seq_name,
        "velodyne_points",
        "data",
        f"{frame_id}.bin",
    )


def load_lidar_xyzi(path, min_forward_m=0.0, max_forward_m=0.0, max_points=0):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4).astype(np.float64)
    if min_forward_m > 0.0 or max_forward_m > 0.0:
        mask = np.ones(points.shape[0], dtype=bool)
        if min_forward_m > 0.0:
            mask &= points[:, 0] > min_forward_m
        if max_forward_m > 0.0:
            mask &= points[:, 0] < max_forward_m
        points = points[mask]
    max_points = int(max_points)
    if max_points > 0 and points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(np.int64)
        points = points[indices]
    return points


def project_velo_to_image(points_xyz, velo_to_cam, r_rect, p_rect, width, height, min_depth_m, max_depth_m):
    if points_xyz.shape[0] == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=bool), np.empty((0,), dtype=np.float64)

    points_h = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (velo_to_cam @ points_h.T).T[:, :3]
    rect = (r_rect @ cam.T).T
    rect_h = np.concatenate([rect, np.ones((rect.shape[0], 1), dtype=np.float64)], axis=1)
    proj = (p_rect @ rect_h.T).T
    z = proj[:, 2]
    uv = np.full((proj.shape[0], 2), np.nan, dtype=np.float64)
    positive_z = z > 1e-12
    uv[positive_z] = proj[positive_z, :2] / z[positive_z, None]

    valid = np.isfinite(uv).all(axis=1)
    valid &= z > float(min_depth_m)
    if max_depth_m > 0.0:
        valid &= z < float(max_depth_m)
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] < width)
    valid &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv_int = np.zeros_like(uv, dtype=np.int32)
    uv_int[valid] = np.round(uv[valid]).astype(np.int32)
    return uv_int, valid, z


def gndnet_nonground_mask(points_xyzi, cfg, args):
    repo_root = cfg.get("gndnet_repo_root", os.path.join(PROJECT_ROOT, "third_party", "GndNet"))
    keep_outside = cfg.get("gndnet_keep_outside_as_nonground", True)
    if args.no_gndnet_keep_outside:
        keep_outside = False

    threshold_m = cfg.get("gndnet_ground_threshold_m", 0.2)
    if args.gndnet_ground_threshold_m is not None:
        threshold_m = float(args.gndnet_ground_threshold_m)

    predictor = GndNetGroundFilter(
        repo_root=repo_root,
        checkpoint_path=cfg.get(
            "gndnet_checkpoint_path",
            os.path.join(repo_root, "trained_models", "checkpoint.pth.tar"),
        ),
        config_path=cfg.get(
            "gndnet_config_path",
            os.path.join(repo_root, "config", "config_kittiSem.yaml"),
        ),
        device=args.device,
        threshold_m=threshold_m,
        keep_outside_as_nonground=keep_outside,
    )
    return predictor.predict_nonground_mask(points_xyzi)


def draw_points(image, uv, depth, ground_mask, nonground_mask, radius, alpha):
    overlay = image.copy()
    visible = ground_mask | nonground_mask
    order = np.argsort(depth[visible])[::-1]
    visible_indices = np.flatnonzero(visible)[order]

    ground_color = (70, 210, 70)
    nonground_color = (40, 60, 255)
    for idx in visible_indices:
        color = nonground_color if nonground_mask[idx] else ground_color
        point = (int(uv[idx, 0]), int(uv[idx, 1]))
        if radius <= 1:
            overlay[point[1], point[0]] = color
        else:
            cv2.circle(overlay, point, int(radius), color, -1, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, float(alpha), image, 1.0 - float(alpha), 0)
    cv2.rectangle(blended, (10, 10), (360, 82), (0, 0, 0), -1)
    cv2.putText(blended, "ground", (22, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ground_color, 2, cv2.LINE_AA)
    cv2.putText(blended, "non-ground", (22, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, nonground_color, 2, cv2.LINE_AA)
    return blended


def process_frame(frame_id, sequence_dir, output_dir, cfg, calib, args):
    image_path = os.path.join(sequence_dir, "image_00", "data_rect", f"{frame_id}.png")
    lidar_path = lidar_path_for_frame(sequence_dir, frame_id)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not os.path.isfile(lidar_path):
        raise FileNotFoundError(f"Missing lidar: {lidar_path}")

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    points = load_lidar_xyzi(
        lidar_path,
        min_forward_m=args.min_forward_m,
        max_forward_m=args.max_forward_m,
        max_points=args.max_points,
    )
    p_rect, r_rect, velo_to_cam = calib
    nonground, info = gndnet_nonground_mask(points, cfg, args)

    uv, valid_projection, depth = project_velo_to_image(
        points[:, :3],
        velo_to_cam,
        r_rect,
        p_rect,
        width,
        height,
        args.min_depth_m,
        args.max_depth_m,
    )
    visible_ground = valid_projection & ~nonground
    visible_nonground = valid_projection & nonground

    overlay = draw_points(
        image,
        uv,
        depth,
        visible_ground,
        visible_nonground,
        radius=args.point_radius,
        alpha=args.alpha,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_image = os.path.join(output_dir, f"{frame_id}_ground_overlay.png")
    out_summary = os.path.join(output_dir, f"{frame_id}_ground_overlay_summary.json")
    cv2.imwrite(out_image, overlay)
    summary = {
        "frame_id": frame_id,
        "method": args.method,
        "image_path": image_path,
        "lidar_path": lidar_path,
        "output_image": out_image,
        "raw_points": int(points.shape[0]),
        "projected_points": int(np.count_nonzero(valid_projection)),
        "projected_ground_points": int(np.count_nonzero(visible_ground)),
        "projected_nonground_points": int(np.count_nonzero(visible_nonground)),
        "nonground_points": int(np.count_nonzero(nonground)),
        "ground_points": int(points.shape[0] - np.count_nonzero(nonground)),
        "classifier_info": info,
    }
    with open(out_summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Project KITTI-360 LiDAR ground/non-ground labels onto rectified images."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(EXP_ROOT, "configs", "kitti360", "lsgslam_pnp_fused_icp.py"),
        help="Experiment config path.",
    )
    parser.add_argument("--output-dir", default="", help="Output directory. Defaults to <workdir>/<run_name>/lidar_ground_image_overlay.")
    parser.add_argument("--method", default="gndnet", choices=["gndnet"])
    parser.add_argument("--device", default="cpu", help="Device for GndNet inference, e.g. cpu or cuda:0.")
    parser.add_argument("--gndnet-ground-threshold-m", type=float, default=None, help="Override GndNet point-vs-ground height threshold.")
    parser.add_argument("--no-gndnet-keep-outside", action="store_true", help="Do not force GndNet grid-outside points to non-ground.")
    parser.add_argument("--start", type=int, default=None, help="Override config start index into depth_sceneflow frame list.")
    parser.add_argument("--end", type=int, default=None, help="Override config end index into depth_sceneflow frame list.")
    parser.add_argument("--stride", type=int, default=None, help="Override config stride.")
    parser.add_argument("--max-frames", type=int, default=0, help="Process only the first N selected frames.")
    parser.add_argument("--all-lidar-frames", action="store_true", help="Ignore config slicing and process every LiDAR .bin frame.")
    parser.add_argument("--min-forward-m", type=float, default=None)
    parser.add_argument("--max-forward-m", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.85)
    args = parser.parse_args()

    cfg_module = load_config_module(args.config)
    config = cfg_module.config
    data_cfg = config["data"]
    lidar_cfg = config.get("lidar_motion_probe", {})

    start = cfg_module.start_idx if args.start is None else args.start
    end = cfg_module.end_idx if args.end is None else args.end
    stride = cfg_module.stride if args.stride is None else args.stride
    args.min_forward_m = lidar_cfg.get("min_forward_m", 0.0) if args.min_forward_m is None else args.min_forward_m
    args.max_forward_m = lidar_cfg.get("max_forward_m", 0.0) if args.max_forward_m is None else args.max_forward_m
    args.max_points = lidar_cfg.get("max_points", 120000) if args.max_points is None else args.max_points

    sequence_dir = os.path.join(data_cfg["basedir"], data_cfg["sequence"])
    output_dir = args.output_dir or os.path.join(config["workdir"], config["run_name"], "lidar_ground_image_overlay")
    calib_path = os.path.join(PROJECT_ROOT, "data", "kitti360", "calibration", "perspective.txt")
    p_rect, r_rect = parse_perspective_calib(calib_path)
    velo_to_cam = load_kitti360_velo_to_cam(PROJECT_ROOT)
    calib = (p_rect, r_rect, velo_to_cam)

    frame_ids = selected_frame_ids(
        sequence_dir,
        start=start,
        end=end,
        stride=stride,
        all_lidar_frames=args.all_lidar_frames,
    )
    if args.max_frames > 0:
        frame_ids = frame_ids[: args.max_frames]
    if not frame_ids:
        raise ValueError("No frames selected.")

    os.makedirs(output_dir, exist_ok=True)
    all_summaries = []
    for idx, frame_id in enumerate(frame_ids, start=1):
        summary = process_frame(frame_id, sequence_dir, output_dir, lidar_cfg, calib, args)
        all_summaries.append(summary)
        print(
            f"[{idx}/{len(frame_ids)}] {frame_id}: "
            f"projected ground={summary['projected_ground_points']} "
            f"non-ground={summary['projected_nonground_points']} -> {summary['output_image']}"
        )

    index_path = os.path.join(output_dir, "ground_overlay_index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": args.config,
                "method": args.method,
                "sequence_dir": sequence_dir,
                "num_frames": len(all_summaries),
                "frames": all_summaries,
            },
            handle,
            indent=2,
        )
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()

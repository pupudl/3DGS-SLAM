#!/usr/bin/env python3
"""Export dynamic-region edge debug views for RigidMask pair outputs.

For each current-frame dynamic component, this script saves:
  1. the component edge in the current frame,
  2. the edge of the same pixels warped to the previous frame by optical flow,
  3. the edge of the same pixels projected to the previous frame by camera pose.

The flow, depth, and intrinsics handling follows utils.dynamic_mask.se3_filter.
"""

import argparse
import json
import re
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

cv2 = None
np = None
find_pair_file = None
_backproject = None
_load_metric_depth = None
_resize_flow_to_shape = None
_target_to_counterpart_points = None


PAIR_TIME_RE = re.compile(r"^(?P<time_idx>\d+)_")
CURRENT_PAIR_RE = re.compile(r"^(?P<time_idx>\d+)_frame_(?P<target>.+)_from_(?P<counterpart>.+)$")
PREVIOUS_PAIR_RE = re.compile(
    r"^(?P<time_idx>\d+)_frame_(?P<target>.+)_to_(?P<counterpart>.+)_target_prev$"
)


def ensure_runtime_imports():
    global cv2
    global np
    global find_pair_file
    global _backproject
    global _load_metric_depth
    global _resize_flow_to_shape
    global _target_to_counterpart_points

    if cv2 is not None:
        return
    try:
        import cv2 as cv2_mod
        import numpy as np_mod
        from utils.dynamic_mask.fusion import find_pair_file as find_pair_file_func
        from utils.dynamic_mask.se3_filter import (
            _backproject as backproject_func,
            _load_metric_depth as load_metric_depth_func,
            _resize_flow_to_shape as resize_flow_to_shape_func,
            _target_to_counterpart_points as target_to_counterpart_points_func,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing runtime dependency. Run this script in the same Python/conda "
            "environment used for lsgslam_4dgs_online, where numpy and opencv-python "
            "are available."
        ) from exc

    cv2 = cv2_mod
    np = np_mod
    find_pair_file = find_pair_file_func
    _backproject = backproject_func
    _load_metric_depth = load_metric_depth_func
    _resize_flow_to_shape = resize_flow_to_shape_func
    _target_to_counterpart_points = target_to_counterpart_points_func


def parse_int_set(text):
    if not text:
        return None
    values = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.add(int(part))
    return values


def parse_str_set(text):
    if not text:
        return None
    return {part.strip() for part in str(text).split(",") if part.strip()}


def parse_range(text):
    if not text:
        return None
    if ":" in text:
        start_s, end_s = text.split(":", 1)
    elif "-" in text:
        start_s, end_s = text.split("-", 1)
    else:
        value = int(text)
        return value, value
    start = int(start_s) if start_s.strip() else None
    end = int(end_s) if end_s.strip() else None
    return start, end


def in_range(value, range_pair):
    if range_pair is None:
        return True
    start, end = range_pair
    if start is not None and value < start:
        return False
    if end is not None and value > end:
        return False
    return True


def parse_pair_name(pair_dir):
    name = pair_dir.name
    match = CURRENT_PAIR_RE.match(name)
    if match:
        return {
            "time_idx": int(match.group("time_idx")),
            "target_frame_id": match.group("target"),
            "counterpart_frame_id": match.group("counterpart"),
            "target_role": "current",
        }
    match = PREVIOUS_PAIR_RE.match(name)
    if match:
        return {
            "time_idx": int(match.group("time_idx")),
            "target_frame_id": match.group("target"),
            "counterpart_frame_id": match.group("counterpart"),
            "target_role": "previous",
        }
    return {}


def summary_target_role(summary, pair_dir):
    role = summary.get("target_role", summary.get("cost_coordinate_frame"))
    if role:
        return str(role).strip().lower()
    parsed = parse_pair_name(pair_dir)
    return parsed.get("target_role", "")


def summary_time_idx(summary, pair_dir):
    value = summary.get("target_time_idx", summary.get("time_idx"))
    if value is not None:
        return int(value)
    parsed = parse_pair_name(pair_dir)
    return parsed.get("time_idx")


def summary_target_frame_id(summary, pair_dir):
    value = summary.get("target_frame_id", summary.get("curr_frame_id"))
    if value is not None:
        return str(value)
    parsed = parse_pair_name(pair_dir)
    return parsed.get("target_frame_id", "")


def summary_counterpart_frame_id(summary, pair_dir):
    value = summary.get("counterpart_frame_id", summary.get("reference_frame_id"))
    if value is not None:
        return str(value)
    parsed = parse_pair_name(pair_dir)
    return parsed.get("counterpart_frame_id", "")


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_gray_mask(path, shape=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if shape is not None and image.shape != tuple(shape):
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return image > 127


def optional_gray_mask(path, shape):
    if not path.exists():
        return None
    return read_gray_mask(path, shape=shape)


def read_rgb(path):
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return image


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) * 255))


def load_dynamic_source_mask(pair_dir, args):
    final_path = find_pair_file(pair_dir, "dynamic_mask.png")
    if not final_path.exists():
        raise FileNotFoundError(f"Missing dynamic mask: {final_path}")
    final_mask = read_gray_mask(final_path)

    stage = str(args.mask_stage).strip().lower().replace("_", "-")
    if stage == "final":
        return final_mask, {
            "stage": "final",
            "source": str(final_path),
            "description": "dynamic_mask after lidar/image SE3 veto",
        }

    if stage == "pre-lidar":
        pre_lidar_path = find_pair_file(pair_dir, "dynamic_mask_pre_lidar.png")
        if not pre_lidar_path.exists():
            raise FileNotFoundError(f"Missing pre-lidar mask: {pre_lidar_path}")
        return read_gray_mask(pre_lidar_path, shape=final_mask.shape), {
            "stage": "pre-lidar",
            "source": str(pre_lidar_path),
            "description": "dynamic_mask_pre_lidar before lidar residual refinement",
        }

    if stage == "pre-se3-veto":
        mask = final_mask.copy()
        sources = [str(final_path)]
        for filename in ("lidar_se3_static_veto_mask.png", "se3_static_veto_mask.png"):
            veto_path = find_pair_file(pair_dir, filename)
            veto_mask = optional_gray_mask(veto_path, final_mask.shape)
            if veto_mask is None:
                continue
            mask |= veto_mask
            sources.append(str(veto_path))
        return mask, {
            "stage": "pre-se3-veto",
            "sources": sources,
            "description": "reconstructed mask before lidar_se3_static_veto and se3_static_veto",
        }

    raise ValueError(f"Unsupported mask stage: {args.mask_stage}")


def ensure_bgr(image, shape):
    if image is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if image.shape[:2] != tuple(shape):
        image = cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return image.copy()


def binary_edge(mask, width):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    width = max(int(width), 1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = mask.astype(np.uint8)
    for _ in range(width):
        eroded = cv2.erode(eroded, kernel, iterations=1)
    return mask & (~eroded.astype(bool))


def splat_points_to_mask(points_xy, shape, radius):
    mask = np.zeros(shape, dtype=np.uint8)
    if points_xy.size == 0:
        return mask.astype(bool)
    h, w = shape
    pts = np.rint(points_xy).astype(np.int32)
    valid = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < w)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < h)
    )
    pts = pts[valid]
    if pts.size == 0:
        return mask.astype(bool)
    mask[pts[:, 1], pts[:, 0]] = 1
    radius = max(int(radius), 0)
    if radius > 0:
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask.astype(bool)


def overlay_edges(base_bgr, edge_specs):
    overlay = base_bgr.copy()
    for edge, color, alpha in edge_specs:
        edge = np.asarray(edge, dtype=bool)
        if not np.any(edge):
            continue
        color_arr = np.asarray(color, dtype=np.float32)
        overlay[edge] = np.clip(
            overlay[edge].astype(np.float32) * (1.0 - alpha) + color_arr * alpha,
            0.0,
            255.0,
        ).astype(np.uint8)
    return overlay


def project_points(points3d, transform, K, shape):
    if points3d.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    pts4 = np.concatenate(
        [points3d.astype(np.float64), np.ones((points3d.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    cam = (transform.astype(np.float64) @ pts4.T).T[:, :3]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    xy = np.full((points3d.shape[0], 2), np.nan, dtype=np.float32)
    xy[valid, 0] = (float(K[0, 0]) * cam[valid, 0] / z[valid]) + float(K[0, 2])
    xy[valid, 1] = (float(K[1, 1]) * cam[valid, 1] / z[valid]) + float(K[1, 2])
    h, w = shape
    valid &= (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(w - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(h - 1))
    )
    return xy, valid


def quaternion_to_rotation(quat):
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.size != 4:
        return None
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return None
    r, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)],
            [2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)],
            [2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_w2c_from_npz(params, time_idx):
    if params is None or "cam_unnorm_rots" not in params or "cam_trans" not in params:
        return None
    rots = params["cam_unnorm_rots"]
    trans = params["cam_trans"]
    time_idx = int(time_idx)
    if time_idx < 0 or time_idx >= int(rots.shape[-1]) or time_idx >= int(trans.shape[-1]):
        return None
    R = quaternion_to_rotation(rots[..., time_idx])
    t = np.asarray(trans[..., time_idx], dtype=np.float64).reshape(-1)
    if R is None or t.size != 3 or not np.isfinite(t).all():
        return None
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def pose_transform_for_summary(params, summary, pair_dir):
    if params is None:
        return None, {"status": "skipped_missing_params"}
    role = summary_target_role(summary, pair_dir)
    target_time_idx = summary_time_idx(summary, pair_dir)
    if target_time_idx is None:
        return None, {"status": "skipped_missing_target_time_idx"}
    target_time_idx = int(target_time_idx)
    if role in ("current", "curr", "t"):
        counterpart_time_idx = target_time_idx - 1
    elif role in ("previous", "prev", "reference", "t-1"):
        counterpart_time_idx = target_time_idx + 1
    else:
        return None, {"status": "skipped_unknown_target_role", "target_role": role}
    target_w2c = camera_w2c_from_npz(params, target_time_idx)
    counterpart_w2c = camera_w2c_from_npz(params, counterpart_time_idx)
    if target_w2c is None or counterpart_w2c is None:
        return None, {
            "status": "skipped_pose_index_out_of_range",
            "target_time_idx": target_time_idx,
            "counterpart_time_idx": counterpart_time_idx,
        }
    transform = counterpart_w2c @ np.linalg.inv(target_w2c)
    return transform, {
        "status": "ok",
        "source": "params.npz",
        "target_time_idx": target_time_idx,
        "counterpart_time_idx": counterpart_time_idx,
        "target_role": role,
    }


def pair_time_idx(pair_dir, summary=None):
    if summary is not None:
        value = summary_time_idx(summary, pair_dir)
        if value is not None:
            return value
    match = PAIR_TIME_RE.match(pair_dir.name)
    return int(match.group("time_idx")) if match else None


def passes_filters(pair_dir, summary, args):
    role = summary_target_role(summary, pair_dir)
    if role != str(args.target_role).strip().lower():
        return False

    time_idx = pair_time_idx(pair_dir, summary)
    if args.time_idx_set is not None and time_idx not in args.time_idx_set:
        return False
    if time_idx is not None and not in_range(time_idx, args.time_range_pair):
        return False

    if args.frame_id_set is not None:
        target_frame_id = summary_target_frame_id(summary, pair_dir)
        if target_frame_id not in args.frame_id_set:
            return False

    if args.sequence:
        seq = str(args.sequence)
        sequence_text = " ".join(
            [
                str(pair_dir),
                str(summary.get("sequence", "")),
                str(summary.get("data_sequence", "")),
            ]
        )
        if seq not in sequence_text:
            return False
    return True


def discover_pair_roots(results_dir, sequence=None):
    base = Path(results_dir)
    candidates = []
    if base.name == "rigidmask_frontend_probe":
        candidates.append(base)
    for rel in (
        Path("dynamic_mask") / "rigidmask_frontend_probe",
        Path("rigidmask_frontend_probe"),
    ):
        path = base / rel
        if path.is_dir():
            candidates.append(path)
    if not candidates:
        candidates.extend(path for path in base.rglob("rigidmask_frontend_probe") if path.is_dir())

    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if sequence and str(sequence) not in str(path):
            continue
        unique.append(path)
    return sorted(unique)


def params_path_for_pair_root(pair_root, explicit_params=None):
    if explicit_params:
        path = Path(explicit_params)
        return path if path.exists() else None
    search_dirs = [
        pair_root.parent.parent,
        pair_root.parent,
        pair_root,
    ]
    for directory in search_dirs:
        path = directory / "params.npz"
        if path.exists():
            return path
    return None


def load_params(path):
    if path is None:
        return None
    return np.load(str(path))


def process_pair(pair_dir, params, args):
    summary_path = find_pair_file(pair_dir, "rigidmask_frontend_summary.json")
    if not summary_path.exists():
        summary_path = find_pair_file(pair_dir, "dynamic_fusion_summary.json")
    if not summary_path.exists():
        return {"pair_dir": str(pair_dir), "status": "skipped_missing_summary"}
    summary = load_json(summary_path)
    if not passes_filters(pair_dir, summary, args):
        return {"pair_dir": str(pair_dir), "status": "skipped_filtered"}

    arrays_path = find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz")
    if not arrays_path.exists():
        return {"pair_dir": str(pair_dir), "status": "skipped_missing_arrays"}

    arrays = np.load(str(arrays_path))
    mask_bool, mask_source = load_dynamic_source_mask(pair_dir, args)
    target_shape = mask_bool.shape
    depth, K, depth_summary = _load_metric_depth(arrays, summary, target_shape)
    if depth is None or K is None:
        return {
            "pair_dir": str(pair_dir),
            "status": "skipped_missing_geometry",
            "depth": depth_summary,
        }
    if "flow_full_x" not in arrays or "flow_full_y" not in arrays:
        return {"pair_dir": str(pair_dir), "status": "skipped_missing_flow"}

    flow_x, flow_y = _resize_flow_to_shape(
        np.asarray(arrays["flow_full_x"], dtype=np.float32),
        np.asarray(arrays["flow_full_y"], dtype=np.float32),
        target_shape,
    )
    xx, yy, other_x, other_y, direction = _target_to_counterpart_points(
        target_shape,
        flow_x,
        flow_y,
        summary.get("target_role", summary.get("cost_coordinate_frame", "current")),
    )

    min_depth = float(args.min_depth_m)
    valid = (
        np.isfinite(depth)
        & (depth >= min_depth)
        & np.isfinite(flow_x)
        & np.isfinite(flow_y)
        & np.isfinite(other_x)
        & np.isfinite(other_y)
        & (other_x >= 0.0)
        & (other_x <= float(target_shape[1] - 1))
        & (other_y >= 0.0)
        & (other_y <= float(target_shape[0] - 1))
    )
    if args.max_depth_m is not None:
        valid &= depth <= float(args.max_depth_m)

    points3d_map = _backproject(xx, yy, depth, K)
    flow_points2d_map = np.stack((other_x, other_y), axis=-1).astype(np.float32)
    pose_transform, pose_summary = pose_transform_for_summary(params, summary, pair_dir)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    component_ids = list(range(1, num_labels))
    component_ids = sorted(component_ids, key=lambda label: stats[label, cv2.CC_STAT_AREA], reverse=True)
    if args.max_components > 0:
        component_ids = component_ids[: int(args.max_components)]

    current_region = np.zeros_like(mask_bool, dtype=bool)
    prev_flow_region = np.zeros_like(mask_bool, dtype=bool)
    prev_pose_region = np.zeros_like(mask_bool, dtype=bool)
    component_records = []

    for label in component_ids:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(args.min_component_area):
            continue
        component = labels == label
        comp_valid = component & valid
        valid_count = int(np.count_nonzero(comp_valid))
        if valid_count < int(args.min_valid_points):
            component_records.append(
                {
                    "label": int(label),
                    "status": "skipped_too_few_valid_points",
                    "area": area,
                    "valid_points": valid_count,
                    "centroid_px": [float(centroids[label][0]), float(centroids[label][1])],
                }
            )
            continue

        flow_points = flow_points2d_map[comp_valid]
        flow_region = splat_points_to_mask(flow_points, target_shape, args.splat_radius)

        pose_region = np.zeros_like(mask_bool, dtype=bool)
        pose_points_count = 0
        if pose_transform is not None:
            pose_points, pose_valid = project_points(points3d_map[comp_valid], pose_transform, K, target_shape)
            pose_points_count = int(np.count_nonzero(pose_valid))
            pose_region = splat_points_to_mask(pose_points[pose_valid], target_shape, args.splat_radius)

        current_region |= component
        prev_flow_region |= flow_region
        prev_pose_region |= pose_region
        component_records.append(
            {
                "label": int(label),
                "status": "ok",
                "area": area,
                "valid_points": valid_count,
                "flow_projected_pixels": int(np.count_nonzero(flow_region)),
                "pose_projected_pixels": int(np.count_nonzero(pose_region)),
                "pose_projected_points": pose_points_count,
                "bbox_xywh": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "centroid_px": [float(centroids[label][0]), float(centroids[label][1])],
            }
        )

    current_edges = binary_edge(current_region, args.edge_width)
    prev_flow_edges = binary_edge(prev_flow_region, args.edge_width)
    prev_pose_edges = binary_edge(prev_pose_region, args.edge_width)
    overlap_edges = prev_flow_edges & prev_pose_edges

    out_dir = pair_dir / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_mask(out_dir / "current_edges.png", current_edges)
    save_mask(out_dir / "prev_flow_edges.png", prev_flow_edges)
    save_mask(out_dir / "prev_pose_edges.png", prev_pose_edges)
    save_mask(out_dir / "prev_flow_region.png", prev_flow_region)
    save_mask(out_dir / "prev_pose_region.png", prev_pose_region)

    curr_rgb = read_rgb(find_pair_file(pair_dir, "curr_rgb.png"))
    prev_rgb = read_rgb(find_pair_file(pair_dir, "prev_rgb.png"))
    current_overlay = overlay_edges(
        ensure_bgr(curr_rgb, target_shape),
        [(current_edges, (0, 255, 255), 1.0)],
    )
    previous_overlay = overlay_edges(
        ensure_bgr(prev_rgb, target_shape),
        [
            (prev_flow_edges, (255, 128, 0), 0.95),
            (prev_pose_edges, (0, 0, 255), 0.95),
            (overlap_edges, (255, 255, 255), 1.0),
        ],
    )
    cv2.imwrite(str(out_dir / "current_overlay.png"), current_overlay)
    cv2.imwrite(str(out_dir / "previous_overlay.png"), previous_overlay)

    summary_out = {
        "status": "ok",
        "pair_dir": str(pair_dir),
        "time_idx": pair_time_idx(pair_dir, summary),
        "target_role": summary_target_role(summary, pair_dir),
        "target_frame_id": summary_target_frame_id(summary, pair_dir),
        "counterpart_frame_id": summary_counterpart_frame_id(summary, pair_dir),
        "direction": direction,
        "target_shape": list(target_shape),
        "depth": depth_summary,
        "mask_source": mask_source,
        "pose_projection": pose_summary,
        "thresholds": {
            "min_component_area": int(args.min_component_area),
            "min_valid_points": int(args.min_valid_points),
            "min_depth_m": float(args.min_depth_m),
            "max_depth_m": None if args.max_depth_m is None else float(args.max_depth_m),
            "splat_radius": int(args.splat_radius),
            "edge_width": int(args.edge_width),
        },
        "counts": {
            "components_total": int(max(num_labels - 1, 0)),
            "components_exported": int(sum(1 for item in component_records if item.get("status") == "ok")),
            "current_edge_pixels": int(np.count_nonzero(current_edges)),
            "prev_flow_edge_pixels": int(np.count_nonzero(prev_flow_edges)),
            "prev_pose_edge_pixels": int(np.count_nonzero(prev_pose_edges)),
            "prev_edge_overlap_pixels": int(np.count_nonzero(overlap_edges)),
        },
        "colors_bgr": {
            "current_overlay_current_edge": [0, 255, 255],
            "previous_overlay_flow_edge": [255, 128, 0],
            "previous_overlay_pose_edge": [0, 0, 255],
            "previous_overlay_overlap": [255, 255, 255],
        },
        "outputs": {
            "current_edges": str(out_dir / "current_edges.png"),
            "prev_flow_edges": str(out_dir / "prev_flow_edges.png"),
            "prev_pose_edges": str(out_dir / "prev_pose_edges.png"),
            "prev_flow_region": str(out_dir / "prev_flow_region.png"),
            "prev_pose_region": str(out_dir / "prev_pose_region.png"),
            "current_overlay": str(out_dir / "current_overlay.png"),
            "previous_overlay": str(out_dir / "previous_overlay.png"),
        },
        "components": component_records,
    }
    save_json(out_dir / "region_edges_summary.json", summary_out)
    return summary_out


def iter_pair_dirs(pair_root):
    for path in sorted(pair_root.iterdir()):
        if path.is_dir():
            yield path


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Export current/previous dynamic-region edge debug images from LSG-SLAM RigidMask outputs.",
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Run dir, dynamic_mask dir, rigidmask_frontend_probe dir, or a parent results dir to scan.",
    )
    parser.add_argument(
        "--params",
        default="",
        help="Optional params.npz path. If omitted, the script tries to find one next to each run dir.",
    )
    parser.add_argument("--sequence", default="", help="Optional sequence substring used to filter discovered runs.")
    parser.add_argument("--time-idx", default="", help="Comma-separated target time_idx values, e.g. 25,26,30.")
    parser.add_argument("--time-range", default="", help="Inclusive time_idx range, e.g. 20:80.")
    parser.add_argument("--frame-ids", default="", help="Comma-separated target frame ids from summary target_frame_id.")
    parser.add_argument("--target-role", default="current", help="Usually current; previous is also supported.")
    parser.add_argument(
        "--mask-stage",
        default="final",
        choices=("final", "pre-se3-veto", "pre-lidar"),
        help=(
            "Which dynamic mask to use for current-frame components. "
            "pre-se3-veto reconstructs the mask before lidar/image SE3 veto."
        ),
    )
    parser.add_argument("--output-subdir", default="edge_debug", help="Output subdir under each pair dir.")
    parser.add_argument("--min-component-area", type=int, default=80)
    parser.add_argument("--min-valid-points", type=int, default=50)
    parser.add_argument("--max-components", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--splat-radius", type=int, default=1)
    parser.add_argument("--edge-width", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", help="Print skip reasons for non-exported pairs.")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.time_idx_set = parse_int_set(args.time_idx)
    args.time_range_pair = parse_range(args.time_range)
    args.frame_id_set = parse_str_set(args.frame_ids)

    pair_roots = discover_pair_roots(args.results_dir, sequence=args.sequence or None)
    if not pair_roots:
        print(f"No rigidmask_frontend_probe directories found under {args.results_dir}")
        return 1

    try:
        ensure_runtime_imports()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    processed = []
    skipped = 0
    for pair_root in pair_roots:
        params_path = params_path_for_pair_root(pair_root, args.params or None)
        params = load_params(params_path)
        if params_path is None:
            print(f"[warn] No params.npz found for {pair_root}; pose projection will be skipped.")
        for pair_dir in iter_pair_dirs(pair_root):
            try:
                result = process_pair(pair_dir, params, args)
            except Exception as exc:
                result = {"pair_dir": str(pair_dir), "status": "error", "reason": str(exc)}
            if result.get("status") == "ok":
                processed.append(result)
                print(f"[ok] {pair_dir} -> {pair_dir / args.output_subdir}")
            else:
                skipped += 1
                if args.verbose and result.get("status") != "skipped_filtered":
                    reason = result.get("reason", "")
                    suffix = f": {reason}" if reason else ""
                    print(f"[skip] {pair_dir}: {result.get('status')}{suffix}")

    print(f"Done. Exported {len(processed)} pair(s), skipped {skipped}.")
    return 0 if processed else 2


if __name__ == "__main__":
    raise SystemExit(main())

import cv2
import numpy as np


DEFAULT_SE3_STATIC_VETO_CFG = {
    "enabled": False,
    "min_component_area": 80,
    "max_components": 64,
    "min_valid_points": 50,
    "min_bg_points": 200,
    "max_bg_points": 8000,
    "max_obj_points": 3000,
    "min_depth_m": 0.1,
    "max_depth_m": 80.0,
    "pnp_reproj_error_px": 4.0,
    "pnp_confidence": 0.995,
    "pnp_iterations": 100,
    "bg_inlier_px": 3.0,
    "bg_inlier_ratio": 0.70,
    "bg_median_px": 3.0,
    "obj_min_inlier_ratio": 0.35,
    "rel_angle_deg": 1.5,
    "rel_trans_m": 0.15,
    "bg_vs_obj_median_ratio": 1.25,
    "prefer_slam_pose": True,
    "fallback_to_background_pnp": True,
    "seed": 0,
}


DEFAULT_COMPONENT_POSE_INIT_CFG = {
    "enabled": True,
    "min_component_area": 64,
    "max_components": 32,
    "min_valid_points": 50,
    "max_obj_points": 3000,
    "min_depth_m": 0.1,
    "max_depth_m": 80.0,
    "pnp_reproj_error_px": 4.0,
    "pnp_confidence": 0.995,
    "pnp_iterations": 100,
    "min_inlier_ratio": 0.20,
    "max_reproj_median_px": 8.0,
    "seed": 0,
}


def merge_se3_static_veto_cfg(cfg):
    merged = dict(DEFAULT_SE3_STATIC_VETO_CFG)
    merged.update(cfg or {})
    return merged


def merge_component_pose_init_cfg(cfg):
    merged = dict(DEFAULT_COMPONENT_POSE_INIT_CFG)
    merged.update(cfg or {})
    return merged


def _resize_flow_to_shape(flow_x, flow_y, target_shape):
    target_h, target_w = target_shape
    src_h, src_w = flow_x.shape
    if (src_h, src_w) == (target_h, target_w):
        return flow_x.astype(np.float32), flow_y.astype(np.float32)
    resized_x = cv2.resize(flow_x.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_y = cv2.resize(flow_y.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_x *= float(target_w) / max(float(src_w), 1.0)
    resized_y *= float(target_h) / max(float(src_h), 1.0)
    return resized_x.astype(np.float32), resized_y.astype(np.float32)


def _calibration_from_summary(summary):
    calib = summary.get("calibration", {})
    if not isinstance(calib, dict):
        return None
    required = ("fx", "cx", "cy", "baseline")
    if any(key not in calib for key in required):
        return None
    fx = float(calib["fx"])
    fy = float(calib.get("fy", fx))
    cx = float(calib["cx"])
    cy = float(calib["cy"])
    baseline = float(calib["baseline"])
    if min(fx, fy, baseline) <= 0.0:
        return None
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline}


def _scaled_intrinsics(calib, source_shape, target_shape):
    src_h, src_w = source_shape
    target_h, target_w = target_shape
    sx = float(target_w) / max(float(src_w), 1.0)
    sy = float(target_h) / max(float(src_h), 1.0)
    fx = calib["fx"] * sx
    fy = calib["fy"] * sy
    cx = calib["cx"] * sx
    cy = calib["cy"] * sy
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K


def _load_metric_depth(arrays, summary, target_shape):
    calib = _calibration_from_summary(summary)
    if calib is None:
        return None, None, {"status": "missing_calibration"}

    if "depth_metric_input_full" in arrays:
        depth_full = np.asarray(arrays["depth_metric_input_full"], dtype=np.float32)
        source_shape = depth_full.shape
        depth = cv2.resize(depth_full, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        K = _scaled_intrinsics(calib, source_shape, target_shape)
        return depth.astype(np.float32), K, {
            "status": "ok",
            "source": "depth_metric_input_full",
            "source_shape": list(source_shape),
        }

    if "disp_input_full" not in arrays:
        return None, None, {"status": "missing_depth_or_disparity"}

    disp_full = np.asarray(arrays["disp_input_full"], dtype=np.float32)
    source_shape = disp_full.shape
    disp = cv2.resize(disp_full, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    sx = float(target_shape[1]) / max(float(source_shape[1]), 1.0)
    disp = disp * sx
    K = _scaled_intrinsics(calib, source_shape, target_shape)
    depth = np.full(target_shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disp) & (disp > 1e-6)
    depth[valid] = float(K[0, 0] * calib["baseline"]) / disp[valid]
    return depth, K, {
        "status": "ok",
        "source": "disp_input_full",
        "source_shape": list(source_shape),
    }


def _target_to_counterpart_points(target_shape, flow_x, flow_y, target_role):
    h, w = target_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    role = str(target_role or "current").strip().lower()
    if role in ("current", "curr", "t"):
        other_x = xx - flow_x
        other_y = yy - flow_y
        direction = "current_to_previous"
    else:
        other_x = xx + flow_x
        other_y = yy + flow_y
        direction = "previous_to_current"
    return xx, yy, other_x, other_y, direction


def _backproject(xx, yy, depth, K):
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth
    return np.stack((x, y, depth), axis=-1).astype(np.float32)


def _sample_indices(indices, max_points, rng):
    if max_points <= 0 or indices.size <= max_points:
        return indices
    return rng.choice(indices, size=int(max_points), replace=False)


def _fit_pnp_ransac(points3d, points2d, K, cfg, rng, max_points):
    count = int(points3d.shape[0])
    if count < int(cfg["min_valid_points"]):
        return None
    sample_idx = _sample_indices(np.arange(count), int(max_points), rng)
    obj = points3d[sample_idx].astype(np.float32)
    img = points2d[sample_idx].astype(np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K.astype(np.float64),
        None,
        iterationsCount=int(cfg["pnp_iterations"]),
        reprojectionError=float(cfg["pnp_reproj_error_px"]),
        confidence=float(cfg["pnp_confidence"]),
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None

    inlier_idx = sample_idx[inliers[:, 0]]
    try:
        ok_refine, rvec, tvec = cv2.solvePnP(
            points3d[inlier_idx].astype(np.float32),
            points2d[inlier_idx].astype(np.float32),
            K.astype(np.float64),
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        ok = bool(ok_refine)
    except cv2.error:
        ok = True
    if not ok:
        return None

    R = cv2.Rodrigues(rvec)[0].astype(np.float64)
    t = tvec.reshape(3).astype(np.float64)
    return {
        "R": R,
        "t": t,
        "inliers": inlier_idx,
        "inlier_ratio": float(len(inlier_idx) / max(sample_idx.size, 1)),
        "sampled_points": int(sample_idx.size),
    }


def _reprojection_errors(points3d, points2d, K, model):
    R = model["R"]
    t = model["t"]
    cam = (R @ points3d.astype(np.float64).T).T + t[None, :]
    z = cam[:, 2]
    valid = np.isfinite(cam).all(axis=1) & (z > 1e-6)
    projected = np.full((points3d.shape[0], 2), np.nan, dtype=np.float32)
    projected[valid, 0] = (K[0, 0] * cam[valid, 0] / z[valid]) + K[0, 2]
    projected[valid, 1] = (K[1, 1] * cam[valid, 1] / z[valid]) + K[1, 2]
    errors = np.full(points3d.shape[0], np.inf, dtype=np.float32)
    diff = projected[valid] - points2d[valid].astype(np.float32)
    errors[valid] = np.linalg.norm(diff, axis=1)
    return errors


def _rotation_angle_deg(R):
    trace_value = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_value)))


def _model_delta(bg_model, obj_model):
    rel_R = obj_model["R"] @ bg_model["R"].T
    angle_deg = _rotation_angle_deg(rel_R)
    trans_m = float(np.linalg.norm(obj_model["t"] - bg_model["t"]))
    return angle_deg, trans_m


def _background_model_from_transform(background_transform):
    if background_transform is None:
        return None, {"status": "missing"}
    if not isinstance(background_transform, dict):
        return None, {"status": "invalid_type"}

    if "target_to_counterpart" in background_transform:
        T = np.asarray(background_transform["target_to_counterpart"], dtype=np.float64)
        if T.shape != (4, 4):
            return None, {"status": "invalid_matrix_shape", "shape": list(T.shape)}
        R = T[:3, :3]
        t = T[:3, 3]
    elif "R" in background_transform and "t" in background_transform:
        R = np.asarray(background_transform["R"], dtype=np.float64)
        t = np.asarray(background_transform["t"], dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            return None, {"status": "invalid_rotation_shape", "shape": list(R.shape)}
    else:
        return None, {"status": "missing_R_t"}

    if not np.isfinite(R).all() or not np.isfinite(t).all():
        return None, {"status": "nonfinite_transform"}
    return {
        "R": R,
        "t": t,
        "inliers": np.zeros(0, dtype=np.int64),
        "inlier_ratio": None,
        "sampled_points": 0,
    }, {
        "status": "ok",
        "source": background_transform.get("source", "slam_tracking_pose"),
        "target_time_idx": background_transform.get("target_time_idx"),
        "counterpart_time_idx": background_transform.get("counterpart_time_idx"),
    }


def _component_bbox_xyxy(stats, label):
    left = int(stats[label, cv2.CC_STAT_LEFT])
    top = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    return [left, top, left + width, top + height]


def _component_centroid_xy(labels, label):
    ys, xs = np.nonzero(labels == label)
    if xs.size == 0:
        return [0.0, 0.0]
    return [float(xs.mean()), float(ys.mean())]


def estimate_component_motion_poses(mask, arrays, summary, cfg=None):
    cfg = merge_component_pose_init_cfg(cfg)
    base_summary = {
        "enabled": bool(cfg.get("enabled", False)),
        "status": "disabled",
        "components": [],
        "num_components": 0,
        "num_ok": 0,
    }
    if not cfg.get("enabled", False):
        return base_summary

    mask_bool = np.asarray(mask > 0.5)
    if not np.any(mask_bool):
        return {**base_summary, "status": "skipped_empty_mask"}
    if "flow_full_x" not in arrays or "flow_full_y" not in arrays:
        return {**base_summary, "status": "skipped_missing_flow"}

    target_shape = mask_bool.shape
    depth, K, depth_summary = _load_metric_depth(arrays, summary, target_shape)
    if depth is None or K is None:
        return {
            **base_summary,
            "status": "skipped_missing_geometry",
            "depth": depth_summary,
            "target_shape": list(target_shape),
        }

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

    min_depth = float(cfg["min_depth_m"])
    max_depth = cfg.get("max_depth_m")
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
    if max_depth is not None:
        valid &= depth <= float(max_depth)

    points3d_map = _backproject(xx, yy, depth, K)
    points2d_map = np.stack((other_x, other_y), axis=-1).astype(np.float32)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    component_ids = list(range(1, num_labels))
    component_ids = sorted(component_ids, key=lambda label: stats[label, cv2.CC_STAT_AREA], reverse=True)
    max_components = int(cfg.get("max_components", 0))
    if max_components > 0:
        component_ids = component_ids[:max_components]

    components = []
    for label in component_ids:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(cfg["min_component_area"]):
            continue
        component = labels == label
        comp_valid = component & valid
        valid_count = int(np.count_nonzero(comp_valid))
        record = {
            "label": int(label),
            "status": "skipped",
            "reason": "",
            "area": area,
            "valid_points": valid_count,
            "bbox_xyxy": _component_bbox_xyxy(stats, label),
            "centroid_px": _component_centroid_xy(labels, label),
        }
        if valid_count < int(cfg["min_valid_points"]):
            record["reason"] = "too_few_valid_points"
            components.append(record)
            continue

        comp_points3d = points3d_map[comp_valid]
        comp_points2d = points2d_map[comp_valid]
        obj_model = _fit_pnp_ransac(comp_points3d, comp_points2d, K, cfg, rng, int(cfg["max_obj_points"]))
        if obj_model is None:
            record["reason"] = "pnp_failed"
            components.append(record)
            continue

        obj_errors = _reprojection_errors(comp_points3d, comp_points2d, K, obj_model)
        finite_obj = np.isfinite(obj_errors)
        if np.any(finite_obj):
            obj_values = obj_errors[finite_obj]
            obj_median = float(np.median(obj_values))
            obj_p90 = float(np.percentile(obj_values, 90.0))
        else:
            obj_median = None
            obj_p90 = None

        inlier_ratio = float(obj_model["inlier_ratio"])
        quality_ok = (
            obj_median is not None
            and inlier_ratio >= float(cfg["min_inlier_ratio"])
            and obj_median <= float(cfg["max_reproj_median_px"])
        )
        if not quality_ok:
            record.update(
                {
                    "reason": "low_quality_pnp",
                    "obj_median_px": obj_median,
                    "obj_p90_px": obj_p90,
                    "obj_inlier_ratio": inlier_ratio,
                    "sampled_points": int(obj_model["sampled_points"]),
                }
            )
            components.append(record)
            continue

        target_to_counterpart = np.eye(4, dtype=np.float64)
        target_to_counterpart[:3, :3] = obj_model["R"]
        target_to_counterpart[:3, 3] = obj_model["t"]
        record.update(
            {
                "status": "ok",
                "reason": "component_pnp",
                "source": "rigidmask_component_pnp",
                "direction": direction,
                "target_role": summary.get("target_role", summary.get("cost_coordinate_frame", "current")),
                "target_time_idx": summary.get("target_time_idx", summary.get("time_idx")),
                "target_frame_id": summary.get("target_frame_id", summary.get("curr_frame_id")),
                "counterpart_frame_id": summary.get("counterpart_frame_id", summary.get("reference_frame_id")),
                "R_target_to_counterpart": obj_model["R"].tolist(),
                "t_target_to_counterpart": obj_model["t"].tolist(),
                "target_to_counterpart": target_to_counterpart.tolist(),
                "obj_median_px": obj_median,
                "obj_p90_px": obj_p90,
                "obj_inlier_ratio": inlier_ratio,
                "sampled_points": int(obj_model["sampled_points"]),
            }
        )
        components.append(record)

    return {
        **base_summary,
        "status": "ok",
        "direction": direction,
        "depth": depth_summary,
        "target_shape": list(target_shape),
        "valid_points": int(np.count_nonzero(valid)),
        "num_components": int(len(components)),
        "num_ok": int(sum(1 for item in components if item.get("status") == "ok")),
        "thresholds": {
            "min_component_area": int(cfg["min_component_area"]),
            "min_valid_points": int(cfg["min_valid_points"]),
            "min_inlier_ratio": float(cfg["min_inlier_ratio"]),
            "max_reproj_median_px": float(cfg["max_reproj_median_px"]),
        },
        "components": components,
    }


def refine_mask_with_se3_static_veto(mask, arrays, summary, cfg=None, background_transform=None):
    cfg = merge_se3_static_veto_cfg(cfg)
    base_summary = {
        "enabled": bool(cfg.get("enabled", False)),
        "status": "disabled",
        "removed_components": 0,
        "removed_pixels": 0,
    }
    diagnostics = {}
    if not cfg.get("enabled", False):
        return mask.astype(np.float32), base_summary, diagnostics

    mask_bool = np.asarray(mask > 0.5)
    if not np.any(mask_bool):
        summary_out = {**base_summary, "status": "skipped_empty_mask"}
        return mask.astype(np.float32), summary_out, diagnostics

    if "flow_full_x" not in arrays or "flow_full_y" not in arrays:
        summary_out = {**base_summary, "status": "skipped_missing_flow"}
        return mask.astype(np.float32), summary_out, diagnostics

    target_shape = mask_bool.shape
    depth, K, depth_summary = _load_metric_depth(arrays, summary, target_shape)
    if depth is None or K is None:
        summary_out = {**base_summary, "status": "skipped_missing_geometry", "depth": depth_summary}
        return mask.astype(np.float32), summary_out, diagnostics

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

    min_depth = float(cfg["min_depth_m"])
    max_depth = cfg.get("max_depth_m")
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
    if max_depth is not None:
        valid &= depth <= float(max_depth)

    points3d_map = _backproject(xx, yy, depth, K)
    points2d_map = np.stack((other_x, other_y), axis=-1).astype(np.float32)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))

    bg_fit_mask = valid & (~mask_bool)
    bg_model = None
    bg_pose_summary = {"status": "disabled"}
    if cfg.get("prefer_slam_pose", True):
        bg_model, bg_pose_summary = _background_model_from_transform(background_transform)

    if bg_model is None:
        if not cfg.get("fallback_to_background_pnp", True):
            summary_out = {
                **base_summary,
                "status": "skipped_missing_slam_pose",
                "valid_points": int(np.count_nonzero(valid)),
                "background_points": int(np.count_nonzero(bg_fit_mask)),
                "depth": depth_summary,
                "background_pose": bg_pose_summary,
            }
            return mask.astype(np.float32), summary_out, diagnostics

        if int(np.count_nonzero(bg_fit_mask)) < int(cfg["min_bg_points"]):
            bg_fit_mask = valid
        if int(np.count_nonzero(bg_fit_mask)) < int(cfg["min_bg_points"]):
            summary_out = {
                **base_summary,
                "status": "skipped_too_few_background_points",
                "valid_points": int(np.count_nonzero(valid)),
                "background_points": int(np.count_nonzero(bg_fit_mask)),
                "depth": depth_summary,
                "background_pose": bg_pose_summary,
            }
            return mask.astype(np.float32), summary_out, diagnostics

        bg_points3d = points3d_map[bg_fit_mask]
        bg_points2d = points2d_map[bg_fit_mask]
        bg_model = _fit_pnp_ransac(bg_points3d, bg_points2d, K, cfg, rng, int(cfg["max_bg_points"]))
        if bg_model is None:
            summary_out = {
                **base_summary,
                "status": "skipped_background_pnp_failed",
                "valid_points": int(np.count_nonzero(valid)),
                "background_points": int(np.count_nonzero(bg_fit_mask)),
                "depth": depth_summary,
                "background_pose": bg_pose_summary,
            }
            return mask.astype(np.float32), summary_out, diagnostics
        bg_pose_summary = {
            "status": "ok",
            "source": "background_pnp_fallback",
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=8)
    component_ids = list(range(1, num_labels))
    component_ids = sorted(component_ids, key=lambda label: stats[label, cv2.CC_STAT_AREA], reverse=True)
    max_components = int(cfg.get("max_components", 0))
    if max_components > 0:
        component_ids = component_ids[:max_components]

    veto_mask = np.zeros_like(mask_bool, dtype=bool)
    inspected = []
    for label in component_ids:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(cfg["min_component_area"]):
            continue
        component = labels == label
        comp_valid = component & valid
        valid_count = int(np.count_nonzero(comp_valid))
        record = {
            "label": int(label),
            "area": area,
            "valid_points": valid_count,
            "decision": "keep",
            "reason": "",
        }
        if valid_count < int(cfg["min_valid_points"]):
            record["decision"] = "keep"
            record["reason"] = "too_few_valid_points"
            inspected.append(record)
            continue

        comp_points3d = points3d_map[comp_valid]
        comp_points2d = points2d_map[comp_valid]
        bg_errors = _reprojection_errors(comp_points3d, comp_points2d, K, bg_model)
        finite_bg = np.isfinite(bg_errors)
        if not np.any(finite_bg):
            record["reason"] = "background_projection_invalid"
            inspected.append(record)
            continue
        bg_values = bg_errors[finite_bg]
        bg_median = float(np.median(bg_values))
        bg_p90 = float(np.percentile(bg_values, 90.0))
        bg_inlier_ratio = float(np.mean(bg_values <= float(cfg["bg_inlier_px"])))
        record.update(
            {
                "bg_median_px": bg_median,
                "bg_p90_px": bg_p90,
                "bg_inlier_ratio": bg_inlier_ratio,
            }
        )

        bg_explains = (
            bg_median <= float(cfg["bg_median_px"])
            and bg_inlier_ratio >= float(cfg["bg_inlier_ratio"])
        )
        if bg_explains:
            veto_mask[component] = True
            record["decision"] = "remove_static"
            record["reason"] = "background_se3_explains_component"
            inspected.append(record)
            continue

        obj_model = _fit_pnp_ransac(comp_points3d, comp_points2d, K, cfg, rng, int(cfg["max_obj_points"]))
        if obj_model is None:
            record["reason"] = "object_pnp_failed"
            inspected.append(record)
            continue

        obj_errors = _reprojection_errors(comp_points3d, comp_points2d, K, obj_model)
        finite_obj = np.isfinite(obj_errors)
        if np.any(finite_obj):
            obj_values = obj_errors[finite_obj]
            obj_median = float(np.median(obj_values))
            obj_p90 = float(np.percentile(obj_values, 90.0))
            obj_projection_valid = True
        else:
            obj_median = None
            obj_p90 = None
            obj_projection_valid = False
        rel_angle_deg, rel_trans_m = _model_delta(bg_model, obj_model)
        record.update(
            {
                "obj_median_px": obj_median,
                "obj_p90_px": obj_p90,
                "obj_inlier_ratio": float(obj_model["inlier_ratio"]),
                "rel_angle_deg": rel_angle_deg,
                "rel_trans_m": rel_trans_m,
            }
        )
        static_like_obj = (
            obj_projection_valid
            and obj_model["inlier_ratio"] >= float(cfg["obj_min_inlier_ratio"])
            and rel_angle_deg <= float(cfg["rel_angle_deg"])
            and rel_trans_m <= float(cfg["rel_trans_m"])
            and bg_median <= obj_median * float(cfg["bg_vs_obj_median_ratio"])
        )
        if static_like_obj:
            veto_mask[component] = True
            record["decision"] = "remove_static"
            record["reason"] = "object_se3_matches_background"
        else:
            record["decision"] = "keep"
            record["reason"] = "independent_motion_or_uncertain"
        inspected.append(record)

    refined = mask_bool & (~veto_mask)
    removed_pixels = int(np.count_nonzero(veto_mask & mask_bool))
    removed_components = sum(1 for item in inspected if item.get("decision") == "remove_static")
    diagnostics = {
        "veto_mask": veto_mask.astype(np.float32),
        "kept_mask": refined.astype(np.float32),
        "component_labels": labels.astype(np.float32) / max(float(labels.max()), 1.0),
    }
    summary_out = {
        **base_summary,
        "status": "ok",
        "direction": direction,
        "depth": depth_summary,
        "target_shape": list(target_shape),
        "valid_points": int(np.count_nonzero(valid)),
        "background_points": int(np.count_nonzero(bg_fit_mask)),
        "background_pnp": {
            "inlier_ratio": None if bg_model["inlier_ratio"] is None else float(bg_model["inlier_ratio"]),
            "sampled_points": int(bg_model["sampled_points"]),
            "translation_norm_m": float(np.linalg.norm(bg_model["t"])),
        },
        "background_pose": bg_pose_summary,
        "removed_components": int(removed_components),
        "inspected_components": int(len(inspected)),
        "removed_pixels": removed_pixels,
        "removed_fraction_of_mask": float(removed_pixels / max(int(np.count_nonzero(mask_bool)), 1)),
        "thresholds": {
            "bg_inlier_px": float(cfg["bg_inlier_px"]),
            "bg_inlier_ratio": float(cfg["bg_inlier_ratio"]),
            "bg_median_px": float(cfg["bg_median_px"]),
            "rel_angle_deg": float(cfg["rel_angle_deg"]),
            "rel_trans_m": float(cfg["rel_trans_m"]),
        },
        "components": inspected,
    }
    return refined.astype(np.float32), summary_out, diagnostics

import os

import cv2
import numpy as np


DEFAULT_LIDAR_SE3_STATIC_VETO_CFG = {
    "enabled": False,
    "filename": "lidar_se3_points.npz",
    "use_nonground_only": True,
    "fallback_to_visible_points": True,
    "min_component_area": 80,
    "max_components": 64,
    "min_lidar_points": 25,
    "min_reference_points": 200,
    "component_association_radius_px": 2.0,
    "bg_inlier_dist_m": 0.35,
    "bg_inlier_ratio": 0.65,
    "bg_median_dist_m": 0.25,
    "bg_p90_dist_m": None,
    "object_icp": {
        "enabled": True,
        "min_points": 30,
        "max_points": 1500,
        "local_margin_m": 1.0,
        "max_corr_m": 0.75,
        "max_iteration": 30,
        "min_fitness": 0.25,
        "max_rmse_m": 0.40,
        "rel_angle_deg": 2.0,
        "rel_trans_m": 0.25,
        "median_dist_m": 0.35,
        "bg_vs_obj_median_ratio": 1.20,
    },
}


def merge_lidar_se3_static_veto_cfg(cfg):
    merged = {}
    for key, value in DEFAULT_LIDAR_SE3_STATIC_VETO_CFG.items():
        merged[key] = dict(value) if isinstance(value, dict) else value
    for key, value in (cfg or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            updated = dict(merged[key])
            updated.update(value)
            merged[key] = updated
        else:
            merged[key] = value
    return merged


def _load_lidar_se3_npz(lidar_pair_dir, filename):
    if not lidar_pair_dir:
        return None, {
            "status": "skipped_missing_lidar_pair_dir",
            "path": "",
        }
    path = os.path.join(str(lidar_pair_dir), filename)
    if not os.path.exists(path):
        return None, {
            "status": "skipped_missing_lidar_se3_points",
            "path": path,
        }
    try:
        return np.load(path), {
            "status": "ok",
            "path": path,
        }
    except Exception as exc:
        return None, {
            "status": "skipped_lidar_se3_points_load_error",
            "path": path,
            "reason": str(exc),
        }


def _array(data, key, default=None):
    if data is None or key not in data.files:
        return default
    return np.asarray(data[key])


def _finite_points(points):
    points = np.asarray(points, dtype=np.float32)
    return points[np.isfinite(points).all(axis=1)]


def _target_prefix_from_role(target_role):
    role = str(target_role or "current").strip().lower()
    if role in ("previous", "prev", "reference", "t-1"):
        return "prev", "curr"
    return "curr", "prev"


def _select_point_set(data, cfg, target_role):
    target_prefix, reference_prefix = _target_prefix_from_role(target_role)
    preferred = "nonground" if cfg.get("use_nonground_only", True) else "visible"
    fallback = "visible" if preferred == "nonground" else "nonground"
    choices = [preferred]
    if cfg.get("fallback_to_visible_points", True) and fallback not in choices:
        choices.append(fallback)

    for name in choices:
        target_points = _array(data, f"{target_prefix}_{name}_bev")
        reference_points = _array(data, f"{reference_prefix}_{name}_bev")
        uv = _array(data, f"{target_prefix}_{name}_uv")
        valid = _array(data, f"{target_prefix}_{name}_valid_projection")
        if target_points is None or reference_points is None or uv is None or valid is None:
            continue
        target_points = np.asarray(target_points, dtype=np.float32)
        reference_points = _finite_points(reference_points)
        uv = np.asarray(uv, dtype=np.float32)
        valid = np.asarray(valid).astype(bool)
        if target_points.shape[0] != uv.shape[0] or target_points.shape[0] != valid.shape[0]:
            continue
        finite = np.isfinite(target_points).all(axis=1) & np.isfinite(uv).all(axis=1)
        keep = valid & finite
        if np.count_nonzero(keep) == 0 or reference_points.shape[0] == 0:
            continue
        return {
            "source": name,
            "target_prefix": target_prefix,
            "reference_prefix": reference_prefix,
            "target_points": target_points[keep],
            "reference_points": reference_points,
            "uv": uv[keep],
        }

    return None


def _points_to_mask_indices(uv, image_shape, mask_shape):
    img_h, img_w = int(image_shape[0]), int(image_shape[1])
    mask_h, mask_w = int(mask_shape[0]), int(mask_shape[1])
    if img_h <= 0 or img_w <= 0 or mask_h <= 0 or mask_w <= 0:
        return None, None, np.zeros((uv.shape[0],), dtype=bool)
    xs = np.floor(uv[:, 0] * float(mask_w) / float(img_w)).astype(np.int64)
    ys = np.floor(uv[:, 1] * float(mask_h) / float(img_h)).astype(np.int64)
    valid = (xs >= 0) & (xs < mask_w) & (ys >= 0) & (ys < mask_h)
    return xs, ys, valid


def _build_kdtree(points):
    try:
        import open3d as o3d
    except ImportError:
        return None, None, "missing_open3d"
    if points.shape[0] == 0:
        return None, None, "empty_reference_points"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return pcd, o3d.geometry.KDTreeFlann(pcd), None


def _nearest_distances(points, kd_tree):
    distances = np.full((points.shape[0],), np.inf, dtype=np.float32)
    for idx, point in enumerate(points.astype(np.float64)):
        _count, _indices, sq_dists = kd_tree.search_knn_vector_3d(point, 1)
        if sq_dists:
            distances[idx] = float(np.sqrt(max(sq_dists[0], 0.0)))
    return distances


def _rotation_angle_deg(rotation):
    trace_value = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(trace_value)))


def _sample_points(points, max_points, seed):
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indices]


def _local_reference_points(component_points, reference_points, margin_m):
    if component_points.shape[0] == 0 or reference_points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32)
    lo = component_points.min(axis=0) - float(margin_m)
    hi = component_points.max(axis=0) + float(margin_m)
    keep = np.all((reference_points >= lo[None, :]) & (reference_points <= hi[None, :]), axis=1)
    return reference_points[keep]


def _icp_component_motion(component_points, reference_points, cfg, seed):
    icp_cfg = cfg.get("object_icp", {})
    if not icp_cfg.get("enabled", True):
        return None, "disabled"
    if component_points.shape[0] < int(icp_cfg.get("min_points", 30)):
        return None, "too_few_component_points"

    local_ref = _local_reference_points(
        component_points,
        reference_points,
        margin_m=float(icp_cfg.get("local_margin_m", 1.0)),
    )
    if local_ref.shape[0] < int(icp_cfg.get("min_points", 30)):
        return None, "too_few_reference_points"

    try:
        import open3d as o3d
    except ImportError:
        return None, "missing_open3d"

    src_points = _sample_points(
        component_points,
        int(icp_cfg.get("max_points", 1500)),
        seed,
    )
    ref_points = _sample_points(
        local_ref,
        int(icp_cfg.get("max_points", 1500)),
        seed + 17,
    )
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_points.astype(np.float64))
    target.points = o3d.utility.Vector3dVector(ref_points.astype(np.float64))
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(icp_cfg.get("max_corr_m", 0.75)),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(icp_cfg.get("max_iteration", 30)),
        ),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)
    transformed = (transform[:3, :3] @ component_points.astype(np.float64).T).T + transform[:3, 3][None, :]
    pcd, kd_tree, error = _build_kdtree(local_ref)
    if error is not None:
        return None, error
    distances = _nearest_distances(transformed.astype(np.float32), kd_tree)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return None, "no_finite_icp_distances"
    return {
        "fitness": float(result.fitness),
        "rmse_m": float(result.inlier_rmse),
        "median_dist_m": float(np.median(finite)),
        "p90_dist_m": float(np.percentile(finite, 90.0)),
        "rel_angle_deg": _rotation_angle_deg(transform[:3, :3]),
        "rel_trans_m": float(np.linalg.norm(transform[:3, 3])),
        "local_reference_points": int(local_ref.shape[0]),
        "sampled_source_points": int(src_points.shape[0]),
        "sampled_reference_points": int(ref_points.shape[0]),
    }, "ok"


def refine_mask_with_lidar_se3_static_veto(mask, lidar_pair_dir, cfg=None, target_role="current"):
    cfg = merge_lidar_se3_static_veto_cfg(cfg)
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
        return mask.astype(np.float32), {**base_summary, "status": "skipped_empty_mask"}, diagnostics

    data, load_summary = _load_lidar_se3_npz(lidar_pair_dir, cfg.get("filename", "lidar_se3_points.npz"))
    if data is None:
        return mask.astype(np.float32), {**base_summary, **load_summary}, diagnostics

    target_prefix, _reference_prefix = _target_prefix_from_role(target_role)
    image_shape = _array(data, f"{target_prefix}_image_shape")
    if image_shape is None:
        image_shape = _array(data, "image_shape")
    if image_shape is None or len(image_shape) < 2:
        return mask.astype(np.float32), {
            **base_summary,
            "status": "skipped_missing_image_shape",
            "path": load_summary.get("path", ""),
        }, diagnostics
    image_shape = [int(image_shape[0]), int(image_shape[1])]

    points = _select_point_set(data, cfg, target_role)
    if points is None:
        return mask.astype(np.float32), {
            **base_summary,
            "status": "skipped_missing_valid_lidar_points",
            "path": load_summary.get("path", ""),
        }, diagnostics

    curr_points = points["target_points"]
    prev_points = points["reference_points"]
    uv = points["uv"]
    if prev_points.shape[0] < int(cfg.get("min_reference_points", 200)):
        return mask.astype(np.float32), {
            **base_summary,
            "status": "skipped_too_few_reference_points",
            "path": load_summary.get("path", ""),
            "point_source": points["source"],
            "reference_points": int(prev_points.shape[0]),
        }, diagnostics

    _pcd, kd_tree, kd_error = _build_kdtree(prev_points)
    if kd_error is not None:
        return mask.astype(np.float32), {
            **base_summary,
            "status": f"skipped_{kd_error}",
            "path": load_summary.get("path", ""),
            "point_source": points["source"],
        }, diagnostics

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    component_ids = list(range(1, num_labels))
    component_ids = sorted(component_ids, key=lambda label: stats[label, cv2.CC_STAT_AREA], reverse=True)
    max_components = int(cfg.get("max_components", 0))
    if max_components > 0:
        component_ids = component_ids[:max_components]

    xs, ys, point_valid = _points_to_mask_indices(uv, image_shape, mask_bool.shape)
    if xs is None:
        return mask.astype(np.float32), {
            **base_summary,
            "status": "skipped_invalid_projection_shape",
            "path": load_summary.get("path", ""),
        }, diagnostics

    curr_points = curr_points[point_valid]
    xs = xs[point_valid]
    ys = ys[point_valid]
    point_labels = labels[ys, xs]

    veto_mask = np.zeros_like(mask_bool, dtype=bool)
    inspected = []
    association_radius = int(round(float(cfg.get("component_association_radius_px", 2.0))))
    kernel = None
    if association_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * association_radius + 1, 2 * association_radius + 1),
        )

    icp_cfg = cfg.get("object_icp", {})
    for label in component_ids:
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(cfg.get("min_component_area", 80)):
            continue
        component = labels == label
        if kernel is None:
            component_for_points = component
            point_indices = np.where(point_labels == label)[0]
        else:
            component_for_points = cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
            point_indices = np.where(component_for_points[ys, xs])[0]

        record = {
            "label": int(label),
            "area": area,
            "lidar_points": int(point_indices.size),
            "decision": "keep",
            "reason": "",
        }
        if point_indices.size < int(cfg.get("min_lidar_points", 25)):
            record["reason"] = "too_few_lidar_points"
            inspected.append(record)
            continue

        comp_points = curr_points[point_indices]
        distances = _nearest_distances(comp_points, kd_tree)
        finite = distances[np.isfinite(distances)]
        if finite.size == 0:
            record["reason"] = "no_finite_background_distances"
            inspected.append(record)
            continue

        bg_median = float(np.median(finite))
        bg_p90 = float(np.percentile(finite, 90.0))
        bg_inlier_ratio = float(np.mean(finite <= float(cfg.get("bg_inlier_dist_m", 0.35))))
        record.update(
            {
                "bg_median_dist_m": bg_median,
                "bg_p90_dist_m": bg_p90,
                "bg_inlier_ratio": bg_inlier_ratio,
            }
        )

        bg_p90_thresh = cfg.get("bg_p90_dist_m", None)
        bg_explains = (
            bg_median <= float(cfg.get("bg_median_dist_m", 0.25))
            and bg_inlier_ratio >= float(cfg.get("bg_inlier_ratio", 0.65))
            and (bg_p90_thresh is None or bg_p90 <= float(bg_p90_thresh))
        )
        if bg_explains:
            veto_mask[component] = True
            record["decision"] = "remove_static"
            record["reason"] = "lidar_background_se3_explains_component"
            inspected.append(record)
            continue

        icp_result, icp_status = _icp_component_motion(
            comp_points,
            prev_points,
            cfg,
            seed=int(label),
        )
        record["object_icp_status"] = icp_status
        if icp_result is not None:
            record.update({f"object_icp_{key}": value for key, value in icp_result.items()})
            static_like_icp = (
                icp_result["fitness"] >= float(icp_cfg.get("min_fitness", 0.25))
                and icp_result["rmse_m"] <= float(icp_cfg.get("max_rmse_m", 0.40))
                and icp_result["median_dist_m"] <= float(icp_cfg.get("median_dist_m", 0.35))
                and icp_result["rel_angle_deg"] <= float(icp_cfg.get("rel_angle_deg", 2.0))
                and icp_result["rel_trans_m"] <= float(icp_cfg.get("rel_trans_m", 0.25))
                and bg_median <= icp_result["median_dist_m"] * float(icp_cfg.get("bg_vs_obj_median_ratio", 1.20))
            )
            if static_like_icp:
                veto_mask[component] = True
                record["decision"] = "remove_static"
                record["reason"] = "lidar_object_icp_matches_background"
            else:
                record["reason"] = "lidar_motion_independent_or_uncertain"
        else:
            record["reason"] = "lidar_motion_independent_or_uncertain"
        inspected.append(record)

    refined = mask_bool & (~veto_mask)
    removed_pixels = int(np.count_nonzero(veto_mask & mask_bool))
    removed_components = int(sum(1 for item in inspected if item.get("decision") == "remove_static"))
    diagnostics = {
        "veto_mask": veto_mask.astype(np.float32),
        "kept_mask": refined.astype(np.float32),
        "component_labels": labels.astype(np.float32) / max(float(labels.max()), 1.0),
    }
    summary_out = {
        **base_summary,
        "status": "ok",
        "path": load_summary.get("path", ""),
        "point_source": points["source"],
        "target_role": str(target_role or "current"),
        "target_prefix": points["target_prefix"],
        "reference_prefix": points["reference_prefix"],
        "image_shape": image_shape,
        "target_shape": list(mask_bool.shape),
        "current_points": int(curr_points.shape[0]),
        "reference_points": int(prev_points.shape[0]),
        "inspected_components": int(len(inspected)),
        "removed_components": removed_components,
        "removed_pixels": removed_pixels,
        "removed_fraction_of_mask": float(removed_pixels / max(int(np.count_nonzero(mask_bool)), 1)),
        "thresholds": {
            "min_component_area": int(cfg.get("min_component_area", 80)),
            "min_lidar_points": int(cfg.get("min_lidar_points", 25)),
            "bg_inlier_dist_m": float(cfg.get("bg_inlier_dist_m", 0.35)),
            "bg_inlier_ratio": float(cfg.get("bg_inlier_ratio", 0.65)),
            "bg_median_dist_m": float(cfg.get("bg_median_dist_m", 0.25)),
        },
        "components": inspected,
    }
    return refined.astype(np.float32), summary_out, diagnostics

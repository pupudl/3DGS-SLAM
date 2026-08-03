import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


PAIR_OUTPUT_GROUPS = {
    "inputs": {
        "prev_rgb.png",
        "curr_rgb.png",
    },
    "costs": {
        "homography_cost.png",
        "epipolar_cost.png",
        "pp2d_cost.png",
        "pp3d_orth_cost.png",
        "pp3d_dir_cost.png",
        "depth_contrast_cost.png",
        "oor2_cost_grid.png",
        "dc_unc_cost_grid.png",
        "tau_cost_grid.png",
        "flow_magnitude.png",
        "tau_full.png",
    },
    "filters": {
        "depth_mask_full.png",
        "sky_mask_full.png",
        "lidar_static_exclusion_mask.png",
        "lidar_above_range_mask.png",
        "lidar_visible_mask.png",
    },
    "dynamic": {
        "dynamic_score.png",
        "dynamic_score_pre_lidar.png",
        "dynamic_score_geom.png",
        "dynamic_gate.png",
        "dynamic_mask.png",
        "dynamic_mask_pre_lidar.png",
        "appearance_score.png",
        "similarity_score.png",
        "lidar_residual_score.png",
        "lidar_residual_confidence.png",
        "lidar_suppression_mask.png",
    },
    "raw": {
        "rigidmask_frontend_arrays.npz",
    },
    "metadata": {
        "rigidmask_frontend_summary.json",
        "dynamic_fusion_summary.json",
    },
}

PAIR_OUTPUT_CATEGORY_BY_NAME = {
    filename: category
    for category, filenames in PAIR_OUTPUT_GROUPS.items()
    for filename in filenames
}


def pair_output_path(pair_dir, filename, organize_outputs=True):
    if not organize_outputs:
        return pair_dir / filename
    category = PAIR_OUTPUT_CATEGORY_BY_NAME.get(filename)
    if category is None:
        return pair_dir / filename
    out_dir = pair_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def find_pair_file(pair_dir, filename):
    candidates = [pair_dir / filename]
    category = PAIR_OUTPUT_CATEGORY_BY_NAME.get(filename)
    if category is not None:
        candidates.append(pair_dir / category / filename)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def organize_pair_outputs(pair_dir):
    moved = []
    for filename, category in PAIR_OUTPUT_CATEGORY_BY_NAME.items():
        src = pair_dir / filename
        if not src.exists():
            continue
        dst_dir = pair_dir / category
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / filename
        if src.resolve() == dst.resolve():
            continue
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
        moved.append(str(dst.relative_to(pair_dir)))
    return moved


#把数组归一化到 [0,1]，使用低分位数和高分位数作为上下限，排除 NaN / Inf
def robust_normalize(array, low_q=5.0, high_q=95.0, finite_mask=None):
    if finite_mask is None:
        finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.float32), 0.0, 1.0
    values = array[finite_mask]
    lo = float(np.percentile(values, low_q))
    hi = float(np.percentile(values, high_q))
    if hi <= lo:
        hi = lo + 1e-6
    normalized = np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized, lo, hi

#输入图像数组，保存为灰度图像，范围 [0,1] 映射到 [0,255]
def save_gray_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), image_u8)

#计算旋转向量的模长，返回弧度值
def angle_magnitude(rot_vec):
    return float(np.linalg.norm(np.asarray(rot_vec, dtype=np.float32)))


def compute_motion_factors(summary, arrays):
    #读取旋转和平移
    rot = np.asarray(summary.get("rot", [0.0, 0.0, 0.0]), dtype=np.float32)
    trans = np.asarray(summary.get("trans", [0.0, 0.0, 1.0]), dtype=np.float32)
    #计算光流大小，每个像素会得到一个光流模长
    flow_mag = np.sqrt(arrays["flow_full_x"] ** 2 + arrays["flow_full_y"] ** 2)

    #计算旋转角度、平移量和光流幅值
    theta = angle_magnitude(rot) #得到旋转向量模长
    t_lateral = float(np.linalg.norm(trans[:2])) #得到平移向量的横向分量模长
    t_forward = float(abs(trans[2])) #得到平移向量的前向分量模长
    flow_p75 = float(np.percentile(flow_mag[np.isfinite(flow_mag)], 75.0)) #得到光流幅值的75%分位数

    #旋转因子、横向平移因子、前向平移因子、光流因子和视差因子，把物理量映射到 [0,1] 权重因子
    rot_factor = np.clip(theta / np.deg2rad(2.0), 0.0, 1.0) #计算旋转因子，2度以内的旋转被认为是小旋转，超过2度的旋转被认为是大旋转
    lateral_factor = np.clip(t_lateral / 0.5, 0.0, 1.0) #计算横向平移因子，0.5以内的平移被认为是小平移，超过0.5的平移被认为是大平移
    forward_factor = np.clip(t_forward / 0.7, 0.0, 1.0) #计算前向平移因子，0.7以内的平移被认为是小平移，超过0.7的平移被认为是大平移
    flow_factor = np.clip(flow_p75 / 4.0, 0.0, 1.0) #计算光流因子，4.0以内的光流被认为是小光流，超过4.0的光流被认为是大光流
    parallax_factor = np.clip(0.6 * lateral_factor + 0.4 * flow_factor, 0.0, 1.0) #计算视差因子，结合横向平移和光流因子

    return {
        "theta_rad": theta,
        "theta_deg": float(np.rad2deg(theta)),
        "t_lateral": t_lateral,
        "t_forward": t_forward,
        "flow_p75": flow_p75,
        "rot_factor": float(rot_factor),
        "lateral_factor": float(lateral_factor),
        "forward_factor": float(forward_factor),
        "flow_factor": float(flow_factor),
        "parallax_factor": float(parallax_factor),
    }


def compute_weights(motion):
    rot_factor = motion["rot_factor"]
    lateral_factor = motion["lateral_factor"]
    forward_factor = motion["forward_factor"]
    flow_factor = motion["flow_factor"]
    parallax_factor = motion["parallax_factor"]

    weights = {
        "homography_cost": 0.12 + 0.18 * rot_factor, #相机旋转较大，则增加单应性代价的权重
        "epipolar_cost": 0.22 + 0.10 * flow_factor, #光流较大，则增加极线代价的权重
        "depth_contrast_cost": 0.12 + 0.18 * forward_factor, #相机前向平移较大，则增加深度对比代价的权重
        "pp3d_orth_cost": 0.10 + 0.20 * parallax_factor, #视差较大，则增加3D正交代价的权重
        "pp2d_cost": 0.16 + 0.14 * lateral_factor, #相机横向平移较大，则增加2D投影代价的权重
        "pp3d_dir_cost": 0.14 + 0.14 * parallax_factor, #视差较大，则增加3D方向代价的权重
    }
    #归一化权重，使其总和为 1
    weight_sum = sum(weights.values())
    return {key: float(value / weight_sum) for key, value in weights.items()}


def reliable_mask_from_uncertainty(arrays):
    #取两种不确定性图
    oor = arrays["oor2_cost_grid"]
    dc_unc = arrays["dc_unc_cost_grid"]

    #排除 NaN / Inf
    oor_finite = np.isfinite(oor)
    dc_finite = np.isfinite(dc_unc)

    #用 60% 分位数​ 作为阈值
    oor_thr = float(np.percentile(oor[oor_finite], 60.0)) if np.any(oor_finite) else 0.0
    dc_thr = float(np.percentile(np.abs(dc_unc[dc_finite]), 60.0)) if np.any(dc_finite) else 0.0

    #可靠区域要求：有限值、不确定性较低
    reliable = oor_finite & dc_finite & (oor <= oor_thr) & (np.abs(dc_unc) <= dc_thr)

    #如果可靠区域过少，则仅要求有限值
    if reliable.mean() < 0.05:
        reliable = oor_finite & dc_finite
    return reliable

#计算方向异常，返回归一化异常值、中心值、低分位数和高分位数
#在某些代价图中，异常值可能是由于方向上的偏差引起的，而不是绝对值的偏差。
#通过计算，与可靠背景的典型值偏离多少，可以更好地捕捉到这些异常情况，从而提高动态区域检测的准确性。
def directional_anomaly(array, reliable_mask):
    finite = np.isfinite(array)
    if np.any(reliable_mask):
        #先在可靠区域中计算中位数作为中心值
        center = float(np.median(array[reliable_mask]))
    elif np.any(finite):
        center = float(np.median(array[finite]))
    else:
        center = 0.0
    #计算每个像素与中心值的绝对差异，作为异常值
    anomaly = np.abs(array - center)
    normalized, lo, hi = robust_normalize(anomaly)
    return normalized, center, lo, hi


def gate_from_uncertainty(arrays):
    #读取两种不确定性代价图，分别进行归一化到[0,1]
    oor_score, oor_lo, oor_hi = robust_normalize(arrays["oor2_cost_grid"])
    dc_score, dc_lo, dc_hi = robust_normalize(np.abs(arrays["dc_unc_cost_grid"]))
    #计算门控因子，结合两种不确定性代价图，越不确定的区域门控因子越小，最低为 0.05，最高为 1.0
    gate = np.clip((1.0 - oor_score) * (1.0 - 0.7 * dc_score), 0.05, 1.0).astype(np.float32)
    return gate, {
        "oor_low": oor_lo,
        "oor_high": oor_hi,
        "dc_unc_low": dc_lo,
        "dc_unc_high": dc_hi,
    }

#把连续的动态分数图变成最终二值掩码
def clean_binary_mask(binary, min_component_area=32):
    binary = binary.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    #形态学操作：先开运算去除小的噪声，再闭运算填充小的孔洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #连通域过滤，去除小的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            cleaned[labels == label] = 1
    return cleaned.astype(np.float32)


#把连续的动态分数图变成最终二值掩码
def postprocess_mask(score, threshold):
    return clean_binary_mask(score >= threshold)

#判断是否强制输出空掩码，依据是全局动态分数的均值、90%分位数、99%分位数和高分数区域的比例是否都低于设定阈值
def should_force_empty_mask(
    score,
    mean_thresh,
    p90_thresh,
    p99_thresh,
    high_score_thresh,
    high_score_area_thresh,
):
    finite = np.isfinite(score)
    if not np.any(finite):
        return True, {
            "score_mean": 0.0,
            "score_p90": 0.0,
            "score_p99": 0.0,
            "high_score_area_fraction": 0.0,
        }

    score_mean = float(score[finite].mean())
    score_p90 = float(np.percentile(score[finite], 90.0))
    score_p99 = float(np.percentile(score[finite], 99.0))
    high_score_area_fraction = float((score[finite] >= high_score_thresh).mean())

    force_empty = (
        score_mean < mean_thresh
        and score_p90 < p90_thresh
        and score_p99 < p99_thresh
        and high_score_area_fraction < high_score_area_thresh
    )
    return force_empty, {
        "score_mean": score_mean,
        "score_p90": score_p90,
        "score_p99": score_p99,
        "high_score_area_fraction": high_score_area_fraction,
    }

#低分辨率代价图缩放到原图大小
def resize_to_image(score, image_shape):
    height, width = image_shape[:2]
    return cv2.resize(score, (width, height), interpolation=cv2.INTER_LINEAR)

#找到并读取当前帧对应的特征相似度图
def load_similarity_map_for_pair(pair_dir, summary, feature_probe_subdir):
    time_idx = summary.get("time_idx")
    curr_frame_id = summary.get("curr_frame_id")
    if time_idx is None or curr_frame_id is None:
        return None, None

    feature_root = pair_dir.parent.parent / feature_probe_subdir
    frame_name = f"{int(time_idx):06d}_frame_{curr_frame_id}"
    frame_dir = feature_root / frame_name

    similarity_npy = frame_dir / "similarity.npy"
    if similarity_npy.exists():
        similarity = np.load(similarity_npy).astype(np.float32)
        return np.clip(similarity, 0.0, 1.0), str(similarity_npy)

    similarity_png = frame_dir / "similarity.png"
    if similarity_png.exists():
        similarity = cv2.imread(str(similarity_png), cv2.IMREAD_GRAYSCALE)
        if similarity is not None:
            return similarity.astype(np.float32) / 255.0, str(similarity_png)

    return None, None

#把相似度图转换为外观异常分数，范围 [0,1]，并返回归一化后的相似度图和低高分位数
def appearance_score_from_similarity(similarity_map, target_shape):
    target_h, target_w = target_shape
    similarity_resized = cv2.resize(
        similarity_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    similarity_resized = np.clip(similarity_resized, 0.0, 1.0)
    appearance_cost = 1.0 - similarity_resized
    #归一化外观异常分数到 [0,1]，并计算低高分位数
    appearance_score, app_lo, app_hi = robust_normalize(appearance_cost)
    return appearance_score, similarity_resized, app_lo, app_hi

#前端计算了深度掩码或天空掩码，作为最终掩码的过滤器，进一步去除不可靠区域。
def final_mask_filter_from_saved_masks(arrays, summary):
    depth_summary = summary.get("depth_mask", {})
    if depth_summary.get("apply_stage") != "post_dynamic_mask":
        return None, None
    filters = []
    sources = []
    if "depth_mask_cost_grid" in arrays:
        filters.append(arrays["depth_mask_cost_grid"].astype(bool))
        sources.append("depth_mask_cost_grid")
    if "sky_mask_cost_grid" in arrays:
        filters.append(~arrays["sky_mask_cost_grid"].astype(bool))
        sources.append("~sky_mask_cost_grid")
    if not filters:
        return None, None
    return np.logical_and.reduce(filters), "&".join(sources)


def frame_id_variants(frame_id):
    variants = []
    if frame_id is None:
        return variants
    raw = str(frame_id)
    variants.append(raw)
    try:
        variants.append(f"{int(frame_id):010d}")
    except (TypeError, ValueError):
        pass
    deduped = []
    for value in variants:
        if value not in deduped:
            deduped.append(value)
    return deduped


def load_preview_shape(pair_dir, arrays):
    for filename in ("prev_rgb.png", "curr_rgb.png"):
        image_path = find_pair_file(pair_dir, filename)
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                return image.shape
    return arrays["flow_full_x"].shape


def find_lidar_motion_pair_dir_for_pair(pair_dir, summary, lidar_motion_subdir):
    curr_frame_id = summary.get("curr_frame_id")
    curr_variants = frame_id_variants(curr_frame_id)
    if not curr_variants:
        return None, {
            "enabled": True,
            "status": "skipped_missing_curr_frame_id",
        }

    lidar_root = pair_dir.parent.parent / lidar_motion_subdir
    if not lidar_root.is_dir():
        return None, {
            "enabled": True,
            "status": "skipped_missing_lidar_motion_root",
            "root": str(lidar_root),
        }

    candidate_dirs = []
    for curr_variant in curr_variants:
        candidate_dirs.extend(path for path in lidar_root.glob(f"*_{curr_variant}") if path.is_dir())
    candidate_dirs = sorted(set(candidate_dirs))
    if not candidate_dirs:
        return None, {
            "enabled": True,
            "status": "skipped_missing_lidar_pair",
            "root": str(lidar_root),
            "curr_frame_id": curr_frame_id,
        }

    return candidate_dirs[-1], None


def load_lidar_projection_for_pair(pair_dir, summary, lidar_motion_subdir, projection_filename):
    lidar_pair_dir, load_error = find_lidar_motion_pair_dir_for_pair(
        pair_dir,
        summary,
        lidar_motion_subdir,
    )
    if lidar_pair_dir is None:
        return None, load_error

    projection_path = lidar_pair_dir / projection_filename
    if not projection_path.exists():
        projection_path = lidar_pair_dir / "projection" / projection_filename
    if not projection_path.exists():
        return None, {
            "enabled": True,
            "status": "skipped_missing_projection_npz",
            "pair_dir": str(lidar_pair_dir),
            "projection_filename": projection_filename,
        }

    projection = np.load(projection_path)
    required = {"uv", "residuals"}
    missing = sorted(required.difference(projection.files))
    if missing:
        return None, {
            "enabled": True,
            "status": "skipped_projection_npz_missing_keys",
            "source": str(projection_path),
            "missing": missing,
        }

    uv = projection["uv"].astype(np.float32)
    residuals = projection["residuals"].astype(np.float32)
    if "visible" in projection.files:
        visible = projection["visible"].astype(bool)
    elif "valid_projection" in projection.files:
        visible = projection["valid_projection"].astype(bool)
    else:
        visible = np.ones(residuals.shape[0], dtype=bool)
    depth = (
        projection["depth"].astype(np.float32)
        if "depth" in projection.files
        else np.ones(residuals.shape[0], dtype=np.float32)
    )

    return {
        "uv": uv,
        "residuals": residuals,
        "visible": visible,
        "depth": depth,
        "source": str(projection_path),
        "pair_dir": str(lidar_pair_dir),
    }, None


def resize_bool_mask(mask, target_shape):
    target_h, target_w = target_shape
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def build_above_lidar_range_mask_from_projection(
    lidar_projection,
    image_shape,
    row_percentile,
    row_margin_px,
):
    height, width = image_shape[:2]
    above_mask = np.zeros((height, width), dtype=bool)
    uv = lidar_projection["uv"]
    visible = (
        lidar_projection["visible"]
        & np.isfinite(uv).all(axis=1)
        & np.isfinite(lidar_projection["depth"])
    )
    if not np.any(visible):
        return above_mask, {
            "status": "skipped_no_visible_projection_points",
            "top_row": None,
            "row_cutoff": 0,
            "visible_points": 0,
        }

    rows = uv[visible, 1]
    top_row = float(np.percentile(rows, float(row_percentile)))
    row_cutoff = int(np.floor(top_row - float(row_margin_px)))
    row_cutoff = max(0, min(height, row_cutoff))
    above_mask[:row_cutoff, :] = True
    return above_mask, {
        "status": "fallback_from_residual_projection",
        "top_row": top_row,
        "row_cutoff": int(row_cutoff),
        "visible_points": int(np.count_nonzero(visible)),
    }


def load_lidar_static_masks_for_pair(
    lidar_projection,
    image_shape,
    static_mask_filename,
    above_row_percentile,
    above_row_margin_px,
):
    height, width = image_shape[:2]
    empty = np.zeros((height, width), dtype=bool)
    lidar_pair_dir = Path(lidar_projection["pair_dir"])
    static_mask_path = lidar_pair_dir / static_mask_filename
    if static_mask_path.exists():
        data = np.load(static_mask_path)
        above_lidar_range_mask = data["above_lidar_range_mask"].astype(bool)
        masks = {
            "static_exclusion_mask": above_lidar_range_mask,
            "above_lidar_range_mask": above_lidar_range_mask,
            "visible_mask": data["visible_mask"].astype(bool),
        }
        return masks, {
            "enabled": True,
            "status": "ok",
            "source": str(static_mask_path),
            "top_row": None if float(data["top_row"]) < 0.0 else float(data["top_row"]),
            "row_cutoff": int(data["row_cutoff"]),
            "visible_points": int(data["visible_points"]),
            "nonground_points": int(data["nonground_points"]) if "nonground_points" in data.files else None,
        }

    above_mask, above_summary = build_above_lidar_range_mask_from_projection(
        lidar_projection,
        image_shape,
        row_percentile=above_row_percentile,
        row_margin_px=above_row_margin_px,
    )
    return {
        "static_exclusion_mask": above_mask,
        "above_lidar_range_mask": above_mask,
        "visible_mask": empty.copy(),
    }, {
        "enabled": True,
        "status": above_summary["status"],
        "source": str(static_mask_path),
        "fallback_source": lidar_projection["source"],
        "top_row": above_summary["top_row"],
        "row_cutoff": above_summary["row_cutoff"],
        "visible_points": above_summary["visible_points"],
        "nonground_points": None,
    }


def build_lidar_residual_maps(
    lidar_projection,
    target_shape,
    image_shape,
    splat_radius,
    confidence_norm,
    static_residual_m,
    dynamic_residual_m,
    residual_high_q,
    residual_mad_scale,
    min_visible_points,
):
    target_h, target_w = target_shape
    image_h, image_w = image_shape[:2]
    uv = lidar_projection["uv"]
    residuals = lidar_projection["residuals"]
    depth = lidar_projection["depth"]
    visible = (
        lidar_projection["visible"]
        & np.isfinite(residuals)
        & np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
    )
    if np.count_nonzero(visible) < int(min_visible_points):
        return None, None, {
            "enabled": True,
            "status": "skipped_too_few_visible_points",
            "source": lidar_projection["source"],
            "visible_points": int(np.count_nonzero(visible)),
            "min_visible_points": int(min_visible_points),
        }

    uv = uv[visible]
    residuals = residuals[visible]
    finite = residuals[np.isfinite(residuals)]
    residual_median = float(np.median(finite))
    residual_mad = float(np.median(np.abs(finite - residual_median)) + 1e-6)
    residual_low = max(float(static_residual_m), residual_median + float(residual_mad_scale) * residual_mad)
    residual_high = max(float(dynamic_residual_m), float(np.percentile(finite, residual_high_q)))
    if residual_high <= residual_low:
        residual_high = residual_low + 1e-6
    point_scores = np.clip((residuals - residual_low) / (residual_high - residual_low), 0.0, 1.0)

    score_sum = np.zeros((target_h, target_w), dtype=np.float32)
    weight_sum = np.zeros((target_h, target_w), dtype=np.float32)
    radius = max(1, int(round(float(splat_radius))))
    sigma = max(float(radius) * 0.5, 1.0)
    x_scale = float(target_w) / max(float(image_w), 1.0)
    y_scale = float(target_h) / max(float(image_h), 1.0)

    for point_uv, point_score in zip(uv, point_scores):
        x = int(round(float(point_uv[0]) * x_scale))
        y = int(round(float(point_uv[1]) * y_scale))
        if x < 0 or x >= target_w or y < 0 or y >= target_h:
            continue
        x0 = max(0, x - radius)
        x1 = min(target_w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(target_h, y + radius + 1)
        for yy in range(y0, y1):
            dy2 = float((yy - y) ** 2)
            for xx in range(x0, x1):
                dist2 = dy2 + float((xx - x) ** 2)
                if dist2 > radius * radius:
                    continue
                weight = float(np.exp(-dist2 / (2.0 * sigma * sigma)))
                score_sum[yy, xx] += float(point_score) * weight
                weight_sum[yy, xx] += weight

    covered = weight_sum > 1e-6
    lidar_score = np.zeros((target_h, target_w), dtype=np.float32)
    lidar_score[covered] = score_sum[covered] / weight_sum[covered]
    lidar_confidence = np.clip(weight_sum / max(float(confidence_norm), 1e-6), 0.0, 1.0).astype(np.float32)
    return lidar_score, lidar_confidence, {
        "enabled": True,
        "status": "ok",
        "source": lidar_projection["source"],
        "pair_dir": lidar_projection["pair_dir"],
        "visible_points": int(finite.size),
        "covered_cells": int(np.count_nonzero(covered)),
        "covered_fraction": float(np.count_nonzero(covered) / max(lidar_score.size, 1)),
        "residual_median_m": residual_median,
        "residual_mad_m": residual_mad,
        "residual_low_m": float(residual_low),
        "residual_high_m": float(residual_high),
        "residual_p90_m": float(np.percentile(finite, 90.0)),
        "residual_p95_m": float(np.percentile(finite, 95.0)),
        "splat_radius": radius,
    }


def build_lidar_component_suppression_mask(
    mask_pre_lidar,
    lidar_score,
    lidar_confidence,
    confidence_thresh,
    component_min_covered_cells,
    component_min_covered_fraction,
    suppress_uncovered_components,
    uncovered_min_area,
    dark_mean_thresh,
    dark_p95_thresh,
    dark_high_score_thresh,
    dark_max_high_fraction,
):
    covered = lidar_confidence >= float(confidence_thresh)
    suppression_mask = np.zeros_like(mask_pre_lidar, dtype=bool)
    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        mask_pre_lidar.astype(np.uint8),
        connectivity=8,
    )
    suppressed_components = 0
    dark_suppressed_components = 0
    uncovered_suppressed_components = 0
    inspected_components = 0
    low_coverage_components = 0
    low_coverage_with_high_residual_components = 0
    for label in range(1, num_labels):
        component = labels == label
        component_covered = component & covered
        covered_cells = int(np.count_nonzero(component_covered))
        component_area = int(np.count_nonzero(component))
        covered_fraction = covered_cells / max(component_area, 1)
        if (
            covered_cells < int(component_min_covered_cells)
            or covered_fraction < float(component_min_covered_fraction)
        ):
            low_coverage_components += 1
            has_high_lidar = False
            if covered_cells > 0:
                values = lidar_score[component_covered]
                has_high_lidar = bool(np.any(values >= float(dark_high_score_thresh)))
            if has_high_lidar:
                low_coverage_with_high_residual_components += 1
            elif (
                suppress_uncovered_components
                and component_area >= int(uncovered_min_area)
            ):
                suppression_mask[component] = True
                suppressed_components += 1
                uncovered_suppressed_components += 1
            continue
        inspected_components += 1
        values = lidar_score[component_covered]
        mean_lidar = float(values.mean())
        p95_lidar = float(np.percentile(values, 95.0))
        high_fraction = float((values >= float(dark_high_score_thresh)).mean())
        if (
            mean_lidar <= float(dark_mean_thresh)
            and p95_lidar <= float(dark_p95_thresh)
            and high_fraction <= float(dark_max_high_fraction)
        ):
            suppression_mask[component] = True
            suppressed_components += 1
            dark_suppressed_components += 1

    return suppression_mask, {
        "inspected_components": int(inspected_components),
        "low_coverage_components": int(low_coverage_components),
        "low_coverage_with_high_residual_components": int(low_coverage_with_high_residual_components),
        "suppressed_components": int(suppressed_components),
        "dark_suppressed_components": int(dark_suppressed_components),
        "uncovered_suppressed_components": int(uncovered_suppressed_components),
        "suppressed_pixels": int(np.count_nonzero(suppression_mask)),
    }


def apply_lidar_residual_to_score(
    score,
    lidar_score,
    lidar_confidence,
    down_weight,
    up_weight,
    suppression_mask=None,
):
    score = score.astype(np.float32)
    lidar_score = np.clip(lidar_score.astype(np.float32), 0.0, 1.0)
    lidar_confidence = np.clip(lidar_confidence.astype(np.float32), 0.0, 1.0)
    if suppression_mask is None:
        suppression = np.zeros_like(score, dtype=np.float32)
    else:
        suppression = suppression_mask.astype(np.float32)
    fused = (
        score * (1.0 - float(down_weight) * suppression * (1.0 - lidar_score))
        + (1.0 - score) * float(up_weight) * lidar_confidence * lidar_score
    )
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def refine_mask_with_lidar_residual(
    mask,
    mask_pre_lidar,
    score_pre_lidar,
    lidar_score,
    lidar_confidence,
    suppression_mask,
    confidence_thresh,
    promote_score_thresh,
    promote_visual_thresh,
):
    refined = mask.astype(bool).copy()
    covered = lidar_confidence >= float(confidence_thresh)
    if suppression_mask is None:
        suppression_mask = np.zeros_like(mask_pre_lidar, dtype=bool)
    else:
        suppression_mask = suppression_mask.astype(bool)
    promote = (
        covered
        & (lidar_score >= float(promote_score_thresh))
        & (score_pre_lidar >= float(promote_visual_thresh))
    )

    num_labels, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        mask_pre_lidar.astype(np.uint8),
        connectivity=8,
    )
    removed_components = 0
    for label in range(1, num_labels):
        component = labels == label
        if np.any(suppression_mask & component):
            refined[component] = False
            removed_components += 1

    refined[promote] = True
    refined = clean_binary_mask(refined)
    return refined, {
        "low_residual_policy": "component_only",
        "veto_pixels": 0,
        "promote_pixels": int(np.count_nonzero(promote)),
        "removed_components": int(removed_components),
        "suppressed_component_pixels": int(np.count_nonzero(suppression_mask)),
    }


def process_pair_dir(
    pair_dir,
    mask_percentile,
    empty_score_mean_thresh,
    empty_score_p90_thresh,
    empty_score_p99_thresh,
    empty_high_score_thresh,
    empty_high_score_area_thresh,
    appearance_boost_alpha,
    feature_probe_subdir,
    lidar_residual_enabled=True,
    lidar_motion_subdir="lidar_motion_probe",
    lidar_projection_filename="image_residual_nonground_features.npz",
    lidar_splat_radius=2,
    lidar_confidence_norm=1.5,
    lidar_static_residual_m=0.15,
    lidar_dynamic_residual_m=0.70,
    lidar_residual_high_q=95.0,
    lidar_residual_mad_scale=2.0,
    lidar_min_visible_points=20,
    lidar_static_mask_enabled=True,
    lidar_static_mask_filename="image_lidar_static_masks.npz",
    lidar_static_filter_above_range=True,
    lidar_static_above_row_percentile=0.1,
    lidar_static_above_row_margin_px=8.0,
    lidar_down_weight=0.65,
    lidar_up_weight=0.45,
    lidar_confidence_thresh=0.25,
    lidar_promote_score_thresh=0.65,
    lidar_promote_visual_thresh=0.20,
    lidar_component_min_covered_cells=3,
    lidar_component_min_covered_fraction=0.03,
    lidar_suppress_uncovered_components=True,
    lidar_uncovered_component_min_area=32,
    lidar_component_dark_mean_thresh=0.08,
    lidar_component_dark_p95_thresh=0.18,
    lidar_component_dark_high_score_thresh=0.35,
    lidar_component_dark_max_high_fraction=0.02,
    lidar_allow_empty_override=False,
    lidar_empty_override_min_cells=3,
    organize_outputs=True,
):
    arrays_path = find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz")
    summary_path = find_pair_file(pair_dir, "rigidmask_frontend_summary.json")

    if not arrays_path.exists() or not summary_path.exists():
        return None

    #加载：前端代价图（npz）、运动摘要（json）
    arrays = dict(np.load(arrays_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if organize_outputs:
        organize_pair_outputs(pair_dir)

    #计算可靠区域掩码
    reliable = reliable_mask_from_uncertainty(arrays)
    #计算运动因子
    motion = compute_motion_factors(summary, arrays)
    #计算各个代价图的权重，根据运动因子调整 6 个代价的权重
    weights = compute_weights(motion)
    #计算不确定性门控因子
    gate, gate_summary = gate_from_uncertainty(arrays)

    #归一化 各个代价图到 [0,1]，并计算方向异常
    homography_score, h_lo, h_hi = robust_normalize(arrays["homography_cost"])
    epipolar_score, e_lo, e_hi = robust_normalize(arrays["epipolar_cost"])
    depth_score, d_lo, d_hi = robust_normalize(np.abs(arrays["depth_contrast_cost"]))
    orth_score, o_lo, o_hi = robust_normalize(np.abs(arrays["pp3d_orth_cost"]))
    pp2d_score, pp2d_center, p2_lo, p2_hi = directional_anomaly(arrays["pp2d_cost"], reliable)
    pp3d_dir_score, pp3d_center, p3_lo, p3_hi = directional_anomaly(arrays["pp3d_dir_cost"], reliable)

    #先融合几何代价，再用外观异常做几何放大。
    geom_score = gate * (
        weights["homography_cost"] * homography_score
        + weights["epipolar_cost"] * epipolar_score
        + weights["depth_contrast_cost"] * depth_score
        + weights["pp3d_orth_cost"] * orth_score
        + weights["pp2d_cost"] * pp2d_score
        + weights["pp3d_dir_cost"] * pp3d_dir_score
    )
    geom_score = np.clip(geom_score, 0.0, 1.0).astype(np.float32)
    
    appearance_score = None
    similarity_score = None
    appearance_source = None
    app_lo = None
    app_hi = None
    #只有当外观增强系数大于 0 时，才尝试读取相似度图
    if appearance_boost_alpha > 0.0:
        similarity_map, appearance_source = load_similarity_map_for_pair(
            pair_dir,
            summary,
            feature_probe_subdir,
        )
        if similarity_map is not None:
            appearance_score, similarity_score, app_lo, app_hi = appearance_score_from_similarity(
                similarity_map,
                geom_score.shape,
            )

    if appearance_score is not None:
        score = geom_score * (1.0 + float(appearance_boost_alpha) * appearance_score)
    else:
        score = geom_score

    #裁剪动态分数图到 [0,1]，并转换为 float32 类型
    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    score_pre_lidar = score.copy()
    lidar_projection = None
    lidar_score = None
    lidar_confidence = None
    lidar_summary = {
        "enabled": bool(lidar_residual_enabled),
        "status": "disabled",
    }
    lidar_static_filter_summary = {
        "enabled": bool(lidar_static_mask_enabled),
        "status": "disabled",
    }
    lidar_static_exclusion_mask = None
    lidar_above_range_mask = None
    lidar_visible_mask = None
    suppression_mask = None
    suppression_summary = None
    preview_shape = load_preview_shape(pair_dir, arrays)
    if lidar_residual_enabled or lidar_static_mask_enabled:
        lidar_projection, lidar_load_summary = load_lidar_projection_for_pair(
            pair_dir,
            summary,
            lidar_motion_subdir,
            lidar_projection_filename,
        )
        if lidar_projection is None:
            lidar_summary = lidar_load_summary
            if lidar_static_mask_enabled:
                lidar_static_filter_summary = {
                    **lidar_load_summary,
                    "enabled": True,
                }
        elif lidar_residual_enabled:
            lidar_score, lidar_confidence, lidar_summary = build_lidar_residual_maps(
                lidar_projection,
                target_shape=score.shape,
                image_shape=preview_shape,
                splat_radius=lidar_splat_radius,
                confidence_norm=lidar_confidence_norm,
                static_residual_m=lidar_static_residual_m,
                dynamic_residual_m=lidar_dynamic_residual_m,
                residual_high_q=lidar_residual_high_q,
                residual_mad_scale=lidar_residual_mad_scale,
                min_visible_points=lidar_min_visible_points,
            )

    if lidar_static_mask_enabled and lidar_projection is not None:
        lidar_static_masks, lidar_static_filter_summary = load_lidar_static_masks_for_pair(
            lidar_projection,
            image_shape=preview_shape,
            static_mask_filename=lidar_static_mask_filename,
            above_row_percentile=lidar_static_above_row_percentile,
            above_row_margin_px=lidar_static_above_row_margin_px,
        )
        lidar_above_range_mask = resize_bool_mask(lidar_static_masks["above_lidar_range_mask"], score.shape)
        lidar_visible_mask = resize_bool_mask(lidar_static_masks["visible_mask"], score.shape)
        static_parts = []
        if lidar_static_filter_above_range:
            static_parts.append(lidar_above_range_mask)
        if static_parts:
            lidar_static_exclusion_mask = np.logical_or.reduce(static_parts)
        else:
            lidar_static_exclusion_mask = np.zeros_like(score, dtype=bool)
        lidar_static_filter_summary.update(
            {
                "filter_above_range": bool(lidar_static_filter_above_range),
                "masked_cells": int(np.count_nonzero(lidar_static_exclusion_mask)),
                "masked_fraction": float(np.count_nonzero(lidar_static_exclusion_mask) / max(score.size, 1)),
                "above_range_cells": int(np.count_nonzero(lidar_above_range_mask)),
            }
        )

    #判断是否强制输出空掩码：只看几何分数，避免外观异常改变整帧静态判定。
    force_empty_mask, global_stats = should_force_empty_mask(
        geom_score,
        mean_thresh=empty_score_mean_thresh,
        p90_thresh=empty_score_p90_thresh,
        p99_thresh=empty_score_p99_thresh,
        high_score_thresh=empty_high_score_thresh,
        high_score_area_thresh=empty_high_score_area_thresh,
    )
    force_empty_pre_lidar = force_empty_mask
    empty_override_by_lidar = False
    high_lidar_cells = 0
    if (
        lidar_allow_empty_override
        and force_empty_mask
        and lidar_score is not None
        and lidar_confidence is not None
    ):
        high_lidar_cells = int(
            np.count_nonzero(
                (lidar_confidence >= float(lidar_confidence_thresh))
                & (lidar_score >= float(lidar_promote_score_thresh))
            )
        )
        if high_lidar_cells >= int(lidar_empty_override_min_cells):
            force_empty_mask = False
            empty_override_by_lidar = True

    if force_empty_pre_lidar:
        threshold_pre_lidar = None
        mask_pre_lidar = np.zeros_like(score_pre_lidar, dtype=np.float32)
    else:
        finite_pre_lidar = np.isfinite(score_pre_lidar)
        threshold_pre_lidar = (
            float(np.percentile(score_pre_lidar[finite_pre_lidar], mask_percentile))
            if np.any(finite_pre_lidar)
            else 0.5
        )
        threshold_pre_lidar = max(0.2, min(0.9, threshold_pre_lidar))
        mask_pre_lidar = postprocess_mask(score_pre_lidar, threshold_pre_lidar)

    if lidar_score is not None and lidar_confidence is not None:
        suppression_mask, suppression_summary = build_lidar_component_suppression_mask(
            mask_pre_lidar,
            lidar_score,
            lidar_confidence,
            confidence_thresh=lidar_confidence_thresh,
            component_min_covered_cells=lidar_component_min_covered_cells,
            component_min_covered_fraction=lidar_component_min_covered_fraction,
            suppress_uncovered_components=lidar_suppress_uncovered_components,
            uncovered_min_area=lidar_uncovered_component_min_area,
            dark_mean_thresh=lidar_component_dark_mean_thresh,
            dark_p95_thresh=lidar_component_dark_p95_thresh,
            dark_high_score_thresh=lidar_component_dark_high_score_thresh,
            dark_max_high_fraction=lidar_component_dark_max_high_fraction,
        )
        score = apply_lidar_residual_to_score(
            score_pre_lidar,
            lidar_score,
            lidar_confidence,
            down_weight=lidar_down_weight,
            up_weight=lidar_up_weight,
            suppression_mask=suppression_mask,
        )
        lidar_summary.update(
            {
                "down_weight": float(lidar_down_weight),
                "up_weight": float(lidar_up_weight),
                "confidence_thresh": float(lidar_confidence_thresh),
                "low_residual_policy": "suppress_pre_lidar_component_when_lidar_dark_or_uncovered",
                "component_suppression": suppression_summary,
                "component_dark_thresholds": {
                    "min_covered_cells": int(lidar_component_min_covered_cells),
                    "min_covered_fraction": float(lidar_component_min_covered_fraction),
                    "suppress_uncovered_components": bool(lidar_suppress_uncovered_components),
                    "uncovered_component_min_area": int(lidar_uncovered_component_min_area),
                    "mean": float(lidar_component_dark_mean_thresh),
                    "p95": float(lidar_component_dark_p95_thresh),
                    "high_score": float(lidar_component_dark_high_score_thresh),
                    "max_high_fraction": float(lidar_component_dark_max_high_fraction),
                },
            }
        )

    if lidar_static_exclusion_mask is not None:
        score[lidar_static_exclusion_mask] = 0.0

    if force_empty_mask:
        threshold = None
        mask = np.zeros_like(score, dtype=np.float32)
        lidar_mask_summary = None
    else:
        finite = np.isfinite(score)
        #比mask_percentile%的动态分数高的像素被认为是动态区域，作为初始掩码阈值（可能为0.1、0.5、0.9...）
        threshold = float(np.percentile(score[finite], mask_percentile)) if np.any(finite) else 0.5
        #限制阈值在 [0.2, 0.8] 范围内，避免过低或过高的阈值导致掩码过于稀疏或过于密集
        threshold = max(0.2, min(0.9, threshold))
        #根据阈值生成二值掩码，并进行形态学处理和连通域过滤
        mask = postprocess_mask(score, threshold)
        lidar_mask_summary = None
        if lidar_score is not None and lidar_confidence is not None:
            mask, lidar_mask_summary = refine_mask_with_lidar_residual(
                mask,
                mask_pre_lidar,
                score_pre_lidar,
                lidar_score,
                lidar_confidence,
                suppression_mask,
                confidence_thresh=lidar_confidence_thresh,
                promote_score_thresh=lidar_promote_score_thresh,
                promote_visual_thresh=lidar_promote_visual_thresh,
            )

    if lidar_static_exclusion_mask is not None:
        mask = mask * (~lidar_static_exclusion_mask).astype(np.float32)

    final_mask_filter, final_mask_filter_source = final_mask_filter_from_saved_masks(arrays, summary)
    final_mask_filter_ratio = None
    if final_mask_filter is not None:
        mask = mask * final_mask_filter.astype(np.float32)
        final_mask_filter_ratio = float(final_mask_filter.mean())

    score_full = resize_to_image(score, preview_shape)
    score_pre_lidar_full = resize_to_image(score_pre_lidar, preview_shape)
    geom_score_full = resize_to_image(geom_score, preview_shape)
    gate_full = resize_to_image(gate, preview_shape)
    mask_full = resize_to_image(mask, preview_shape)
    mask_pre_lidar_full = resize_to_image(mask_pre_lidar, preview_shape)

    save_gray_image(pair_output_path(pair_dir, "dynamic_score.png", organize_outputs), score_full)
    save_gray_image(pair_output_path(pair_dir, "dynamic_score_pre_lidar.png", organize_outputs), score_pre_lidar_full)
    save_gray_image(pair_output_path(pair_dir, "dynamic_score_geom.png", organize_outputs), geom_score_full)
    save_gray_image(pair_output_path(pair_dir, "dynamic_gate.png", organize_outputs), gate_full)
    save_gray_image(pair_output_path(pair_dir, "dynamic_mask.png", organize_outputs), mask_full)
    save_gray_image(pair_output_path(pair_dir, "dynamic_mask_pre_lidar.png", organize_outputs), mask_pre_lidar_full)
    if lidar_static_exclusion_mask is not None:
        save_gray_image(
            pair_output_path(pair_dir, "lidar_static_exclusion_mask.png", organize_outputs),
            resize_to_image(lidar_static_exclusion_mask.astype(np.float32), preview_shape),
        )
        if lidar_above_range_mask is not None:
            save_gray_image(
                pair_output_path(pair_dir, "lidar_above_range_mask.png", organize_outputs),
                resize_to_image(lidar_above_range_mask.astype(np.float32), preview_shape),
            )
        if lidar_visible_mask is not None:
            save_gray_image(
                pair_output_path(pair_dir, "lidar_visible_mask.png", organize_outputs),
                resize_to_image(lidar_visible_mask.astype(np.float32), preview_shape),
            )
    if appearance_score is not None and similarity_score is not None:
        save_gray_image(
            pair_output_path(pair_dir, "appearance_score.png", organize_outputs),
            resize_to_image(appearance_score, preview_shape),
        )
        save_gray_image(
            pair_output_path(pair_dir, "similarity_score.png", organize_outputs),
            resize_to_image(similarity_score, preview_shape),
        )
    if lidar_score is not None and lidar_confidence is not None:
        save_gray_image(
            pair_output_path(pair_dir, "lidar_residual_score.png", organize_outputs),
            resize_to_image(lidar_score, preview_shape),
        )
        save_gray_image(
            pair_output_path(pair_dir, "lidar_residual_confidence.png", organize_outputs),
            resize_to_image(lidar_confidence, preview_shape),
        )
        if suppression_mask is not None:
            save_gray_image(
                pair_output_path(pair_dir, "lidar_suppression_mask.png", organize_outputs),
                resize_to_image(suppression_mask.astype(np.float32), preview_shape),
            )

    fusion_summary = {
        "pair_name": summary.get("pair_name", pair_dir.name),
        "motion": motion,
        "weights": weights,
        "appearance_boost": {
            "enabled": appearance_score is not None,
            "alpha": float(appearance_boost_alpha),
            "source": appearance_source,
        },
        "lidar_residual_fusion": {
            **lidar_summary,
            "mask_refinement": lidar_mask_summary,
            "empty_override_enabled": bool(lidar_allow_empty_override),
            "empty_override_by_lidar": bool(empty_override_by_lidar),
            "empty_override_high_cells": int(high_lidar_cells),
        },
        "lidar_static_filter": lidar_static_filter_summary,
        "force_empty_mask": force_empty_mask,
        "force_empty_mask_pre_lidar": force_empty_pre_lidar,
        "mask_threshold": threshold,
        "mask_threshold_pre_lidar": threshold_pre_lidar,
        "reliable_fraction": float(reliable.mean()),
        "empty_mask_score_source": "geom_score",
        "score_mean": global_stats["score_mean"],
        "score_p90": global_stats["score_p90"],
        "score_p99": global_stats["score_p99"],
        "high_score_area_fraction": global_stats["high_score_area_fraction"],
        "empty_mask_gate": {
            "score_mean_thresh": empty_score_mean_thresh,
            "score_p90_thresh": empty_score_p90_thresh,
            "score_p99_thresh": empty_score_p99_thresh,
            "high_score_thresh": empty_high_score_thresh,
            "high_score_area_thresh": empty_high_score_area_thresh,
        },
        "final_mask_filter": {
            "enabled": final_mask_filter is not None,
            "source": final_mask_filter_source,
            "keep_ratio": final_mask_filter_ratio,
        },
        "component_stats": {
            "homography_cost": {"low": h_lo, "high": h_hi},
            "epipolar_cost": {"low": e_lo, "high": e_hi},
            "depth_contrast_cost": {"low": d_lo, "high": d_hi},
            "pp3d_orth_cost": {"low": o_lo, "high": o_hi},
            "pp2d_cost": {"center": pp2d_center, "low": p2_lo, "high": p2_hi},
            "pp3d_dir_cost": {"center": pp3d_center, "low": p3_lo, "high": p3_hi},
            "appearance_cost": None if appearance_score is None else {"low": app_lo, "high": app_hi},
            "uncertainty_gate": gate_summary,
        },
    }
    with open(pair_output_path(pair_dir, "dynamic_fusion_summary.json", organize_outputs), "w", encoding="utf-8") as handle:
        json.dump(fusion_summary, handle, indent=2)
    return fusion_summary


def collect_pair_dirs(root):
    pair_dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and find_pair_file(path, "rigidmask_frontend_arrays.npz").exists():
            pair_dirs.append(path)
    return pair_dirs


def main():
    parser = argparse.ArgumentParser(description="Fuse rigidmask frontend costs into heuristic dynamic scores")
    parser.add_argument("probe_root", help="Path to a rigidmask_frontend_probe directory")
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=85.0,
        help="Percentile over dynamic score used as the initial mask threshold",
    )
    parser.add_argument(
        "--empty-score-mean-thresh",
        type=float,
        default=0.06,
        help="If global dynamic evidence is weaker than this and the other empty-mask conditions also hold, output an empty mask",
    )
    parser.add_argument(
        "--empty-score-p90-thresh",
        type=float,
        default=0.15,
        help="P90 threshold used by the empty-mask gate",
    )
    parser.add_argument(
        "--empty-score-p99-thresh",
        type=float,
        default=0.25,
        help="P99 threshold used by the empty-mask gate",
    )
    parser.add_argument(
        "--empty-high-score-thresh",
        type=float,
        default=0.25,
        help="Score level used to measure whether enough pixels are strongly dynamic-like",
    )
    parser.add_argument(
        "--empty-high-score-area-thresh",
        type=float,
        default=0.01,
        help="Minimum fraction of pixels above --empty-high-score-thresh required to avoid forcing an empty mask",
    )
    parser.add_argument(
        "--appearance-boost-alpha",
        type=float,
        default=0.5,
        help="Geometry amplification factor used as final_score = geom_score * (1 + alpha * appearance_score)",
    )
    parser.add_argument(
        "--feature-probe-subdir",
        default="stage1_feature_probe",
        help="Sibling output subdirectory under the run root that stores similarity maps",
    )
    parser.add_argument(
        "--lidar-motion-subdir",
        default="lidar_motion_probe",
        help="Sibling output subdirectory under the run root that stores LiDAR motion residuals",
    )
    parser.add_argument(
        "--lidar-projection-filename",
        default="image_residual_nonground_features.npz",
        help="LiDAR residual projection npz filename inside each lidar_motion_probe pair directory",
    )
    parser.add_argument(
        "--no-lidar-residual",
        action="store_true",
        help="Disable LiDAR residual fusion and use only geometry plus appearance scores",
    )
    parser.add_argument(
        "--lidar-splat-radius",
        type=float,
        default=2.0,
        help="Radius in the low-resolution cost grid used to splat each projected LiDAR residual",
    )
    parser.add_argument(
        "--lidar-confidence-norm",
        type=float,
        default=1.5,
        help="Accumulated splat weight that maps to full LiDAR confidence",
    )
    parser.add_argument(
        "--lidar-static-residual-m",
        type=float,
        default=0.15,
        help="Residual below this value is strong static evidence before robust adaptation",
    )
    parser.add_argument(
        "--lidar-dynamic-residual-m",
        type=float,
        default=0.70,
        help="Residual above this value is strong dynamic evidence before robust adaptation",
    )
    parser.add_argument(
        "--lidar-residual-high-q",
        type=float,
        default=95.0,
        help="Residual percentile used as the adaptive high residual reference",
    )
    parser.add_argument(
        "--lidar-residual-mad-scale",
        type=float,
        default=2.0,
        help="MAD multiplier used for the adaptive low residual reference",
    )
    parser.add_argument(
        "--lidar-min-visible-points",
        type=int,
        default=20,
        help="Minimum projected visible LiDAR points required before residual fusion is used",
    )
    parser.add_argument(
        "--no-lidar-static-mask",
        action="store_true",
        help="Disable hard static filtering above the LiDAR sampling row range",
    )
    parser.add_argument(
        "--lidar-static-mask-filename",
        default="image_lidar_static_masks.npz",
        help="LiDAR static projection mask npz filename inside each lidar_motion_probe pair directory",
    )
    parser.add_argument(
        "--no-lidar-static-above-range",
        action="store_true",
        help="Do not force pixels above the LiDAR sampling row range to static",
    )
    parser.add_argument(
        "--lidar-static-above-row-percentile",
        type=float,
        default=0.1,
        help="Visible LiDAR row percentile used to estimate the top of the LiDAR sampling range",
    )
    parser.add_argument(
        "--lidar-static-above-row-margin-px",
        type=float,
        default=8.0,
        help="Pixel margin subtracted from the LiDAR top-row estimate before masking the region above it",
    )
    parser.add_argument(
        "--lidar-down-weight",
        type=float,
        default=0.65,
        help="How strongly a LiDAR-dark pre-LiDAR component suppresses the visual dynamic score",
    )
    parser.add_argument(
        "--lidar-up-weight",
        type=float,
        default=0.45,
        help="How strongly high LiDAR residual boosts the visual dynamic score",
    )
    parser.add_argument(
        "--lidar-confidence-thresh",
        type=float,
        default=0.25,
        help="Minimum LiDAR confidence for component checks and mask promotion",
    )
    parser.add_argument(
        "--lidar-promote-score-thresh",
        type=float,
        default=0.65,
        help="LiDAR residual score required to promote a visually plausible dynamic pixel",
    )
    parser.add_argument(
        "--lidar-promote-visual-thresh",
        type=float,
        default=0.20,
        help="Pre-LiDAR visual score required before high LiDAR residual can promote a mask pixel",
    )
    parser.add_argument(
        "--lidar-component-min-covered-cells",
        type=int,
        default=3,
        help="Minimum confident LiDAR-covered cells required before a pre-LiDAR component can be suppressed",
    )
    parser.add_argument(
        "--lidar-component-min-covered-fraction",
        type=float,
        default=0.03,
        help="Minimum confident LiDAR coverage fraction required before a pre-LiDAR component can be suppressed",
    )
    parser.add_argument(
        "--keep-uncovered-components",
        action="store_true",
        help="Do not suppress low-coverage pre-LiDAR components that have no high LiDAR residual support",
    )
    parser.add_argument(
        "--lidar-uncovered-component-min-area",
        type=int,
        default=64,
        help="Minimum pre-LiDAR component area before a low-coverage component can be suppressed",
    )
    parser.add_argument(
        "--lidar-component-dark-mean-thresh",
        type=float,
        default=0.08,
        help="Maximum mean LiDAR residual score for an inspected component to count as almost black",
    )
    parser.add_argument(
        "--lidar-component-dark-p95-thresh",
        type=float,
        default=0.18,
        help="Maximum P95 LiDAR residual score for an inspected component to count as almost black",
    )
    parser.add_argument(
        "--lidar-component-dark-high-score-thresh",
        type=float,
        default=0.35,
        help="LiDAR residual score treated as high when checking whether a component is almost black",
    )
    parser.add_argument(
        "--lidar-component-dark-max-high-fraction",
        type=float,
        default=0.02,
        help="Maximum high-residual fraction allowed for an inspected component to count as almost black",
    )
    parser.add_argument(
        "--lidar-allow-empty-override",
        action="store_true",
        help="Allow strong LiDAR residual evidence to disable the global empty-mask gate",
    )
    parser.add_argument(
        "--flat-outputs",
        action="store_true",
        help="Write outputs directly in each pair directory instead of categorizing files into subdirectories",
    )
    args = parser.parse_args()

    probe_root = Path(args.probe_root)
    if not probe_root.is_dir():
        raise FileNotFoundError(f"Probe root not found: {probe_root}")

    pair_dirs = collect_pair_dirs(probe_root)
    if not pair_dirs:
        raise FileNotFoundError(f"No pair directories with rigidmask_frontend_arrays.npz found under {probe_root}")

    run_summary = []
    for pair_dir in pair_dirs:
        fusion_summary = process_pair_dir(
            pair_dir,
            args.mask_percentile,
            args.empty_score_mean_thresh,
            args.empty_score_p90_thresh,
            args.empty_score_p99_thresh,
            args.empty_high_score_thresh,
            args.empty_high_score_area_thresh,
            args.appearance_boost_alpha,
            args.feature_probe_subdir,
            lidar_residual_enabled=not args.no_lidar_residual,
            lidar_motion_subdir=args.lidar_motion_subdir,
            lidar_projection_filename=args.lidar_projection_filename,
            lidar_splat_radius=args.lidar_splat_radius,
            lidar_confidence_norm=args.lidar_confidence_norm,
            lidar_static_residual_m=args.lidar_static_residual_m,
            lidar_dynamic_residual_m=args.lidar_dynamic_residual_m,
            lidar_residual_high_q=args.lidar_residual_high_q,
            lidar_residual_mad_scale=args.lidar_residual_mad_scale,
            lidar_min_visible_points=args.lidar_min_visible_points,
            lidar_static_mask_enabled=not args.no_lidar_static_mask,
            lidar_static_mask_filename=args.lidar_static_mask_filename,
            lidar_static_filter_above_range=not args.no_lidar_static_above_range,
            lidar_static_above_row_percentile=args.lidar_static_above_row_percentile,
            lidar_static_above_row_margin_px=args.lidar_static_above_row_margin_px,
            lidar_down_weight=args.lidar_down_weight,
            lidar_up_weight=args.lidar_up_weight,
            lidar_confidence_thresh=args.lidar_confidence_thresh,
            lidar_promote_score_thresh=args.lidar_promote_score_thresh,
            lidar_promote_visual_thresh=args.lidar_promote_visual_thresh,
            lidar_component_min_covered_cells=args.lidar_component_min_covered_cells,
            lidar_component_min_covered_fraction=args.lidar_component_min_covered_fraction,
            lidar_suppress_uncovered_components=not args.keep_uncovered_components,
            lidar_uncovered_component_min_area=args.lidar_uncovered_component_min_area,
            lidar_component_dark_mean_thresh=args.lidar_component_dark_mean_thresh,
            lidar_component_dark_p95_thresh=args.lidar_component_dark_p95_thresh,
            lidar_component_dark_high_score_thresh=args.lidar_component_dark_high_score_thresh,
            lidar_component_dark_max_high_fraction=args.lidar_component_dark_max_high_fraction,
            lidar_allow_empty_override=args.lidar_allow_empty_override,
            organize_outputs=not args.flat_outputs,
        )
        if fusion_summary is not None:
            run_summary.append(fusion_summary)

    with open(probe_root / "dynamic_fusion_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(f"Processed {len(run_summary)} pair directories under {probe_root}")


if __name__ == "__main__":
    main()

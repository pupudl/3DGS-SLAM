import argparse
import json
from pathlib import Path

import cv2
import numpy as np

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
def postprocess_mask(score, threshold):
    binary = (score >= threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    #形态学操作：先开运算去除小的噪声，再闭运算填充小的孔洞
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    #连通域过滤，去除小的连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= 32:
            cleaned[labels == label] = 1
    return cleaned.astype(np.float32)

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
    if "depth_mask_cost_grid" in arrays:
        return arrays["depth_mask_cost_grid"].astype(bool), "depth_mask_cost_grid"
    if "sky_mask_cost_grid" in arrays:
        return ~arrays["sky_mask_cost_grid"].astype(bool), "sky_mask_cost_grid"
    return None, None


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
):
    arrays_path = pair_dir / "rigidmask_frontend_arrays.npz"
    summary_path = pair_dir / "rigidmask_frontend_summary.json"
    prev_rgb_path = pair_dir / "prev_rgb.png"

    if not arrays_path.exists() or not summary_path.exists():
        return None

    #加载：前端代价图（npz）、运动摘要（json）
    arrays = dict(np.load(arrays_path))
    with open(summary_path, "r", encoding="utf-8") as handle:
        summary = json.load(handle)

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

    #判断是否强制输出空掩码：只看几何分数，避免外观异常改变整帧静态判定。
    force_empty_mask, global_stats = should_force_empty_mask(
        geom_score,
        mean_thresh=empty_score_mean_thresh,
        p90_thresh=empty_score_p90_thresh,
        p99_thresh=empty_score_p99_thresh,
        high_score_thresh=empty_high_score_thresh,
        high_score_area_thresh=empty_high_score_area_thresh,
    )

    if force_empty_mask:
        threshold = None
        mask = np.zeros_like(score, dtype=np.float32)
    else:
        finite = np.isfinite(score)
        #比mask_percentile%的动态分数高的像素被认为是动态区域，作为初始掩码阈值（可能为0.1、0.5、0.9...）
        threshold = float(np.percentile(score[finite], mask_percentile)) if np.any(finite) else 0.5
        #限制阈值在 [0.2, 0.8] 范围内，避免过低或过高的阈值导致掩码过于稀疏或过于密集
        threshold = max(0.2, min(0.9, threshold))
        #根据阈值生成二值掩码，并进行形态学处理和连通域过滤
        mask = postprocess_mask(score, threshold)

    final_mask_filter, final_mask_filter_source = final_mask_filter_from_saved_masks(arrays, summary)
    final_mask_filter_ratio = None
    if final_mask_filter is not None:
        mask = mask * final_mask_filter.astype(np.float32)
        final_mask_filter_ratio = float(final_mask_filter.mean())

    #确定输出尺寸
    preview_shape = arrays["flow_full_x"].shape
    if prev_rgb_path.exists():
        prev_rgb = cv2.imread(str(prev_rgb_path), cv2.IMREAD_COLOR)
        if prev_rgb is not None:
            preview_shape = prev_rgb.shape

    score_full = resize_to_image(score, preview_shape)
    geom_score_full = resize_to_image(geom_score, preview_shape)
    gate_full = resize_to_image(gate, preview_shape)
    mask_full = resize_to_image(mask, preview_shape)

    save_gray_image(pair_dir / "dynamic_score.png", score_full)
    save_gray_image(pair_dir / "dynamic_score_geom.png", geom_score_full)
    save_gray_image(pair_dir / "dynamic_gate.png", gate_full)
    save_gray_image(pair_dir / "dynamic_mask.png", mask_full)
    if appearance_score is not None and similarity_score is not None:
        save_gray_image(
            pair_dir / "appearance_score.png",
            resize_to_image(appearance_score, preview_shape),
        )
        save_gray_image(
            pair_dir / "similarity_score.png",
            resize_to_image(similarity_score, preview_shape),
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
        "force_empty_mask": force_empty_mask,
        "mask_threshold": threshold,
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
    with open(pair_dir / "dynamic_fusion_summary.json", "w", encoding="utf-8") as handle:
        json.dump(fusion_summary, handle, indent=2)
    return fusion_summary


def collect_pair_dirs(root):
    pair_dirs = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "rigidmask_frontend_arrays.npz").exists():
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
        )
        if fusion_summary is not None:
            run_summary.append(fusion_summary)

    with open(probe_root / "dynamic_fusion_run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)

    print(f"Processed {len(run_summary)} pair directories under {probe_root}")


if __name__ == "__main__":
    main()

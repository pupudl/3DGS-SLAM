import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.dynamic_mask.fusion import find_pair_file
from utils.slam_external import build_rotation
from utils.slam_helpers import get_depth_and_silhouette, matrix_to_quaternion, quat_mult


DEFAULT_DYNAMIC_4DGS_CFG = {
    "enabled": False,
    "mask_root": "",
    "mask_subdir": "dynamic_mask",
    "rigidmask_subdir": "rigidmask_frontend_probe",
    "mask_filename": "dynamic_mask.png",
    "score_filename": "dynamic_score.png",
    "require_appearance_mask": False,
    "mask_threshold": 0.5,
    "score_threshold": 0.2,
    "min_component_area": 64,
    "min_component_points": 24,
    "max_components_per_frame": 16,
    "max_new_gaussians_per_component": 1200,
    "association_dist_m": 4.0,
    "max_inactive_frames": 5,
    "num_iters": 40,
    "min_loss_pixels": 16,
    "window": {
        "min_score": 0.05,
        "use_mask_confidence": True,
        "weight_power": 1.0,
        "weight_epsilon": 1e-8,
    },
    "rigidmask_pose_init": {
        "enabled": True,
        "pose_filename": "dynamic_component_poses.json",
        "use_rotation": True,
        "use_translation": True,
        "fallback_to_centroid_translation": True,
        "copy_previous_rotation_on_missing": True,
        "min_inlier_ratio": 0.20,
        "max_reproj_median_px": 8.0,
        "max_translation_residual_m": 5.0,
        "max_component_center_dist_px": 80.0,
        "min_component_bbox_iou": 0.05,
    },
    "lrs": {
        "dyn_means3D_canon": 0.0001,
        "dyn_rgb_colors": 0.0025,
        "dyn_unnorm_rotations": 0.0005,
        "dyn_logit_opacities": 0.02,
        "dyn_log_scales": 0.0005,
        "dyn_obj_unnorm_rots": 0.0004,
        "dyn_obj_trans": 0.002,
    },
    "loss_weights": {
        "im": 1.0,
        "depth": 0.5,
        "sil": 0.2,
        "outside_sil": 0.1,
        "motion_smooth": 0.02,
        "scale": 0.001,
    },
}


def merge_dynamic_4dgs_config(cfg):
    return _deep_merge(DEFAULT_DYNAMIC_4DGS_CFG, cfg or {})


def _deep_merge(base, override):
    merged = {}
    for key, value in base.items():
        merged[key] = _deep_merge(value, {}) if isinstance(value, dict) else value
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resize_bool_mask(mask, target_height, target_width):
    if mask is None:
        return None
    if mask.shape[-2:] == (target_height, target_width):
        return mask
    resized = F.interpolate(
        mask.float().unsqueeze(0),
        size=(target_height, target_width),
        mode="nearest",
    )[0]
    return resized > 0.5


def load_dynamic_observation(output_dir, dynamic_cfg, time_idx, frame_id, reference_frame_id, image_hw, device):
    if not dynamic_cfg.get("enabled", False) or time_idx <= 0:
        return None

    mask_root = dynamic_cfg.get("mask_root") or os.path.join(
        output_dir,
        dynamic_cfg.get("mask_subdir", "dynamic_mask"),
    )
    pair_root = Path(mask_root) / dynamic_cfg.get("rigidmask_subdir", "rigidmask_frontend_probe")
    if not pair_root.is_dir():
        return None

    pair_dirs = []
    if reference_frame_id is not None:
        pair_name = f"{int(time_idx):06d}_frame_{frame_id}_from_{reference_frame_id}"
        pair_dirs.append(pair_root / pair_name)
    pair_dirs.extend(sorted(path for path in pair_root.glob(f"{int(time_idx):06d}_frame_{frame_id}_from_*") if path.is_dir()))

    pair_dir = next((path for path in pair_dirs if path.is_dir()), None)
    if pair_dir is None:
        return None

    summary_path = find_pair_file(pair_dir, "dynamic_fusion_summary.json")
    if dynamic_cfg.get("require_appearance_mask", False):
        if not summary_path.exists():
            return None
        with open(summary_path, "r", encoding="utf-8") as handle:
            fusion_summary = json.load(handle)
        appearance_summary = fusion_summary.get("appearance_fusion", {})
        if not appearance_summary.get("enabled", False):
            return None

    mask_path = find_pair_file(pair_dir, dynamic_cfg.get("mask_filename", "dynamic_mask.png"))
    if not mask_path.exists():
        return None

    height, width = image_hw
    mask_np = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_np is None:
        return None
    if mask_np.shape != (height, width):
        mask_np = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = torch.from_numpy(mask_np.astype(np.float32) / 255.0).to(device)
    mask = (mask >= float(dynamic_cfg.get("mask_threshold", 0.5))).unsqueeze(0)

    score = None
    score_path = find_pair_file(pair_dir, dynamic_cfg.get("score_filename", "dynamic_score.png"))
    if score_path.exists():
        score_np = cv2.imread(str(score_path), cv2.IMREAD_GRAYSCALE)
        if score_np is not None:
            if score_np.shape != (height, width):
                score_np = cv2.resize(score_np, (width, height), interpolation=cv2.INTER_LINEAR)
            score = torch.from_numpy(score_np.astype(np.float32) / 255.0).to(device).unsqueeze(0)

    component_poses = None
    pose_init_cfg = dynamic_cfg.get("rigidmask_pose_init", {})
    if pose_init_cfg.get("enabled", False):
        pose_path = find_pair_file(pair_dir, pose_init_cfg.get("pose_filename", "dynamic_component_poses.json"))
        if pose_path.exists():
            try:
                with open(pose_path, "r", encoding="utf-8") as handle:
                    component_poses = json.load(handle)
                component_poses["path"] = str(pose_path)
            except (OSError, json.JSONDecodeError) as exc:
                component_poses = {
                    "enabled": True,
                    "status": "load_error",
                    "reason": str(exc),
                    "path": str(pose_path),
                    "components": [],
                }

    return {
        "dynamic_mask": mask.bool(),
        "dynamic_score": score,
        "dynamic_mask_path": str(mask_path),
        "dynamic_pair_dir": str(pair_dir),
        "dynamic_fusion_summary_path": str(summary_path) if summary_path.exists() else "",
        "dynamic_component_poses": component_poses,
    }


def apply_dynamic_mask_to_color(color, dynamic_mask):
    if color is None or dynamic_mask is None:
        return color
    masked = color.clone()
    masked[:, dynamic_mask[0]] = 0
    return masked


def apply_dynamic_mask_to_depth(depth, dynamic_mask):
    if depth is None or dynamic_mask is None:
        return depth
    masked = depth.clone()
    masked[:, dynamic_mask[0]] = 0
    return masked


def has_dynamic_gaussians(dynamic_params):
    return (
        dynamic_params is not None
        and "dyn_means3D_canon" in dynamic_params
        and dynamic_params["dyn_means3D_canon"].numel() > 0
    )


def pack_dynamic_params(output_params, dynamic_params):
    if not has_dynamic_gaussians(dynamic_params):
        return output_params
    for key, value in dynamic_params.items():
        output_params[key] = value
    return output_params


def initialize_dynamic_optimizer(dynamic_params, lrs_dict):
    trainable = [
        "dyn_means3D_canon",
        "dyn_rgb_colors",
        "dyn_unnorm_rotations",
        "dyn_logit_opacities",
        "dyn_log_scales",
        "dyn_obj_unnorm_rots",
        "dyn_obj_trans",
    ]
    param_groups = []
    for key in trainable:
        if key in dynamic_params:
            param_groups.append({"params": [dynamic_params[key]], "name": key, "lr": float(lrs_dict.get(key, 0.0))})
    if not param_groups:
        return None
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def camera_w2c_from_params(params, time_idx, camera_grad=False):
    if camera_grad:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx])
        cam_tran = params["cam_trans"][..., time_idx]
    else:
        cam_rot = F.normalize(params["cam_unnorm_rots"][..., time_idx].detach())
        cam_tran = params["cam_trans"][..., time_idx].detach()
    w2c = torch.eye(4, device=cam_tran.device).float()
    w2c[:3, :3] = build_rotation(cam_rot)
    w2c[:3, 3] = cam_tran
    return w2c


def pointcloud_from_mask(color, depth, intrinsics, w2c, mask, mean_sq_dist_method="projective"):
    device = color.device
    width, height = color.shape[2], color.shape[1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]
    fx = intrinsics[0][0]
    fy = intrinsics[1][1]

    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    xx = ((x_grid - cx) / fx).reshape(-1)
    yy = ((y_grid - cy) / fy).reshape(-1)
    depth_z = depth[0].reshape(-1)

    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat((pts_cam, torch.ones(height * width, 1, device=device).float()), dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    if mean_sq_dist_method != "projective":
        raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    mean3_sq_dist = (depth_z / ((fx + fy) / 2.0)) ** 2

    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
    flat_mask = mask.reshape(-1).bool()
    return torch.cat((pts, cols), dim=-1)[flat_mask], mean3_sq_dist[flat_mask]


def _identity_quat(device):
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).float()


def _component_bbox_from_stats(stats, label):
    left = float(stats[label, cv2.CC_STAT_LEFT])
    top = float(stats[label, cv2.CC_STAT_TOP])
    width = float(stats[label, cv2.CC_STAT_WIDTH])
    height = float(stats[label, cv2.CC_STAT_HEIGHT])
    return [left, top, left + width, top + height]


def _bbox_iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def _scaled_pose_record_geometry(record, pose_summary, image_hw):
    source_shape = record.get("image_shape", pose_summary.get("target_shape", image_hw))
    if source_shape is None or len(source_shape) < 2:
        source_shape = image_hw
    src_h, src_w = float(source_shape[0]), float(source_shape[1])
    dst_h, dst_w = float(image_hw[0]), float(image_hw[1])
    sx = dst_w / max(src_w, 1.0)
    sy = dst_h / max(src_h, 1.0)
    bbox = record.get("bbox_xyxy")
    centroid = record.get("centroid_px")
    scaled_bbox = None
    scaled_centroid = None
    if bbox is not None and len(bbox) == 4:
        scaled_bbox = [float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy]
    if centroid is not None and len(centroid) == 2:
        scaled_centroid = [float(centroid[0]) * sx, float(centroid[1]) * sy]
    return scaled_bbox, scaled_centroid


def _pose_record_is_usable(record, pose_cfg):
    if not isinstance(record, dict) or record.get("status") != "ok":
        return False
    if str(record.get("direction", "")).lower() not in ("current_to_previous", ""):
        return False
    inlier_ratio = record.get("obj_inlier_ratio")
    if inlier_ratio is not None and float(inlier_ratio) < float(pose_cfg.get("min_inlier_ratio", 0.0)):
        return False
    median_px = record.get("obj_median_px")
    if median_px is not None and float(median_px) > float(pose_cfg.get("max_reproj_median_px", 1e9)):
        return False
    return "target_to_counterpart" in record or (
        "R_target_to_counterpart" in record and "t_target_to_counterpart" in record
    )


def _match_component_pose_record(pose_summary, label, stats, cc_centroid, image_hw, used_indices, pose_cfg):
    if not pose_cfg.get("enabled", False) or not isinstance(pose_summary, dict):
        return None, None
    components = pose_summary.get("components", [])
    if not isinstance(components, list):
        return None, None

    for index, record in enumerate(components):
        if index in used_indices or not _pose_record_is_usable(record, pose_cfg):
            continue
        if int(record.get("label", -1)) == int(label):
            return record, index

    comp_bbox = _component_bbox_from_stats(stats, label)
    comp_centroid = [float(cc_centroid[0]), float(cc_centroid[1])]
    image_diag = float(np.hypot(float(image_hw[0]), float(image_hw[1])))
    max_center_dist = float(pose_cfg.get("max_component_center_dist_px", max(32.0, 0.08 * image_diag)))
    min_iou = float(pose_cfg.get("min_component_bbox_iou", 0.05))
    best = None
    for index, record in enumerate(components):
        if index in used_indices or not _pose_record_is_usable(record, pose_cfg):
            continue
        pose_bbox, pose_centroid = _scaled_pose_record_geometry(record, pose_summary, image_hw)
        if pose_bbox is None or pose_centroid is None:
            continue
        iou = _bbox_iou_xyxy(comp_bbox, pose_bbox)
        dist = float(np.linalg.norm(np.asarray(comp_centroid) - np.asarray(pose_centroid)))
        if iou < min_iou and dist > max_center_dist:
            continue
        score = (iou, -dist)
        if best is None or score > best[0]:
            best = (score, record, index)
    if best is None:
        return None, None
    return best[1], best[2]


def _pose_record_matrix(record, device):
    if "target_to_counterpart" in record:
        matrix = torch.as_tensor(record["target_to_counterpart"], device=device).float()
    else:
        matrix = torch.eye(4, device=device).float()
        matrix[:3, :3] = torch.as_tensor(record["R_target_to_counterpart"], device=device).float()
        matrix[:3, 3] = torch.as_tensor(record["t_target_to_counterpart"], device=device).float()
    if matrix.shape != (4, 4) or not bool(torch.isfinite(matrix).all()):
        return None
    return matrix


def _previous_or_identity_object_rotation(dynamic_params, obj_id, time_idx, pose_cfg, device):
    if (
        dynamic_params is not None
        and int(time_idx) > 0
        and pose_cfg.get("copy_previous_rotation_on_missing", True)
        and bool(dynamic_params["dyn_obj_visible"][int(obj_id), int(time_idx) - 1].item())
    ):
        return F.normalize(dynamic_params["dyn_obj_unnorm_rots"][int(obj_id), :, int(time_idx) - 1].detach(), dim=0)
    return _identity_quat(device)


def _rigidmask_initialized_object_pose(dynamic_params, camera_params, obj_id, time_idx, centroid, pose_record, cfg, device):
    pose_cfg = cfg.get("rigidmask_pose_init", {})
    fallback_rot = _previous_or_identity_object_rotation(dynamic_params, obj_id, time_idx, pose_cfg, device)
    fallback_trans = centroid
    if not pose_cfg.get("enabled", False) or pose_record is None or int(time_idx) <= 0:
        return fallback_rot, fallback_trans
    if not bool(dynamic_params["dyn_obj_visible"][int(obj_id), int(time_idx) - 1].item()):
        return fallback_rot, fallback_trans

    target_to_counterpart = _pose_record_matrix(pose_record, device)
    if target_to_counterpart is None:
        return fallback_rot, fallback_trans

    try:
        curr_w2c = camera_w2c_from_params(camera_params, int(time_idx), camera_grad=False)
        prev_w2c = camera_w2c_from_params(camera_params, int(time_idx) - 1, camera_grad=False)
        prev_obj_rot = F.normalize(dynamic_params["dyn_obj_unnorm_rots"][int(obj_id), :, int(time_idx) - 1].detach(), dim=0)
        prev_obj_trans = dynamic_params["dyn_obj_trans"][int(obj_id), :, int(time_idx) - 1].detach()
        prev_obj_pose = torch.eye(4, device=device).float()
        prev_obj_pose[:3, :3] = build_rotation(prev_obj_rot.view(1, 4))[0]
        prev_obj_pose[:3, 3] = prev_obj_trans

        curr_obj_pose = (
            torch.linalg.inv(curr_w2c)
            @ torch.linalg.inv(target_to_counterpart)
            @ prev_w2c
            @ prev_obj_pose
        )
        if not bool(torch.isfinite(curr_obj_pose).all()):
            return fallback_rot, fallback_trans

        init_rot = matrix_to_quaternion(curr_obj_pose[:3, :3].unsqueeze(0))[0]
        init_rot = F.normalize(init_rot, dim=0)
        init_trans = curr_obj_pose[:3, 3]
        if not bool(torch.isfinite(init_rot).all()) or not bool(torch.isfinite(init_trans).all()):
            return fallback_rot, fallback_trans

        if not pose_cfg.get("use_rotation", True):
            init_rot = fallback_rot
        if not pose_cfg.get("use_translation", True):
            init_trans = fallback_trans

        max_residual = pose_cfg.get("max_translation_residual_m", 5.0)
        if max_residual is not None and float(max_residual) >= 0.0:
            residual = torch.linalg.norm(init_trans - centroid)
            if float(residual.item()) > float(max_residual):
                if pose_cfg.get("fallback_to_centroid_translation", True):
                    init_trans = fallback_trans
                else:
                    return fallback_rot, fallback_trans
        return init_rot, init_trans
    except RuntimeError:
        return fallback_rot, fallback_trans


def _points_to_object_local(points_world, obj_rot, obj_trans):
    obj_rot_mat = build_rotation(F.normalize(obj_rot.detach(), dim=0).view(1, 4))[0]
    return (points_world - obj_trans.detach()[None, :]) @ obj_rot_mat


def update_dynamic_params_from_frame(
    dynamic_params,
    curr_data,
    camera_params,
    time_idx,
    num_frames,
    mean_sq_dist_method,
    gaussian_distribution,
    cfg,
):
    dynamic_mask = curr_data.get("dynamic_mask")
    if dynamic_mask is None or not bool(dynamic_mask.any()):
        return dynamic_params, False

    device = curr_data["im"].device
    mask_np = dynamic_mask[0].detach().cpu().numpy().astype(np.uint8)
    num_labels, labels, stats, cc_centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    if num_labels <= 1:
        return dynamic_params, False

    components = []
    min_area = int(cfg.get("min_component_area", 64))
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((label, area))
    components = sorted(components, key=lambda item: item[1], reverse=True)[: int(cfg.get("max_components_per_frame", 16))]
    if not components:
        return dynamic_params, False

    w2c = camera_w2c_from_params(camera_params, time_idx, camera_grad=False)
    used_obj_ids = set()
    used_pose_record_indices = set()
    pose_cfg = cfg.get("rigidmask_pose_init", {})
    pose_summary = curr_data.get("dynamic_component_poses")
    new_chunks = {
        "dyn_means3D_canon": [],
        "dyn_rgb_colors": [],
        "dyn_unnorm_rotations": [],
        "dyn_logit_opacities": [],
        "dyn_log_scales": [],
        "dyn_obj_ids": [],
        "dyn_birth_time": [],
    }
    changed = False

    for label, _area in components:
        comp_mask_np = labels == label
        comp_mask = torch.from_numpy(comp_mask_np).to(device).bool().unsqueeze(0)
        comp_mask = comp_mask & (curr_data["depth"] > 0)
        if curr_data.get("sky_mask") is not None:
            comp_mask = comp_mask & (~curr_data["sky_mask"].bool())
        if curr_data.get("dynamic_score") is not None:
            comp_mask = comp_mask & (curr_data["dynamic_score"] >= float(cfg.get("score_threshold", 0.2)))
        if int(comp_mask.sum().item()) < int(cfg.get("min_component_points", 24)):
            continue

        comp_mask = _subsample_mask(comp_mask, int(cfg.get("max_new_gaussians_per_component", 1200)))
        point_cld, mean3_sq_dist = pointcloud_from_mask(
            curr_data["im"],
            curr_data["depth"],
            curr_data["intrinsics"],
            w2c,
            comp_mask,
            mean_sq_dist_method=mean_sq_dist_method,
        )
        if point_cld.shape[0] < int(cfg.get("min_component_points", 24)):
            continue

        centroid = point_cld[:, :3].mean(dim=0)
        obj_id, dynamic_params = _associate_or_create_object(
            dynamic_params,
            centroid,
            time_idx,
            num_frames,
            used_obj_ids,
            cfg,
            device,
        )
        if obj_id is None:
            continue
        used_obj_ids.add(int(obj_id))

        pose_record, pose_record_index = _match_component_pose_record(
            pose_summary,
            label,
            stats,
            cc_centroids[label],
            mask_np.shape,
            used_pose_record_indices,
            pose_cfg,
        )
        if pose_record_index is not None:
            used_pose_record_indices.add(pose_record_index)
        obj_rot, obj_trans = _rigidmask_initialized_object_pose(
            dynamic_params,
            camera_params,
            obj_id,
            time_idx,
            centroid,
            pose_record,
            cfg,
            device,
        )

        with torch.no_grad():
            dynamic_params["dyn_obj_unnorm_rots"][int(obj_id), :, time_idx] = obj_rot
            dynamic_params["dyn_obj_trans"][int(obj_id), :, time_idx] = obj_trans
            dynamic_params["dyn_obj_visible"][int(obj_id), time_idx] = True
            dynamic_params["dyn_obj_last_seen"][int(obj_id)] = int(time_idx)
            dynamic_params["dyn_obj_obs_count"][int(obj_id)] += 1

        local_means = _points_to_object_local(point_cld[:, :3], obj_rot, obj_trans)
        new_chunks["dyn_means3D_canon"].append(local_means)
        new_chunks["dyn_rgb_colors"].append(point_cld[:, 3:6])
        new_chunks["dyn_unnorm_rotations"].append(torch.tile(torch.tensor([1, 0, 0, 0], device=device).float(), (point_cld.shape[0], 1)))
        new_chunks["dyn_logit_opacities"].append(torch.zeros((point_cld.shape[0], 1), device=device).float())
        if gaussian_distribution == "isotropic":
            new_chunks["dyn_log_scales"].append(torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1))
        elif gaussian_distribution == "anisotropic":
            new_chunks["dyn_log_scales"].append(torch.tile(torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1), (1, 3)))
        else:
            raise ValueError(f"Unknown gaussian_distribution {gaussian_distribution}")
        new_chunks["dyn_obj_ids"].append(torch.full((point_cld.shape[0],), int(obj_id), device=device).long())
        new_chunks["dyn_birth_time"].append(torch.full((point_cld.shape[0],), int(time_idx), device=device).long())
        changed = True

    if changed:
        dynamic_params = _append_dynamic_gaussians(dynamic_params, new_chunks)
    return dynamic_params, changed


def _subsample_mask(mask, max_points):
    if max_points <= 0:
        return mask
    flat = mask.reshape(-1)
    indices = torch.where(flat)[0]
    if indices.numel() <= max_points:
        return mask
    selected = indices[torch.randperm(indices.numel(), device=indices.device)[:max_points]]
    sampled = torch.zeros_like(flat)
    sampled[selected] = True
    return sampled.reshape_as(mask)


def _associate_or_create_object(dynamic_params, centroid, time_idx, num_frames, used_obj_ids, cfg, device):
    if dynamic_params is not None and "dyn_obj_trans" in dynamic_params:
        num_objects = int(dynamic_params["dyn_obj_trans"].shape[0])
    else:
        num_objects = 0

    best_obj = None
    if num_objects > 0:
        last_seen = dynamic_params["dyn_obj_last_seen"].detach().long()
        valid = (int(time_idx) - last_seen) <= int(cfg.get("max_inactive_frames", 5))
        if used_obj_ids:
            used = torch.tensor(sorted(used_obj_ids), device=device).long()
            valid[used] = False
        if bool(valid.any()):
            obj_indices = torch.arange(num_objects, device=device).long()
            centers_by_time = dynamic_params["dyn_obj_trans"].detach().permute(0, 2, 1)
            centers = centers_by_time[obj_indices, last_seen]
            distances = torch.norm(centers - centroid[None, :], dim=1)
            distances[~valid] = 1e9
            min_dist, min_idx = torch.min(distances, dim=0)
            if float(min_dist.item()) <= float(cfg.get("association_dist_m", 4.0)):
                best_obj = int(min_idx.item())

    if best_obj is not None:
        return best_obj, dynamic_params

    return num_objects, _append_dynamic_object(dynamic_params, centroid, num_frames, device)


def _append_dynamic_object(dynamic_params, centroid, num_frames, device):
    obj_rot = torch.tile(torch.tensor([1, 0, 0, 0], device=device).float().view(1, 4, 1), (1, 1, num_frames))
    obj_trans = centroid.view(1, 3, 1).repeat(1, 1, num_frames)
    obj_visible = torch.zeros((1, num_frames), device=device).bool()
    obj_last_seen = torch.full((1,), -1, device=device).long()
    obj_obs_count = torch.zeros((1,), device=device).long()

    if dynamic_params is None:
        return {
            "dyn_obj_unnorm_rots": torch.nn.Parameter(obj_rot.requires_grad_(True)),
            "dyn_obj_trans": torch.nn.Parameter(obj_trans.requires_grad_(True)),
            "dyn_obj_visible": obj_visible,
            "dyn_obj_last_seen": obj_last_seen,
            "dyn_obj_obs_count": obj_obs_count,
        }

    dynamic_params["dyn_obj_unnorm_rots"] = torch.nn.Parameter(
        torch.cat((dynamic_params["dyn_obj_unnorm_rots"].detach(), obj_rot), dim=0).requires_grad_(True)
    )
    dynamic_params["dyn_obj_trans"] = torch.nn.Parameter(
        torch.cat((dynamic_params["dyn_obj_trans"].detach(), obj_trans), dim=0).requires_grad_(True)
    )
    dynamic_params["dyn_obj_visible"] = torch.cat((dynamic_params["dyn_obj_visible"], obj_visible), dim=0)
    dynamic_params["dyn_obj_last_seen"] = torch.cat((dynamic_params["dyn_obj_last_seen"], obj_last_seen), dim=0)
    dynamic_params["dyn_obj_obs_count"] = torch.cat((dynamic_params["dyn_obj_obs_count"], obj_obs_count), dim=0)
    return dynamic_params


def _append_dynamic_gaussians(dynamic_params, chunks):
    if dynamic_params is None:
        raise ValueError("dynamic_params must be initialized with at least one object before adding gaussians.")

    tensor_chunks = {key: torch.cat(value, dim=0) for key, value in chunks.items() if value}
    trainable = {"dyn_means3D_canon", "dyn_rgb_colors", "dyn_unnorm_rotations", "dyn_logit_opacities", "dyn_log_scales"}
    for key, value in tensor_chunks.items():
        if key in trainable:
            if key in dynamic_params:
                value = torch.cat((dynamic_params[key].detach(), value), dim=0)
            dynamic_params[key] = torch.nn.Parameter(value.requires_grad_(True))
        else:
            if key in dynamic_params:
                value = torch.cat((dynamic_params[key], value), dim=0)
            dynamic_params[key] = value
    return dynamic_params


def transform_dynamic_to_frame(dynamic_params, camera_params, time_idx, gaussians_grad, camera_grad=False):
    if not has_dynamic_gaussians(dynamic_params):
        return None, None

    obj_ids_all = dynamic_params["dyn_obj_ids"].long()
    birth = dynamic_params["dyn_birth_time"].long()
    visible = dynamic_params["dyn_obj_visible"].bool()
    active = (birth <= int(time_idx)) & visible[obj_ids_all, int(time_idx)]
    if not bool(active.any()):
        return None, active

    obj_ids = obj_ids_all[active]
    if gaussians_grad:
        local_pts = dynamic_params["dyn_means3D_canon"][active]
        local_rots = dynamic_params["dyn_unnorm_rotations"][active]
        obj_rots = F.normalize(dynamic_params["dyn_obj_unnorm_rots"][obj_ids, :, int(time_idx)])
        obj_trans = dynamic_params["dyn_obj_trans"][obj_ids, :, int(time_idx)]
    else:
        local_pts = dynamic_params["dyn_means3D_canon"][active].detach()
        local_rots = dynamic_params["dyn_unnorm_rotations"][active].detach()
        obj_rots = F.normalize(dynamic_params["dyn_obj_unnorm_rots"][obj_ids, :, int(time_idx)].detach())
        obj_trans = dynamic_params["dyn_obj_trans"][obj_ids, :, int(time_idx)].detach()

    obj_rot_mats = build_rotation(obj_rots)
    world_pts = torch.bmm(obj_rot_mats, local_pts.unsqueeze(-1)).squeeze(-1) + obj_trans

    w2c = camera_w2c_from_params(camera_params, time_idx, camera_grad=camera_grad)
    pts4 = torch.cat((world_pts, torch.ones(world_pts.shape[0], 1, device=world_pts.device).float()), dim=1)
    cam_pts = (w2c @ pts4.T).T[:, :3]

    if camera_grad:
        cam_rot = F.normalize(camera_params["cam_unnorm_rots"][..., time_idx])
    else:
        cam_rot = F.normalize(camera_params["cam_unnorm_rots"][..., time_idx].detach())
    world_rots = quat_mult(obj_rots, F.normalize(local_rots))
    cam_rots = quat_mult(cam_rot, world_rots)

    return {
        "means3D": cam_pts,
        "unnorm_rotations": cam_rots,
        "active_mask": active,
    }, active


def dynamic_params2rendervar(dynamic_params, transformed_dynamic, active_mask):
    if transformed_dynamic is None:
        return None
    active = active_mask.bool()
    log_scales = dynamic_params["dyn_log_scales"][active]
    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    return {
        "means3D": transformed_dynamic["means3D"],
        "colors_precomp": dynamic_params["dyn_rgb_colors"][active],
        "rotations": F.normalize(transformed_dynamic["unnorm_rotations"]),
        "opacities": torch.sigmoid(dynamic_params["dyn_logit_opacities"][active]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(transformed_dynamic["means3D"], requires_grad=True, device=transformed_dynamic["means3D"].device) + 0,
    }


def dynamic_params2depthplussilhouette(dynamic_params, transformed_dynamic, active_mask, w2c):
    if transformed_dynamic is None:
        return None
    active = active_mask.bool()
    log_scales = dynamic_params["dyn_log_scales"][active]
    if log_scales.shape[1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))
    return {
        "means3D": transformed_dynamic["means3D"],
        "colors_precomp": get_depth_and_silhouette(transformed_dynamic["means3D"], w2c),
        "rotations": F.normalize(transformed_dynamic["unnorm_rotations"]),
        "opacities": torch.sigmoid(dynamic_params["dyn_logit_opacities"][active]),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(transformed_dynamic["means3D"], requires_grad=True, device=transformed_dynamic["means3D"].device) + 0,
    }


def merge_rendervars(static_rendervar, dynamic_rendervar):
    if dynamic_rendervar is None:
        return static_rendervar
    merged = {}
    for key, value in static_rendervar.items():
        merged[key] = torch.cat((value, dynamic_rendervar[key]), dim=0)
    return merged


def get_dynamic_loss(dynamic_params, camera_params, curr_data, time_idx, cfg):
    dynamic_mask = curr_data.get("dynamic_mask")
    if dynamic_mask is None or not has_dynamic_gaussians(dynamic_params):
        return None, {}

    transformed_dynamic, active_mask = transform_dynamic_to_frame(
        dynamic_params,
        camera_params,
        time_idx,
        gaussians_grad=True,
        camera_grad=False,
    )
    if transformed_dynamic is None:
        return None, {}

    rendervar = dynamic_params2rendervar(dynamic_params, transformed_dynamic, active_mask)
    depth_sil_rendervar = dynamic_params2depthplussilhouette(dynamic_params, transformed_dynamic, active_mask, curr_data["w2c"])
    im, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**rendervar)
    depth_sil, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]

    dyn_mask = dynamic_mask.bool()
    valid_area = curr_data["depth"] > 0
    if curr_data.get("sky_mask") is not None:
        valid_area = valid_area & (~curr_data["sky_mask"].bool())
    loss_mask = dyn_mask & valid_area
    if int(loss_mask.sum().item()) < int(cfg.get("min_loss_pixels", 16)):
        return None, {}

    color_mask = loss_mask.repeat(3, 1, 1)
    losses = {}
    losses["im"] = torch.abs(im - curr_data["im"])[color_mask].mean()
    losses["depth"] = torch.abs(depth - curr_data["depth"])[loss_mask].mean()

    sil_valid = valid_area[0]
    target_sil = dyn_mask[0].float()
    losses["sil"] = torch.abs(silhouette[sil_valid] - target_sil[sil_valid]).mean()

    outside = (~dyn_mask) & valid_area
    if bool(outside.any()):
        losses["outside_sil"] = silhouette[outside[0]].mean()
    else:
        losses["outside_sil"] = torch.zeros((), device=im.device)

    losses["motion_smooth"] = _dynamic_motion_smoothness(dynamic_params, time_idx)
    losses["scale"] = torch.exp(dynamic_params["dyn_log_scales"]).mean()

    weights = cfg.get("loss_weights", {})
    weighted = {key: value * float(weights.get(key, 0.0)) for key, value in losses.items()}
    weighted["loss"] = sum(weighted.values())
    return weighted["loss"], weighted


def _dynamic_motion_smoothness(dynamic_params, time_idx):
    if time_idx <= 0 or "dyn_obj_trans" not in dynamic_params:
        return torch.zeros((), device=dynamic_params["dyn_obj_trans"].device)
    visible = dynamic_params["dyn_obj_visible"].bool()
    active = visible[:, int(time_idx)] & visible[:, int(time_idx) - 1]
    if not bool(active.any()):
        return torch.zeros((), device=dynamic_params["dyn_obj_trans"].device)
    trans = dynamic_params["dyn_obj_trans"][active]
    vel = trans[:, :, int(time_idx)] - trans[:, :, int(time_idx) - 1]
    if time_idx > 1:
        active_prev = active & visible[:, int(time_idx) - 2]
        if bool(active_prev.any()):
            trans_prev = dynamic_params["dyn_obj_trans"][active_prev]
            accel = (
                trans_prev[:, :, int(time_idx)]
                - 2.0 * trans_prev[:, :, int(time_idx) - 1]
                + trans_prev[:, :, int(time_idx) - 2]
            )
            return accel.norm(dim=1).mean()
    return vel.norm(dim=1).mean()

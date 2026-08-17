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
from utils.slam_helpers import get_depth_and_silhouette, quat_mult


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

    return {
        "dynamic_mask": mask.bool(),
        "dynamic_score": score,
        "dynamic_mask_path": str(mask_path),
        "dynamic_pair_dir": str(pair_dir),
        "dynamic_fusion_summary_path": str(summary_path) if summary_path.exists() else "",
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
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
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

        with torch.no_grad():
            dynamic_params["dyn_obj_trans"][int(obj_id), :, time_idx] = centroid
            dynamic_params["dyn_obj_visible"][int(obj_id), time_idx] = True
            dynamic_params["dyn_obj_last_seen"][int(obj_id)] = int(time_idx)
            dynamic_params["dyn_obj_obs_count"][int(obj_id)] += 1

        local_means = point_cld[:, :3] - centroid[None, :]
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

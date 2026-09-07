import json
import os
import shutil
from pathlib import Path

import cv2
import torch

from utils.pnp_fused_icp_utils import get_dataset_frame_id

from .appearance import AppearanceSimilarityProbe
from .fastsam_probe import FastSAMProbe
from .fusion import find_pair_file, load_lidar_projection_for_pair, process_pair_dir
from .lidar_motion import LidarMotionProbe
from .rigidmask_frontend import RigidMaskFrontendProbe


DEFAULT_DYNAMIC_MASK_CFG = {
    "enabled": False,
    "output_subdir": "dynamic_mask",
    "minimal_storage": False,
    "fail_on_error": False,
    "require_lidar_residual": False,
    "require_fastsam": False,
    "rigidmask": {
        "enabled": True,
        "output_subdir": "rigidmask_frontend_probe",
        "run_every": 1,
        "repo_root": "third_party/rigidmask",
        "checkpoint_path": "",
        "calibration_path": "",
        "disparity_dir": "disparity_sceneflow",
        "sensor": "stereo",
        "use_opencv_essential_mat": True,
        "testres": 1.0,
        "maxdisp": 256,
        "fac": 1.0,
        "lazy_load": True,
        "offload_after_use": True,
        "save_raw_tensors": True,
        "save_visualizations": False,
        "save_input_rgbs": True,
        "require_cuda": True,
        "depth_mask": {
            "enabled": False,
            "min_depth_m": 0.1,
            "max_depth_m": None,
            "min_disp": 1e-6,
            "mask_sky": False,
            "apply_stage": "post_dynamic_mask",
            "save_visualizations": False,
            "save_raw_tensors": True,
        },
    },
    "lidar_residual": {
        "enabled": True,
        "output_subdir": "lidar_motion_probe",
        "run_every": 1,
        "max_points": 120000,
        "min_forward_m": 0.0,
        "max_forward_m": 70.0,
        "bev_reference_frame": "prev_camera",
        "min_height_m": -3.0,
        "max_height_m": 3.0,
        "bev_xlim": (-35.0, 35.0),
        "bev_zlim": (0.0, 75.0),
        "save_npz": True,
        "compute_nonground_residual": True,
        "save_visualizations": False,
        "save_nonground_visualizations": False,
        "nonground_filter_method": "gndnet",
        "gndnet_repo_root": "third_party/GndNet",
        "gndnet_checkpoint_path": "third_party/GndNet/trained_models/checkpoint.pth.tar",
        "gndnet_config_path": "third_party/GndNet/config/config_kittiSem.yaml",
        "gndnet_ground_threshold_m": 0.2,
        "gndnet_keep_outside_as_nonground": True,
        "save_static_projection_masks": True,
        "static_projection_point_radius": 3,
        "static_projection_top_row_percentile": 0.1,
        "static_projection_top_row_margin_px": 8.0,
        "save_lidar_se3_points": True,
        "lidar_se3_points_filename": "lidar_se3_points.npz",
        "save_feature_residual": True,
        "save_feature_residual_image_projection": True,
        "save_feature_residual_projection_npz": True,
        "feature_residual_projection_point_radius": 3,
        "feature_residual_projection_alpha": 0.88,
        "feature_residual_projection_min_depth_m": 0.1,
        "feature_residual_projection_max_depth_m": 80.0,
        "save_feature_mask_visualization": False,
        "feature_residual_max_points": 60000,
        "feature_residual_min_points": 200,
        "feature_residual_mode": "hybrid",
        "feature_residual_euclidean_weight": 0.35,
        "feature_residual_vis_max_m": 1.5,
        "feature_knn": 20,
        "feature_min_linearity": 0.55,
        "feature_min_planarity": 0.45,
        "feature_max_scattering": 0.12,
        "feature_max_curvature": 0.08,
        "feature_max_neighbor_radius_m": 1.5,
    },
    "appearance": {
        "enabled": True,
        "output_subdir": "appearance_similarity",
        "run_every": 1,
        "checkpoint_path": "checkpoints/dinov2_reg_small_finetuned.pth",
        "model_name": "vit_small_patch14_reg4_dinov2.lvd142m",
        "save_raw_tensors": True,
        "save_feature_tensors": False,
        "save_visualizations": False,
        "save_input_rgbs": False,
        "offload_after_use": True,
        "require_appearance": True,
        "use_mapping_render": True,
        "defer_fusion_until_appearance": True,
    },
    "fastsam": {
        "enabled": False,
        "mode": "online",
        "precomputed_root": "",
        "precomputed_subdir": "fastsam_masks",
        "require_precomputed": False,
        "repo_root": "third_party/FastSAM",
        "checkpoint_path": "checkpoints/FastSAM-x.pt",
        "source_image": "anchor_rgb.png",
        "run_every": 1,
        "imgsz": 1024,
        "conf": 0.4,
        "iou": 0.9,
        "retina_masks": True,
        "min_area_ratio": 0.0005,
        "max_area_ratio": 0.80,
        "max_masks": 128,
        "save_visualization": True,
        "offload_after_use": False,
        "overwrite": False,
    },
    "fusion": {
        "enabled": True,
        "mask_percentile": 85.0,
        "empty_score_mean_thresh": 0.06,
        "empty_score_p90_thresh": 0.15,
        "empty_score_p99_thresh": 0.25,
        "empty_high_score_thresh": 0.25,
        "empty_high_score_area_thresh": 0.01,
        "appearance_boost_alpha": 0.5,
        "lidar_residual_enabled": True,
        "lidar_motion_subdir": "lidar_motion_probe",
        "lidar_projection_filename": "image_residual_nonground_features.npz",
        "lidar_splat_radius": 2.0,
        "lidar_confidence_norm": 1.5,
        "lidar_static_residual_m": 0.15,
        "lidar_dynamic_residual_m": 0.70,
        "lidar_residual_high_q": 95.0,
        "lidar_residual_mad_scale": 2.0,
        "lidar_min_visible_points": 20,
        "lidar_static_mask_enabled": True,
        "lidar_static_mask_filename": "image_lidar_static_masks.npz",
        "lidar_static_filter_above_range": True,
        "lidar_static_above_row_percentile": 0.1,
        "lidar_static_above_row_margin_px": 8.0,
        "lidar_down_weight": 0.65,
        "lidar_up_weight": 0.45,
        "lidar_confidence_thresh": 0.25,
        "lidar_promote_score_thresh": 0.65,
        "lidar_promote_visual_thresh": 0.20,
        "lidar_component_min_covered_cells": 3,
        "lidar_component_min_covered_fraction": 0.03,
        "lidar_suppress_uncovered_components": True,
        "lidar_uncovered_component_min_area": 64,
        "lidar_component_dark_mean_thresh": 0.08,
        "lidar_component_dark_p95_thresh": 0.18,
        "lidar_component_dark_high_score_thresh": 0.35,
        "lidar_component_dark_max_high_fraction": 0.02,
        "lidar_allow_empty_override": False,
        "lidar_empty_override_min_cells": 3,
        "fastsam_enabled": False,
        "fastsam_min_overlap_fraction": 0.60,
        "fastsam_min_score_mean": 0.35,
        "fastsam_min_score_p90": 0.55,
        "fastsam_min_area_cells": 32,
        "fastsam_max_area_fraction": 0.80,
        "lidar_se3_static_veto": {
            "enabled": False,
            "filename": "lidar_se3_points.npz",
            "use_nonground_only": True,
            "fallback_to_visible_points": True,
            "min_component_area": 80,
            "max_components": 64,
            "min_lidar_points": 25,
            "min_reference_points": 200,
            "component_association_radius_px": 2.0,
            "bg_inlier_dist_m": 0.25,
            "bg_inlier_ratio": 0.85,
            "bg_median_dist_m": 0.15,
            "bg_p90_dist_m": 0.35,
            "object_icp": {
                "enabled": True,
                "min_points": 30,
                "max_points": 1500,
                "local_margin_m": 1.0,
                "max_corr_m": 0.50,
                "max_iteration": 30,
                "min_fitness": 0.45,
                "max_rmse_m": 0.25,
                "rel_angle_deg": 1.0,
                "rel_trans_m": 0.10,
                "median_dist_m": 0.20,
                "bg_vs_obj_median_ratio": 0.80,
            },
        },
        "se3_static_veto": {
            "enabled": False,
            "min_component_area": 80,
            "min_valid_points": 50,
            "min_bg_points": 200,
            "max_bg_points": 8000,
            "max_obj_points": 3000,
            "min_depth_m": 0.1,
            "max_depth_m": 80.0,
            "pnp_reproj_error_px": 4.0,
            "bg_inlier_px": 3.0,
            "bg_inlier_ratio": 0.70,
            "bg_median_px": 3.0,
            "obj_min_inlier_ratio": 0.35,
            "rel_angle_deg": 1.5,
            "rel_trans_m": 0.15,
            "bg_vs_obj_median_ratio": 1.25,
            "prefer_slam_pose": True,
            "fallback_to_background_pnp": True,
            "edge_guard_enabled": True,
            "edge_guard_min_static_edge_iou": 0.45,
            "edge_guard_max_static_symdiff_ratio": 0.10,
            "edge_guard_splat_radius": 1,
            "edge_guard_edge_width": 2,
        },
        "component_pose_init": {
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
        },
        "save_diagnostics": False,
        "organize_outputs": True,
    },
}

MINIMAL_PAIR_KEEP_FILES = {
    "dynamic_mask.png",
    "dynamic_score.png",
    "dynamic_fusion_summary.json",
    "dynamic_component_poses.json",
}


def _deep_merge(base, override):
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = _deep_merge(value, {})
        else:
            merged[key] = value
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_project_path(path, project_root):
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def _quaternion_to_rotation_numpy(quat):
    q = quat.detach().float().reshape(-1).cpu()
    if q.numel() != 4:
        return None
    norm = torch.linalg.norm(q)
    if float(norm.item()) <= 1e-12:
        return None
    q = (q / norm).numpy()
    r, x, y, z = [float(value) for value in q]
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - r * z), 2.0 * (x * z + r * y)],
        [2.0 * (x * y + r * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - r * x)],
        [2.0 * (x * z - r * y), 2.0 * (y * z + r * x), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _camera_w2c_numpy_from_params(params, time_idx):
    if params is None or "cam_unnorm_rots" not in params or "cam_trans" not in params:
        return None
    try:
        num_frames = int(params["cam_unnorm_rots"].shape[-1])
        time_idx = int(time_idx)
        if time_idx < 0 or time_idx >= num_frames:
            return None
        rot = _quaternion_to_rotation_numpy(params["cam_unnorm_rots"][..., time_idx])
        if rot is None:
            return None
        tran = params["cam_trans"][..., time_idx].detach().float().reshape(-1).cpu().numpy()
        if tran.shape[0] != 3:
            return None
    except Exception:
        return None
    import numpy as np

    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(rot, dtype=np.float64)
    w2c[:3, 3] = tran.astype(np.float64)
    return w2c


def _slam_background_transform_for_target(params, target_record):
    if params is None or target_record is None:
        return None
    target_time_idx = target_record.get("target_time_idx")
    if target_time_idx is None:
        return None
    role = str(target_record.get("target_role", "current")).strip().lower()
    target_time_idx = int(target_time_idx)
    if role in ("current", "curr", "t"):
        counterpart_time_idx = target_time_idx - 1
    elif role in ("previous", "prev", "reference", "t-1"):
        counterpart_time_idx = target_time_idx + 1
    else:
        return None
    target_w2c = _camera_w2c_numpy_from_params(params, target_time_idx)
    counterpart_w2c = _camera_w2c_numpy_from_params(params, counterpart_time_idx)
    if target_w2c is None or counterpart_w2c is None:
        return None
    import numpy as np

    target_to_counterpart = counterpart_w2c @ np.linalg.inv(target_w2c)
    return {
        "source": "slam_tracking_pose",
        "target_time_idx": int(target_time_idx),
        "counterpart_time_idx": int(counterpart_time_idx),
        "target_role": role,
        "target_to_counterpart": target_to_counterpart.tolist(),
    }


class DynamicMaskManager:
    def __init__(self, cfg, data_cfg, dataset, output_dir, project_root, device):
        self.cfg = _deep_merge(DEFAULT_DYNAMIC_MASK_CFG, cfg or {})
        self.minimal_storage = bool(self.cfg.get("minimal_storage", False))
        if self.minimal_storage:
            self._apply_minimal_storage_defaults()
        self.enabled = bool(self.cfg.get("enabled", False))
        self.fail_on_error = bool(self.cfg.get("fail_on_error", False))
        self.require_lidar_residual = bool(self.cfg.get("require_lidar_residual", False))
        fastsam_cfg = self.cfg.get("fastsam", {})
        self.require_fastsam = bool(
            self.cfg.get("require_fastsam", False)
            or (
                str(fastsam_cfg.get("mode", "online")).lower() == "precomputed"
                and fastsam_cfg.get("require_precomputed", False)
            )
        )
        self.data_cfg = dict(data_cfg)
        self.dataset = dataset
        self.output_dir = output_dir
        self.project_root = project_root
        self.device = device
        self.root_dir = os.path.join(output_dir, self.cfg.get("output_subdir", "dynamic_mask"))
        self.rigidmask_output_dir = os.path.join(
            self.root_dir,
            self.cfg["rigidmask"].get("output_subdir", "rigidmask_frontend_probe"),
        )
        self.appearance_output_dir = os.path.join(
            self.root_dir,
            self.cfg["appearance"].get("output_subdir", "appearance_similarity"),
        )
        self.lidar_output_dir = os.path.join(
            self.root_dir,
            self.cfg["lidar_residual"].get("output_subdir", "lidar_motion_probe"),
        )
        self.rigidmask = None
        self.appearance = None
        self.fastsam = None
        self.lidar_residual = None
        self._summary = []
        self._init_error = None
        self._sky_masks_by_time_idx = {}

        if self.enabled:
            os.makedirs(self.root_dir, exist_ok=True)
            self._write_config_snapshot(data_cfg)

    def _apply_minimal_storage_defaults(self):
        self.cfg["rigidmask"]["save_visualizations"] = False
        self.cfg["rigidmask"]["depth_mask"]["save_visualizations"] = False
        self.cfg["lidar_residual"]["save_npz"] = False
        self.cfg["lidar_residual"]["save_visualizations"] = False
        self.cfg["lidar_residual"]["save_nonground_visualizations"] = False
        self.cfg["lidar_residual"]["save_feature_mask_visualization"] = False
        self.cfg["appearance"]["save_feature_tensors"] = False
        self.cfg["appearance"]["save_visualizations"] = False
        self.cfg["appearance"]["save_input_rgbs"] = False
        self.cfg["fastsam"]["save_visualization"] = False
        self.cfg["fusion"]["save_diagnostics"] = False

    def run_for_frame(self, time_idx, num_frames, params, sky_mask=None):
        if not self.enabled:
            return None
        record = {
            "time_idx": int(time_idx),
            "status": "skipped",
            "rigidmask_status": "not_run",
            "lidar_residual_status": "not_run",
            "fastsam_status": "not_run",
            "fusion_status": "not_run",
        }
        try:
            result = self._run_for_frame(time_idx, num_frames, params, sky_mask)
            record.update(result or {})
        except Exception as exc:
            record.update({"status": "error", "reason": str(exc)})
            if self.fail_on_error:
                raise
            print(f"Dynamic mask skipped at frame {time_idx}: {exc}")
        self._summary.append(record)
        self._write_run_summary()
        return record

    def _run_for_frame(self, time_idx, num_frames, params, sky_mask):
        curr_frame_id = get_dataset_frame_id(self.dataset, time_idx)
        if sky_mask is not None:
            self._sky_masks_by_time_idx[int(time_idx)] = sky_mask.detach().cpu() if torch.is_tensor(sky_mask) else sky_mask
        if time_idx <= 0:
            return {
                "status": "skipped",
                "reason": "first_frame_has_no_previous_rgb_pair",
                "curr_frame_id": curr_frame_id,
            }
        lidar_summary = None
        if self.cfg["lidar_residual"].get("enabled", True) and time_idx > 0:
            lidar = self._ensure_lidar_residual()
            if lidar is not None and lidar.should_run(time_idx):
                os.makedirs(self.lidar_output_dir, exist_ok=True)
                lidar_summary = lidar.save_pair(
                    output_root=self.lidar_output_dir,
                    dataset=self.dataset,
                    params=params,
                    prev_time_idx=time_idx - 1,
                    curr_time_idx=time_idx,
                    device=self.device,
                )

        rigidmask_summary = None
        rigidmask = self._ensure_rigidmask()
        if rigidmask is None or not rigidmask.should_run(time_idx, num_frames):
            return {
                "status": "skipped",
                "reason": "rigidmask_disabled_or_not_scheduled",
                "curr_frame_id": curr_frame_id,
                "lidar_residual_status": self._status(lidar_summary),
            }

        reference_frame_id = get_dataset_frame_id(self.dataset, time_idx - 1)
        os.makedirs(self.rigidmask_output_dir, exist_ok=True)
        rigidmask_summary = rigidmask.save_pair(
            output_root=self.rigidmask_output_dir,
            time_idx=time_idx,
            curr_frame_id=curr_frame_id,
            reference_frame_id=reference_frame_id,
            sky_mask=sky_mask,
            reference_sky_mask=self._sky_masks_by_time_idx.get(int(time_idx) - 1),
        )
        if rigidmask_summary.get("status") != "ok":
            return {
                "status": "skipped",
                "reason": rigidmask_summary.get("reason", "rigidmask_skipped"),
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }

        pair_name = rigidmask_summary.get(
            "pair_name",
            f"{time_idx:06d}_frame_{curr_frame_id}_from_{reference_frame_id}",
        )
        pair_dir = Path(self.rigidmask_output_dir) / pair_name
        target_records = self._target_records_from_rigidmask_summary(rigidmask_summary)
        lidar_ok = lidar_summary is not None and lidar_summary.get("status") == "ok"
        if not find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz").exists():
            return {
                "status": "skipped",
                "reason": "missing_rigidmask_arrays",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }
        missing_target_arrays = [
            target["target_role"]
            for target in target_records
            if not find_pair_file(Path(target["pair_dir"]), "rigidmask_frontend_arrays.npz").exists()
        ]
        if missing_target_arrays:
            return {
                "status": "skipped",
                "reason": "missing_target_rigidmask_arrays",
                "missing_targets": missing_target_arrays,
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }
        fastsam_results = self._save_fastsam_for_targets(target_records)
        if self.require_fastsam and self._has_missing_required_fastsam(fastsam_results):
            return {
                "status": "skipped",
                "reason": "missing_required_fastsam",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "targets": target_records,
                "fastsam_results": fastsam_results,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fastsam_status": "missing",
            }
        if self.require_lidar_residual:
            if not lidar_ok:
                return {
                    "status": "skipped",
                    "reason": "missing_required_lidar_residual",
                    "curr_frame_id": curr_frame_id,
                    "reference_frame_id": reference_frame_id,
                    "targets": target_records,
                    "rigidmask_status": self._status(rigidmask_summary),
                    "lidar_residual_status": self._status(lidar_summary),
                }
            missing_lidar = self._missing_required_lidar_targets(target_records)
            if missing_lidar:
                return {
                    "status": "skipped",
                    "reason": "missing_required_lidar_residual",
                    "curr_frame_id": curr_frame_id,
                    "reference_frame_id": reference_frame_id,
                    "targets": target_records,
                    "missing_lidar_targets": missing_lidar,
                    "rigidmask_status": self._status(rigidmask_summary),
                    "lidar_residual_status": "missing",
                }
        if not self.cfg["fusion"].get("enabled", True):
            return {
                "status": "skipped",
                "reason": "fusion_disabled",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "targets": target_records,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fastsam_results": fastsam_results,
                "fastsam_status": self._targets_status(fastsam_results),
                "fusion_status": "disabled",
            }

        if self.cfg["appearance"].get("enabled", True) and self.cfg["appearance"].get(
            "defer_fusion_until_appearance",
            self.cfg["appearance"].get("require_appearance", True),
        ):
            return {
                "status": "pending_appearance",
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "targets": target_records,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fastsam_results": fastsam_results,
                "fastsam_status": self._targets_status(fastsam_results),
                "fusion_status": "deferred",
            }

        target_results = self._process_target_pair_dirs(target_records, params=params)
        if not target_results or not target_results.get("current", {}).get("fusion_ok", False):
            return {
                "status": "skipped",
                "reason": "fusion_returned_none",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fastsam_results": fastsam_results,
                "fastsam_status": self._targets_status(fastsam_results),
                "fusion_status": "skipped",
            }
        self._compact_storage_after_fusion(target_records, target_results, time_idx, num_frames)

        return {
            "status": "ok" if all(item.get("fusion_ok", False) for item in target_results.values()) else "partial",
            "pair_name": pair_name,
            "pair_dir": str(pair_dir),
            "curr_frame_id": curr_frame_id,
            "reference_frame_id": reference_frame_id,
            "targets": target_records,
            "target_results": target_results,
            "rigidmask_status": self._status(rigidmask_summary),
            "lidar_residual_status": self._status(lidar_summary),
            "fastsam_results": fastsam_results,
            "fastsam_status": self._targets_status(fastsam_results),
            "fusion_status": "ok",
        }

    def finalize_for_frame(self, time_idx, num_frames, params, curr_data, render_pair=None):
        if not self.enabled:
            return None
        record = {
            "time_idx": int(time_idx),
            "phase": "appearance_and_fusion",
            "status": "skipped",
            "appearance_status": "not_run",
            "fastsam_status": "not_run",
            "fusion_status": "not_run",
        }
        try:
            result = self._finalize_for_frame(time_idx, num_frames, params, curr_data, render_pair)
            record.update(result or {})
        except Exception as exc:
            record.update({"status": "error", "reason": str(exc)})
            if self.fail_on_error:
                raise
            print(f"Dynamic mask finalization skipped at frame {time_idx}: {exc}")
        self._summary.append(record)
        self._write_run_summary()
        return record

    def _finalize_for_frame(self, time_idx, num_frames, params, curr_data, render_pair):
        if not self.cfg["appearance"].get("enabled", True):
            return {
                "status": "skipped",
                "reason": "appearance_disabled",
                "fusion_status": "not_run",
            }
        curr_frame_id = get_dataset_frame_id(self.dataset, time_idx)
        appearance_summary = self._save_appearance_for_frame(
            time_idx,
            curr_frame_id,
            params,
            curr_data,
            render_pair,
        )
        if time_idx <= 0:
            return {
                "status": "appearance_only",
                "reason": "first_frame_has_no_previous_rgb_pair",
                "curr_frame_id": curr_frame_id,
                "appearance_status": self._status(appearance_summary),
                "fusion_status": "not_run",
            }
        reference_frame_id = get_dataset_frame_id(self.dataset, time_idx - 1)
        pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_from_{reference_frame_id}"
        pair_dir = Path(self.rigidmask_output_dir) / pair_name
        target_records = self._target_records_for_temporal_pair(time_idx, curr_frame_id, reference_frame_id)
        if not find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz").exists():
            return {
                "status": "skipped",
                "reason": "missing_rigidmask_arrays",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "appearance_status": self._status(appearance_summary),
            }
        missing_target_arrays = [
            target["target_role"]
            for target in target_records
            if not find_pair_file(Path(target["pair_dir"]), "rigidmask_frontend_arrays.npz").exists()
        ]
        if missing_target_arrays:
            return {
                "status": "skipped",
                "reason": "missing_target_rigidmask_arrays",
                "missing_targets": missing_target_arrays,
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "appearance_status": self._status(appearance_summary),
            }

        if self.require_lidar_residual:
            missing_lidar = self._missing_required_lidar_targets(target_records)
            if missing_lidar:
                return {
                    "status": "skipped",
                    "reason": "missing_required_lidar_residual",
                    "curr_frame_id": curr_frame_id,
                    "reference_frame_id": reference_frame_id,
                    "pair_name": pair_name,
                    "pair_dir": str(pair_dir),
                    "targets": target_records,
                    "missing_lidar_targets": missing_lidar,
                    "appearance_status": self._status(appearance_summary),
                    "lidar_residual_status": "missing",
                }

        missing_appearance = [
            target["target_role"]
            for target in target_records
            if not self._appearance_available(target["target_time_idx"], target["target_frame_id"])
        ]
        if self.cfg["appearance"].get("require_appearance", True) and missing_appearance:
            return {
                "status": "skipped",
                "reason": "missing_required_appearance_similarity",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "targets": target_records,
                "missing_appearance_targets": missing_appearance,
                "appearance_status": self._status(appearance_summary),
            }
        fastsam_results = self._save_fastsam_for_targets(target_records)
        if self.require_fastsam and self._has_missing_required_fastsam(fastsam_results):
            return {
                "status": "skipped",
                "reason": "missing_required_fastsam",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "targets": target_records,
                "fastsam_results": fastsam_results,
                "appearance_status": self._status(appearance_summary),
                "fastsam_status": "missing",
            }
        if not self.cfg["fusion"].get("enabled", True):
            return {
                "status": "skipped",
                "reason": "fusion_disabled",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "targets": target_records,
                "appearance_status": self._status(appearance_summary),
                "fastsam_results": fastsam_results,
                "fastsam_status": self._targets_status(fastsam_results),
                "fusion_status": "disabled",
            }

        target_results = self._process_target_pair_dirs(target_records, params=params)
        if not target_results or not target_results.get("current", {}).get("fusion_ok", False):
            return {
                "status": "skipped",
                "reason": "fusion_returned_none",
                "curr_frame_id": curr_frame_id,
                "reference_frame_id": reference_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "targets": target_records,
                "target_results": target_results,
                "appearance_status": self._status(appearance_summary),
                "fastsam_results": fastsam_results,
                "fastsam_status": self._targets_status(fastsam_results),
                "fusion_status": "skipped",
            }
        self._compact_storage_after_fusion(target_records, target_results, time_idx, num_frames)
        return {
            "status": "ok" if all(item.get("fusion_ok", False) for item in target_results.values()) else "partial",
            "curr_frame_id": curr_frame_id,
            "reference_frame_id": reference_frame_id,
            "pair_name": pair_name,
            "pair_dir": str(pair_dir),
            "targets": target_records,
            "target_results": target_results,
            "appearance_status": self._status(appearance_summary),
            "fastsam_results": fastsam_results,
            "fastsam_status": self._targets_status(fastsam_results),
            "fusion_status": "ok",
        }

    def _ensure_rigidmask(self):
        if self.rigidmask is not None:
            return self.rigidmask
        rigidmask_cfg = dict(self.cfg["rigidmask"])
        if not rigidmask_cfg.get("enabled", True):
            return None
        rigidmask_cfg["repo_root"] = _resolve_project_path(
            rigidmask_cfg.get("repo_root"),
            self.project_root,
        )
        rigidmask_cfg["checkpoint_path"] = _resolve_project_path(
            rigidmask_cfg.get("checkpoint_path"),
            self.project_root,
        )
        rigidmask_cfg["calibration_path"] = _resolve_project_path(
            rigidmask_cfg.get("calibration_path"),
            self.project_root,
        )
        try:
            self.rigidmask = RigidMaskFrontendProbe(rigidmask_cfg, self._dynamic_data_cfg())
        except Exception as exc:
            self._init_error = str(exc)
            if self.fail_on_error:
                raise
            print(f"Dynamic mask RigidMask backend disabled: {exc}")
            return None
        return self.rigidmask

    def _ensure_lidar_residual(self):
        if self.lidar_residual is not None:
            return self.lidar_residual
        lidar_cfg = dict(self.cfg["lidar_residual"])
        if not lidar_cfg.get("enabled", True):
            return None
        for key in ("gndnet_repo_root", "gndnet_checkpoint_path", "gndnet_config_path"):
            lidar_cfg[key] = _resolve_project_path(lidar_cfg.get(key), self.project_root)
        self.lidar_residual = LidarMotionProbe(lidar_cfg, self.project_root)
        return self.lidar_residual

    def _ensure_appearance(self):
        if self.appearance is not None:
            return self.appearance
        appearance_cfg = dict(self.cfg["appearance"])
        if not appearance_cfg.get("enabled", True):
            return None
        appearance_cfg["checkpoint_path"] = _resolve_project_path(
            appearance_cfg.get("checkpoint_path"),
            self.project_root,
        )
        try:
            self.appearance = AppearanceSimilarityProbe(appearance_cfg, self.device)
        except Exception as exc:
            self._init_error = str(exc)
            if self.fail_on_error:
                raise
            print(f"Dynamic mask appearance backend disabled: {exc}")
            return None
        return self.appearance

    def _ensure_fastsam(self):
        if self.fastsam is not None:
            return self.fastsam
        fastsam_cfg = dict(self.cfg["fastsam"])
        if not fastsam_cfg.get("enabled", False):
            return None
        if str(fastsam_cfg.get("mode", "online")).lower() == "precomputed":
            return None
        for key in ("repo_root", "checkpoint_path"):
            fastsam_cfg[key] = _resolve_project_path(fastsam_cfg.get(key), self.project_root)
        try:
            self.fastsam = FastSAMProbe(fastsam_cfg, self.device)
        except Exception as exc:
            self._init_error = str(exc)
            if self.fail_on_error:
                raise
            print(f"Dynamic mask FastSAM backend disabled: {exc}")
            return None
        return self.fastsam

    def _render_current_rgb(self, params, curr_data, time_idx):
        import torch
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        from utils.slam_helpers import transformed_params2rendervar, transform_to_frame

        with torch.no_grad():
            transformed_gaussians = transform_to_frame(
                params,
                time_idx,
                gaussians_grad=False,
                camera_grad=False,
            )
            rendervar = transformed_params2rendervar(params, transformed_gaussians)
            image, _, _, _ = Renderer(raster_settings=curr_data["cam"])(**rendervar)
        return image.detach().cpu()

    def _dynamic_data_cfg(self):
        data_cfg = dict(getattr(self, "data_cfg", {}) or {})
        if not data_cfg:
            data_cfg = {}
        data_cfg.setdefault("sequence", getattr(self.dataset, "sequence", None) or getattr(self.dataset, "seq", ""))
        data_cfg.setdefault("basedir", getattr(self.dataset, "basedir", None) or getattr(self.dataset, "input_folder", ""))
        return data_cfg

    def _process_pair_dir(self, pair_dir, params=None, target_record=None):
        fusion_cfg = self.cfg["fusion"]
        slam_background_transform = _slam_background_transform_for_target(params, target_record)
        return process_pair_dir(
            pair_dir=pair_dir,
            mask_percentile=fusion_cfg["mask_percentile"],
            empty_score_mean_thresh=fusion_cfg["empty_score_mean_thresh"],
            empty_score_p90_thresh=fusion_cfg["empty_score_p90_thresh"],
            empty_score_p99_thresh=fusion_cfg["empty_score_p99_thresh"],
            empty_high_score_thresh=fusion_cfg["empty_high_score_thresh"],
            empty_high_score_area_thresh=fusion_cfg["empty_high_score_area_thresh"],
            appearance_boost_alpha=fusion_cfg.get("appearance_boost_alpha", 0.5),
            appearance_subdir=self.cfg["appearance"].get("output_subdir", "appearance_similarity"),
            lidar_residual_enabled=fusion_cfg.get("lidar_residual_enabled", True),
            require_lidar_residual=self.require_lidar_residual,
            lidar_motion_subdir=self.cfg["lidar_residual"].get("output_subdir", "lidar_motion_probe"),
            lidar_projection_filename=fusion_cfg["lidar_projection_filename"],
            lidar_splat_radius=fusion_cfg["lidar_splat_radius"],
            lidar_confidence_norm=fusion_cfg["lidar_confidence_norm"],
            lidar_static_residual_m=fusion_cfg["lidar_static_residual_m"],
            lidar_dynamic_residual_m=fusion_cfg["lidar_dynamic_residual_m"],
            lidar_residual_high_q=fusion_cfg["lidar_residual_high_q"],
            lidar_residual_mad_scale=fusion_cfg["lidar_residual_mad_scale"],
            lidar_min_visible_points=fusion_cfg["lidar_min_visible_points"],
            lidar_static_mask_enabled=fusion_cfg["lidar_static_mask_enabled"],
            lidar_static_mask_filename=fusion_cfg["lidar_static_mask_filename"],
            lidar_static_filter_above_range=fusion_cfg["lidar_static_filter_above_range"],
            lidar_static_above_row_percentile=fusion_cfg["lidar_static_above_row_percentile"],
            lidar_static_above_row_margin_px=fusion_cfg["lidar_static_above_row_margin_px"],
            lidar_down_weight=fusion_cfg["lidar_down_weight"],
            lidar_up_weight=fusion_cfg["lidar_up_weight"],
            lidar_confidence_thresh=fusion_cfg["lidar_confidence_thresh"],
            lidar_promote_score_thresh=fusion_cfg["lidar_promote_score_thresh"],
            lidar_promote_visual_thresh=fusion_cfg["lidar_promote_visual_thresh"],
            lidar_component_min_covered_cells=fusion_cfg["lidar_component_min_covered_cells"],
            lidar_component_min_covered_fraction=fusion_cfg["lidar_component_min_covered_fraction"],
            lidar_suppress_uncovered_components=fusion_cfg["lidar_suppress_uncovered_components"],
            lidar_uncovered_component_min_area=fusion_cfg["lidar_uncovered_component_min_area"],
            lidar_component_dark_mean_thresh=fusion_cfg["lidar_component_dark_mean_thresh"],
            lidar_component_dark_p95_thresh=fusion_cfg["lidar_component_dark_p95_thresh"],
            lidar_component_dark_high_score_thresh=fusion_cfg["lidar_component_dark_high_score_thresh"],
            lidar_component_dark_max_high_fraction=fusion_cfg["lidar_component_dark_max_high_fraction"],
            lidar_allow_empty_override=fusion_cfg["lidar_allow_empty_override"],
            lidar_empty_override_min_cells=fusion_cfg["lidar_empty_override_min_cells"],
            fastsam_enabled=fusion_cfg.get("fastsam_enabled", False),
            fastsam_min_overlap_fraction=fusion_cfg.get("fastsam_min_overlap_fraction", 0.60),
            fastsam_min_score_mean=fusion_cfg.get("fastsam_min_score_mean", 0.35),
            fastsam_min_score_p90=fusion_cfg.get("fastsam_min_score_p90", 0.55),
            fastsam_min_area_cells=fusion_cfg.get("fastsam_min_area_cells", 32),
            fastsam_max_area_fraction=fusion_cfg.get("fastsam_max_area_fraction", 0.80),
            fastsam_masks_path=(target_record or {}).get("fastsam_masks_path"),
            fastsam_summary_path=(target_record or {}).get("fastsam_summary_path"),
            lidar_se3_static_veto=fusion_cfg.get("lidar_se3_static_veto", {}),
            se3_static_veto=fusion_cfg.get("se3_static_veto", {}),
            component_pose_init=fusion_cfg.get("component_pose_init", {}),
            slam_background_transform=slam_background_transform,
            save_diagnostics=fusion_cfg.get("save_diagnostics", False),
            organize_outputs=fusion_cfg.get("organize_outputs", True),
        )

    def _target_records_from_rigidmask_summary(self, rigidmask_summary):
        targets = rigidmask_summary.get("targets")
        if isinstance(targets, dict) and targets:
            records = []
            for role in ("previous", "current"):
                target = targets.get(role)
                if not isinstance(target, dict):
                    continue
                record = dict(target)
                record.setdefault("target_role", role)
                if "pair_name" not in record and record.get("pair_dir"):
                    record["pair_name"] = Path(record["pair_dir"]).name
                if "pair_dir" not in record and record.get("pair_name"):
                    record["pair_dir"] = str(Path(self.rigidmask_output_dir) / record["pair_name"])
                if "target_frame_id" in record:
                    record["target_frame_id"] = str(record["target_frame_id"])
                if "counterpart_frame_id" in record:
                    record["counterpart_frame_id"] = str(record["counterpart_frame_id"])
                if "target_time_idx" in record and record["target_time_idx"] is not None:
                    record["target_time_idx"] = int(record["target_time_idx"])
                records.append(record)
            if records:
                return records

        time_idx = rigidmask_summary.get("time_idx", rigidmask_summary.get("target_time_idx"))
        curr_frame_id = rigidmask_summary.get("target_frame_id", rigidmask_summary.get("curr_frame_id"))
        reference_frame_id = rigidmask_summary.get(
            "reference_frame_id",
            rigidmask_summary.get("counterpart_frame_id"),
        )
        if time_idx is None or curr_frame_id is None or reference_frame_id is None:
            pair_dir = rigidmask_summary.get("pair_dir")
            pair_name = rigidmask_summary.get("pair_name", Path(pair_dir).name if pair_dir else "")
            return [
                {
                    "target_role": rigidmask_summary.get("target_role", "current"),
                    "target_time_idx": int(time_idx) if time_idx is not None else None,
                    "target_frame_id": str(curr_frame_id) if curr_frame_id is not None else "",
                    "counterpart_frame_id": str(reference_frame_id) if reference_frame_id is not None else "",
                    "pair_name": pair_name,
                    "pair_dir": pair_dir or str(Path(self.rigidmask_output_dir) / pair_name),
                }
            ]
        return self._target_records_for_temporal_pair(
            int(time_idx),
            str(curr_frame_id),
            str(reference_frame_id),
        )

    def _target_records_for_temporal_pair(self, time_idx, curr_frame_id, reference_frame_id):
        current_pair_name = f"{int(time_idx):06d}_frame_{curr_frame_id}_from_{reference_frame_id}"
        previous_pair_name = f"{int(time_idx):06d}_frame_{reference_frame_id}_to_{curr_frame_id}_target_prev"
        return [
            {
                "target_role": "previous",
                "target_time_idx": int(time_idx) - 1,
                "target_frame_id": str(reference_frame_id),
                "counterpart_frame_id": str(curr_frame_id),
                "pair_name": previous_pair_name,
                "pair_dir": str(Path(self.rigidmask_output_dir) / previous_pair_name),
            },
            {
                "target_role": "current",
                "target_time_idx": int(time_idx),
                "target_frame_id": str(curr_frame_id),
                "counterpart_frame_id": str(reference_frame_id),
                "pair_name": current_pair_name,
                "pair_dir": str(Path(self.rigidmask_output_dir) / current_pair_name),
            },
        ]

    def _process_target_pair_dirs(self, target_records, params=None):
        results = {}
        for target in target_records:
            role = str(target.get("target_role", "current"))
            pair_dir = Path(target["pair_dir"])
            fusion_summary = self._process_pair_dir(pair_dir, params=params, target_record=target)
            mask_path = find_pair_file(pair_dir, "dynamic_mask.png")
            summary_path = find_pair_file(pair_dir, "dynamic_fusion_summary.json")
            results[role] = {
                "fusion_ok": fusion_summary is not None and mask_path.exists(),
                "pair_name": target.get("pair_name", pair_dir.name),
                "pair_dir": str(pair_dir),
                "target_time_idx": target.get("target_time_idx"),
                "target_frame_id": target.get("target_frame_id"),
                "dynamic_mask_path": str(mask_path) if mask_path.exists() else "",
                "dynamic_fusion_summary_path": str(summary_path) if summary_path.exists() else "",
            }
            if fusion_summary is None:
                results[role]["reason"] = "fusion_returned_none"
        return results

    def _compact_storage_after_fusion(self, target_records, target_results, time_idx, num_frames):
        if not self.minimal_storage:
            return
        for target in target_records:
            role = str(target.get("target_role", "current"))
            if target_results.get(role, {}).get("fusion_ok", False):
                self._compact_pair_dir(Path(target["pair_dir"]))
        if target_results and all(item.get("fusion_ok", False) for item in target_results.values()):
            self._compact_lidar_dirs(target_records)
            self._compact_appearance_dirs(time_idx, num_frames)

    def _compact_pair_dir(self, pair_dir):
        if not pair_dir.is_dir():
            return
        for path in list(pair_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.name in MINIMAL_PAIR_KEEP_FILES:
                continue
            self._remove_path(path)
        self._remove_empty_dirs(pair_dir)

    def _compact_lidar_dirs(self, target_records):
        lidar_root = Path(self.lidar_output_dir)
        if not lidar_root.is_dir():
            return
        pair_names = set()
        for target in target_records:
            role = str(target.get("target_role", "current")).strip().lower()
            target_frame_id = target.get("target_frame_id")
            counterpart_frame_id = target.get("counterpart_frame_id")
            if target_frame_id is None or counterpart_frame_id is None:
                continue
            if role in ("current", "curr", "t"):
                pair_names.add(f"{counterpart_frame_id}_{target_frame_id}")
            else:
                pair_names.add(f"{target_frame_id}_{counterpart_frame_id}")
        for pair_name in pair_names:
            self._remove_path(lidar_root / pair_name)

    def _compact_appearance_dirs(self, time_idx, num_frames):
        appearance_root = Path(self.appearance_output_dir)
        if not appearance_root.is_dir():
            return
        keep_from_time_idx = int(time_idx)
        if int(time_idx) >= int(num_frames) - 1:
            keep_from_time_idx = int(time_idx) + 1
        for frame_dir in appearance_root.iterdir():
            if not frame_dir.is_dir():
                continue
            try:
                frame_time_idx = int(frame_dir.name.split("_", 1)[0])
            except (IndexError, ValueError):
                continue
            if frame_time_idx < keep_from_time_idx:
                self._remove_path(frame_dir)

    @staticmethod
    def _remove_path(path):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Dynamic mask minimal-storage cleanup skipped {path}: {exc}")

    @staticmethod
    def _remove_empty_dirs(root):
        for path in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass

    def _save_appearance_for_frame(self, time_idx, frame_id, params, curr_data, render_pair):
        appearance = self._ensure_appearance()
        if appearance is None:
            return {
                "status": "skipped",
                "reason": "appearance_backend_unavailable",
                "time_idx": int(time_idx),
                "frame_id": str(frame_id),
            }
        if not appearance.should_run(time_idx):
            return {
                "status": "skipped",
                "reason": "appearance_not_scheduled",
                "time_idx": int(time_idx),
                "frame_id": str(frame_id),
            }

        os.makedirs(self.appearance_output_dir, exist_ok=True)
        if render_pair is not None and "gt_image" in render_pair and "render_image" in render_pair:
            gt_image = render_pair["gt_image"].detach()
            render_image = render_pair["render_image"].detach()
        else:
            gt_image = curr_data["im"].detach()
            render_image = self._render_current_rgb(params, curr_data, time_idx)
            sky_mask = curr_data.get("sky_mask")
            if sky_mask is not None:
                sky_mask_t = sky_mask.detach().cpu() if torch.is_tensor(sky_mask) else torch.as_tensor(sky_mask)
                sky_mask_t = sky_mask_t.bool()
                if sky_mask_t.dim() == 2:
                    sky_mask_t = sky_mask_t.unsqueeze(0)
                if sky_mask_t.dim() == 3 and sky_mask_t.shape[0] != 1:
                    sky_mask_t = sky_mask_t[:1]
                keep_mask = (~sky_mask_t).repeat(3, 1, 1)
                gt_image = gt_image.detach().cpu() * keep_mask
                render_image = render_image.detach().cpu() * keep_mask

        return appearance.save_pair(
            gt_image=gt_image,
            render_image=render_image,
            output_root=self.appearance_output_dir,
            time_idx=time_idx,
            frame_id=frame_id,
        )

    def _save_fastsam_for_targets(self, target_records):
        fastsam_cfg = self.cfg["fastsam"]
        if not fastsam_cfg.get("enabled", False):
            return {}
        if str(fastsam_cfg.get("mode", "online")).lower() == "precomputed":
            return self._load_precomputed_fastsam_for_targets(target_records)

        fastsam = self._ensure_fastsam()
        if fastsam is None:
            return {}
        results = {}
        for target in target_records:
            role = str(target.get("target_role", "current"))
            time_idx = target.get("target_time_idx")
            if time_idx is not None and not fastsam.should_run(time_idx):
                results[role] = {
                    "status": "skipped",
                    "reason": "fastsam_not_scheduled",
                    "target_time_idx": time_idx,
                }
                continue
            try:
                results[role] = fastsam.save_for_pair_dir(
                    target["pair_dir"],
                    time_idx=time_idx,
                    frame_id=target.get("target_frame_id"),
                    force=bool(self.cfg["fastsam"].get("overwrite", False)),
                )
            except Exception as exc:
                if self.fail_on_error:
                    raise
                results[role] = {
                    "status": "error",
                    "reason": str(exc),
                    "pair_dir": target.get("pair_dir", ""),
                    "target_time_idx": time_idx,
                }
        return results

    def _load_precomputed_fastsam_for_targets(self, target_records):
        cfg = self.cfg["fastsam"]
        sequence_root = cfg.get("precomputed_root") or getattr(
            self.dataset,
            "input_folder",
            "",
        )
        output_subdir = cfg.get("precomputed_subdir", "fastsam_masks")
        results = {}
        for target in target_records:
            role = str(target.get("target_role", "current"))
            frame_id = str(target.get("target_frame_id", ""))
            time_idx = target.get("target_time_idx")
            if time_idx is not None and int(time_idx) % int(cfg.get("run_every", 1)) != 0:
                results[role] = {
                    "status": "skipped",
                    "reason": "fastsam_not_scheduled",
                    "target_time_idx": time_idx,
                }
                continue
            if not sequence_root or not frame_id:
                results[role] = {
                    "status": "missing",
                    "reason": "missing_sequence_root_or_frame_id",
                }
                continue

            masks_path = Path(sequence_root) / output_subdir / f"{frame_id}.npz"
            summary_path = Path(sequence_root) / output_subdir / f"{frame_id}.json"
            if not masks_path.is_file() or (
                cfg.get("require_precomputed", False) and not summary_path.is_file()
            ):
                results[role] = {
                    "status": "missing",
                    "reason": "missing_precomputed_fastsam",
                    "masks_path": str(masks_path),
                    "summary_path": str(summary_path),
                }
                continue

            target["fastsam_masks_path"] = str(masks_path)
            target["fastsam_summary_path"] = str(summary_path) if summary_path.is_file() else ""
            results[role] = {
                "status": "ok",
                "reason": "precomputed",
                "masks_path": str(masks_path),
                "summary_path": str(summary_path) if summary_path.is_file() else "",
                "target_time_idx": time_idx,
                "target_frame_id": frame_id,
            }
        return results

    @staticmethod
    def _has_missing_required_fastsam(fastsam_results):
        if not fastsam_results:
            return True
        return any(result.get("status") != "ok" for result in fastsam_results.values())

    @staticmethod
    def _targets_status(results):
        if not results:
            return "not_run"
        statuses = {result.get("status", "unknown") for result in results.values()}
        if statuses == {"ok"}:
            return "ok"
        if "ok" in statuses:
            return "partial"
        if "error" in statuses:
            return "error"
        return sorted(statuses)[0]

    def _appearance_available(self, time_idx, frame_id):
        frame_name = f"{int(time_idx):06d}_frame_{frame_id}"
        frame_dir = Path(self.appearance_output_dir) / frame_name
        return (frame_dir / "similarity.npy").exists() or (frame_dir / "similarity.png").exists()

    def _missing_required_lidar_targets(self, target_records):
        missing = []
        for target in target_records:
            pair_dir = Path(target["pair_dir"])
            summary_path = find_pair_file(pair_dir, "rigidmask_frontend_summary.json")
            role = target.get("target_role", "current")
            if not summary_path.exists():
                missing.append({"target_role": role, "status": "missing_rigidmask_summary"})
                continue
            with open(summary_path, "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            lidar_projection, lidar_load_summary = load_lidar_projection_for_pair(
                pair_dir,
                summary,
                self.cfg["lidar_residual"].get("output_subdir", "lidar_motion_probe"),
                self.cfg["fusion"]["lidar_projection_filename"],
            )
            if lidar_projection is None:
                missing.append(
                    {
                        "target_role": role,
                        "status": lidar_load_summary.get("status", "missing")
                        if lidar_load_summary
                        else "missing",
                    }
                )
        return missing

    def _write_config_snapshot(self, data_cfg):
        path = os.path.join(self.root_dir, "dynamic_mask_config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.cfg, handle, indent=2, default=str)

    def _write_run_summary(self):
        path = os.path.join(self.root_dir, "dynamic_mask_run_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._summary, handle, indent=2)

    def load_dynamic_mask(self, record, image_height, image_width):
        if not record or record.get("status") not in ("ok", "partial") or "pair_dir" not in record:
            return None
        mask_path = find_pair_file(Path(record["pair_dir"]), "dynamic_mask.png")
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (image_height, image_width):
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask > 0).to(device=self.device)
        return mask.unsqueeze(0)

    def load_dynamic_mask_for_time_idx(self, time_idx, num_frames, image_height, image_width, target_role="current"):
        if not self.enabled or time_idx <= 0:
            return None
        curr_frame_id = get_dataset_frame_id(self.dataset, time_idx)
        reference_frame_id = get_dataset_frame_id(self.dataset, time_idx - 1)
        role = str(target_role).strip().lower()
        if role in ("current", "curr", "t"):
            pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_from_{reference_frame_id}"
        elif role in ("previous", "prev", "reference", "t-1"):
            pair_name = f"{time_idx:06d}_frame_{reference_frame_id}_to_{curr_frame_id}_target_prev"
        else:
            raise ValueError(f"Unsupported dynamic-mask target_role: {target_role}")
        pair_dir = Path(self.rigidmask_output_dir) / pair_name
        return self.load_dynamic_mask(
            {"status": "ok", "pair_dir": str(pair_dir)},
            image_height,
            image_width,
        )

    @staticmethod
    def _status(summary):
        if summary is None:
            return "not_run"
        return summary.get("status", "unknown")

import json
import os
from pathlib import Path

import cv2
import torch

from utils.pnp_fused_icp_utils import get_dataset_frame_id

from .appearance import AppearanceSimilarityProbe
from .fusion import find_pair_file, load_lidar_projection_for_pair, process_pair_dir
from .lidar_motion import LidarMotionProbe
from .rigidmask_frontend import RigidMaskFrontendProbe


DEFAULT_DYNAMIC_MASK_CFG = {
    "enabled": False,
    "output_subdir": "dynamic_mask",
    "fail_on_error": False,
    "require_lidar_residual": False,
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
        "save_diagnostics": False,
        "organize_outputs": True,
    },
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


class DynamicMaskManager:
    def __init__(self, cfg, data_cfg, dataset, output_dir, project_root, device):
        self.cfg = _deep_merge(DEFAULT_DYNAMIC_MASK_CFG, cfg or {})
        self.enabled = bool(self.cfg.get("enabled", False))
        self.fail_on_error = bool(self.cfg.get("fail_on_error", False))
        self.require_lidar_residual = bool(self.cfg.get("require_lidar_residual", False))
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
        self.lidar_residual = None
        self._summary = []
        self._init_error = None

        if self.enabled:
            os.makedirs(self.root_dir, exist_ok=True)
            self._write_config_snapshot(data_cfg)

    def run_for_frame(self, time_idx, num_frames, params, sky_mask=None):
        if not self.enabled:
            return None
        record = {
            "time_idx": int(time_idx),
            "status": "skipped",
            "rigidmask_status": "not_run",
            "lidar_residual_status": "not_run",
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
        if time_idx >= num_frames - 1:
            return {
                "status": "skipped",
                "reason": "last_frame_has_no_next_rgb_pair",
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

        next_frame_id = get_dataset_frame_id(self.dataset, time_idx + 1)
        os.makedirs(self.rigidmask_output_dir, exist_ok=True)
        rigidmask_summary = rigidmask.save_pair(
            output_root=self.rigidmask_output_dir,
            time_idx=time_idx,
            curr_frame_id=curr_frame_id,
            next_frame_id=next_frame_id,
            sky_mask=sky_mask,
        )
        if rigidmask_summary.get("status") != "ok":
            return {
                "status": "skipped",
                "reason": rigidmask_summary.get("reason", "rigidmask_skipped"),
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }

        pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_to_{next_frame_id}"
        pair_dir = Path(self.rigidmask_output_dir) / pair_name
        lidar_ok = lidar_summary is not None and lidar_summary.get("status") == "ok"
        if self.require_lidar_residual and not lidar_ok:
            return {
                "status": "skipped",
                "reason": "missing_required_lidar_residual",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }
        if not find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz").exists():
            return {
                "status": "skipped",
                "reason": "missing_rigidmask_arrays",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
            }
        if not self.cfg["fusion"].get("enabled", True):
            return {
                "status": "skipped",
                "reason": "fusion_disabled",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fusion_status": "disabled",
            }

        if self.cfg["appearance"].get("enabled", True):
            return {
                "status": "pending_appearance",
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fusion_status": "deferred",
            }

        fusion_summary = self._process_pair_dir(pair_dir)
        if fusion_summary is None:
            return {
                "status": "skipped",
                "reason": "fusion_returned_none",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "rigidmask_status": self._status(rigidmask_summary),
                "lidar_residual_status": self._status(lidar_summary),
                "fusion_status": "skipped",
            }

        return {
            "status": "ok",
            "pair_name": pair_name,
            "pair_dir": str(pair_dir),
            "curr_frame_id": curr_frame_id,
            "next_frame_id": next_frame_id,
            "rigidmask_status": self._status(rigidmask_summary),
            "lidar_residual_status": self._status(lidar_summary),
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
        if time_idx >= num_frames - 1:
            return {
                "status": "skipped",
                "reason": "last_frame_has_no_next_rgb_pair",
                "curr_frame_id": curr_frame_id,
            }
        next_frame_id = get_dataset_frame_id(self.dataset, time_idx + 1)
        pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_to_{next_frame_id}"
        pair_dir = Path(self.rigidmask_output_dir) / pair_name
        if not find_pair_file(pair_dir, "rigidmask_frontend_arrays.npz").exists():
            return {
                "status": "skipped",
                "reason": "missing_rigidmask_arrays",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
            }
        summary_path = find_pair_file(pair_dir, "rigidmask_frontend_summary.json")
        if not summary_path.exists():
            return {
                "status": "skipped",
                "reason": "missing_rigidmask_summary",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
            }
        if self.require_lidar_residual:
            with open(summary_path, "r", encoding="utf-8") as handle:
                rigidmask_summary = json.load(handle)
            lidar_projection, lidar_load_summary = load_lidar_projection_for_pair(
                pair_dir,
                rigidmask_summary,
                self.cfg["lidar_residual"].get("output_subdir", "lidar_motion_probe"),
                self.cfg["fusion"]["lidar_projection_filename"],
            )
            if lidar_projection is None:
                return {
                    "status": "skipped",
                    "reason": "missing_required_lidar_residual",
                    "curr_frame_id": curr_frame_id,
                    "next_frame_id": next_frame_id,
                    "pair_name": pair_name,
                    "pair_dir": str(pair_dir),
                    "lidar_residual_status": lidar_load_summary.get("status", "missing"),
                }

        appearance_summary = None
        if self.cfg["appearance"].get("enabled", True):
            appearance = self._ensure_appearance()
            if appearance is not None and appearance.should_run(time_idx):
                os.makedirs(self.appearance_output_dir, exist_ok=True)
                if render_pair is not None and "gt_image" in render_pair and "render_image" in render_pair:
                    gt_image = render_pair["gt_image"].detach()
                    render_image = render_pair["render_image"].detach()
                else:
                    gt_image = curr_data["im"].detach()
                    render_image = self._render_current_rgb(params, curr_data, time_idx)
                    sky_mask = curr_data.get("sky_mask")
                    if sky_mask is not None:
                        keep_mask = (~sky_mask).repeat(3, 1, 1).detach().cpu()
                        gt_image = gt_image.detach().cpu() * keep_mask
                        render_image = render_image.detach().cpu() * keep_mask
                appearance_summary = appearance.save_pair(
                    gt_image=gt_image,
                    render_image=render_image,
                    output_root=self.appearance_output_dir,
                    time_idx=time_idx,
                    frame_id=curr_frame_id,
                )

        appearance_ok = appearance_summary is not None and appearance_summary.get("status") == "ok"
        if self.cfg["appearance"].get("require_appearance", True) and not appearance_ok:
            return {
                "status": "skipped",
                "reason": "missing_required_appearance_similarity",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "appearance_status": self._status(appearance_summary),
            }
        if not self.cfg["fusion"].get("enabled", True):
            return {
                "status": "skipped",
                "reason": "fusion_disabled",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "appearance_status": self._status(appearance_summary),
                "fusion_status": "disabled",
            }

        fusion_summary = self._process_pair_dir(pair_dir)
        if fusion_summary is None:
            return {
                "status": "skipped",
                "reason": "fusion_returned_none",
                "curr_frame_id": curr_frame_id,
                "next_frame_id": next_frame_id,
                "pair_name": pair_name,
                "pair_dir": str(pair_dir),
                "appearance_status": self._status(appearance_summary),
                "fusion_status": "skipped",
            }
        return {
            "status": "ok",
            "curr_frame_id": curr_frame_id,
            "next_frame_id": next_frame_id,
            "pair_name": pair_name,
            "pair_dir": str(pair_dir),
            "appearance_status": self._status(appearance_summary),
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

    def _process_pair_dir(self, pair_dir):
        fusion_cfg = self.cfg["fusion"]
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
            save_diagnostics=fusion_cfg.get("save_diagnostics", False),
            organize_outputs=fusion_cfg.get("organize_outputs", True),
        )

    def _write_config_snapshot(self, data_cfg):
        path = os.path.join(self.root_dir, "dynamic_mask_config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.cfg, handle, indent=2, default=str)

    def _write_run_summary(self):
        path = os.path.join(self.root_dir, "dynamic_mask_run_summary.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._summary, handle, indent=2)

    def load_dynamic_mask(self, record, image_height, image_width):
        if not self.enabled or not record:
            return None
        pair_dir = record.get("pair_dir")
        if not pair_dir:
            return None
        mask_path = find_pair_file(Path(pair_dir), "dynamic_mask.png")
        if not mask_path.exists():
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != (image_height, image_width):
            mask = cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(mask > 127).to(device=self.device).unsqueeze(0)

    def load_dynamic_mask_for_frame(self, time_idx, num_frames, image_height, image_width):
        if not self.enabled or time_idx >= num_frames - 1:
            return None
        curr_frame_id = get_dataset_frame_id(self.dataset, time_idx)
        next_frame_id = get_dataset_frame_id(self.dataset, time_idx + 1)
        pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_to_{next_frame_id}"
        record = {"pair_dir": str(Path(self.rigidmask_output_dir) / pair_name)}
        return self.load_dynamic_mask(record, image_height, image_width)

    @staticmethod
    def _status(summary):
        if summary is None:
            return "not_run"
        return summary.get("status", "unknown")

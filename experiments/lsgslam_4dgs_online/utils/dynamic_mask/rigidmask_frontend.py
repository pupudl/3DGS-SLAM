import importlib.util
import json
import os
import sys
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_perspective_file(calibration_path):
    values = {}
    with open(calibration_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            tokens = []
            for token in raw.strip().split():
                try:
                    tokens.append(float(token))
                except ValueError:
                    break
            values[key.strip()] = np.asarray(tokens, dtype=np.float64)
    p_rect_00 = values["P_rect_00"].reshape(3, 4)
    p_rect_01 = values["P_rect_01"].reshape(3, 4)
    fx = float(p_rect_00[0, 0])
    fy = float(p_rect_00[1, 1])
    cx = float(p_rect_00[0, 2])
    cy = float(p_rect_00[1, 2])
    tx0 = float(p_rect_00[0, 3] / p_rect_00[0, 0])
    tx1 = float(p_rect_01[0, 3] / p_rect_01[0, 0])
    baseline = abs(tx1 - tx0)
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
    }


def _install_rigidmask_det_stub():
    if "models.det" in sys.modules:
        return
    stub = types.ModuleType("models.det")
    stub.create_model = lambda *_args, **_kwargs: nn.Identity()
    stub.load_model = lambda model, *_args, **_kwargs: model
    stub.save_model = lambda *_args, **_kwargs: None
    sys.modules["models.det"] = stub


class _DummyMiDaS(nn.Module):
    def forward(self, _x):
        raise RuntimeError("Stereo probe should not execute the MiDaS branch.")


def _strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def _maybe_add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def _resize_to_rigidmask_shape(image, testres):
    maxh = image.shape[0] * testres
    maxw = image.shape[1] * testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh:
        max_h += 64
    if max_w < maxw:
        max_w += 64
    return max_w, max_h


def _reconfigure_runtime_modules(model, flow_reg_cls, warp_module_cls, max_w, max_h):
    reg_scales = [64, 32, 16, 8, 4]
    for i, reg_scale in enumerate(reg_scales):
        template = getattr(model, f"flow_reg{reg_scale}")
        model.reg_modules[i] = flow_reg_cls(
            [1, max_w // reg_scale, max_h // reg_scale],
            ent=template.ent,
            maxdisp=template.md,
            fac=template.fac,
        ).cuda()
    for i in range(len(model.warp_modules)):
        warp_scale = 2 ** (6 - i)
        model.warp_modules[i] = warp_module_cls(
            [1, max_w // warp_scale, max_h // warp_scale]
        ).cuda()


def _build_intrinsics_list(calib, input_size, max_w, max_h, sensor):
    baseline = calib["baseline"] if sensor == "stereo" else 1.0
    return [
        torch.tensor([calib["fx"]], device="cuda"),
        torch.tensor([calib["cx"]], device="cuda"),
        torch.tensor([calib["cy"]], device="cuda"),
        torch.tensor([baseline], device="cuda"),
        torch.tensor([1.0], device="cuda"),
        torch.tensor([0.0], device="cuda"),
        torch.tensor([0.0], device="cuda"),
        torch.tensor([1.0], device="cuda"),
        torch.tensor([0.0], device="cuda"),
        torch.tensor([0.0], device="cuda"),
        torch.tensor([input_size[1] / max_w], device="cuda"),
        torch.tensor([input_size[0] / max_h], device="cuda"),
        torch.tensor([calib["fx"]], device="cuda"),
    ]


def _prepare_pair_tensors(prev_rgb, curr_rgb, mean_L, mean_R):
    imgL_noaug = torch.tensor(prev_rgb / 255.0, device="cuda")[None].float()
    imgL = prev_rgb[:, :, ::-1].copy() / 255.0 - np.asarray(mean_L).mean(0)[None, None, :]
    imgR = curr_rgb[:, :, ::-1].copy() / 255.0 - np.asarray(mean_R).mean(0)[None, None, :]
    imgL = torch.tensor(np.transpose(imgL, [2, 0, 1])[None], device="cuda").float()
    imgR = torch.tensor(np.transpose(imgR, [2, 0, 1])[None], device="cuda").float()
    return imgL, imgR, imgL_noaug


def _normalize_for_vis(array):
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[finite]
    lo = np.percentile(values, 1.0)
    hi = np.percentile(values, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    scaled[~finite] = 0.0
    return (255.0 * scaled).astype(np.uint8)


def _compute_metric_depth_from_disp(disp_t, calib, min_disp=1e-6):
    disp_t = disp_t.float()
    depth_t = torch.full_like(disp_t, float("nan"))
    valid_disp_mask = torch.isfinite(disp_t) & (disp_t > float(min_disp))
    depth_scale = float(calib["fx"] * calib["baseline"])
    if depth_scale <= 0:
        raise ValueError("Calibration must provide positive fx and baseline for stereo depth masking.")
    depth_t[valid_disp_mask] = depth_scale / disp_t[valid_disp_mask]
    return depth_t, valid_disp_mask


def _build_depth_mask_from_disp(disp_t, calib, min_depth_m, max_depth_m, min_disp=1e-6):
    depth_t, valid_disp_mask = _compute_metric_depth_from_disp(disp_t, calib, min_disp=min_disp)
    depth_mask = valid_disp_mask.clone()
    if min_depth_m is not None:
        depth_mask = depth_mask & (depth_t >= float(min_depth_m))
    if max_depth_m is not None:
        depth_mask = depth_mask & (depth_t <= float(max_depth_m))
    return depth_mask, depth_t


def _resize_bool_mask(mask, width, height):
    if isinstance(mask, torch.Tensor):
        mask_u8 = mask.detach().cpu().numpy().astype(np.uint8)
    else:
        mask_u8 = np.asarray(mask).astype(np.uint8)
    if mask_u8.ndim == 3 and mask_u8.shape[0] == 1:
        mask_u8 = mask_u8[0]
    resized = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _apply_numpy_mask(array, mask):
    masked = array.copy()
    masked[~mask] = np.nan
    return masked


def _resize_flow_to_shape(flow_x, flow_y, target_shape):
    target_h, target_w = target_shape
    src_h, src_w = flow_x.shape
    if (src_h, src_w) == (target_h, target_w):
        return flow_x.astype(np.float32), flow_y.astype(np.float32)
    resized_x = cv2.resize(flow_x, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_y = cv2.resize(flow_y, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_x = resized_x * (float(target_w) / max(float(src_w), 1.0))
    resized_y = resized_y * (float(target_h) / max(float(src_h), 1.0))
    return resized_x.astype(np.float32), resized_y.astype(np.float32)


def _build_forward_splat(flow_x, flow_y, target_shape):
    src_h, src_w = flow_x.shape
    target_h, target_w = target_shape
    yy, xx = np.mgrid[0:src_h, 0:src_w].astype(np.float32)
    dst_x = xx + flow_x.astype(np.float32)
    dst_y = yy + flow_y.astype(np.float32)
    valid = (
        np.isfinite(dst_x)
        & np.isfinite(dst_y)
        & (dst_x >= 0.0)
        & (dst_x <= float(target_w - 1))
        & (dst_y >= 0.0)
        & (dst_y <= float(target_h - 1))
    )
    valid_src = np.flatnonzero(valid.reshape(-1))
    dst_x = dst_x.reshape(-1)[valid_src]
    dst_y = dst_y.reshape(-1)[valid_src]
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    dx = dst_x - x0
    dy = dst_y - y0

    src_indices = []
    dst_indices = []
    weights = []
    for ox, oy, weight in (
        (0, 0, (1.0 - dx) * (1.0 - dy)),
        (1, 0, dx * (1.0 - dy)),
        (0, 1, (1.0 - dx) * dy),
        (1, 1, dx * dy),
    ):
        xi = x0 + ox
        yi = y0 + oy
        inside = (xi >= 0) & (xi < target_w) & (yi >= 0) & (yi < target_h) & (weight > 0.0)
        src_indices.append(valid_src[inside])
        dst_indices.append((yi[inside] * target_w + xi[inside]).astype(np.int64))
        weights.append(weight[inside].astype(np.float32))

    coverage = np.zeros(target_h * target_w, dtype=np.float32)
    for dst_idx, weight in zip(dst_indices, weights):
        np.add.at(coverage, dst_idx, weight)

    return {
        "source_shape": (src_h, src_w),
        "target_shape": (target_h, target_w),
        "source_indices": src_indices,
        "target_indices": dst_indices,
        "weights": weights,
        "coverage": coverage.reshape(target_h, target_w),
    }


def _forward_splat_array(array, splat):
    flat_src = array.astype(np.float32).reshape(-1)
    target_h, target_w = splat["target_shape"]
    out = np.zeros(target_h * target_w, dtype=np.float32)
    denom = np.zeros(target_h * target_w, dtype=np.float32)
    for src_idx, dst_idx, weight in zip(
        splat["source_indices"],
        splat["target_indices"],
        splat["weights"],
    ):
        values = flat_src[src_idx]
        finite = np.isfinite(values) & np.isfinite(weight)
        if not np.any(finite):
            continue
        np.add.at(out, dst_idx[finite], values[finite] * weight[finite])
        np.add.at(denom, dst_idx[finite], weight[finite])

    result = np.full(target_h * target_w, np.nan, dtype=np.float32)
    covered = denom > 1e-6
    result[covered] = out[covered] / denom[covered]
    return result.reshape(target_h, target_w)


def _warp_raw_arrays_forward_to_target(raw_arrays):
    flow_x_full = raw_arrays["flow_full_x"].astype(np.float32)
    flow_y_full = raw_arrays["flow_full_y"].astype(np.float32)
    full_shape = flow_x_full.shape
    cost_shape = raw_arrays["homography_cost"].shape

    flow_x_cost, flow_y_cost = _resize_flow_to_shape(flow_x_full, flow_y_full, cost_shape)
    cost_splat = _build_forward_splat(flow_x_cost, flow_y_cost, cost_shape)
    full_splat = _build_forward_splat(flow_x_full, flow_y_full, full_shape)

    warped = {}
    for key, value in raw_arrays.items():
        if value.ndim != 2:
            warped[key] = value
        elif value.shape == cost_shape:
            warped[key] = _forward_splat_array(value, cost_splat)
        elif value.shape == full_shape:
            warped[key] = _forward_splat_array(value, full_splat)
        else:
            warped[key] = value

    warped["warp_coverage_cost_grid"] = cost_splat["coverage"].astype(np.float32)
    warped["warp_coverage_full"] = full_splat["coverage"].astype(np.float32)
    return warped, {
        "enabled": True,
        "method": "forward_splat_bilinear",
        "cost_grid_coverage": float(np.mean(cost_splat["coverage"] > 1e-6)),
        "full_coverage": float(np.mean(full_splat["coverage"] > 1e-6)),
    }


def _normalize_depth_mask_apply_stage(depth_mask_cfg):
    stage = str(depth_mask_cfg.get("apply_stage", "pre_fusion")).strip().lower()
    aliases = {
        "pre_fusion": "pre_fusion",
        "pre_cost_fusion": "pre_fusion",
        "pre_score": "pre_fusion",
        "post_dynamic_mask": "post_dynamic_mask",
        "post_mask": "post_dynamic_mask",
        "final_mask_only": "post_dynamic_mask",
    }
    if stage not in aliases:
        raise ValueError(
            "Unsupported rigidmask_frontend_probe.depth_mask.apply_stage "
            f"'{depth_mask_cfg.get('apply_stage')}'."
        )
    return aliases[stage]


class RigidMaskFrontendProbe:
    def __init__(self, probe_cfg, data_cfg):
        self.cfg = probe_cfg
        self.data_cfg = data_cfg
        self.sequence = data_cfg["sequence"]
        self.sequence_dir = os.path.join(data_cfg["basedir"], self.sequence)
        self.image_dir = os.path.join(self.sequence_dir, "image_00", "data_rect")
        self.disp_dir = os.path.join(self.sequence_dir, probe_cfg.get("disparity_dir", "disparity_sceneflow"))
        self.calib = _parse_perspective_file(probe_cfg["calibration_path"])
        self.run_every = int(probe_cfg.get("run_every", 1))
        self.lazy_load = bool(probe_cfg.get("lazy_load", True))
        self.offload_after_use = bool(probe_cfg.get("offload_after_use", True))
        self.depth_mask_cfg = probe_cfg.get("depth_mask", {})
        self.model_bundle = None

        self._preflight()
        if not self.lazy_load:
            self.model_bundle = self._load_model_bundle()

    def _preflight(self):
        if self.cfg.get("require_cuda", True) and not torch.cuda.is_available():
            raise RuntimeError("RigidMask frontend probe currently requires CUDA.")
        checkpoint_path = self.cfg["checkpoint_path"]
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"RigidMask checkpoint not found: {checkpoint_path}")
        if self.cfg["sensor"] == "stereo" and not os.path.isdir(self.disp_dir):
            raise FileNotFoundError(f"Stereo probe requires disparity dir: {self.disp_dir}")
        if not self.cfg.get("use_opencv_essential_mat", False):
            _maybe_add_path(os.path.join(os.path.abspath(self.cfg["repo_root"]), "models", "ngransac"))
            if importlib.util.find_spec("ngransac") is None:
                raise ModuleNotFoundError(
                    "RigidMask frontend probe requires `ngransac`, but it is not importable."
                )
        if self.depth_mask_cfg.get("enabled", False) and self.cfg["sensor"] != "stereo":
            raise ValueError("Depth masking for rigidmask_frontend_probe currently supports stereo mode only.")

    def _load_model_bundle(self):
        rigidmask_root = os.path.abspath(self.cfg["repo_root"])
        _maybe_add_path(rigidmask_root)
        _maybe_add_path(os.path.join(rigidmask_root, "models", "ngransac"))
        _install_rigidmask_det_stub()

        original_hub_load = torch.hub.load
        if self.cfg["sensor"] == "stereo":
            torch.hub.load = lambda *_args, **_kwargs: _DummyMiDaS()
        try:
            from models.VCNplus import VCN, WarpModule, flow_reg, get_grid
            from models.submodule import F_ngransac, compute_geo_costs, get_intrinsics

            checkpoint_path = self.cfg["checkpoint_path"]
            exp_unc = "kitti" not in checkpoint_path.lower()
            model = VCN(
                [1, 1408, 384],
                md=[int(4 * (self.cfg["maxdisp"] / 256)), 4, 4, 4, 4],
                fac=self.cfg["fac"],
                exp_unc=exp_unc,
            )
        finally:
            torch.hub.load = original_hub_load

        checkpoint = torch.load(self.cfg["checkpoint_path"], map_location="cpu")
        model.load_state_dict(_strip_module_prefix(checkpoint["state_dict"]), strict=False)
        model = model.eval()
        return {
            "model": model,
            "flow_reg_cls": flow_reg,
            "warp_module_cls": WarpModule,
            "F_ngransac": F_ngransac,
            "compute_geo_costs": compute_geo_costs,
            "get_intrinsics": get_intrinsics,
            "get_grid": get_grid,
            "mean_L": checkpoint["mean_L"],
            "mean_R": checkpoint["mean_R"],
        }

    def _ensure_model_bundle(self):
        if self.model_bundle is None:
            self.model_bundle = self._load_model_bundle()
        model = self.model_bundle["model"]
        if next(model.parameters()).device.type != "cuda":
            self.model_bundle["model"] = model.cuda().eval()
        return self.model_bundle

    def should_run(self, time_idx, num_frames):
        if not self.cfg.get("enabled", False):
            return False
        if time_idx <= 0:
            return False
        return (time_idx % self.run_every) == 0

    def _load_disp_input(self, frame_id):
        npy_path = os.path.join(self.disp_dir, f"{int(frame_id):010d}.npy")
        png_path = os.path.join(self.disp_dir, f"{int(frame_id):010d}.png")
        if os.path.exists(npy_path):
            disp = np.load(npy_path).astype(np.float32)
        elif os.path.exists(png_path):
            disp = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
            if disp is None:
                raise FileNotFoundError(f"Failed to read disparity file: {png_path}")
            disp = disp.astype(np.float32)
        else:
            raise FileNotFoundError(f"Missing disparity prior for frame {frame_id} in {self.disp_dir}")
        return torch.tensor(disp, device="cuda")[None, None].float()

    def _build_raw_arrays(self, results):
        return {
            "homography_cost": results["homography_cost"][0, 0].detach().cpu().numpy(),
            "epipolar_cost": results["epipolar_cost"][0, 0].detach().cpu().numpy(),
            "pp2d_cost": results["pp2d_cost"][0, 0].detach().cpu().numpy(),
            "pp3d_orth_cost": results["pp3d_orth_cost"][0, 0].detach().cpu().numpy(),
            "pp3d_dir_cost": results["pp3d_dir_cost"][0, 0].detach().cpu().numpy(),
            "depth_contrast_cost": results["depth_contrast_cost"][0, 0].detach().cpu().numpy(),
            "oor2_cost_grid": results["oor2_cost_grid"][0, 0].detach().cpu().numpy(),
            "dc_unc_cost_grid": results["dc_unc_cost_grid"][0, 0].detach().cpu().numpy(),
            "p3dmag": results["p3dmag"][0, 0].detach().cpu().numpy(),
            "tau_cost_grid": results["tau_cost_grid"][0, 0].detach().cpu().numpy(),
            "disp_cost_grid": results["disp_cost_grid"][0, 0].detach().cpu().numpy(),
            "flow_full_x": results["flow_full"][0, 0].detach().cpu().numpy(),
            "flow_full_y": results["flow_full"][0, 1].detach().cpu().numpy(),
            "tau_full": results["tau_full"][0].detach().cpu().numpy(),
            "oor2_full": results["oor2_full"][0].detach().cpu().numpy(),
            "dc_unc_full": results["dc_unc_full"][0].detach().cpu().numpy(),
        }

    def _save_target_pair(
        self,
        pair_dir,
        pair_name,
        raw_arrays,
        target_time_idx,
        target_frame_id,
        counterpart_frame_id,
        target_role,
        target_rgb_orig,
        reference_rgb_orig,
        prev_rgb_orig,
        curr_rgb_orig,
        target_disp_input,
        target_sky_mask,
        results,
        elapsed,
        warp_summary=None,
    ):
        os.makedirs(pair_dir, exist_ok=True)
        raw_arrays = {key: value.copy() for key, value in raw_arrays.items()}
        if self.cfg.get("save_raw_tensors", True) and target_disp_input is not None:
            target_disp_np = target_disp_input[0, 0].detach().cpu().numpy().astype(np.float32)
            target_depth_metric_t, target_valid_disp_t = _compute_metric_depth_from_disp(
                target_disp_input[0, 0],
                self.calib,
                min_disp=self.depth_mask_cfg.get("min_disp", 1e-6),
            )
            raw_arrays["disp_input_full"] = target_disp_np
            raw_arrays["depth_metric_input_full"] = target_depth_metric_t.detach().cpu().numpy().astype(np.float32)
            raw_arrays["valid_disp_input_full"] = target_valid_disp_t.detach().cpu().numpy().astype(np.uint8)

        depth_mask_summary = {
            "enabled": bool(self.depth_mask_cfg.get("enabled", False)),
            "mask_sky": bool(self.depth_mask_cfg.get("mask_sky", False)),
            "apply_stage": _normalize_depth_mask_apply_stage(self.depth_mask_cfg),
        }
        if self.depth_mask_cfg.get("enabled", False):
            full_depth_mask_t, depth_metric_t = _build_depth_mask_from_disp(
                target_disp_input[0, 0],
                self.calib,
                min_depth_m=self.depth_mask_cfg.get("min_depth_m", 0.1),
                max_depth_m=self.depth_mask_cfg.get("max_depth_m"),
                min_disp=self.depth_mask_cfg.get("min_disp", 1e-6),
            )
            input_depth_mask_np = full_depth_mask_t.detach().cpu().numpy().astype(bool)
            cost_h, cost_w = raw_arrays["homography_cost"].shape
            cost_depth_mask_np = _resize_bool_mask(full_depth_mask_t, cost_w, cost_h)
            full_h, full_w = raw_arrays["flow_full_x"].shape
            full_depth_mask_np = _resize_bool_mask(full_depth_mask_t, full_w, full_h)

            if self.depth_mask_cfg.get("mask_sky", False) and target_sky_mask is not None:
                input_sky_mask_np = _resize_bool_mask(
                    target_sky_mask,
                    input_depth_mask_np.shape[1],
                    input_depth_mask_np.shape[0],
                )
                cost_sky_mask_np = _resize_bool_mask(target_sky_mask, cost_w, cost_h)
                full_sky_mask_np = _resize_bool_mask(target_sky_mask, full_w, full_h)

                input_depth_mask_np = input_depth_mask_np & (~input_sky_mask_np)
                cost_depth_mask_np = cost_depth_mask_np & (~cost_sky_mask_np)
                full_depth_mask_np = full_depth_mask_np & (~full_sky_mask_np)
            else:
                input_sky_mask_np = None
                full_sky_mask_np = None

            if depth_mask_summary["apply_stage"] == "pre_fusion":
                cost_keys = [
                    "homography_cost",
                    "epipolar_cost",
                    "pp2d_cost",
                    "pp3d_orth_cost",
                    "pp3d_dir_cost",
                    "depth_contrast_cost",
                    "oor2_cost_grid",
                    "dc_unc_cost_grid",
                    "p3dmag",
                    "tau_cost_grid",
                    "disp_cost_grid",
                ]
                full_keys = [
                    "flow_full_x",
                    "flow_full_y",
                    "tau_full",
                    "oor2_full",
                    "dc_unc_full",
                ]
                for key in cost_keys:
                    raw_arrays[key] = _apply_numpy_mask(raw_arrays[key], cost_depth_mask_np)
                for key in full_keys:
                    raw_arrays[key] = _apply_numpy_mask(raw_arrays[key], full_depth_mask_np)

            if self.depth_mask_cfg.get("save_raw_tensors", True):
                raw_arrays["depth_mask_cost_grid"] = cost_depth_mask_np.astype(np.uint8)
                raw_arrays["depth_mask_full"] = full_depth_mask_np.astype(np.uint8)
                raw_arrays["depth_mask_input_full"] = input_depth_mask_np.astype(np.uint8)
                raw_arrays["depth_metric_input_full"] = depth_metric_t.detach().cpu().numpy()
                if input_sky_mask_np is not None:
                    raw_arrays["sky_mask_cost_grid"] = cost_sky_mask_np.astype(np.uint8)
                    raw_arrays["sky_mask_input_full"] = input_sky_mask_np.astype(np.uint8)
                    raw_arrays["sky_mask_full"] = full_sky_mask_np.astype(np.uint8)

            if self.depth_mask_cfg.get("save_visualizations", True):
                cv2.imwrite(
                    os.path.join(pair_dir, "depth_mask_full.png"),
                    (full_depth_mask_np.astype(np.uint8) * 255),
                )
                if full_sky_mask_np is not None:
                    cv2.imwrite(
                        os.path.join(pair_dir, "sky_mask_full.png"),
                        (full_sky_mask_np.astype(np.uint8) * 255),
                    )

            valid_depth = depth_metric_t[input_depth_mask_np]
            depth_mask_summary.update(
                {
                    "min_depth_m": None
                    if self.depth_mask_cfg.get("min_depth_m") is None
                    else float(self.depth_mask_cfg.get("min_depth_m")),
                    "max_depth_m": None
                    if self.depth_mask_cfg.get("max_depth_m") is None
                    else float(self.depth_mask_cfg.get("max_depth_m")),
                    "mask_ratio_input_full": float(input_depth_mask_np.mean()),
                    "mask_ratio_full": float(full_depth_mask_np.mean()),
                    "mask_ratio_cost_grid": float(cost_depth_mask_np.mean()),
                    "valid_depth_mean_m": float(valid_depth.mean().item()) if valid_depth.numel() > 0 else None,
                }
            )

        if self.cfg.get("save_input_rgbs", True):
            cv2.imwrite(os.path.join(pair_dir, "anchor_rgb.png"), target_rgb_orig[:, :, ::-1])
            cv2.imwrite(os.path.join(pair_dir, "reference_rgb.png"), reference_rgb_orig[:, :, ::-1])
            cv2.imwrite(os.path.join(pair_dir, "prev_rgb.png"), prev_rgb_orig[:, :, ::-1])
            cv2.imwrite(os.path.join(pair_dir, "curr_rgb.png"), curr_rgb_orig[:, :, ::-1])

        if self.cfg.get("save_raw_tensors", True):
            np.savez_compressed(os.path.join(pair_dir, "rigidmask_frontend_arrays.npz"), **raw_arrays)

        if self.cfg.get("save_visualizations", True):
            vis_targets = {
                "homography_cost.png": raw_arrays["homography_cost"],
                "epipolar_cost.png": raw_arrays["epipolar_cost"],
                "pp2d_cost.png": raw_arrays["pp2d_cost"],
                "pp3d_orth_cost.png": raw_arrays["pp3d_orth_cost"],
                "pp3d_dir_cost.png": raw_arrays["pp3d_dir_cost"],
                "depth_contrast_cost.png": raw_arrays["depth_contrast_cost"],
                "oor2_cost_grid.png": raw_arrays["oor2_cost_grid"],
                "dc_unc_cost_grid.png": raw_arrays["dc_unc_cost_grid"],
                "tau_cost_grid.png": raw_arrays["tau_cost_grid"],
                "flow_magnitude.png": np.sqrt(raw_arrays["flow_full_x"] ** 2 + raw_arrays["flow_full_y"] ** 2),
                "tau_full.png": raw_arrays["tau_full"],
            }
            for filename, array in vis_targets.items():
                array_vis = _normalize_for_vis(array)
                if array_vis.shape != target_rgb_orig.shape[:2]:
                    array_vis = cv2.resize(
                        array_vis,
                        (target_rgb_orig.shape[1], target_rgb_orig.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                cv2.imwrite(os.path.join(pair_dir, filename), array_vis)

        summary = {
            "status": "ok",
            "pair_name": pair_name,
            "time_idx": int(target_time_idx),
            "target_time_idx": int(target_time_idx),
            "target_frame_id": str(target_frame_id),
            "target_role": target_role,
            "curr_frame_id": str(target_frame_id),
            "counterpart_frame_id": str(counterpart_frame_id),
            "reference_direction": "previous",
            "inference_direction": "reference_to_current",
            "cost_coordinate_frame": target_role,
            "runtime_sec": elapsed,
            "cost_shape": list(results["cost_shape"]),
            "rot": results["rot"][0].detach().cpu().tolist(),
            "trans": results["trans"][0].detach().cpu().tolist(),
            "calibration": {
                "fx": float(self.calib["fx"]),
                "fy": float(self.calib.get("fy", self.calib["fx"])),
                "cx": float(self.calib["cx"]),
                "cy": float(self.calib["cy"]),
                "baseline": float(self.calib["baseline"]),
            },
            "image_shape": list(target_rgb_orig.shape[:2]),
            "depth_mask": depth_mask_summary,
            "lidar_pair_name": f"{int(counterpart_frame_id):010d}_{int(target_frame_id):010d}"
            if target_role == "current"
            else f"{int(target_frame_id):010d}_{int(counterpart_frame_id):010d}",
            "lidar_projection_filename": "image_residual_nonground_features.npz"
            if target_role == "current"
            else "image_residual_nonground_features_prev.npz",
            "lidar_static_mask_filename": "image_lidar_static_masks.npz"
            if target_role == "current"
            else "image_lidar_static_masks_prev.npz",
        }
        if target_role == "current":
            summary["reference_frame_id"] = str(counterpart_frame_id)
            summary["warp_to_current"] = warp_summary or {}
        else:
            summary["next_frame_id"] = str(counterpart_frame_id)

        with open(os.path.join(pair_dir, "rigidmask_frontend_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary

    def _run_frontend(self, imgLR, disc_aux, disp_input):
        import kornia

        model = self.model_bundle["model"]
        compute_geo_costs = self.model_bundle["compute_geo_costs"]
        get_intrinsics = self.model_bundle["get_intrinsics"]
        F_ngransac = self.model_bundle["F_ngransac"]
        get_grid = self.model_bundle["get_grid"]

        bs = imgLR.shape[0] // 2
        c06, c05, c04, c03, c02 = model.pspnet(imgLR)
        c16, c26 = c06[:bs], c06[bs:]
        c15, c25 = c05[:bs], c05[bs:]
        c14, c24 = c04[:bs], c04[bs:]
        c13, c23 = c03[:bs], c03[bs:]
        c12, c22 = c02[:bs], c02[bs:]

        flow6, flow6h, ent6h, _ = model.cost_matching(None, c16, c26, None, None, level=0)
        up_flow6 = F.upsample(flow6, [imgLR.size()[2] // 32, imgLR.size()[3] // 32], mode="bilinear") * 2
        flow5, flow5h, ent5h, _ = model.cost_matching(up_flow6, c15, c25, flow6h, ent6h, level=1)
        up_flow5 = F.upsample(flow5, [imgLR.size()[2] // 16, imgLR.size()[3] // 16], mode="bilinear") * 2
        flow4, flow4h, ent4h, _ = model.cost_matching(up_flow5, c14, c24, flow5h, ent5h, level=2)
        up_flow4 = F.upsample(flow4, [imgLR.size()[2] // 8, imgLR.size()[3] // 8], mode="bilinear") * 2
        flow3, flow3h, ent3h, _ = model.cost_matching(up_flow4, c13, c23, flow4h, ent4h, level=3)
        up_flow3 = F.upsample(flow3, [imgLR.size()[2] // 4, imgLR.size()[3] // 4], mode="bilinear") * 2
        flow2, _, _, oor2 = model.cost_matching(up_flow3, c12, c22, flow3h, ent3h, level=4)

        b, _, h4, w4 = flow2.shape
        exp2, err2, _ = model.affine(
            get_grid(b, h4, w4)[:, 0].permute(0, 3, 1, 2).repeat(b, 1, 1, 1).clone(),
            flow2.detach(),
            pw=1,
        )
        x = torch.cat((model.f3d2v2(-exp2.log()), model.f3d2v3(err2)), 1)
        dchange2 = -exp2.log() + 1.0 / 200 * model.f3d2(x)[0]
        x = torch.cat(
            (
                model.dcnetv1(c12.detach()),
                model.dcnetv2(dchange2.detach()),
                model.dcnetv3(-exp2.log()),
                model.dcnetv4(err2),
            ),
            1,
        )
        dcneto = 1.0 / 200 * model.dcnet(x)[0]
        dchange2 = dchange2.detach() + dcneto[:, :1]
        dchange2 = F.upsample(dchange2, [imgLR.size()[2], imgLR.size()[3]], mode="bilinear")
        if dcneto.shape[1] > 1:
            dc_unc = dcneto[:, 1:2]
        else:
            dc_unc = torch.zeros_like(dcneto)
        dc_unc = F.upsample(dc_unc, [imgLR.size()[2], imgLR.size()[3]], mode="bilinear")[:, 0]

        Kinv, Kinv_n = get_intrinsics(disc_aux[3], noise=False)
        H, W = imgLR.size()[2:4]
        flow = 4 * F.upsample(flow2, [H, W], mode="bilinear").detach()
        oor2 = F.upsample(oor2[:, np.newaxis], [H, W], mode="bilinear").detach()[:, 0]
        tau_full = (-dchange2[:, 0]).exp().detach()

        fscale = 128.0 / H
        fscalex = 32.0 / H
        hp0o = torch.cat(
            [
                torch.arange(0, W, out=torch.cuda.FloatTensor()).view(1, -1).repeat(H, 1)[np.newaxis],
                torch.arange(0, H, out=torch.cuda.FloatTensor()).view(-1, 1).repeat(1, W)[np.newaxis],
            ],
            0,
        )[np.newaxis]
        hp1o = hp0o + flow
        hp0o[:, 0] *= disc_aux[3][10]
        hp0o[:, 1] *= disc_aux[3][11]
        hp1o[:, 0] *= disc_aux[3][10]
        hp1o[:, 1] *= disc_aux[3][11]

        hp0 = F.interpolate(hp0o, scale_factor=fscale, mode="nearest")
        hp1 = F.interpolate(hp1o, scale_factor=fscale, mode="nearest")
        _, _, h, w = hp0.shape
        hp0 = hp0.view(1, 2, -1).permute(0, 2, 1)
        hp1 = hp1.view(bs, 2, -1).permute(0, 2, 1)
        hp0 = torch.cat((hp0, torch.ones(1, hp0.shape[1], 1).cuda()), -1)
        hp1 = torch.cat((hp1, torch.ones(bs, hp0.shape[1], 1).cuda()), -1)
        unc = torch.cat(
            (
                F.interpolate(oor2[:, np.newaxis], scale_factor=fscale, mode="nearest"),
                F.interpolate(dc_unc[:, np.newaxis].detach(), scale_factor=fscale, mode="nearest"),
            ),
            1,
        )
        tau = F.interpolate(tau_full[:, np.newaxis], scale_factor=fscale, mode="nearest").view(bs, 1, -1)

        hp0x = F.interpolate(hp0o, scale_factor=fscalex, mode="nearest")
        hp1x = F.interpolate(hp1o, scale_factor=fscalex, mode="nearest")
        hp0x = hp0x.view(1, 2, -1).permute(0, 2, 1)
        hp1x = hp1x.view(bs, 2, -1).permute(0, 2, 1)
        hp0x = torch.cat((hp0x, torch.ones(1, hp0x.shape[1], 1).cuda()), -1)
        hp1x = torch.cat((hp1x, torch.ones(bs, hp1x.shape[1], 1).cuda()), -1)

        unc_occ = F.interpolate(oor2[:, np.newaxis], scale_factor=fscalex, mode="nearest").view(bs, -1)
        rotx, transx, Ex = F_ngransac(
            hp0x,
            hp1x,
            Kinv.inverse(),
            False,
            unc_occ,
            Kn=Kinv_n.inverse(),
            cv=self.cfg.get("use_opencv_essential_mat", False),
        )
        rot = rotx.cuda().detach()
        trans = transx.cuda().detach()
        mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, _ = compute_geo_costs(
            rot, trans, Ex, Kinv, hp0, hp1, tau, Kinv_n=Kinv_n
        )

        disp = F.interpolate(disp_input, [h, w], mode="bilinear")
        med_dgt = torch.median(disp.view(bs, -1), dim=-1)[0]
        med_dp3d = torch.median(p3dmag.view(bs, -1), dim=-1)[0]
        med_ratio = (med_dgt / med_dp3d)[:, np.newaxis, np.newaxis, np.newaxis]
        log_dratio = (med_ratio * p3dmag.view(bs, 1, h, w) / disp.view(bs, 1, h, w)).log().abs()

        return {
            "flow_full": flow,
            "tau_full": tau_full,
            "dc_unc_full": dc_unc,
            "oor2_full": oor2,
            "rot": rot,
            "trans": trans,
            "homography_cost": (mcost00 + mcost01).view(bs, 1, h, w),
            "epipolar_cost": mcost1.view(bs, 1, h, w),
            "pp2d_cost": mcost2.view(bs, 1, h, w),
            "pp3d_orth_cost": mcost3.view(bs, 1, h, w),
            "pp3d_dir_cost": mcost4.view(bs, 1, h, w),
            "depth_contrast_cost": log_dratio.view(bs, 1, h, w),
            "oor2_cost_grid": unc[:, :1].view(bs, 1, h, w),
            "dc_unc_cost_grid": unc[:, 1:].view(bs, 1, h, w),
            "p3dmag": p3dmag.view(bs, 1, h, w),
            "tau_cost_grid": tau.view(bs, 1, h, w),
            "disp_cost_grid": disp.view(bs, 1, h, w),
            "cost_shape": (h, w),
        }

    def save_pair(
        self,
        output_root,
        time_idx,
        curr_frame_id,
        reference_frame_id,
        sky_mask=None,
        reference_sky_mask=None,
    ):
        model_bundle = self._ensure_model_bundle()
        current_pair_name = f"{time_idx:06d}_frame_{curr_frame_id}_from_{reference_frame_id}"
        previous_pair_name = f"{time_idx:06d}_frame_{reference_frame_id}_to_{curr_frame_id}_target_prev"
        current_pair_dir = os.path.join(output_root, current_pair_name)
        previous_pair_dir = os.path.join(output_root, previous_pair_name)

        anchor_path = os.path.join(self.image_dir, f"{int(curr_frame_id):010d}.png")
        reference_path = os.path.join(self.image_dir, f"{int(reference_frame_id):010d}.png")
        if not os.path.exists(anchor_path) or not os.path.exists(reference_path):
            return {"status": "skipped", "reason": "missing_image", "pair_name": current_pair_name}

        anchor_rgb_orig = cv2.imread(anchor_path)[:, :, ::-1]
        reference_rgb_orig = cv2.imread(reference_path)[:, :, ::-1]
        resized_w, resized_h = _resize_to_rigidmask_shape(reference_rgb_orig, self.cfg["testres"])
        anchor_rgb = cv2.resize(anchor_rgb_orig, (resized_w, resized_h))
        reference_rgb = cv2.resize(reference_rgb_orig, (resized_w, resized_h))

        _reconfigure_runtime_modules(
            model_bundle["model"],
            model_bundle["flow_reg_cls"],
            model_bundle["warp_module_cls"],
            resized_w,
            resized_h,
        )
        imgL, imgR, imgL_noaug = _prepare_pair_tensors(
            reference_rgb, anchor_rgb, model_bundle["mean_L"], model_bundle["mean_R"]
        )
        disc_aux = [
            None,
            None,
            None,
            _build_intrinsics_list(self.calib, reference_rgb_orig.shape, resized_w, resized_h, self.cfg["sensor"]),
            imgL_noaug,
            None,
        ]
        disp_input = self._load_disp_input(reference_frame_id)
        need_target_depth_tensors = self.cfg.get("save_raw_tensors", True) or self.depth_mask_cfg.get("enabled", False)
        depth_mask_disp_input = (
            self._load_disp_input(curr_frame_id)
            if need_target_depth_tensors
            else None
        )
        reference_depth_mask_disp_input = (
            disp_input
            if need_target_depth_tensors
            else None
        )

        with torch.no_grad():
            imgLR = torch.cat([imgL, imgR], 0)
            torch.cuda.synchronize()
            start_time = time.time()
            results = self._run_frontend(imgLR, disc_aux, disp_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

        previous_raw_arrays = self._build_raw_arrays(results)
        current_raw_arrays, warp_summary = _warp_raw_arrays_forward_to_target(previous_raw_arrays)

        previous_summary = self._save_target_pair(
            pair_dir=previous_pair_dir,
            pair_name=previous_pair_name,
            raw_arrays=previous_raw_arrays,
            target_time_idx=time_idx - 1,
            target_frame_id=reference_frame_id,
            counterpart_frame_id=curr_frame_id,
            target_role="previous",
            target_rgb_orig=reference_rgb_orig,
            reference_rgb_orig=anchor_rgb_orig,
            prev_rgb_orig=reference_rgb_orig,
            curr_rgb_orig=anchor_rgb_orig,
            target_disp_input=reference_depth_mask_disp_input
            if reference_depth_mask_disp_input is not None
            else disp_input,
            target_sky_mask=reference_sky_mask,
            results=results,
            elapsed=elapsed,
        )
        current_summary = self._save_target_pair(
            pair_dir=current_pair_dir,
            pair_name=current_pair_name,
            raw_arrays=current_raw_arrays,
            target_time_idx=time_idx,
            target_frame_id=curr_frame_id,
            counterpart_frame_id=reference_frame_id,
            target_role="current",
            target_rgb_orig=anchor_rgb_orig,
            reference_rgb_orig=reference_rgb_orig,
            prev_rgb_orig=reference_rgb_orig,
            curr_rgb_orig=anchor_rgb_orig,
            target_disp_input=depth_mask_disp_input if depth_mask_disp_input is not None else disp_input,
            target_sky_mask=sky_mask,
            results=results,
            elapsed=elapsed,
            warp_summary=warp_summary,
        )

        if self.offload_after_use and self.model_bundle is not None:
            self.model_bundle["model"] = self.model_bundle["model"].cpu().eval()
            torch.cuda.empty_cache()
        return {
            **current_summary,
            "pair_name": current_pair_name,
            "pair_dir": current_pair_dir,
            "status": "ok",
            "targets": {
                "previous": {
                    **previous_summary,
                    "pair_dir": previous_pair_dir,
                },
                "current": {
                    **current_summary,
                    "pair_dir": current_pair_dir,
                },
            },
        }

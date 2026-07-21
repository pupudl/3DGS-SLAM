import argparse
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


def load_python_config(config_path):
    spec = importlib.util.spec_from_file_location("rigidmask_frontend_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "config"):
        return module.config
    raise ValueError(f"Config file {config_path} does not define `config`.")


def parse_perspective_file(calibration_path):
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
    if "P_rect_00" not in values or "P_rect_01" not in values:
        raise FileNotFoundError(
            f"Expected P_rect_00 and P_rect_01 in calibration file: {calibration_path}"
        )

    p_rect_00 = values["P_rect_00"].reshape(3, 4)
    p_rect_01 = values["P_rect_01"].reshape(3, 4)
    fx = float(p_rect_00[0, 0])
    cx = float(p_rect_00[0, 2])
    cy = float(p_rect_00[1, 2])
    tx0 = float(p_rect_00[0, 3] / p_rect_00[0, 0])
    tx1 = float(p_rect_01[0, 3] / p_rect_01[0, 0])
    baseline = abs(tx1 - tx0)
    return {
        "fx": fx,
        "cx": cx,
        "cy": cy,
        "baseline": baseline,
    }


def collect_frame_pairs(image_dir, start_idx, end_idx, stride):
    frame_ids = list(range(start_idx, end_idx + 1, stride))
    frames = []
    for frame_id in frame_ids:
        frame_path = os.path.join(image_dir, f"{frame_id:010d}.png")
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Missing frame: {frame_path}")
        frames.append((frame_id, frame_path))
    if len(frames) < 2:
        raise ValueError("Need at least two frames to form a sequence pair.")
    return list(zip(frames[:-1], frames[1:]))


def install_rigidmask_det_stub():
    if "models.det" in sys.modules:
        return
    stub = types.ModuleType("models.det")

    def create_model(*_args, **_kwargs):
        return nn.Identity()

    def load_model(model, *_args, **_kwargs):
        return model

    def save_model(*_args, **_kwargs):
        return None

    stub.create_model = create_model
    stub.load_model = load_model
    stub.save_model = save_model
    sys.modules["models.det"] = stub


class DummyMiDaS(nn.Module):
    def forward(self, _x):
        raise RuntimeError(
            "MiDaS was stubbed because this run is in stereo mode. "
            "Switch to `sensor='mono'` only after you have a local MiDaS cache."
        )


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned[key[len("module.") :]] = value
        else:
            cleaned[key] = value
    return cleaned


def maybe_add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_rigidmask_model(rigidmask_cfg):
    rigidmask_root = os.path.abspath(rigidmask_cfg["repo_root"])
    maybe_add_path(rigidmask_root)
    maybe_add_path(os.path.join(rigidmask_root, "models", "ngransac"))
    install_rigidmask_det_stub()

    original_hub_load = torch.hub.load
    if rigidmask_cfg["sensor"] == "stereo":
        torch.hub.load = lambda *_args, **_kwargs: DummyMiDaS()
    try:
        from models.VCNplus import VCN, WarpModule, flow_reg
        from models.submodule import F_ngransac, compute_geo_costs, get_intrinsics

        checkpoint_path = rigidmask_cfg["checkpoint_path"]
        exp_unc = "kitti" not in checkpoint_path.lower()
        model = VCN(
            [1, rigidmask_cfg["max_width"], rigidmask_cfg["max_height"]],
            md=[int(4 * (rigidmask_cfg["maxdisp"] / 256)), 4, 4, 4, 4],
            fac=rigidmask_cfg["fac"],
            exp_unc=exp_unc,
        )
    finally:
        torch.hub.load = original_hub_load

    checkpoint = torch.load(rigidmask_cfg["checkpoint_path"], map_location="cpu")
    state_dict = strip_module_prefix(checkpoint["state_dict"])
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    return {
        "model": model,
        "flow_reg_cls": flow_reg,
        "warp_module_cls": WarpModule,
        "F_ngransac": F_ngransac,
        "compute_geo_costs": compute_geo_costs,
        "get_intrinsics": get_intrinsics,
        "mean_L": checkpoint["mean_L"],
        "mean_R": checkpoint["mean_R"],
    }


def resize_to_rigidmask_shape(image, testres):
    maxh = image.shape[0] * testres
    maxw = image.shape[1] * testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh:
        max_h += 64
    if max_w < maxw:
        max_w += 64
    return max_w, max_h


def reconfigure_runtime_modules(model, flow_reg_cls, warp_module_cls, max_w, max_h):
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


def build_intrinsics_list(calib, input_size, max_w, max_h, sensor):
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


def prepare_pair_tensors(prev_rgb, curr_rgb, mean_L, mean_R):
    imgL_noaug = torch.tensor(prev_rgb / 255.0, device="cuda")[None].float()

    imgL = prev_rgb[:, :, ::-1].copy() / 255.0 - np.asarray(mean_L).mean(0)[None, None, :]
    imgR = curr_rgb[:, :, ::-1].copy() / 255.0 - np.asarray(mean_R).mean(0)[None, None, :]
    imgL = torch.tensor(np.transpose(imgL, [2, 0, 1])[None], device="cuda").float()
    imgR = torch.tensor(np.transpose(imgR, [2, 0, 1])[None], device="cuda").float()
    return imgL, imgR, imgL_noaug


def load_disp_input(disp_dir, frame_id):
    npy_path = os.path.join(disp_dir, f"{frame_id:010d}.npy")
    png_path = os.path.join(disp_dir, f"{frame_id:010d}.png")
    if os.path.exists(npy_path):
        disp = np.load(npy_path).astype(np.float32)
    elif os.path.exists(png_path):
        disp = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(f"Failed to read disparity file: {png_path}")
        disp = disp.astype(np.float32)
    else:
        raise FileNotFoundError(
            f"Missing disparity prior for frame {frame_id:010d} in {disp_dir}"
        )
    return torch.tensor(disp, device="cuda")[None, None].float()


def run_frontend(model_bundle, imgLR, disc_aux, disp_input, use_opencv_essential_mat):
    import kornia
    from models.VCNplus import get_grid

    model = model_bundle["model"]
    compute_geo_costs = model_bundle["compute_geo_costs"]
    get_intrinsics = model_bundle["get_intrinsics"]
    F_ngransac = model_bundle["F_ngransac"]

    bs = imgLR.shape[0] // 2
    c06, c05, c04, c03, c02 = model.pspnet(imgLR)
    c16 = c06[:bs]
    c26 = c06[bs:]
    c15 = c05[:bs]
    c25 = c05[bs:]
    c14 = c04[:bs]
    c24 = c04[bs:]
    c13 = c03[:bs]
    c23 = c03[bs:]
    c12 = c02[:bs]
    c22 = c02[bs:]

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
        cv=use_opencv_essential_mat,
    )
    rot = rotx.cuda().detach()
    trans = transx.cuda().detach()

    mcost00, mcost01, mcost1, mcost2, mcost3, mcost4, p3dmag, _ = compute_geo_costs(
        rot,
        trans,
        Ex,
        Kinv,
        hp0,
        hp1,
        tau,
        Kinv_n=Kinv_n,
    )

    if disp_input is None:
        with torch.no_grad():
            model.midas.eval()
            input_im = (
                disc_aux[4].permute(0, 3, 1, 2)
                - torch.tensor([0.485, 0.456, 0.406], device="cuda")[None, :, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225], device="cuda")[None, :, None, None]
            wsize = int((input_im.shape[3] * 448.0 / input_im.shape[2]) // 32 * 32)
            input_im = F.interpolate(input_im, (448, wsize), mode="bilinear")
            dispo = model.midas.forward(input_im)[None].clamp(1e-6, np.inf)
    else:
        dispo = disp_input

    disp = F.interpolate(dispo, [h, w], mode="bilinear")
    med_dgt = torch.median(disp.view(bs, -1), dim=-1)[0]
    med_dp3d = torch.median(p3dmag.view(bs, -1), dim=-1)[0]
    med_ratio = (med_dgt / med_dp3d)[:, np.newaxis, np.newaxis, np.newaxis]
    log_dratio = (med_ratio * p3dmag.view(bs, 1, h, w) / disp.view(bs, 1, h, w)).log().abs()

    depth = (1.0 / disp).view(bs, 1, -1)
    depth = depth.clamp(depth.median() / 10, depth.median() * 10)
    p03d = depth * Kinv.matmul(hp0.permute(0, 2, 1))
    p13d = depth / tau * Kinv_n.matmul(hp1.permute(0, 2, 1))
    p13d = kornia.angle_axis_to_rotation_matrix(rot).matmul(p13d)

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


def normalize_for_vis(array):
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[finite]
    lo = np.percentile(values, 1.0)
    hi = np.percentile(values, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    return (255.0 * scaled).astype(np.uint8)


def save_pair_outputs(
    output_dir,
    pair_name,
    prev_rgb_orig,
    curr_rgb_orig,
    results,
    save_visualizations,
    save_raw_tensors,
    save_input_rgbs,
):
    pair_dir = os.path.join(output_dir, pair_name)
    os.makedirs(pair_dir, exist_ok=True)

    if save_input_rgbs:
        cv2.imwrite(os.path.join(pair_dir, "prev_rgb.png"), prev_rgb_orig[:, :, ::-1])
        cv2.imwrite(os.path.join(pair_dir, "curr_rgb.png"), curr_rgb_orig[:, :, ::-1])

    raw_arrays = {
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

    if save_raw_tensors:
        np.savez_compressed(os.path.join(pair_dir, "rigidmask_frontend_arrays.npz"), **raw_arrays)

    if save_visualizations:
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
            array_vis = normalize_for_vis(array)
            if array_vis.shape != prev_rgb_orig.shape[:2]:
                array_vis = cv2.resize(
                    array_vis,
                    (prev_rgb_orig.shape[1], prev_rgb_orig.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.imwrite(os.path.join(pair_dir, filename), array_vis)

    summary = {
        "pair_name": pair_name,
        "cost_shape": list(results["cost_shape"]),
        "flow_shape": list(results["flow_full"].shape),
        "rot": results["rot"][0].detach().cpu().tolist(),
        "trans": results["trans"][0].detach().cpu().tolist(),
    }
    with open(os.path.join(pair_dir, "rigidmask_frontend_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def preflight_checks(config):
    rigidmask_cfg = config["rigidmask"]
    data_cfg = config["data"]

    checkpoint_path = rigidmask_cfg["checkpoint_path"]
    if not checkpoint_path:
        raise FileNotFoundError(
            "Missing `rigidmask.checkpoint_path` in config. "
            "Point it to the rigidmask pretrained checkpoint you want to use."
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"RigidMask checkpoint not found: {checkpoint_path}")

    if rigidmask_cfg["sensor"] not in {"mono", "stereo"}:
        raise ValueError("`rigidmask.sensor` must be either `mono` or `stereo`.")

    if rigidmask_cfg["sensor"] == "stereo":
        disp_dir = os.path.join(data_cfg["root"], data_cfg["sequence"], data_cfg["disparity_dir"])
        if not os.path.isdir(disp_dir):
            raise FileNotFoundError(f"Stereo mode requires disparity dir: {disp_dir}")

    if rigidmask_cfg["sensor"] == "mono":
        has_midas_cache = any(
            os.path.exists(path)
            for path in [
                os.path.expanduser("~/.cache/torch/hub/checkpoints/dpt_large-midas-2f21e586.pt"),
                os.path.expanduser("~/.cache/torch/hub/checkpoints/midas_v21-f6b98070.pt"),
                os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_master"),
                os.path.expanduser("~/.cache/torch/hub/intel-isl_MiDaS_main"),
            ]
        )
        if not has_midas_cache:
            raise FileNotFoundError(
                "Mono exact mode requires MiDaS cache under ~/.cache/torch/hub, but no local cache was found."
            )

    if not rigidmask_cfg.get("use_opencv_essential_mat", False):
        maybe_add_path(os.path.join(os.path.abspath(rigidmask_cfg["repo_root"]), "models", "ngransac"))
        spec = importlib.util.find_spec("ngransac")
        if spec is None:
            raise ModuleNotFoundError(
                "Exact pose estimation requires `ngransac`, but it is not importable. "
                "Build it first, for example with: "
                "`cd third_party/rigidmask/models/ngransac && python3 setup.py build_ext --inplace`"
            )

    if rigidmask_cfg.get("require_cuda", True) and not torch.cuda.is_available():
        raise RuntimeError("This exact rigidmask frontend currently requires CUDA.")


def main():
    parser = argparse.ArgumentParser(description="RigidMask exact frontend cost-map probe")
    parser.add_argument("config", help="Path to the python config file.")
    args = parser.parse_args()

    config = load_python_config(args.config)
    preflight_checks(config)

    data_cfg = config["data"]
    rigidmask_cfg = config["rigidmask"]

    sequence_dir = os.path.join(data_cfg["root"], data_cfg["sequence"])
    image_dir = os.path.join(sequence_dir, data_cfg["image_dir"])
    disp_dir = os.path.join(sequence_dir, data_cfg["disparity_dir"])
    frame_pairs = collect_frame_pairs(image_dir, data_cfg["start"], data_cfg["end"], data_cfg["stride"])

    first_image = cv2.imread(frame_pairs[0][0][1])[:, :, ::-1]
    max_width, max_height = resize_to_rigidmask_shape(first_image, rigidmask_cfg["testres"])
    rigidmask_cfg["max_width"] = max_width
    rigidmask_cfg["max_height"] = max_height

    calib = parse_perspective_file(data_cfg["calibration_path"])
    model_bundle = load_rigidmask_model(rigidmask_cfg)

    output_dir = os.path.join(
        config["workdir"],
        config["group_name"],
        config["run_name"],
        rigidmask_cfg["output_subdir"],
    )
    os.makedirs(output_dir, exist_ok=True)

    run_summary = {
        "sequence": data_cfg["sequence"],
        "sensor": rigidmask_cfg["sensor"],
        "num_pairs": len(frame_pairs),
        "checkpoint_path": rigidmask_cfg["checkpoint_path"],
        "use_opencv_essential_mat": rigidmask_cfg.get("use_opencv_essential_mat", False),
        "pairs": [],
    }

    for (left_id, left_path), (right_id, right_path) in frame_pairs:
        pair_name = f"{left_id:010d}_to_{right_id:010d}"
        prev_rgb_orig = cv2.imread(left_path)[:, :, ::-1]
        curr_rgb_orig = cv2.imread(right_path)[:, :, ::-1]
        resized_w, resized_h = resize_to_rigidmask_shape(prev_rgb_orig, rigidmask_cfg["testres"])
        prev_rgb = cv2.resize(prev_rgb_orig, (resized_w, resized_h))
        curr_rgb = cv2.resize(curr_rgb_orig, (resized_w, resized_h))

        reconfigure_runtime_modules(
            model_bundle["model"],
            model_bundle["flow_reg_cls"],
            model_bundle["warp_module_cls"],
            resized_w,
            resized_h,
        )

        imgL, imgR, imgL_noaug = prepare_pair_tensors(
            prev_rgb,
            curr_rgb,
            model_bundle["mean_L"],
            model_bundle["mean_R"],
        )
        disc_aux = [
            None,
            None,
            None,
            build_intrinsics_list(calib, prev_rgb_orig.shape, resized_w, resized_h, rigidmask_cfg["sensor"]),
            imgL_noaug,
            None,
        ]

        disp_input = None
        if rigidmask_cfg["sensor"] == "stereo":
            disp_input = load_disp_input(disp_dir, left_id)

        with torch.no_grad():
            imgLR = torch.cat([imgL, imgR], 0)
            torch.cuda.synchronize()
            start_time = time.time()
            results = run_frontend(
                model_bundle,
                imgLR,
                disc_aux,
                disp_input,
                rigidmask_cfg.get("use_opencv_essential_mat", False),
            )
            torch.cuda.synchronize()
            elapsed = time.time() - start_time

        save_pair_outputs(
            output_dir,
            pair_name,
            prev_rgb_orig,
            curr_rgb_orig,
            results,
            rigidmask_cfg.get("save_visualizations", True),
            rigidmask_cfg.get("save_raw_tensors", True),
            rigidmask_cfg.get("save_input_rgbs", True),
        )
        run_summary["pairs"].append({"pair_name": pair_name, "runtime_sec": elapsed})
        print(f"[RigidMaskFrontend] {pair_name} finished in {elapsed:.3f}s")

    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2)


if __name__ == "__main__":
    main()

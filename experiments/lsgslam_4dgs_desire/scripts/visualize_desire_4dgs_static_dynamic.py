import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.desire_4dgs import get_scaling_t, is_enabled as desire_4dgs_enabled
from utils.recon_helpers import setup_camera
from utils.slam_helpers import transform_to_frame, transformed_params2rendervar


def _load_config(config_path):
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    return experiment.config


def _default_params_path(config):
    return os.path.join(config["workdir"], config["run_name"], "params.npz")


def _to_torch_params(np_params, device):
    params = {}
    for key, value in np_params.items():
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
            params[key] = torch.as_tensor(value, device=device).float()
        else:
            params[key] = value
    return params


def _scalar_int(value, fallback):
    if value is None:
        return int(fallback)
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(np.asarray(value).reshape(-1)[0])


def _parse_frames(frames_arg, num_frames, every, max_frames):
    if frames_arg:
        frames = []
        for item in frames_arg.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item:
                parts = [p.strip() for p in item.split(":")]
                if len(parts) not in (2, 3):
                    raise ValueError(f"Bad frame range: {item}")
                start = int(parts[0])
                stop = int(parts[1])
                step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
                frames.extend(range(start, stop + 1, step))
            else:
                frames.append(int(item))
    else:
        frames = list(range(0, num_frames, every))

    frames = [frame for frame in frames if 0 <= frame < num_frames]
    frames = sorted(dict.fromkeys(frames))
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def _stats(values):
    values = np.asarray(values)
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _gaussian_subset(params, mask):
    num_gaussians = params["means3D"].shape[0]
    subset = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == num_gaussians:
            subset[key] = value[mask]
        else:
            subset[key] = value
    return subset


def _base_camera(params, width, height, depth_threshold=None):
    intrinsics = params["intrinsics"].detach().cpu().numpy()
    if "w2c" in params:
        w2c = params["w2c"].detach().cpu().numpy()
    else:
        w2c = np.eye(4, dtype=np.float32)
    return setup_camera(width, height, intrinsics, w2c, depth_threshold=depth_threshold)


def _render_black(width, height):
    return torch.zeros((3, height, width), dtype=torch.float32, device="cuda")


def _render_rgb(params, cam, frame_idx, desire_4dgs_cfg):
    if params["means3D"].shape[0] == 0:
        return _render_black(cam.image_width, cam.image_height)
    transformed = transform_to_frame(
        params,
        frame_idx,
        gaussians_grad=False,
        camera_grad=False,
        desire_4dgs_cfg=desire_4dgs_cfg,
    )
    rendervar = transformed_params2rendervar(params, transformed)
    image, _, _, _ = Renderer(raster_settings=cam)(**rendervar)
    return torch.clamp(image, 0.0, 1.0)


def _render_with_colors(params, cam, frame_idx, desire_4dgs_cfg, colors):
    if params["means3D"].shape[0] == 0:
        return _render_black(cam.image_width, cam.image_height)
    transformed = transform_to_frame(
        params,
        frame_idx,
        gaussians_grad=False,
        camera_grad=False,
        desire_4dgs_cfg=desire_4dgs_cfg,
    )
    if params["log_scales"].shape[1] == 1:
        log_scales = torch.tile(params["log_scales"], (1, 3))
    else:
        log_scales = params["log_scales"]
    opacities = torch.sigmoid(params["logit_opacities"])
    if "marginal_t" in transformed:
        opacities = opacities * transformed["marginal_t"]
    rendervar = {
        "means3D": transformed["means3D"],
        "colors_precomp": torch.clamp(colors, 0.0, 1.0),
        "rotations": F.normalize(transformed["unnorm_rotations"]),
        "opacities": opacities,
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(params["means3D"], requires_grad=True, device="cuda") + 0,
    }
    image, _, _, _ = Renderer(raster_settings=cam)(**rendervar)
    return torch.clamp(image, 0.0, 1.0)


def _to_uint8_rgb(image):
    image = torch.clamp(image, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def _write_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(_to_uint8_rgb(image), cv2.COLOR_RGB2BGR))


def _label(rgb, text):
    out = rgb.copy()
    h, w = out.shape[:2]
    bar_h = max(28, h // 18)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    font_scale = max(0.45, min(0.8, w / 1300.0))
    cv2.putText(
        out,
        text,
        (10, int(bar_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def _make_panel(items):
    labeled = [_label(_to_uint8_rgb(image), title) for title, image in items]
    top = np.concatenate(labeled[:3], axis=1)
    bottom = np.concatenate(labeled[3:6], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _save_panel(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    panel = _make_panel(items)
    cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _normalization_max(values, user_value):
    if user_value is not None:
        return max(float(user_value), 1e-8)
    return max(float(torch.quantile(values.detach(), 0.99).item()), 1e-8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DeSiRe/4DGS learned static and dynamic Gaussians from a SplaTAM params.npz."
    )
    parser.add_argument("config", type=str, help="Path to the LSG-SLAM config file.")
    parser.add_argument("--params-path", default=None, help="Override params.npz path. Defaults to config workdir/run_name/params.npz.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <result_dir>/desire_4dgs_static_dynamic.")
    parser.add_argument("--frames", default=None, help="Comma list/ranges of frame ids, e.g. '0,5,10:20:2'. Defaults to all frames.")
    parser.add_argument("--every", type=int, default=1, help="Render every N frames when --frames is omitted.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames rendered.")
    parser.add_argument(
        "--dynamic-criterion",
        default="scaling",
        choices=["scaling", "velocity", "scaling_or_velocity", "scaling_and_velocity"],
        help=(
            "Criterion used to split dynamic/static Gaussians. "
            "'scaling' matches the DeSiRe-GS-style temporal-scale proxy; "
            "'scaling_and_velocity' is usually cleaner for qualitative inspection."
        ),
    )
    parser.add_argument("--scaling-t-threshold", type=float, default=None, help="Dynamic if exp(_scaling_t) <= threshold. Defaults to desire_4dgs.separate_scaling_t.")
    parser.add_argument("--velocity-threshold", type=float, default=None, help="Velocity threshold used by velocity-based dynamic criteria.")
    parser.add_argument("--velocity-vis-max", type=float, default=None, help="Velocity map normalization max. Defaults to p99.")
    parser.add_argument("--marginal-vis-max", type=float, default=1.0, help="Marginal_t map normalization max.")
    parser.add_argument("--depth-threshold", type=float, default=None, help="Optional rasterizer depth threshold override.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.config)
    desire_4dgs_cfg = dict(config.get("desire_4dgs", {}))

    params_path = args.params_path or _default_params_path(config)
    params_path = os.path.abspath(params_path)
    result_dir = os.path.dirname(params_path)
    output_dir = os.path.abspath(args.output_dir or os.path.join(result_dir, "desire_4dgs_static_dynamic"))
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda")
    params = _to_torch_params(dict(np.load(params_path, allow_pickle=True)), device=device)
    if not desire_4dgs_enabled(params, desire_4dgs_cfg):
        raise RuntimeError(
            "This params.npz does not contain enabled DeSiRe/4DGS parameters "
            "(_t, _scaling_t, _velocity), or desire_4dgs.enabled is false."
        )

    num_gaussians = int(params["means3D"].shape[0])
    num_frames = int(params["cam_trans"].shape[-1])
    data_cfg = config.get("data", {})
    width = _scalar_int(params.get("org_width"), data_cfg.get("desired_image_width", data_cfg.get("width", 0)))
    height = _scalar_int(params.get("org_height"), data_cfg.get("desired_image_height", data_cfg.get("height", 0)))
    if width <= 0 or height <= 0:
        raise RuntimeError("Could not infer image width/height from params or config.")

    scaling_t_threshold = args.scaling_t_threshold
    if scaling_t_threshold is None:
        scaling_t_threshold = float(desire_4dgs_cfg.get("separate_scaling_t", 0.2))

    scaling_t = get_scaling_t(params).reshape(-1)
    velocity_norm = torch.linalg.norm(params["_velocity"], dim=1)
    scaling_dynamic_mask = scaling_t <= float(scaling_t_threshold)
    velocity_dynamic_mask = None
    if args.dynamic_criterion != "scaling":
        if args.velocity_threshold is None:
            raise ValueError(
                "--velocity-threshold is required when --dynamic-criterion uses velocity."
            )
        velocity_dynamic_mask = velocity_norm >= float(args.velocity_threshold)

    if args.dynamic_criterion == "scaling":
        dynamic_mask = scaling_dynamic_mask
    elif args.dynamic_criterion == "velocity":
        dynamic_mask = velocity_dynamic_mask
    elif args.dynamic_criterion == "scaling_or_velocity":
        dynamic_mask = scaling_dynamic_mask | velocity_dynamic_mask
    elif args.dynamic_criterion == "scaling_and_velocity":
        dynamic_mask = scaling_dynamic_mask & velocity_dynamic_mask
    else:
        raise ValueError(f"Unknown dynamic criterion: {args.dynamic_criterion}")
    static_mask = ~dynamic_mask

    frames = _parse_frames(args.frames, num_frames, max(1, args.every), args.max_frames)
    cam = _base_camera(params, width, height, depth_threshold=args.depth_threshold)

    dynamic_params = _gaussian_subset(params, dynamic_mask)
    static_params = _gaussian_subset(params, static_mask)

    class_colors = torch.empty((num_gaussians, 3), dtype=torch.float32, device=device)
    class_colors[static_mask] = torch.tensor([0.10, 0.55, 1.00], dtype=torch.float32, device=device)
    class_colors[dynamic_mask] = torch.tensor([1.00, 0.18, 0.05], dtype=torch.float32, device=device)

    velocity_vis_max = _normalization_max(velocity_norm, args.velocity_vis_max)
    velocity_gray = (velocity_norm / velocity_vis_max).clamp(0.0, 1.0).unsqueeze(1).repeat(1, 3)

    scale_vis_max = max(float(torch.quantile(scaling_t.detach(), 0.95).item()), float(scaling_t_threshold), 1e-8)
    scale_score = (1.0 - scaling_t / scale_vis_max).clamp(0.0, 1.0)
    scale_gray = scale_score.unsqueeze(1).repeat(1, 3)

    np.savez(
        os.path.join(output_dir, "gaussian_dynamic_static_masks.npz"),
        dynamic_mask=dynamic_mask.detach().cpu().numpy(),
        static_mask=static_mask.detach().cpu().numpy(),
        velocity_norm=velocity_norm.detach().cpu().numpy(),
        scaling_t=scaling_t.detach().cpu().numpy(),
        scaling_t_threshold=np.array([scaling_t_threshold], dtype=np.float32),
        velocity_threshold=np.array(
            [np.nan if args.velocity_threshold is None else float(args.velocity_threshold)],
            dtype=np.float32,
        ),
    )

    frames_dir = os.path.join(output_dir, "frames")
    panels_dir = os.path.join(output_dir, "panels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(panels_dir, exist_ok=True)

    rendered_frames = []
    with torch.no_grad():
        for frame_idx in tqdm(frames, desc="Render static/dynamic 4DGS"):
            all_rgb = _render_rgb(params, cam, frame_idx, desire_4dgs_cfg)
            static_rgb = _render_rgb(static_params, cam, frame_idx, desire_4dgs_cfg)
            dynamic_rgb = _render_rgb(dynamic_params, cam, frame_idx, desire_4dgs_cfg)
            class_map = _render_with_colors(params, cam, frame_idx, desire_4dgs_cfg, class_colors)
            velocity_map = _render_with_colors(params, cam, frame_idx, desire_4dgs_cfg, velocity_gray)

            transformed = transform_to_frame(
                params,
                frame_idx,
                gaussians_grad=False,
                camera_grad=False,
                desire_4dgs_cfg=desire_4dgs_cfg,
            )
            marginal = transformed["marginal_t"].reshape(-1)
            marginal_gray = (marginal / max(float(args.marginal_vis_max), 1e-8)).clamp(0.0, 1.0).unsqueeze(1).repeat(1, 3)
            marginal_map = _render_with_colors(params, cam, frame_idx, desire_4dgs_cfg, marginal_gray)

            prefix = os.path.join(frames_dir, f"{frame_idx:04d}")
            _write_rgb(prefix + "_all.png", all_rgb)
            _write_rgb(prefix + "_static.png", static_rgb)
            _write_rgb(prefix + "_dynamic.png", dynamic_rgb)
            _write_rgb(prefix + "_class_map.png", class_map)
            _write_rgb(prefix + "_velocity_map.png", velocity_map)
            _write_rgb(prefix + "_marginal_t_map.png", marginal_map)

            panel_items = [
                ("all render", all_rgb),
                ("static only", static_rgb),
                ("dynamic only", dynamic_rgb),
                ("class map: blue static / red dynamic", class_map),
                (f"velocity map, max={velocity_vis_max:.3g}", velocity_map),
                ("marginal_t map", marginal_map),
            ]
            _save_panel(os.path.join(panels_dir, f"{frame_idx:04d}_panel.png"), panel_items)
            rendered_frames.append(frame_idx)

    summary = {
        "params_path": params_path,
        "output_dir": output_dir,
        "num_gaussians": num_gaussians,
        "num_static_gaussians": int(static_mask.sum().item()),
        "num_dynamic_gaussians": int(dynamic_mask.sum().item()),
        "dynamic_fraction": float(dynamic_mask.float().mean().item()),
        "criteria": {
            "dynamic_criterion": args.dynamic_criterion,
            "dynamic_if_scaling_t_lte": float(scaling_t_threshold),
            "dynamic_if_velocity_norm_gte": None if args.velocity_threshold is None else float(args.velocity_threshold),
        },
        "scaling_t_stats": _stats(scaling_t.detach().cpu().numpy()),
        "velocity_norm_stats": _stats(velocity_norm.detach().cpu().numpy()),
        "rendered_frames": rendered_frames,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

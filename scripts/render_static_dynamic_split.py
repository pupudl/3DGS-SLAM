import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import torch
from tqdm import tqdm

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset,
    KittiDataset,
    Kitti360Dataset,
    EurocDataset,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.dynamic_gs import (
    dynamic_params2depthplussilhouette,
    dynamic_params2rendervar,
    has_dynamic_gaussians,
    merge_rendervars,
    transform_dynamic_to_frame,
)
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2depthplussilhouette,
    transformed_params2rendervar,
)


def get_dataset(config_dict, basedir, sequence, **kwargs):
    name = config_dict["dataset_name"].lower()
    if name == "icl":
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    if name == "replica":
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    if name == "replicav2":
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    if name in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    if name == "scannet":
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    if name == "ai2thor":
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    if name == "record3d":
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    if name == "realsense":
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    if name == "tum":
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    if name == "scannetpp":
        return ScannetPPDataset(basedir, sequence, **kwargs)
    if name == "nerfcapture":
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    if name == "kitti":
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    if name == "kitti360":
        return Kitti360Dataset(config_dict, basedir, sequence, **kwargs)
    if name == "euroc":
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def load_params(scene_path):
    raw = dict(np.load(scene_path, allow_pickle=True))
    params = {}
    for key, value in raw.items():
        if isinstance(value, np.ndarray) and value.dtype != object:
            params[key] = torch.tensor(value).cuda().float()
        else:
            params[key] = value
    return params


def render_pair(cam, rgb_rendervar, depth_rendervar):
    image, _, _, _ = Renderer(raster_settings=cam)(**rgb_rendervar)
    depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
    depth = depth_sil[0].unsqueeze(0)
    sil = depth_sil[1].unsqueeze(0)
    return image, depth, sil


def zeros_like_frame(height, width, device):
    return (
        torch.zeros((3, height, width), device=device),
        torch.zeros((1, height, width), device=device),
        torch.zeros((1, height, width), device=device),
    )


def save_rgb(path, image):
    image_np = torch.clamp(image, 0, 1).detach().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255.0).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    return image_np


def save_depth(path, depth, vmax):
    depth_np = depth[0].detach().cpu().numpy()
    norm = np.clip(depth_np / max(float(vmax), 1e-6), 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(path, color)


def save_sil(path, sil):
    sil_np = np.clip(sil[0].detach().cpu().numpy(), 0.0, 1.0)
    cv2.imwrite(path, (sil_np * 255).astype(np.uint8))


def add_label(image, label):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (max(120, 12 * len(label)), 28), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Path to the experiment config used for the run.")
    parser.add_argument("--scene-path", default="", help="Path to params.npz. Defaults to workdir/run_name/params.npz.")
    parser.add_argument("--out-dir", default="", help="Output directory for split renders.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every N local dataset frames.")
    parser.add_argument("--max-frames", type=int, default=50, help="Maximum number of frames to render.")
    parser.add_argument("--depth-vmax", type=float, default=80.0, help="Depth visualization maximum in meters.")
    args = parser.parse_args()

    experiment = SourceFileLoader(os.path.basename(args.experiment), args.experiment).load_module()
    config = experiment.config
    dataset_config = config["data"]
    gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=torch.device(config["primary_device"]),
        relative_pose=True,
        ignore_bad=dataset_config.get("ignore_bad", False),
        use_train_split=dataset_config.get("use_train_split", True),
    )
    num_frames = len(dataset) if dataset_config.get("num_frames", -1) == -1 else dataset_config["num_frames"]

    results_dir = os.path.join(config["workdir"], config["run_name"])
    scene_path = args.scene_path or os.path.join(results_dir, "params.npz")
    out_dir = args.out_dir or os.path.join(results_dir, "eval", "split_static_dynamic")
    os.makedirs(out_dir, exist_ok=True)

    params = load_params(scene_path)
    dynamic_available = has_dynamic_gaussians(params)
    if dynamic_available:
        print(
            "Dynamic 4DGS:",
            int(params["dyn_means3D_canon"].shape[0]),
            "gaussians,",
            int(params["dyn_obj_trans"].shape[0]),
            "objects",
        )
    else:
        print("Dynamic 4DGS: no dynamic gaussians found in params.npz")

    subdirs = [
        "gt_rgb",
        "static_rgb",
        "dynamic_rgb",
        "composite_rgb",
        "static_depth",
        "dynamic_depth",
        "composite_depth",
        "dynamic_sil",
        "comparison",
    ]
    for subdir in subdirs:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    rendered = []
    for time_idx in tqdm(range(0, num_frames, max(1, args.frame_stride))):
        if len(rendered) >= args.max_frames:
            break
        color, depth, intrinsics, pose, _depth_original, _global_feature = dataset[time_idx]
        color = color.permute(2, 0, 1) / 255.0
        intrinsics = intrinsics[:3, :3]
        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = setup_camera(
                color.shape[2],
                color.shape[1],
                intrinsics.cpu().numpy(),
                first_frame_w2c.detach().cpu().numpy(),
            )

        transformed_static = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
        static_rgb_var = transformed_params2rendervar(params, transformed_static)
        static_depth_var = transformed_params2depthplussilhouette(params, first_frame_w2c, transformed_static)
        static_rgb, static_depth, _static_sil = render_pair(cam, static_rgb_var, static_depth_var)

        dynamic_rgb, dynamic_depth, dynamic_sil = zeros_like_frame(color.shape[1], color.shape[2], color.device)
        composite_rgb_var = static_rgb_var
        composite_depth_var = static_depth_var
        if dynamic_available:
            transformed_dynamic, active_dynamic = transform_dynamic_to_frame(
                params,
                params,
                time_idx,
                gaussians_grad=False,
                camera_grad=False,
            )
            if transformed_dynamic is not None:
                dynamic_rgb_var = dynamic_params2rendervar(params, transformed_dynamic, active_dynamic)
                dynamic_depth_var = dynamic_params2depthplussilhouette(
                    params,
                    transformed_dynamic,
                    active_dynamic,
                    first_frame_w2c,
                )
                dynamic_rgb, dynamic_depth, dynamic_sil = render_pair(cam, dynamic_rgb_var, dynamic_depth_var)
                composite_rgb_var = merge_rendervars(static_rgb_var, dynamic_rgb_var)
                composite_depth_var = merge_rendervars(static_depth_var, dynamic_depth_var)
        composite_rgb, composite_depth, _composite_sil = render_pair(cam, composite_rgb_var, composite_depth_var)

        stem = f"{time_idx:04d}.png"
        gt_np = save_rgb(os.path.join(out_dir, "gt_rgb", stem), color)
        static_np = save_rgb(os.path.join(out_dir, "static_rgb", stem), static_rgb)
        dynamic_np = save_rgb(os.path.join(out_dir, "dynamic_rgb", stem), dynamic_rgb)
        composite_np = save_rgb(os.path.join(out_dir, "composite_rgb", stem), composite_rgb)
        save_depth(os.path.join(out_dir, "static_depth", stem), static_depth, args.depth_vmax)
        save_depth(os.path.join(out_dir, "dynamic_depth", stem), dynamic_depth, args.depth_vmax)
        save_depth(os.path.join(out_dir, "composite_depth", stem), composite_depth, args.depth_vmax)
        save_sil(os.path.join(out_dir, "dynamic_sil", stem), dynamic_sil)

        comparison = np.concatenate(
            [
                add_label(gt_np, "GT"),
                add_label(static_np, "Static"),
                add_label(dynamic_np, "Dynamic 4DGS"),
                add_label(composite_np, "Composite"),
            ],
            axis=1,
        )
        cv2.imwrite(os.path.join(out_dir, "comparison", stem), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        rendered.append(time_idx)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene_path": scene_path,
                "dynamic_available": bool(dynamic_available),
                "num_dynamic_gaussians": int(params["dyn_means3D_canon"].shape[0]) if dynamic_available else 0,
                "num_dynamic_objects": int(params["dyn_obj_trans"].shape[0]) if dynamic_available else 0,
                "rendered_time_indices": rendered,
            },
            handle,
            indent=2,
        )
    print(f"Saved split renders to: {out_dir}")


if __name__ == "__main__":
    main()

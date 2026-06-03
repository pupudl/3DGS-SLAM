import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_BASE_DIR))

sys.path.insert(0, _BASE_DIR)
sys.path.insert(1, PROJECT_ROOT)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from datasets.gradslam_datasets import (
    Kitti360Dataset,
    KittiDataset,
    EurocDataset,
    load_dataset_config,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from utils.recon_helpers import setup_camera
from utils.slam_external import calc_psnr
from utils.slam_helpers import (
    transform_to_frame,
    transformed_params2depthplussilhouette,
    transformed_params2rendervar,
)


def get_dataset(config_dict, basedir, sequence, **kwargs):
    dataset_name = config_dict["dataset_name"].lower()
    if dataset_name == "kitti":
        return KittiDataset(config_dict, basedir, sequence, **kwargs)
    if dataset_name == "kitti360":
        return Kitti360Dataset(config_dict, basedir, sequence, **kwargs)
    if dataset_name == "euroc":
        return EurocDataset(config_dict, basedir, sequence, **kwargs)
    raise ValueError(f"Unsupported dataset for this script: {config_dict['dataset_name']}")


def get_dataset_flag(dataset_cfg, key, default):
    value = dataset_cfg.get(key, default)
    if value is None:
        return default
    return value


def load_experiment_config(config_path):
    experiment = SourceFileLoader(
        os.path.basename(config_path), config_path
    ).load_module()
    return experiment.config


def load_params(npz_path, device):
    data = np.load(npz_path, allow_pickle=True)
    params = {}
    for key in data.files:
        value = data[key]
        if np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_:
            params[key] = torch.from_numpy(value).to(device)
        else:
            params[key] = value
    return params


def render_frame(params, cam, canonical_w2c, frame_idx):
    transformed_gaussians = transform_to_frame(
        params,
        frame_idx,
        gaussians_grad=False,
        camera_grad=False,
    )
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_rendervar = transformed_params2depthplussilhouette(
        params, canonical_w2c, transformed_gaussians
    )
    image, _, _, _ = Renderer(raster_settings=cam)(**rendervar)
    depth_sil, _, _, _ = Renderer(raster_settings=cam)(**depth_rendervar)
    depth = depth_sil[0:1]
    silhouette = depth_sil[1]
    return image, depth, silhouette


def compute_depth_l1(rendered_depth, gt_depth):
    valid_mask = gt_depth > 0
    if int(valid_mask.sum().item()) == 0:
        return float("nan"), torch.zeros_like(gt_depth)
    diff = torch.abs(rendered_depth - gt_depth) * valid_mask
    value = (diff.sum() / valid_mask.sum()).item()
    return value, diff


def tensor_rgb_to_numpy(image):
    return torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).detach().cpu().numpy()


def tensor_depth_to_numpy(depth, valid_mask, vmax=None):
    depth_cpu = depth[0].detach().cpu().numpy()
    mask_cpu = valid_mask[0].detach().cpu().numpy().astype(bool)
    if vmax is None:
        valid_values = depth_cpu[mask_cpu]
        vmax = float(np.percentile(valid_values, 98)) if valid_values.size else 1.0
    depth_vis = np.clip(depth_cpu / max(vmax, 1e-6), 0.0, 1.0)
    colored = plt.cm.jet(depth_vis)[..., :3]
    colored[~mask_cpu] = 0.0
    return colored, vmax


def tensor_depth_to_numpy_raw(depth, vmax=None):
    depth_cpu = depth[0].detach().cpu().numpy()
    positive_mask = depth_cpu > 0
    if vmax is None:
        valid_values = depth_cpu[positive_mask]
        vmax = float(np.percentile(valid_values, 98)) if valid_values.size else 1.0
    depth_vis = np.clip(depth_cpu / max(vmax, 1e-6), 0.0, 1.0)
    colored = plt.cm.jet(depth_vis)[..., :3]
    colored[~positive_mask] = 0.0
    return colored, vmax


def make_frame_figure(
    frame_idx,
    gt_rgb,
    gt_depth,
    full_rgb,
    full_depth,
    filtered_rgb,
    filtered_depth,
    full_psnr,
    filtered_psnr,
    full_depth_l1,
    filtered_depth_l1,
    out_path,
):
    valid_mask = gt_depth > 0
    gt_rgb_np = tensor_rgb_to_numpy(gt_rgb)
    full_rgb_np = tensor_rgb_to_numpy(full_rgb)
    filtered_rgb_np = tensor_rgb_to_numpy(filtered_rgb)

    gt_depth_np, depth_vmax = tensor_depth_to_numpy(gt_depth, valid_mask)
    full_depth_np, _ = tensor_depth_to_numpy_raw(full_depth, vmax=depth_vmax)
    filtered_depth_np, _ = tensor_depth_to_numpy_raw(filtered_depth, vmax=depth_vmax)

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(
        (
            f"Frame {frame_idx} | "
            f"Full PSNR {full_psnr:.2f}, Full Depth L1 {full_depth_l1:.4f} | "
            f"Filtered PSNR {filtered_psnr:.2f}, Filtered Depth L1 {filtered_depth_l1:.4f}"
        ),
        fontsize=14,
        y=0.98,
    )

    axes[0, 0].imshow(gt_rgb_np)
    axes[0, 0].set_title("GT RGB")
    axes[0, 1].imshow(full_rgb_np)
    axes[0, 1].set_title("Full RGB")
    axes[0, 2].imshow(filtered_rgb_np)
    axes[0, 2].set_title("Filtered RGB")

    axes[1, 0].imshow(gt_depth_np)
    axes[1, 0].set_title("GT Depth")
    axes[1, 1].imshow(full_depth_np)
    axes[1, 1].set_title("Full Depth")
    axes[1, 2].imshow(filtered_depth_np)
    axes[1, 2].set_title("Filtered Depth")

    for ax in axes.flat:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_plot(output_dir, frames, full_psnr, filtered_psnr, full_l1, filtered_l1):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(frames, full_psnr, label="full")
    axes[0].plot(frames, filtered_psnr, label="filtered")
    axes[0].set_title("Per-frame PSNR")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("PSNR")
    axes[0].legend()

    axes[1].plot(frames, full_l1, label="full")
    axes[1].plot(frames, filtered_l1, label="filtered")
    axes[1].set_title("Per-frame Depth L1")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Depth L1")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics.png"), bbox_inches="tight")
    plt.close(fig)


def maybe_write_video(frames_dir, output_path, fps):
    frame_files = sorted(
        [
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if name.endswith(".png")
        ]
    )
    if not frame_files:
        return

    first = cv2.imread(frame_files[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for path in frame_files:
        frame = cv2.imread(path)
        writer.write(frame)
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, help="Directory containing params_full.npz and params.npz")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for comparison frames")
    parser.add_argument("--fps", type=int, default=5, help="FPS for optional mp4 export")
    parser.add_argument("--no-video", action="store_true", help="Skip writing mp4")
    args = parser.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    config_path = os.path.join(result_dir, "config.py")
    full_npz = os.path.join(result_dir, "params_full.npz")
    filtered_npz = os.path.join(result_dir, "params.npz")
    output_dir = args.output_dir or os.path.join(result_dir, "compare_full_vs_filtered")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    config = load_experiment_config(config_path)
    dataset_cfg = config["data"]
    gradslam_cfg = load_dataset_config(dataset_cfg["gradslam_data_cfg"])

    device = torch.device(config.get("primary_device", "cuda:0"))
    full_params = load_params(full_npz, device)
    filtered_params = load_params(filtered_npz, device)

    dataset = get_dataset(
        config_dict=gradslam_cfg,
        basedir=dataset_cfg["basedir"],
        sequence=os.path.basename(dataset_cfg["sequence"]),
        start=dataset_cfg["start"],
        end=dataset_cfg["end"],
        stride=dataset_cfg["stride"],
        desired_height=dataset_cfg["desired_image_height"],
        desired_width=dataset_cfg["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=get_dataset_flag(dataset_cfg, "ignore_bad", False),
        use_train_split=get_dataset_flag(dataset_cfg, "use_train_split", True),
    )

    num_frames = len(dataset) if dataset_cfg["num_frames"] == -1 else dataset_cfg["num_frames"]
    canonical_w2c = full_params["w2c"].float()
    intrinsics = full_params["intrinsics"].float()

    first_color, _, _, first_pose, _, _ = dataset[0]
    first_color = first_color.permute(2, 0, 1) / 255.0
    _ = first_pose
    cam = setup_camera(
        first_color.shape[2],
        first_color.shape[1],
        intrinsics.detach().cpu().numpy(),
        canonical_w2c.detach().cpu().numpy(),
    )

    frame_ids = []
    full_psnr_values = []
    filtered_psnr_values = []
    full_l1_values = []
    filtered_l1_values = []

    for frame_idx in tqdm(range(num_frames), desc="Rendering comparison"):
        color, depth, _, _, _, _ = dataset[frame_idx]
        gt_rgb = (color.permute(2, 0, 1) / 255.0).to(device)
        gt_depth = depth.permute(2, 0, 1).to(device)
        valid_mask = gt_depth > 0

        with torch.no_grad():
            full_rgb, full_depth, _ = render_frame(full_params, cam, canonical_w2c, frame_idx)
            filtered_rgb, filtered_depth, _ = render_frame(filtered_params, cam, canonical_w2c, frame_idx)

            full_psnr = calc_psnr(full_rgb * valid_mask, gt_rgb * valid_mask).mean().item()
            filtered_psnr = calc_psnr(filtered_rgb * valid_mask, gt_rgb * valid_mask).mean().item()
            full_depth_l1, _ = compute_depth_l1(full_depth, gt_depth)
            filtered_depth_l1, _ = compute_depth_l1(filtered_depth, gt_depth)

        frame_ids.append(frame_idx)
        full_psnr_values.append(full_psnr)
        filtered_psnr_values.append(filtered_psnr)
        full_l1_values.append(full_depth_l1)
        filtered_l1_values.append(filtered_depth_l1)

        out_path = os.path.join(frames_dir, f"{frame_idx:04d}.png")
        make_frame_figure(
            frame_idx,
            gt_rgb,
            gt_depth,
            full_rgb,
            full_depth,
            filtered_rgb,
            filtered_depth,
            full_psnr,
            filtered_psnr,
            full_depth_l1,
            filtered_depth_l1,
            out_path,
        )

    np.savetxt(
        os.path.join(output_dir, "metrics.csv"),
        np.column_stack(
            [
                np.array(frame_ids),
                np.array(full_psnr_values),
                np.array(filtered_psnr_values),
                np.array(full_l1_values),
                np.array(filtered_l1_values),
            ]
        ),
        delimiter=",",
        header="frame,full_psnr,filtered_psnr,full_depth_l1,filtered_depth_l1",
        comments="",
    )
    save_metrics_plot(
        output_dir,
        frame_ids,
        full_psnr_values,
        filtered_psnr_values,
        full_l1_values,
        filtered_l1_values,
    )

    if not args.no_video:
        maybe_write_video(frames_dir, os.path.join(output_dir, "comparison.mp4"), args.fps)

    print(f"Saved comparison to: {output_dir}")


if __name__ == "__main__":
    main()

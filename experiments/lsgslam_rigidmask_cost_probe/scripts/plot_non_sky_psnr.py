#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


FRAME_DIR_PATTERN = re.compile(r"(?P<probe_idx>\d+)_frame_(?P<frame_id>\d+)")
SUPPORTED_METRICS = ("psnr", "ssim", "lpips")


@dataclass(frozen=True)
class FrameMetric:
    probe_idx: int
    frame_id: int
    valid_pixel_count: int
    values: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot image-quality metrics excluding sky pixels for multiple result folders."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/lsgslam_stage1_feature_probe/results"),
        help="Root directory containing experiment result folders.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "kitti360-0000-opt_local_map_qugaosi",
            "kitti360-0000-qugaosi-wuopt",
            "kitti360-0000-wuqugaosi-wuopt",
        ],
        help="Experiment folder names under results-root.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr"],
        help=f"Metrics to compute. Supported: {', '.join(SUPPORTED_METRICS)}",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="kitti360-0000_non_sky_psnr_comparison",
        help="Prefix for output PNG/CSV/TXT files.",
    )
    parser.add_argument(
        "--x-axis",
        choices=["frame_id", "index"],
        default="frame_id",
        help="Use original frame ids or 1-based frame indices on the plot x-axis.",
    )
    return parser.parse_args()


def validate_metrics(metrics: Iterable[str]) -> list[str]:
    normalized = []
    for metric in metrics:
        metric_name = metric.lower()
        if metric_name not in SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics: {SUPPORTED_METRICS}")
        if metric_name not in normalized:
            normalized.append(metric_name)
    return normalized


def find_single_run_dir(experiment_dir: Path) -> Path:
    candidates = sorted([path for path in experiment_dir.iterdir() if path.is_dir()])
    probe_candidates = [path for path in candidates if (path / "stage1_feature_probe").is_dir()]
    if len(probe_candidates) == 1:
        return probe_candidates[0]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one run directory under {experiment_dir}, found {len(candidates)}."
        )
    return candidates[0]


def load_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def load_sky_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read sky mask: {path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return Variable(window_2d.expand(channel, 1, window_size, window_size).contiguous())


def calc_ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11, size_average: bool = True) -> torch.Tensor:
    channel = img1.size(-3)
    window = create_window(window_size, channel).type_as(img1)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window: torch.Tensor,
    window_size: int,
    channel: int,
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def compute_non_sky_metrics(
    render_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    sky_mask: np.ndarray,
    metrics: list[str],
    device: torch.device,
    lpips_model: lpips.LPIPS | None,
) -> tuple[Dict[str, float], int]:
    non_sky_mask = ~sky_mask
    valid_pixel_count = int(non_sky_mask.sum())
    if valid_pixel_count == 0:
        raise RuntimeError("Sky mask removed all pixels; cannot compute metrics.")

    results: Dict[str, float] = {}
    if "psnr" in metrics:
        diff = render_rgb - gt_rgb
        mse = float((diff[non_sky_mask] ** 2).mean())
        results["psnr"] = float("inf") if mse <= 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse))

    if "ssim" in metrics or "lpips" in metrics:
        render_tensor = torch.from_numpy(render_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        mask_tensor = (
            torch.from_numpy(non_sky_mask.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )
        masked_render = render_tensor * mask_tensor
        masked_gt = gt_tensor * mask_tensor

        with torch.inference_mode():
            if "ssim" in metrics:
                results["ssim"] = float(calc_ssim(masked_render, masked_gt).item())
            if "lpips" in metrics:
                if lpips_model is None:
                    raise RuntimeError("LPIPS requested but model is not initialized.")
                results["lpips"] = float(lpips_model(masked_render, masked_gt, normalize=True).item())

    return results, valid_pixel_count


def index_sky_masks(run_dir: Path) -> dict[int, Path]:
    sky_masks_dir = run_dir / "sky_masks"
    if not sky_masks_dir.exists():
        raise RuntimeError(f"Sky mask directory not found: {sky_masks_dir}")

    indexed: dict[int, Path] = {}
    for path in sky_masks_dir.rglob("*.png"):
        stem = path.stem
        if stem.isdigit():
            indexed[int(stem)] = path
    if not indexed:
        raise RuntimeError(f"No sky mask PNG files found under: {sky_masks_dir}")
    return indexed


def collect_frame_metrics(
    run_dir: Path,
    metrics: list[str],
    device: torch.device,
    lpips_model: lpips.LPIPS | None,
) -> list[FrameMetric]:
    probe_dir = run_dir / "stage1_feature_probe"
    if not probe_dir.exists():
        raise RuntimeError(f"Stage1 feature probe directory not found: {probe_dir}")

    mask_index = index_sky_masks(run_dir)
    frame_metrics: list[FrameMetric] = []

    for frame_dir in sorted([path for path in probe_dir.iterdir() if path.is_dir()]):
        match = FRAME_DIR_PATTERN.fullmatch(frame_dir.name)
        if match is None:
            continue

        probe_idx = int(match.group("probe_idx"))
        frame_id = int(match.group("frame_id"))
        gt_path = frame_dir / "gt_rgb.png"
        render_path = frame_dir / "render_rgb.png"
        mask_path = mask_index.get(frame_id)
        if mask_path is None:
            raise RuntimeError(f"Sky mask for frame {frame_id} not found under {run_dir / 'sky_masks'}")

        gt_rgb = load_rgb_image(gt_path)
        render_rgb = load_rgb_image(render_path)
        if gt_rgb.shape != render_rgb.shape:
            raise RuntimeError(
                f"Image shape mismatch for frame {frame_id}: gt {gt_rgb.shape} vs render {render_rgb.shape}"
            )

        sky_mask = load_sky_mask(mask_path, gt_rgb.shape[:2])
        values, valid_pixel_count = compute_non_sky_metrics(
            render_rgb=render_rgb,
            gt_rgb=gt_rgb,
            sky_mask=sky_mask,
            metrics=metrics,
            device=device,
            lpips_model=lpips_model,
        )
        frame_metrics.append(
            FrameMetric(
                probe_idx=probe_idx,
                frame_id=frame_id,
                valid_pixel_count=valid_pixel_count,
                values=values,
            )
        )

    if not frame_metrics:
        raise RuntimeError(f"No frame metrics found under {probe_dir}")
    return frame_metrics


def write_csv(output_path: Path, experiment_to_metrics: dict[str, list[FrameMetric]], metrics: list[str]) -> None:
    common_frame_ids = sorted(
        set.intersection(*[set(metric.frame_id for metric in frame_metrics) for frame_metrics in experiment_to_metrics.values()])
    )
    metric_tables = {
        name: {metric.frame_id: metric for metric in frame_metrics}
        for name, frame_metrics in experiment_to_metrics.items()
    }

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["probe_idx", "frame_id"]
        for name in experiment_to_metrics:
            for metric_name in metrics:
                header.append(f"{name}_{metric_name}")
            header.append(f"{name}_valid_pixels")
        writer.writerow(header)

        for frame_id in common_frame_ids:
            first_metric = next(iter(metric_tables.values()))[frame_id]
            row = [first_metric.probe_idx, frame_id]
            for name in experiment_to_metrics:
                metric = metric_tables[name][frame_id]
                for metric_name in metrics:
                    row.append(f"{metric.values[metric_name]:.6f}")
                row.append(metric.valid_pixel_count)
            writer.writerow(row)


def write_summary(output_path: Path, experiment_to_metrics: dict[str, list[FrameMetric]], metrics: list[str]) -> None:
    title_map = {
        "psnr": "Non-sky PSNR summary",
        "ssim": "Non-sky SSIM summary",
        "lpips": "Non-sky LPIPS summary",
    }

    lines: list[str] = []
    for metric_name in metrics:
        lines.append(title_map[metric_name])
        lines.append("")
        for name, frame_metrics in experiment_to_metrics.items():
            values = np.array([metric.values[metric_name] for metric in frame_metrics], dtype=np.float64)
            lines.append(
                f"{name}: frames={len(frame_metrics)}, mean={values.mean():.4f}, "
                f"median={np.median(values):.4f}, min={values.min():.4f}, max={values.max():.4f}"
            )
        lines.append("")
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def plot_metric(
    output_path: Path,
    experiment_to_metrics: dict[str, list[FrameMetric]],
    metric_name: str,
    x_axis_mode: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("tab10")

    for idx, (name, frame_metrics) in enumerate(experiment_to_metrics.items()):
        color = cmap(idx % cmap.N)
        if x_axis_mode == "index":
            frame_ids = [metric.probe_idx + 1 for metric in frame_metrics]
        else:
            frame_ids = [metric.frame_id for metric in frame_metrics]
        values = [metric.values[metric_name] for metric in frame_metrics]
        avg_value = float(np.mean(values))
        if metric_name == "psnr":
            label = f"{name} (mean={avg_value:.2f} dB)"
        else:
            label = f"{name} (mean={avg_value:.4f})"
        ax.plot(
            frame_ids,
            values,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=color,
            label=label,
        )

    if metric_name == "psnr":
        ax.set_title("Per-frame PSNR on Non-sky Pixels")
        ax.set_ylabel("PSNR (dB)")
    elif metric_name == "ssim":
        ax.set_title("Per-frame SSIM on Non-sky Pixels")
        ax.set_ylabel("SSIM")
    else:
        ax.set_title("Per-frame LPIPS on Non-sky Pixels")
        ax.set_ylabel("LPIPS (lower is better)")

    ax.set_xlabel("Frame Index" if x_axis_mode == "index" else "Frame ID")
    ax.legend()
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plot_path(output_dir: Path, output_prefix: str, metric_name: str, total_metrics: int) -> Path:
    if total_metrics == 1:
        return output_dir / f"{output_prefix}.png"
    return output_dir / f"{output_prefix}_{metric_name}.png"


def main() -> None:
    args = parse_args()
    metrics = validate_metrics(args.metrics)

    results_root = args.results_root.resolve()
    output_dir = results_root
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = None
    if "lpips" in metrics:
        lpips_model = lpips.LPIPS(net="alex").to(device)
        lpips_model.eval()

    experiment_to_metrics: dict[str, list[FrameMetric]] = {}
    for experiment_name in args.experiments:
        experiment_dir = results_root / experiment_name
        run_dir = find_single_run_dir(experiment_dir)
        experiment_to_metrics[experiment_name] = collect_frame_metrics(
            run_dir=run_dir,
            metrics=metrics,
            device=device,
            lpips_model=lpips_model,
        )

    output_prefix = args.output_prefix
    csv_path = output_dir / f"{output_prefix}.csv"
    summary_path = output_dir / f"{output_prefix}_summary.txt"

    write_csv(csv_path, experiment_to_metrics, metrics)
    write_summary(summary_path, experiment_to_metrics, metrics)
    print(f"Wrote csv: {csv_path}")
    print(f"Wrote summary: {summary_path}")

    for metric_name in metrics:
        plot_path = make_plot_path(output_dir, output_prefix, metric_name, len(metrics))
        plot_metric(plot_path, experiment_to_metrics, metric_name, args.x_axis)
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="ascii") as f:
        return json.load(f)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def _save_intersection_visual(
    path: Path,
    render_rgb: np.ndarray,
    intersection_mask: np.ndarray,
) -> None:
    output = np.zeros_like(render_rgb)
    output[intersection_mask] = render_rgb[intersection_mask]
    Image.fromarray(output).save(path)


def _resolve_depth_dir(results_root: Path, feature_frame_dir: Path) -> Path:
    new_layout_dir = results_root / "depth_probe" / feature_frame_dir.name
    if (new_layout_dir / "depth_diff_abs.png").exists():
        return new_layout_dir

    legacy_layout_dir = feature_frame_dir / "depth_probe"
    if (legacy_layout_dir / "depth_diff_abs.png").exists():
        return legacy_layout_dir

    return new_layout_dir


def _collect_feature_frame_dirs(results_root: Path) -> list[Path]:
    stage1_root = results_root / "stage1_feature_probe"
    if stage1_root.is_dir():
        search_root = stage1_root
    else:
        search_root = results_root

    return sorted(
        {
            path.parent.parent
            for path in search_root.rglob("gaussian_anomaly/matched_gaussians_render.png")
        }
    )


def process_frame(results_root: Path, frame_dir: Path, yellow_threshold: float) -> dict:
    depth_dir = _resolve_depth_dir(results_root, frame_dir)
    gaussian_dir = frame_dir / "gaussian_anomaly"

    depth_diff_path = depth_dir / "depth_diff_abs.png"
    depth_summary_path = depth_dir / "depth_probe_summary.json"
    depth_tensors_path = depth_dir / "depth_probe_tensors.pt"
    matched_render_path = gaussian_dir / "matched_gaussians_render.png"

    if not depth_diff_path.exists():
        raise FileNotFoundError(f"Missing file: {depth_diff_path}")
    if not depth_summary_path.exists():
        raise FileNotFoundError(f"Missing file: {depth_summary_path}")
    if not depth_tensors_path.exists():
        raise FileNotFoundError(f"Missing file: {depth_tensors_path}")
    if not matched_render_path.exists():
        raise FileNotFoundError(f"Missing file: {matched_render_path}")

    depth_summary = _load_json(depth_summary_path)
    vis_max = float(depth_summary["abs_diff_vis_max"])
    if vis_max <= 0.0:
        raise ValueError(f"Invalid abs_diff_vis_max={vis_max} in {depth_summary_path}")

    tensors = torch.load(depth_tensors_path, map_location="cpu", weights_only=True)
    abs_diff = tensors["abs_diff"].squeeze(0).numpy()
    valid_mask = tensors["valid_mask"].squeeze(0).numpy().astype(bool)

    depth_diff_rgb = np.array(Image.open(depth_diff_path).convert("RGB"))
    matched_render_rgb = np.array(Image.open(matched_render_path).convert("RGB"))

    if depth_diff_rgb.shape[:2] != matched_render_rgb.shape[:2]:
        raise ValueError(
            f"Image size mismatch in {frame_dir.name}: "
            f"{depth_diff_rgb.shape[:2]} vs {matched_render_rgb.shape[:2]}"
        )
    if abs_diff.shape != depth_diff_rgb.shape[:2]:
        raise ValueError(
            f"Tensor/image size mismatch in {frame_dir.name}: "
            f"{abs_diff.shape} vs {depth_diff_rgb.shape[:2]}"
        )

    normalized_abs_diff = np.clip(abs_diff / vis_max, 0.0, 1.0)
    yellow_mask = valid_mask & (normalized_abs_diff >= yellow_threshold)
    selected_mask = np.any(matched_render_rgb > 0, axis=2)
    intersection_mask = yellow_mask & selected_mask

    _save_mask(gaussian_dir / "depth_diff_yellow_mask.png", yellow_mask)
    _save_mask(gaussian_dir / "matched_gaussians_selected_mask.png", selected_mask)
    _save_mask(gaussian_dir / "depth_gaussian_intersection_mask.png", intersection_mask)
    _save_intersection_visual(
        gaussian_dir / "depth_gaussian_intersection.png",
        render_rgb=matched_render_rgb,
        intersection_mask=intersection_mask,
    )

    yellow_pixels = int(yellow_mask.sum())
    selected_pixels = int(selected_mask.sum())
    intersection_pixels = int(intersection_mask.sum())

    summary = {
        "frame_name": frame_dir.name,
        "yellow_threshold": float(yellow_threshold),
        "abs_diff_vis_max": vis_max,
        "yellow_pixels": yellow_pixels,
        "selected_pixels": selected_pixels,
        "intersection_pixels": intersection_pixels,
        "intersection_over_yellow": (
            float(intersection_pixels / yellow_pixels) if yellow_pixels > 0 else 0.0
        ),
        "intersection_over_selected": (
            float(intersection_pixels / selected_pixels) if selected_pixels > 0 else 0.0
        ),
    }

    with open(
        gaussian_dir / "depth_gaussian_intersection_summary.json",
        "w",
        encoding="ascii",
    ) as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Intersect yellow-highlighted depth-diff pixels with selected-gaussian "
            "pixels for every frame in a probe result tree."
        )
    )
    parser.add_argument("results_root", type=Path)
    parser.add_argument(
        "--yellow-threshold",
        type=float,
        default=0.90,
        help=(
            "Normalized inferno colormap threshold used to define yellow pixels in "
            "depth_diff_abs.png. Default: 0.90"
        ),
    )
    args = parser.parse_args()

    results_root = args.results_root
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    frame_dirs = [
        frame_dir
        for frame_dir in _collect_feature_frame_dirs(results_root)
        if (_resolve_depth_dir(results_root, frame_dir) / "depth_diff_abs.png").exists()
    ]
    if not frame_dirs:
        raise FileNotFoundError(
            "No matching frame directories with depth_probe/depth_diff_abs.png and "
            "stage1_feature_probe/*/gaussian_anomaly/matched_gaussians_render.png "
            "were found."
        )

    summaries = []
    for frame_dir in frame_dirs:
        summaries.append(
            process_frame(
                results_root,
                frame_dir,
                yellow_threshold=args.yellow_threshold,
            )
        )

    csv_path = results_root / "depth_gaussian_intersection_summary.csv"
    with open(csv_path, "w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_name",
                "yellow_threshold",
                "abs_diff_vis_max",
                "yellow_pixels",
                "selected_pixels",
                "intersection_pixels",
                "intersection_over_yellow",
                "intersection_over_selected",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    total_yellow = sum(item["yellow_pixels"] for item in summaries)
    total_selected = sum(item["selected_pixels"] for item in summaries)
    total_intersection = sum(item["intersection_pixels"] for item in summaries)
    aggregate = {
        "results_root": str(results_root),
        "frame_count": len(summaries),
        "yellow_threshold": float(args.yellow_threshold),
        "total_yellow_pixels": int(total_yellow),
        "total_selected_pixels": int(total_selected),
        "total_intersection_pixels": int(total_intersection),
        "intersection_over_yellow": (
            float(total_intersection / total_yellow) if total_yellow > 0 else 0.0
        ),
        "intersection_over_selected": (
            float(total_intersection / total_selected) if total_selected > 0 else 0.0
        ),
        "summary_csv": str(csv_path),
    }
    with open(
        results_root / "depth_gaussian_intersection_summary.json",
        "w",
        encoding="ascii",
    ) as f:
        json.dump(aggregate, f, indent=2)

    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

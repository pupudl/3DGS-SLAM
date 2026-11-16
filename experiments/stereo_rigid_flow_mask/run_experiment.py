#!/usr/bin/env python3
"""Dense rigid projection from SGBM depth, compared at sparse LK tracks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Use the lsgslam conda environment: "
            "conda run -n lsgslam python ..."
        ) from exc
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_path(output_root: str | Path, sequence: str, first: int, second: int, depth_source: str) -> Path:
    """Build a distinct directory for each frame pair and depth source."""
    run_name = f"{sequence}_{first:010d}_{second:010d}_{depth_source}"
    return project_path(output_root) / run_name


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def read_calibration(path: Path) -> tuple[np.ndarray, float]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            if key in {"P_rect_00", "P_rect_01"}:
                values[key] = np.fromstring(value, sep=" ", dtype=np.float64)
    try:
        p0 = values["P_rect_00"].reshape(3, 4)
        p1 = values["P_rect_01"].reshape(3, 4)
    except KeyError as exc:
        raise ValueError(f"Expected P_rect_00 and P_rect_01 in {path}") from exc
    K = p0[:, :3].copy()
    baseline = abs(float(p1[0, 3])) / float(K[0, 0])
    return K, baseline


def read_poses(path: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) != 17:
            raise ValueError(f"Expected id + 16 c2w values at {path}:{line_num}")
        poses[int(values[0])] = np.asarray(values[1:], dtype=np.float64).reshape(4, 4)
    return poses


def ensure_sky_masks(image_paths: list[Path], sequence_root: Path, cfg: dict[str, Any]) -> Path:
    """Reuse existing masks; use the lsgslam_windows preprocessor only if needed."""
    mask_dir = sequence_root / str(cfg.get("directory", "sky_masks"))
    needed = [mask_dir / f"{image_path.stem}.png" for image_path in image_paths]
    missing = [path for path in needed if not path.is_file()]
    if not missing:
        return mask_dir
    if str(cfg.get("mode", "precomputed_or_generate")) != "precomputed_or_generate":
        raise FileNotFoundError(f"Missing sky masks: {', '.join(str(p) for p in missing)}")
    config_path, checkpoint_path = cfg.get("mmseg_config"), cfg.get("mmseg_checkpoint")
    if not config_path or not checkpoint_path:
        raise FileNotFoundError("Sky masks are missing and MMSeg config/checkpoint are not configured.")
    windows_root = PROJECT_ROOT / "experiments" / "lsgslam_windows"
    if not windows_root.is_dir():
        raise FileNotFoundError(f"Could not find reusable preprocessor: {windows_root}")
    sys.path.insert(0, str(windows_root))
    from tools.preprocessing.frame_mask_preprocessor import precompute_sky_masks

    summary = precompute_sky_masks(
        image_paths=[str(path) for path in image_paths],
        sequence_root=str(sequence_root),
        mmseg_config=str(project_path(config_path)),
        mmseg_checkpoint=str(project_path(checkpoint_path)),
        device=str(cfg.get("device", "cuda")),
        sky_class_id=int(cfg.get("class_id", 10)),
        output_subdir=str(cfg.get("directory", "sky_masks")),
        overwrite=False,
    )
    print(f"Sky masks: {summary}")
    missing = [path for path in needed if not path.is_file()]
    if missing:
        raise RuntimeError(f"Sky-mask generation did not create: {missing}")
    return mask_dir


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read sky mask: {path}")
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def sgbm_depth(left: np.ndarray, right: np.ndarray, K: np.ndarray, baseline: float, cfg: dict[str, Any]):
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    requested = int(cfg.get("num_disparities", 128))
    disparities = max(16, requested // 16 * 16)
    if disparities >= gray_left.shape[1]:
        disparities = max(16, (gray_left.shape[1] - 1) // 16 * 16)
    block = max(3, int(cfg.get("block_size", 5)) | 1)
    matcher = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=disparities, blockSize=block,
        P1=8 * block * block, P2=32 * block * block,
        disp12MaxDiff=1, uniquenessRatio=int(cfg.get("uniqueness_ratio", 10)),
        speckleWindowSize=int(cfg.get("speckle_window_size", 100)),
        speckleRange=int(cfg.get("speckle_range", 2)), mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = float(K[0, 0] * baseline) / disparity[valid]
    return disparity, depth


def load_first_depth(
    source: str, sequence_root: Path, first: int, image_shape: tuple[int, int], left0: np.ndarray,
    right0: np.ndarray | None, K: np.ndarray, baseline: float, stereo_cfg: dict[str, Any], depth_cfg: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray, dict[str, str]]:
    """Load or estimate the first-frame depth selected in the run configuration."""
    if source == "sgbm":
        if right0 is None:
            raise FileNotFoundError("SGBM depth requires the first-frame right stereo image.")
        disparity, depth = sgbm_depth(left0, right0, K, baseline, stereo_cfg)
        return disparity, depth, {"source": source}
    if source == "precomputed_igev":
        subdir = str(depth_cfg.get("precomputed_igev_subdir", "depth_sceneflow"))
        depth_path = sequence_root / subdir / f"{first:010d}.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"Precomputed IGEV depth is missing: {depth_path}")
        depth = np.load(depth_path).astype(np.float32)
        if depth.shape != image_shape:
            raise ValueError(f"Precomputed IGEV depth has shape {depth.shape}, expected {image_shape}: {depth_path}")
        return None, depth, {"source": source, "path": str(depth_path)}
    raise ValueError("depth.source must be one of: sgbm, precomputed_igev")


def project(points0: np.ndarray, K: np.ndarray, transform_1_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points1 = (transform_1_0[:3, :3] @ points0.T).T + transform_1_0[:3, 3]
    pixels = np.full((len(points0), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(points1).all(axis=1) & (points1[:, 2] > 1e-6)
    pixels[valid, 0] = K[0, 0] * points1[valid, 0] / points1[valid, 2] + K[0, 2]
    pixels[valid, 1] = K[1, 1] * points1[valid, 1] / points1[valid, 2] + K[1, 2]
    return pixels, points1


def dense_rigid_projection(
    depth: np.ndarray, valid_depth: np.ndarray, K: np.ndarray, transform_1_0: np.ndarray,
) -> dict[str, np.ndarray]:
    """Back-project every valid first-frame depth pixel and project it into frame 1."""
    h, w = depth.shape
    points0 = np.full((h, w, 3), np.nan, dtype=np.float32)
    points1 = np.full((h, w, 3), np.nan, dtype=np.float32)
    predicted = np.full((h, w, 2), np.nan, dtype=np.float32)
    rigid_flow = np.full((h, w, 2), np.nan, dtype=np.float32)
    source_rows, source_cols = np.nonzero(valid_depth)
    z = depth[source_rows, source_cols]
    source_points0 = np.column_stack((
        (source_cols.astype(np.float32) - K[0, 2]) * z / K[0, 0],
        (source_rows.astype(np.float32) - K[1, 2]) * z / K[1, 1],
        z,
    )).astype(np.float32)
    source_predicted, source_points1 = project(source_points0, K, transform_1_0)
    projected_cols = np.rint(source_predicted[:, 0]).astype(np.int32)
    projected_rows = np.rint(source_predicted[:, 1]).astype(np.int32)
    projection_valid = (
        np.isfinite(source_predicted).all(axis=1)
        & (projected_cols >= 0) & (projected_cols < w)
        & (projected_rows >= 0) & (projected_rows < h)
    )
    points0[source_rows, source_cols] = source_points0
    points1[source_rows, source_cols] = source_points1
    predicted[source_rows, source_cols] = source_predicted
    rigid_flow[source_rows[projection_valid], source_cols[projection_valid], 0] = (
        source_predicted[projection_valid, 0] - source_cols[projection_valid]
    )
    rigid_flow[source_rows[projection_valid], source_cols[projection_valid], 1] = (
        source_predicted[projection_valid, 1] - source_rows[projection_valid]
    )
    projected_mask = np.zeros((h, w), dtype=bool)
    projected_mask[source_rows[projection_valid], source_cols[projection_valid]] = True
    return {
        "source_valid": valid_depth.copy(),
        "projection_valid": projected_mask,
        "points0": points0,
        "points1_static": points1,
        "predicted_p1": predicted,
        "rigid_flow_px": rigid_flow,
    }


def compute_tracks(
    image0: np.ndarray, image1: np.ndarray, valid_depth: np.ndarray, sky0: np.ndarray,
    sky1: np.ndarray, dense_projection: dict[str, np.ndarray], cfg: dict[str, Any],
) -> dict[str, np.ndarray]:
    gray0, gray1 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray0, maxCorners=int(cfg.get("max_corners", 12000)), qualityLevel=float(cfg.get("quality_level", 0.01)),
        minDistance=float(cfg.get("min_distance", 5.0)), blockSize=7, mask=(valid_depth & ~sky0).astype(np.uint8) * 255,
    )
    if corners is None:
        raise RuntimeError("No valid Shi-Tomasi features were found.")
    window = int(cfg.get("window_size", 21))
    lk = dict(winSize=(window, window), maxLevel=int(cfg.get("max_level", 3)),
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, status01, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, corners, None, **lk)
    p0_back, status10, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, p1, None, **lk)
    p0, observed = corners.reshape(-1, 2), p1.reshape(-1, 2)
    fb_error = np.linalg.norm(p0 - p0_back.reshape(-1, 2), axis=1)
    h, w = valid_depth.shape
    x0, y0 = np.rint(p0[:, 0]).astype(int), np.rint(p0[:, 1]).astype(int)
    x1, y1 = np.rint(observed[:, 0]).astype(int), np.rint(observed[:, 1]).astype(int)
    valid = (status01.ravel() > 0) & (status10.ravel() > 0)
    valid &= (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h) & (x1 >= 0) & (x1 < w) & (y1 >= 0) & (y1 < h)
    valid &= fb_error <= float(cfg.get("forward_backward_threshold_px", 1.0))
    safe_x0, safe_y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
    safe_x1, safe_y1 = np.clip(x1, 0, w - 1), np.clip(y1, 0, h - 1)
    valid &= (
        valid_depth[safe_y0, safe_x0]
        & dense_projection["projection_valid"][safe_y0, safe_x0]
        & ~sky1[safe_y1, safe_x1]
    )
    p0, observed, fb_error, x0, y0 = p0[valid], observed[valid], fb_error[valid], x0[valid], y0[valid]
    predicted = dense_projection["predicted_p1"][y0, x0]
    points0 = dense_projection["points0"][y0, x0]
    points1 = dense_projection["points1_static"][y0, x0]
    return {
        "p0": p0.astype(np.float32), "observed_p1": observed.astype(np.float32),
        "predicted_p1": predicted.astype(np.float32), "points0": points0.astype(np.float32),
        "points1_static": points1.astype(np.float32), "fb_error_px": fb_error.astype(np.float32),
        "residual_px": np.linalg.norm(observed - predicted, axis=1).astype(np.float32),
    }


def make_sparse_mask(points: np.ndarray, shape: tuple[int, int], radius: int, min_area: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x, y in np.rint(points).astype(np.int32):
        cv2.circle(mask, (int(x), int(y)), max(radius, 0), 255, -1)
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def forward_warp_first_to_second(image0: np.ndarray, dense_projection: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-pixel, z-buffered visualization of the dense first-frame projection."""
    h, w = image0.shape[:2]
    source_rows, source_cols = np.nonzero(dense_projection["projection_valid"])
    predicted = dense_projection["predicted_p1"][source_rows, source_cols]
    target_cols = np.rint(predicted[:, 0]).astype(np.int32)
    target_rows = np.rint(predicted[:, 1]).astype(np.int32)
    target_depth = dense_projection["points1_static"][source_rows, source_cols, 2]
    draw_order = np.argsort(target_depth)[::-1]
    warped = np.zeros_like(image0)
    coverage = np.zeros((h, w), dtype=np.uint8)
    warped[target_rows[draw_order], target_cols[draw_order]] = image0[source_rows[draw_order], source_cols[draw_order]]
    coverage[target_rows, target_cols] = 255
    return warped, coverage


def save_depth_visualization(depth: np.ndarray, output_path: Path, min_depth_m: float, max_depth_m: float) -> None:
    """Save metric depth with a fixed range; invalid pixels are black."""
    if max_depth_m <= min_depth_m:
        raise ValueError("Depth visualization requires max_depth_m > min_depth_m.")
    valid = np.isfinite(depth)
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - min_depth_m) / (max_depth_m - min_depth_m) * 255.0, 0, 255
    ).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    cv2.imwrite(str(output_path), colored)


def save_outputs(
    output: Path, image0: np.ndarray, image1: np.ndarray, right0: np.ndarray | None, sky0: np.ndarray, sky1: np.ndarray,
    disparity: np.ndarray | None, depth: np.ndarray, depth_range_m: tuple[float, float],
    dense_projection: dict[str, np.ndarray], tracks: dict[str, np.ndarray], dynamic: np.ndarray,
    mask0: np.ndarray, mask1: np.ndarray,
) -> None:
    inputs_dir = output / "inputs"
    depth_dir = output / "depth"
    rigid_dir = output / "rigid_projection"
    sparse_dir = output / "sparse_comparison"
    masks_dir = output / "masks"
    for directory in (inputs_dir, depth_dir, rigid_dir, sparse_dir, masks_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_depth_visualization(depth, depth_dir / "depth_first_meters.png", *depth_range_m)
    if disparity is not None:
        finite = np.isfinite(disparity) & (disparity > 0)
        displayed = np.zeros(disparity.shape, dtype=np.uint8)
        if finite.any():
            displayed[finite] = cv2.normalize(disparity[finite], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8).ravel()
        cv2.imwrite(str(depth_dir / "sgbm_disparity.png"), cv2.applyColorMap(displayed, cv2.COLORMAP_TURBO))

    rigid_magnitude = np.linalg.norm(dense_projection["rigid_flow_px"], axis=2)
    finite_flow = np.isfinite(rigid_magnitude)
    displayed_flow = np.zeros(rigid_magnitude.shape, dtype=np.uint8)
    if finite_flow.any():
        displayed_flow[finite_flow] = cv2.normalize(
            rigid_magnitude[finite_flow], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8).ravel()
    cv2.imwrite(str(rigid_dir / "rigid_flow_magnitude_first.png"), cv2.applyColorMap(displayed_flow, cv2.COLORMAP_TURBO))
    warped, coverage = forward_warp_first_to_second(image0, dense_projection)
    cv2.imwrite(str(rigid_dir / "rigid_warp_first_to_second.png"), warped)
    cv2.imwrite(str(rigid_dir / "rigid_projection_coverage_second.png"), coverage)

    # The SGBM input is the first-frame stereo pair. The second left image is
    # retained separately because it is the temporal target for LK flow.
    stereo_left = image0.copy(); stereo_left[sky0] = 0
    temporal_second = image1.copy(); temporal_second[sky1] = 0
    cv2.imwrite(str(inputs_dir / "stereo_left_first_no_sky.png"), stereo_left)
    if right0 is not None:
        stereo_right = right0.copy(); stereo_right[sky0] = 0
        cv2.imwrite(str(inputs_dir / "stereo_right_first_no_sky.png"), stereo_right)
    cv2.imwrite(str(inputs_dir / "temporal_left_second_no_sky.png"), temporal_second)
    cv2.imwrite(str(inputs_dir / "sky_mask_first.png"), sky0.astype(np.uint8) * 255)
    cv2.imwrite(str(inputs_dir / "sky_mask_second.png"), sky1.astype(np.uint8) * 255)
    for suffix, image, mask in (("first", image0, mask0), ("second", image1, mask1)):
        overlay = image.copy(); overlay[mask > 0] = (0.35 * overlay[mask > 0] + np.array([0, 0, 166])).astype(np.uint8)
        cv2.imwrite(str(masks_dir / f"dynamic_mask_{suffix}.png"), mask)
        cv2.imwrite(str(masks_dir / f"dynamic_overlay_{suffix}.png"), overlay)

    track_view = image0.copy()
    rigid_vs_lk_view = image0.copy()
    for p0, observed_p1, predicted_p1, is_dynamic in zip(
        tracks["p0"], tracks["observed_p1"], tracks["predicted_p1"], dynamic,
    ):
        source = tuple(np.rint(p0).astype(int))
        observed = tuple(np.rint(observed_p1).astype(int))
        static_prediction = tuple(np.rint(predicted_p1).astype(int))
        color = (0, 0, 255) if is_dynamic else (0, 200, 0)
        cv2.line(track_view, source, observed, color, 1, cv2.LINE_AA)
        cv2.line(rigid_vs_lk_view, source, static_prediction, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(rigid_vs_lk_view, static_prediction, 2, (255, 255, 0), -1, cv2.LINE_AA)
        cv2.line(rigid_vs_lk_view, source, observed, color, 1, cv2.LINE_AA)
        cv2.circle(rigid_vs_lk_view, observed, 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(sparse_dir / "sparse_tracks_first.png"), track_view)
    cv2.imwrite(str(sparse_dir / "sparse_rigid_vs_lk_first.png"), rigid_vs_lk_view)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXPERIMENT_ROOT / "configs" / "kitti360_0000.yaml")
    args = parser.parse_args()
    config_path, config = args.config.resolve(), load_config(args.config.resolve())
    dataset, frames = config.get("dataset", {}), config["frames"]
    if dataset.get("type") != "kitti360":
        raise ValueError("This experiment currently supports dataset.type: kitti360 only.")
    data_root, sequence = project_path(dataset["data_root"]), str(dataset["sequence"])
    first, second = int(frames["first"]), int(frames["second"])
    if first == second:
        raise ValueError("frames.first and frames.second must be different.")
    depth_cfg = config.get("depth", {})
    depth_source = str(depth_cfg.get("source", "sgbm"))
    sequence_root = data_root / "data_2d_raw" / sequence
    left_dir, right_dir = sequence_root / "image_00" / "data_rect", sequence_root / "image_01" / "data_rect"
    left0_path = left_dir / f"{first:010d}.png"
    left1_path = left_dir / f"{second:010d}.png"
    right0_path = right_dir / f"{first:010d}.png"
    required_paths = [left0_path, left1_path]
    if depth_source == "sgbm":
        required_paths.append(right0_path)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required image is missing: {path}")
    image0, image1 = read_image(left0_path), read_image(left1_path)
    right0 = read_image(right0_path) if right0_path.is_file() else None
    if image0.shape != image1.shape or (right0 is not None and image0.shape != right0.shape):
        raise ValueError("Stereo and temporal images must have identical shapes.")
    K, baseline = read_calibration(data_root / "calibration" / "perspective.txt")
    poses = read_poses(data_root / "data_poses" / sequence / "cam0_to_world.txt")
    if first not in poses or second not in poses:
        raise KeyError(f"GT c2w pose missing for pair {first}->{second}.")
    sky_dir = ensure_sky_masks([left0_path, left1_path], sequence_root, config.get("sky_mask", {}))
    sky0, sky1 = read_mask(sky_dir / f"{first:010d}.png", image0.shape[:2]), read_mask(sky_dir / f"{second:010d}.png", image1.shape[:2])
    processing_start = time.perf_counter()
    stereo0 = image0.copy(); stereo0[sky0] = 0
    stereo_right0 = None if right0 is None else right0.copy()
    if stereo_right0 is not None:
        stereo_right0[sky0] = 0
    disparity, depth, depth_info = load_first_depth(
        depth_source, sequence_root, first, image0.shape[:2], stereo0, stereo_right0,
        K, baseline, config.get("stereo", {}), depth_cfg,
    )
    stereo = config.get("stereo", {})
    valid_depth = np.isfinite(depth) & ~sky0 & (depth >= float(stereo.get("min_depth_m", 1.0))) & (depth <= float(stereo.get("max_depth_m", 80.0)))
    depth[~valid_depth] = np.nan
    transform_1_0 = np.linalg.inv(poses[second]) @ poses[first]
    dense_projection = dense_rigid_projection(depth, valid_depth, K, transform_1_0)
    tracks = compute_tracks(image0, image1, valid_depth, sky0, sky1, dense_projection, config.get("sparse_flow", {}))
    dynamic_cfg = config.get("dynamic", {})
    dynamic = tracks["residual_px"] > float(dynamic_cfg.get("residual_threshold_px", 3.0))
    radius, min_area = int(dynamic_cfg.get("splat_radius_px", 4)), int(dynamic_cfg.get("min_component_area_px", 50))
    mask0, mask1 = make_sparse_mask(tracks["p0"][dynamic], depth.shape, radius, min_area), make_sparse_mask(tracks["observed_p1"][dynamic], depth.shape, radius, min_area)
    mask0[sky0] = 0; mask1[sky1] = 0
    computation_time_seconds = time.perf_counter() - processing_start
    output_subdir = config.get("output_subdir")
    output = (
        project_path(config["output_dir"]) / str(output_subdir)
        if output_subdir is not None
        else output_path(config["output_dir"], sequence, first, second, depth_source)
    )
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "run_config.yaml")
    (output / "depth").mkdir(exist_ok=True)
    (output / "rigid_projection").mkdir(exist_ok=True)
    (output / "sparse_comparison").mkdir(exist_ok=True)
    np.save(output / "depth" / "depth_first.npy", depth)
    (output / "depth" / "depth_source.json").write_text(json.dumps(depth_info, indent=2), encoding="utf-8")
    np.savez_compressed(output / "rigid_projection" / "dense_rigid_projection.npz", **dense_projection, K=K, transform_1_0=transform_1_0)
    np.savez_compressed(output / "sparse_comparison" / "tracks_and_residuals.npz", **tracks, dynamic=dynamic, K=K, transform_1_0=transform_1_0)
    save_outputs(
        output, image0, image1, right0, sky0, sky1, disparity, depth,
        (float(stereo.get("min_depth_m", 1.0)), float(stereo.get("max_depth_m", 80.0))),
        dense_projection, tracks, dynamic, mask0, mask1,
    )
    summary = {
        "sequence": sequence, "first_frame": first, "second_frame": second, "baseline_m": baseline,
        "depth_source": depth_info,
        "valid_depth_pixels": int(valid_depth.sum()), "projected_depth_pixels": int(dense_projection["projection_valid"].sum()),
        "tracked_points": int(len(dynamic)), "dynamic_points": int(dynamic.sum()),
        "dynamic_mask_first_pixels": int((mask0 > 0).sum()), "dynamic_mask_second_pixels": int((mask1 > 0).sum()),
        "residual_px_median": float(np.median(tracks["residual_px"])) if len(dynamic) else None,
        "computation_time_seconds_excluding_sky_mask": computation_time_seconds,
        "transform_1_0": transform_1_0.tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"Outputs: {output}")


if __name__ == "__main__":
    main()

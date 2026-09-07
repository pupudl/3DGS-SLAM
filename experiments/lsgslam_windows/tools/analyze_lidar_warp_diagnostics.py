#!/usr/bin/env python3

import argparse
import csv
import glob
import json
import os
import re
import sys


EXPERIMENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, EXPERIMENT_ROOT)

from utils.lidar_warp_diagnostics import CSV_FIELDS, LidarWarpDiagnostics


BOOL_FIELDS = {"gradient_sampled", "active"}
INT_FIELDS = {"frame", "tracking_iteration", "num_correspondences"}
TEXT_FIELDS = {"frame_id", "mode"}


def _parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _parse_row(row, start_idx, stride):
    parsed = {}
    for field in CSV_FIELDS:
        value = row.get(field, "")
        if field in BOOL_FIELDS:
            parsed[field] = _parse_bool(value)
        elif field in INT_FIELDS:
            parsed[field] = int(float(value))
        elif field in TEXT_FIELDS:
            parsed[field] = str(value)
        else:
            parsed[field] = float(value) if value not in ("", None) else float("nan")

    # A chunk stores a local dataset time index. Use the raw KITTI frame id
    # when possible, otherwise reconstruct it from chunk metadata.
    try:
        parsed["frame"] = int(parsed["frame_id"])
    except (TypeError, ValueError):
        parsed["frame"] = int(start_idx) + int(parsed["frame"]) * int(stride)
    return parsed


def _discover_chunks(base_folder, scene_name):
    pattern = re.compile(rf"^{re.escape(scene_name)}_(\d+)_(\d+)_(\d+)$")
    chunks = []
    for path in glob.glob(os.path.join(base_folder, f"{scene_name}_*")):
        if not os.path.isdir(path):
            continue
        match = pattern.match(os.path.basename(path))
        if match is None:
            continue
        stats_path = os.path.join(path, "eval", "lidar_warp_stats.csv")
        if os.path.isfile(stats_path):
            chunks.append((int(match.group(1)), int(match.group(3)), path, stats_path))
    return sorted(chunks, key=lambda item: item[0])


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-chunk LiDAR warp diagnostics for one sequence."
    )
    parser.add_argument("--base-folder", required=True, help="Sequence result group directory.")
    parser.add_argument("--scene-name", required=True, help="Sequence prefix, e.g. 04.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to BASE_FOLDER/LidarWarpDiagnostics.",
    )
    args = parser.parse_args()

    base_folder = os.path.abspath(args.base_folder)
    chunks = _discover_chunks(base_folder, args.scene_name)
    if not chunks:
        raise RuntimeError(
            f"No per-chunk eval/lidar_warp_stats.csv found under {base_folder} "
            f"for scene {args.scene_name}."
        )

    output_dir = args.output_dir or os.path.join(base_folder, "LidarWarpDiagnostics")
    lidar_weight = 1.0
    all_rows = []
    chunk_names = []
    for start_idx, stride, chunk_path, stats_path in chunks:
        chunk_names.append(os.path.basename(chunk_path))
        with open(stats_path, "r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                all_rows.append(_parse_row(row, start_idx, stride))
        summary_path = os.path.join(chunk_path, "eval", "lidar_warp_summary.json")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as handle:
                lidar_weight = float(json.load(handle).get("lidar_warp_weight", lidar_weight))

    diagnostics = LidarWarpDiagnostics(
        output_dir,
        {
            "enabled": True,
            "weight": lidar_weight,
            "diagnostics": {"enabled": True},
        },
    )
    diagnostics.rows = sorted(
        all_rows,
        key=lambda row: (row["frame"], row["tracking_iteration"]),
    )
    summary = diagnostics.finalize()
    summary["scene_name"] = str(args.scene_name)
    summary["chunks"] = chunk_names
    summary_path = os.path.join(output_dir, "lidar_warp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(f"Aggregated {len(chunks)} chunks and {len(all_rows)} rows")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

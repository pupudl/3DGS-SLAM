#!/usr/bin/env python3
"""Compare sparse-to-Gaussian ownership with ownership after initial joint fitting.

Both stages use the same first-frame Gaussians and the same motion IDs. This
visualization deliberately stops before render-guided supplemental matching.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pair_utils import load_depth, make_gaussians, resolve, stereo_io


PALETTE_RGB = np.array([
    [.18, .68, 1.], [1., .52, .16], [.30, .88, .42], [.92, .37, .82],
    [1., .85, .24], [.28, .86, .87], [.75, .56, 1.], [1., .47, .52],
], np.float32)
# Structural edits create new lineage identities even when the active motion
# count stays small. Keep the original eight colors and extend deterministically.
PALETTE_RGB = np.vstack((PALETTE_RGB, np.array([
    colorsys.hsv_to_rgb((index * .6180339887498949) % 1., .62, .95)
    for index in range(8, 32)
], np.float32)))


def load_first_map(cfg: dict) -> tuple[dict, np.ndarray]:
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    first = int(cfg['frames']['first'])
    left = stereo_io.read_image(sequence_root / 'image_00/data_rect' / f'{first:010d}.png')
    right = stereo_io.read_image(sequence_root / 'image_01/data_rect' / f'{first:010d}.png')
    K, baseline = stereo_io.read_calibration(data_root / 'calibration/perspective.txt')
    depth = load_depth(sequence_root, first, left, right, K, baseline, cfg['depth'])
    sky = sequence_root / 'sky_masks' / f'{first:010d}.png'
    if sky.is_file():
        depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
    gc = cfg['gaussians']
    gmap = make_gaussians(left, depth, K, int(gc['stride']),
                          float(gc['max_depth_jump_m']),
                          float(gc.get('projected_sigma_px', 1.6)),
                          gc.get('initialization', 'surface_aligned'))
    return gmap, left


def caption(image: np.ndarray, title: str, detail: str = '') -> np.ndarray:
    bar = np.full((54, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (25, 25, 25), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(bar, detail, (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def save(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


def pixel_coordinates(gmap: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    uv = np.rint(gmap['uv']).astype(np.int32)
    if np.any((uv[:, 0] < 0) | (uv[:, 0] >= w) |
              (uv[:, 1] < 0) | (uv[:, 1] >= h)):
        raise ValueError('A Gaussian source pixel lies outside the first image')
    return uv[:, 1], uv[:, 0]


def ownership_image(image: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                    labels: np.ndarray) -> np.ndarray:
    result = image.copy()
    palette = PALETTE_RGB[labels, ::-1] * 255
    result[rows, cols] = (0.35 * image[rows, cols] + 0.65 * palette).astype(np.uint8)
    return result


def confidence_image(image: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                     confidence: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    values = np.rint(np.clip(confidence, 0, 1) * 255).astype(np.uint8)
    result[rows, cols] = cv2.applyColorMap(values[:, None], cv2.COLORMAP_VIRIDIS)[:, 0]
    return result


def changed_image(image: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                  changed: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    selected_rows, selected_cols = rows[changed], cols[changed]
    result[selected_rows, selected_cols] = (
        0.2 * result[selected_rows, selected_cols] +
        0.8 * np.array([0, 0, 255], np.float32)).astype(np.uint8)
    return result


def visualize(result_dir: Path, output_dir: Path | None = None) -> dict:
    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / 'ownership_stage_comparison').resolve()
    cfg = yaml.safe_load((result_dir / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    with np.load(result_dir / 'pairwise_state.npz') as saved:
        before = saved['q_gaussian_sparse'].copy()
        after = saved['q_gaussian_joint'].copy()
        H_before = saved['H_sparse'].copy()
        H_after = saved['H_joint'].copy()
        initial_count = int(summary['initial_3d_anchors'])
        initial_matches = saved['anchors_final'][:initial_count].copy()
        anchor_state = ({name: saved[f'anchor_gaussian_{name}_final'][:initial_count].copy()
                         for name in ('indices', 'xyz', 'uv', 'color', 'scale')}
                        if 'anchor_gaussian_indices_final' in saved.files else None)
    gmap, image = load_first_map(cfg)
    if before.shape != after.shape or len(before) != len(gmap['xyz']):
        raise ValueError('Sparse/joint ownership does not match the first-frame map')
    n, k = before.shape
    if H_before.shape != H_after.shape or len(H_before) != k or k > len(PALETTE_RGB):
        raise ValueError('Sparse/joint motion IDs do not align')
    if not (np.allclose(before.sum(axis=1), 1, atol=1e-3) and
            np.allclose(after.sum(axis=1), 1, atol=1e-3)):
        raise ValueError('Invalid Gaussian motion probabilities')
    if anchor_state is not None:
        ids = anchor_state['indices'].astype(np.int64)
        if (len(ids) != initial_count or len(np.unique(ids)) != len(ids) or
                np.any(ids < 0) or np.any(ids >= n) or
                not np.allclose(anchor_state['uv'], initial_matches[:, :2], atol=1e-4)):
            raise ValueError('Initial anchor Gaussian state cannot be reconstructed')
        for name in ('xyz', 'uv', 'color', 'scale'):
            gmap[name][ids] = anchor_state[name]
    rows, cols = pixel_coordinates(gmap, image.shape[:2])
    before_labels, after_labels = before.argmax(axis=1), after.argmax(axis=1)
    changed = before_labels != after_labels
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {
        'initial_hard.png': ownership_image(image, rows, cols, before_labels),
        'joint_hard.png': ownership_image(image, rows, cols, after_labels),
        'changed_hard.png': changed_image(image, rows, cols, changed),
        'initial_confidence.png': confidence_image(image, rows, cols, before.max(axis=1)),
        'joint_confidence.png': confidence_image(image, rows, cols, after.max(axis=1)),
    }
    for filename, panel in images.items():
        save(output_dir / filename, panel)
    transitions = np.zeros((k, k), np.int64)
    np.add.at(transitions, (before_labels, after_labels), 1)
    report = {
        'frames': summary['frames'],
        'stages': ['after sparse-to-Gaussian ownership initialization',
                   'after initial joint optimization, before render-guided feedback'],
        'gaussians': n,
        'initial_3d_matches': initial_count,
        'motions': k,
        'reference_model': int(summary['reference_sparse']),
        'hard_changed_gaussians': int(changed.sum()),
        'hard_changed_fraction': float(changed.mean()),
        'initial_hard_counts': np.bincount(before_labels, minlength=k).tolist(),
        'joint_hard_counts': np.bincount(after_labels, minlength=k).tolist(),
        'hard_transitions_initial_rows_joint_columns': transitions.tolist(),
        'mean_probability_l1_change': float(np.abs(before - after).sum(axis=1).mean()),
        'interpretation': ('Both maps show the same first-frame Gaussians and motion IDs. '
                           'Hard labels are argmax of soft probabilities. Red indicates '
                           'a changed hard label; it does not indicate ground-truth motion. '
                           'Feedback matches and final structural changes are excluded.'),
    }
    (output_dir / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    panel_size = (704, 188)
    def small(name: str) -> np.ndarray:
        return cv2.resize(images[name], panel_size, interpolation=cv2.INTER_AREA)
    top = np.hstack((
        caption(small('initial_hard.png'), '1  Sparse evidence -> Gaussians',
                'Initial hard ownership; before joint optimization'),
        caption(small('joint_hard.png'), '2  After initial joint optimization',
                'Same first-frame Gaussians and same M colors'),
        caption(small('changed_hard.png'), '3  Changed hard assignment',
                f'Red = switched M; {changed.sum():,}/{n:,} ({changed.mean():.1%})'),
    ))
    stats = np.full((188, 704, 3), 255, np.uint8)
    cv2.putText(stats, f'{n:,} Gaussians; {initial_count} initial 3D matches',
                (13, 29), cv2.FONT_HERSHEY_SIMPLEX, .59, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(stats, f'Mean probability L1 change: {report["mean_probability_l1_change"]:.3f}',
                (13, 61), cv2.FONT_HERSHEY_SIMPLEX, .52, (50, 50, 50), 1, cv2.LINE_AA)
    for model in range(k):
        y = 86 + 24 * model
        color = tuple(int(round(x * 255)) for x in PALETTE_RGB[model, ::-1])
        cv2.rectangle(stats, (14, y - 12), (35, y + 5), color, -1)
        cv2.putText(stats, f'M{model}: {report["initial_hard_counts"][model]:,} -> '
                   f'{report["joint_hard_counts"][model]:,}', (44, y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, .5, (35, 35, 35), 1, cv2.LINE_AA)
    bottom = np.hstack((
        caption(small('initial_confidence.png'), 'Initial confidence',
                'Brighter = higher maximum motion probability'),
        caption(small('joint_confidence.png'), 'Joint confidence',
                'A hard color can change even for a small probability shift'),
        caption(stats, 'Counts by motion', 'M IDs are comparable only within these two stages'),
    ))
    save(output_dir / 'overview.png', np.vstack((top, bottom)))
    print(f'{output_dir}: {n:,} Gaussians; {int(changed.sum()):,} hard labels changed')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--result', type=Path, help='One saved pair/full_feedback directory')
    choice.add_argument('--group', type=Path, help='Four-pair output directory')
    parser.add_argument('--output', type=Path, help='Override output for a single result')
    args = parser.parse_args()
    if args.group:
        if args.output:
            parser.error('--output is only supported with --result')
        results = sorted(p.parent for p in args.group.glob('*/full_feedback/pairwise_state.npz'))
        if not results:
            parser.error(f'No saved pair results found under {args.group}')
        for result in results:
            visualize(result)
    else:
        visualize(args.result, args.output)


if __name__ == '__main__':
    main()

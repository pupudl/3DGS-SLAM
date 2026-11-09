#!/usr/bin/env python3
"""Evaluate saved two-frame motion ownership on projected Gaussian centers.

Evaluation masks are read only after estimation. This is a center-pixel,
nearest-depth diagnostic, not a rasterized semantic segmentation metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pair_utils import bind_anchor_gaussians, lift, load_depth, resolve, stereo_io
from visualize_motion import source_map


def projected_labels(xyz: np.ndarray, H: np.ndarray, q: np.ndarray,
                     K: np.ndarray, shape: tuple[int, int], reference: int) -> np.ndarray:
    h, w = shape
    owner = q.argmax(axis=1)
    transformed = np.einsum('nij,nj->ni', H[owner, :3, :3], xyz) + H[owner, :3, 3]
    depth = transformed[:, 2]
    x = np.rint(K[0, 0] * transformed[:, 0] / np.maximum(depth, 1e-6) + K[0, 2]).astype(np.int32)
    y = np.rint(K[1, 1] * transformed[:, 1] / np.maximum(depth, 1e-6) + K[1, 2]).astype(np.int32)
    valid = (depth > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    source_ids = np.flatnonzero(valid)
    pixels = y[valid].astype(np.int64) * w + x[valid]
    order = np.lexsort((depth[valid], pixels))
    sorted_pixels = pixels[order]
    unique_pixels, first = np.unique(sorted_pixels, return_index=True)
    winner = source_ids[order[first]]
    result = np.full(h * w, -1, np.int8)
    result[unique_pixels] = (owner[winner] != reference).astype(np.int8)
    return result.reshape(h, w)


def metrics(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
    dynamic = prediction == 1
    tp = int((valid & dynamic & truth).sum())
    fp = int((valid & dynamic & ~truth).sum())
    fn = int((valid & ~dynamic & truth).sum())
    tn = int((valid & ~dynamic & ~truth).sum())
    return dict(covered=int(valid.sum()), predicted_dynamic=tp + fp,
                tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                iou=tp / max(tp + fp + fn, 1),
                static_false_positive_rate=fp / max(fp + tn, 1))


def analyze(result: Path, mask_path: Path) -> dict:
    cfg = yaml.safe_load((result / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
    with np.load(result / 'pairwise_state.npz') as archive:
        data = {key: archive[key].copy() for key in archive.files
                if key in ('H_joint', 'H_final', 'q_gaussian_joint', 'q_gaussian_final',
                           'anchors_final', 'anchor_gaussian_indices_final',
                           'anchor_gaussian_xyz_final')}
    base_map, K, first_image, second_image = source_map(cfg)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != second_image.shape[:2]:
        raise ValueError(f'Missing or mismatched target mask: {mask_path}')
    truth = mask > 127
    initial_count = int(summary['initial_3d_anchors'])
    first, second = map(int, summary['frames'])
    sequence_root = (resolve(cfg['dataset']['data_root']) / 'data_2d_raw' /
                     cfg['dataset']['sequence'])
    right_dir = sequence_root / 'image_01' / 'data_rect'
    right_first = stereo_io.read_image(right_dir / f'{first:010d}.png')
    right_second = stereo_io.read_image(right_dir / f'{second:010d}.png')
    _, baseline = stereo_io.read_calibration(resolve(cfg['dataset']['data_root']) /
                                              'calibration' / 'perspective.txt')
    d0 = load_depth(sequence_root, first, first_image, right_first, K, baseline, cfg['depth'])
    d1 = load_depth(sequence_root, second, second_image, right_second, K, baseline, cfg['depth'])
    if cfg['joint'].get('anchor_gaussians', False):
        initial_anchors = lift(data['anchors_final'][:initial_count], d0, d1, K)
        initial_map, _ = bind_anchor_gaussians(base_map, initial_anchors, first_image, K)
    else:
        initial_map = base_map
    final_xyz = base_map['xyz'].copy()
    if 'anchor_gaussian_indices_final' in data:
        final_xyz[data['anchor_gaussian_indices_final']] = data['anchor_gaussian_xyz_final']
    joint = projected_labels(initial_map['xyz'], data['H_joint'], data['q_gaussian_joint'],
                             K, truth.shape, int(summary['reference_sparse']))
    final = projected_labels(final_xyz, data['H_final'], data['q_gaussian_final'],
                             K, truth.shape, int(summary['reference_final']))
    common = (joint >= 0) & (final >= 0)
    return dict(result=str(result), frames=summary['frames'],
                initial_models=len(data['H_joint']), final_models=len(data['H_final']),
                joint=metrics(joint, truth, common), final=metrics(final, truth, common),
                initial_anchors=initial_count, final_anchors=int(summary['final_3d_anchors']),
                accepted_updates=[dict(cycle=cycle['cycle'], kind=proposal['kind'])
                                  for cycle in summary['cycles'] for proposal in cycle['proposals']
                                  if proposal.get('accepted', False)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--mask', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze(args.result, args.mask)
    content = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n', encoding='utf-8')
    print(content)


if __name__ == '__main__':
    main()

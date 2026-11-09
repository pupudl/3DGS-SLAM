#!/usr/bin/env python3
"""Compare saved three-point sparse hypotheses with a new sampling config.

Uses the exact initial 3D matches from the saved runs. This CPU-only check
does not rerun feature matching, Gaussian optimization, or feedback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from pair_utils import discover, errors, resolve
from visualize_sparse_hypotheses import load_inputs


def measurements(anchors: dict, models: list[np.ndarray], supports: list[int],
                 threshold: float) -> dict:
    residuals = np.stack([errors(anchors['x0'], anchors['x1'], H)
                          for H in models], axis=1)
    best = residuals.min(axis=1)
    return {
        'model_count': len(models),
        'supports': [int(n) for n in supports],
        'median_best_error_m': float(np.median(best)),
        'p90_best_error_m': float(np.quantile(best, 0.9)),
        'explained_fraction': float(np.mean(best < threshold)),
        'median_support_error_m': [float(np.median(residuals[residuals[:, k] < threshold, k]))
                                   for k in range(len(models))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Directory containing the four saved full_feedback runs')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    baseline = resolve(args.baseline)
    result = {'baseline': str(baseline), 'config': str(args.config.resolve()), 'pairs': {}}
    for old in sorted(baseline.glob('*/full_feedback')):
        summary = json.loads((old / 'summary.json').read_text(encoding='utf-8'))
        old_config = yaml.safe_load((old / 'config.yaml').read_text(encoding='utf-8'))
        with np.load(old / 'pairwise_state.npz') as saved:
            matches = saved['anchors_final'][:int(summary['initial_3d_anchors'])].copy()
            saved_H = saved['H_sparse'].copy()
        _, _, anchors = load_inputs(old_config, matches)
        if len(anchors['x0']) != len(matches):
            raise ValueError(f'Initial 3D matches changed for {old}')
        old_models, _, old_supports = discover(
            anchors, old_config['motion'], int(old_config.get('seed', 7)))
        if (old_supports != summary['initial_model_supports'] or
                not np.allclose(np.stack(old_models), saved_H, atol=1e-7, rtol=1e-7)):
            raise ValueError(f'Could not reproduce saved baseline for {old}')
        same_budget_cfg = {**config['motion'],
                           'ransac_sample_size': int(config['motion'].get('ransac_seed_size', 3))}
        same_budget_models, _, same_budget_supports = discover(
            anchors, same_budget_cfg, int(config.get('seed', 7)))
        new_models, _, new_supports = discover(
            anchors, config['motion'], int(config.get('seed', 7)))
        threshold = float(config['motion']['inlier_threshold_m'])
        result['pairs'][old.parent.name] = {
            'initial_3d_matches': len(matches),
            'baseline': measurements(anchors, old_models, old_supports, threshold),
            'three_point_same_budget': measurements(
                anchors, same_budget_models, same_budget_supports, threshold),
            'ransac8': measurements(anchors, new_models, new_supports, threshold),
        }
        pair = result['pairs'][old.parent.name]
        print(f'{old.parent.name}: baseline {pair["baseline"]["supports"]}, '
              f'3-point same budget {pair["three_point_same_budget"]["supports"]}, '
              f'8-point {pair["ransac8"]["supports"]}', flush=True)
    if not result['pairs']:
        raise FileNotFoundError(f'No saved runs in {baseline}')
    output = resolve(config['output_dir']) / 'sparse_sampling_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()

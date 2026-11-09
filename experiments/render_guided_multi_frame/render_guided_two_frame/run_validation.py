"""Run reproducible, grouped ablations of the frozen-map two-frame experiment."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DEFAULT_PAIRS = [(2433, 2435), (2437, 2439), (2445, 2447)]


def mask_for_pair(data_root: Path, sequence: str, first: int, second: int) -> Path:
    probe = (data_root / 'data_2d_raw' / sequence /
             'gt_dynamic_masks_2435_2463_stride2' / 'rigidmask_frontend_probe')
    paths = sorted(probe.glob(f'*_frame_{second:010d}_from_{first:010d}/dynamic/dynamic_mask.png'))
    if len(paths) != 1:
        raise FileNotFoundError(f'Expected one evaluation mask for {first}->{second}; found {paths}')
    return paths[0]


def largest_object_box(mask: np.ndarray, margin: int = 10) -> list[int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        raise ValueError('No dynamic component available for controlled holdout')
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = [int(v) for v in stats[i]]
    height, width = mask.shape
    return [max(0, x-margin), max(0, y-margin), min(width, x+w+margin), min(height, y+h+margin)]


def variant_config(base: dict, variant: str) -> dict:
    cfg = copy.deepcopy(base)
    if variant in ('single_rigid',):
        cfg['motion']['max_components'] = 1
        cfg['discovery']['max_components_total'] = 1
        cfg['discovery']['max_cycles'] = 0
    elif variant in ('multi_no_feedback', 'holdout_no_feedback'):
        cfg['discovery']['max_cycles'] = 0
    elif variant in ('full_feedback', 'holdout_full_feedback'):
        pass
    else:
        raise ValueError(variant)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/kitti360_0000.yaml')
    parser.add_argument('--group', default='validation_20261007')
    parser.add_argument('--pairs', nargs='*', default=[f'{a}:{b}' for a, b in DEFAULT_PAIRS])
    parser.add_argument('--holdout-pairs', nargs='*', default=['2433:2435', '2445:2447'])
    args = parser.parse_args()
    if Path(args.group).name != args.group or args.group in ('.', '..'):
        raise ValueError('Group must be one directory name')
    pairs = [tuple(map(int, value.split(':'))) for value in args.pairs]
    holdout_pairs = {tuple(map(int, value.split(':'))) for value in args.holdout_pairs}
    if not holdout_pairs.issubset(pairs):
        raise ValueError('Holdout pairs must be included in pairs')
    base = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    data_root = ROOT / base['dataset']['data_root']
    sequence = base['dataset']['sequence']
    group = HERE / 'outputs' / args.group
    configs = group / 'configs'
    logs = group / 'logs'
    pairs_root = group / 'pairs'
    for path in (configs, logs, pairs_root):
        path.mkdir(parents=True, exist_ok=True)
    manifest = {'group': args.group, 'sequence': sequence,
                'base_config': str(args.config.resolve()),
                'evaluation_masks_are_estimation_inputs': False,
                'pairs': []}
    for first, second in pairs:
        if first == second:
            raise ValueError('Pair must contain distinct frames')
        mask_path = mask_for_pair(data_root, sequence, first, second)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Unreadable mask: {mask_path}')
        pair_key = f'{sequence}_{first:010d}_{second:010d}'
        holdout = largest_object_box(mask > 127) if (first, second) in holdout_pairs else None
        record = {'first': first, 'second': second, 'pair_key': pair_key,
                  'evaluation_mask': str(mask_path), 'holdout_roi': holdout,
                  'variants': []}
        manifest['pairs'].append(record)
        (group / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        variants = ['single_rigid', 'multi_no_feedback', 'full_feedback']
        if holdout is not None:
            variants += ['holdout_no_feedback', 'holdout_full_feedback']
        for variant in variants:
            cfg = variant_config(base, variant)
            cfg['frames'] = {'first': first, 'second': second}
            cfg['output_dir'] = str(pairs_root)
            cfg['holdout_roi_second'] = None
            config_path = configs / f'{first:010d}_{second:010d}_{variant}.yaml'
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
            result_dir = pairs_root / pair_key / variant
            log_path = logs / f'{first:010d}_{second:010d}_{variant}.log'
            command = [sys.executable, str(HERE / 'run_experiment.py'),
                       '--config', str(config_path), '--tag', variant]
            if variant.startswith('holdout_'):
                command += ['--holdout-roi', *map(str, holdout)]
            print(f'RUN {first}->{second} {variant}', flush=True)
            if not (result_dir / 'summary.json').is_file():
                with log_path.open('w', encoding='utf-8') as log:
                    completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                               stderr=subprocess.STDOUT, check=False)
                if completed.returncode:
                    tail = '\n'.join(log_path.read_text(errors='replace').splitlines()[-35:])
                    raise RuntimeError(f'{variant} {first}->{second} failed:\n{tail}')
            record['variants'].append(variant)
            (group / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Completed {len(pairs)} pairs: {group}', flush=True)


if __name__ == '__main__':
    main()

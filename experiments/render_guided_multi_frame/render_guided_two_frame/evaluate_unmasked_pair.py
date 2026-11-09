"""Compare the three two-frame variants when no dynamic evaluation mask exists."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from evaluate_validation import STANDARD, load_result, valid_depth_mask

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    args = parser.parse_args()
    group = args.group.resolve()
    manifest = json.loads((group / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('dynamic_evaluation_mask') is not None:
        raise ValueError('Use evaluate_validation.py for a group with dynamic masks')
    sequence, first, second = (manifest['sequence'], int(manifest['first']),
                               int(manifest['second']))
    pair_key = f'{sequence}_{first:010d}_{second:010d}'
    pair_dir = group / 'pairs' / pair_key
    results = {name: load_result(pair_dir / name) for name in STANDARD}
    cfg = yaml.safe_load((pair_dir / 'full_feedback/config.yaml').read_text(encoding='utf-8'))
    sequence_root = ROOT / cfg['dataset']['data_root'] / 'data_2d_raw' / sequence
    target = cv2.imread(str(sequence_root / 'image_00/data_rect' / f'{second:010d}.png'))
    if target is None:
        raise FileNotFoundError(f'Missing target image {second}')
    depth, observed = valid_depth_mask(sequence_root, second, target.shape[:2], cfg)
    common = observed.copy()
    for name in STANDARD:
        if results[name]['image'].shape != target.shape:
            raise ValueError(f'{name} render has a different shape')
        common &= results[name]['maps']['coverage_final']
    if not common.any():
        raise RuntimeError('No shared observed and rendered pixels')
    initial = results[STANDARD[0]]['anchors'][:results[STANDARD[0]]['summary']['initial_3d_anchors']]
    for name in STANDARD[1:]:
        item = results[name]
        if not np.array_equal(initial, item['anchors'][:len(initial)]):
            raise ValueError(f'{name} does not share the same initial anchors')

    records = []
    for name in STANDARD:
        item = results[name]
        summary, maps = item['summary'], item['maps']
        diff = (item['image'].astype(np.float32)-target.astype(np.float32))/255.0
        rgb_l1 = np.abs(diff).mean(axis=2)
        coverage = maps['coverage_final'] & observed
        row = {
            'variant': name,
            'initial_3d_anchors': summary['initial_3d_anchors'],
            'final_3d_anchors': summary['final_3d_anchors'],
            'final_models': summary['final_models'],
            'observed_pixels': int(observed.sum()),
            'common_pixels': int(common.sum()),
            'coverage_fraction': float(coverage.sum()/observed.sum()),
            'rgb_l1_common': float(rgb_l1[common].mean()),
            'rgb_l1_all_observed': float(rgb_l1[observed].mean()),
            'depth_abs_common_m': float(np.abs(maps['rendered_depth_final'][common]-depth[common]).mean()),
            'camera_rotation_error_deg': summary['pose_error_final']['rotation_deg'],
            'camera_translation_error_m': summary['pose_error_final']['translation_m'],
        }
        records.append(row)

    report_dir = group / 'reports' / f'{first:010d}_{second:010d}'
    report_dir.mkdir(parents=True, exist_ok=True)
    strips = []
    for name, image in [('Target (no dynamic mask available)', target),
                        *((name, results[name]['image']) for name in STANDARD)]:
        h, w = image.shape[:2]
        bar = np.full((35, w, 3), 255, dtype=np.uint8)
        cv2.putText(bar, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (15, 15, 15), 2, cv2.LINE_AA)
        strips.extend((bar, image))
    if not cv2.imwrite(str(report_dir / 'comparison.png'), np.vstack(strips)):
        raise OSError('Failed to save comparison image')
    with (group / 'evaluation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (group / 'evaluation.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    lines = [f'# 两帧对比：{first}→{second}', '',
             '目标帧没有可用的动态评估掩码。下表在三组方法共同覆盖、且目标帧深度有效的相同像素上比较；另报告所有有效深度像素的 RGB 误差与各自覆盖率。',
             '', '| 方法 | 模型数 | 3D 锚点 初始→最终 | 覆盖率 | 共同像素 RGB L1 ↓ | 全部有效像素 RGB L1 ↓ | 共同像素深度误差 m ↓ | 相机旋转误差 ° ↓ | 相机平移误差 m ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['variant']} | {row['final_models']} | {row['initial_3d_anchors']}→{row['final_3d_anchors']} | "
                     f"{100*row['coverage_fraction']:.1f}% | {row['rgb_l1_common']:.4f} | "
                     f"{row['rgb_l1_all_observed']:.4f} | {row['depth_abs_common_m']:.3f} | "
                     f"{row['camera_rotation_error_deg']:.3f} | {row['camera_translation_error_m']:.3f} |")
    lines += ['', f'[四行渲染对比](reports/{first:010d}_{second:010d}/comparison.png)。',
              '当前没有动态区域真值，因此全图误差不能回答运动物体区域是否改善，也不能评价运动归属准确性。']
    motion_links = []
    for variant in STANDARD:
        motion_rel = f'pairs/{pair_key}/{variant}/motion_visualization'
        if (group / motion_rel / 'overview.png').is_file():
            motion_links.append(f'[{variant} 运动归属]({motion_rel}/overview.png) · '
                                f'[参数]({motion_rel}/motions.json)')
    if motion_links:
        lines += ['', '## 各方法的运动及高斯归属', '',
                  '；'.join(motion_links) + '。',
                  '颜色表示各方法内部的运动编号，不同方法的同色编号不保证对应同一运动。']
    (group / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote {group / "report.md"}')


if __name__ == '__main__':
    main()

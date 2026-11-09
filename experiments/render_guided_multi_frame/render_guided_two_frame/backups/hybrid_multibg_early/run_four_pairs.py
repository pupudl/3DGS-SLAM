#!/usr/bin/env python3
"""Rerun four complete two-frame feedback experiments and motion overviews."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from pair_utils import ROOT, resolve


PAIRS = ((2433, 2435), (2437, 2439), (2445, 2447), (2463, 2465))
HERE = Path(__file__).resolve().parent


def write_hybrid_report(output_root, baseline):
    lines = ['# 多背景早期版本（恢复复跑）：四组结果', '',
             '本目录保存早期多背景算法的四组恢复复跑结果。初始匹配全部保留；2D 本质矩阵、PnP 与 3D 刚体假设共同竞争，'
             '重投影为主、深度一致性为辅，最低 6 个核心支持点。'
             '未加入当前版的背景等价候选合并、未截断尺度核心支持要求，以及合并后的候选数量限制。初始联合优化使用相同几何评分；后续补点沿用原流程。'
             '未使用历史动态物体 mask。', '',
             '这是恢复版与当前版的对照，两版使用同样的全量初始匹配。恢复来源及首组历史复现检查见 [恢复说明](RESTORATION.md)；三版流程见 [方法对比](METHOD_COMPARISON.md)。'
             '候选数量和支持点数不等同于真实物体数或分割准确率。', '',
             '| 帧对 | 旧初始锚点 / 新 | 初始候选数 旧→新 | 新候选核心支持 | 新未归属 | 最终候选数 |',
             '|---|---:|---:|---|---:|---:|']
    records = []
    for first, second in PAIRS:
        summary_path = next(output_root.glob(f'*_{first:010d}_{second:010d}/*/summary.json'))
        result = summary_path.parent
        row = json.loads(summary_path.read_text())
        old_paths = list(baseline.glob(f'*_{first:010d}_{second:010d}/*/summary.json'))
        old = json.loads(old_paths[0].read_text()) if old_paths else {}
        report = row['initialization_report']
        lines.append(f"| {first}→{second} | {old.get('initial_3d_anchors', '—')} / "
                     f"{row['initial_3d_anchors']} | {len(old.get('initial_model_supports', []))} → "
                     f"{len(row['initial_model_supports'])} | {row['initial_model_supports']} | "
                     f"{report['unassigned_count']} | {row['final_models']} |")
        records.append((first, second, result.relative_to(output_root)))
    lines += ['', '表中“旧”为当前留存版，“新”为本目录早期多背景版。两版支持点定义不同：当前版额外要求未截断尺度一致性，多背景版只要求混合代价及归属区分度。'
              '未归属点包括候选间模糊点和高混合代价点；渲染与最终运动图属于后续优化结果，应与初始候选图分开查看。', '',
              '## 各阶段 overview', '',
              '| 帧对 | 初始候选 | 初始联合优化 | 补匹配 | 反馈前后 | 最终运动 |',
              '|---|---|---|---|---|---|']
    directories = ('sparse_motion_visualization', 'ownership_stage_comparison',
                   'feedback_match_visualization', 'feedback_final_visualization',
                   'motion_visualization')
    for first, second, result in records:
        links = ' | '.join(f'[查看]({result / directory / "overview.png"})' for directory in directories)
        lines.append(f'| {first}→{second} | {links} |')
    lines += ['', '初始候选图中的灰点表示未解释或候选间归属不明确。'
              '补匹配总览中的红点仍表示最终接纳的新匹配。'
              '每组 initialization_report.json 记录提案来源、竞争支持和未归属数量。', '']
    (output_root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=HERE / 'configs/kitti360_full_feedback_four_pairs_hybrid_multibg_early.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    output_root = resolve(config['output_dir'])
    log_dir = output_root / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sequence = config['dataset']['sequence']
    tag = config['run_tag']

    for first, second in PAIRS:
        name = f'{first}_{second}'
        result = output_root / f'{sequence}_{first:010d}_{second:010d}' / tag
        print(f'[{name}] Estimating motions and ownership...', flush=True)
        with (log_dir / f'{name}_estimate.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(HERE / 'strict_two_frame.py'),
                            '--config', str(args.config.resolve()), '--frames',
                            str(first), str(second)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        summary = json.loads((result / 'summary.json').read_text(encoding='utf-8'))
        print(f'[{name}] {summary["final_models"]} motions, '
              f'{summary["final_3d_anchors"]} anchors; rendering overview...', flush=True)
        if config.get('output', {}).get('compact', False):
            with (log_dir / f'{name}_visualize.log').open('w', encoding='utf-8') as log:
                for script, directory in (
                    ('visualize_sparse_hypotheses', 'sparse_motion_visualization'),
                    ('visualize_ownership_stages', 'ownership_stage_comparison'),
                    ('visualize_feedback_matches', 'feedback_match_visualization'),
                    ('visualize_feedback_final', 'feedback_final_visualization'),
                    ('visualize_motion', 'motion_visualization'),
                ):
                    with tempfile.TemporaryDirectory(prefix='two_frame_overview_') as temporary:
                        subprocess.run([sys.executable, str(HERE / f'{script}.py'),
                                        '--result', str(result), '--output', temporary],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                        overview = Path(temporary) / 'overview.png'
                        if not overview.is_file() or not overview.stat().st_size:
                            raise RuntimeError(f'Missing or empty overview: {directory}')
                        destination = result / directory
                        destination.mkdir(parents=True, exist_ok=True)
                        for source in [overview, *Path(temporary).glob('*.json')]:
                            shutil.copy2(source, destination / source.name)
            print(f'[{name}] All five stage overviews saved.', flush=True)
            continue
        with (log_dir / f'{name}_visualize.log').open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, str(HERE / 'visualize_motion.py'),
                            '--result', str(result)], cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        overview = result / 'motion_visualization/overview.png'
        if not overview.is_file() or overview.stat().st_size == 0:
            raise RuntimeError(f'Missing or empty overview: {overview}')
        print(f'[{name}] {overview}', flush=True)
        if config['discovery'].get('feedback_mode') in ('global_mask_repair', 'masked_repair'):
            with (log_dir / f'{name}_feedback_matches.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(HERE / 'visualize_feedback_matches.py'),
                                '--result', str(result)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            mask_overview = result / 'feedback_match_visualization/overview.png'
            if not mask_overview.is_file() or mask_overview.stat().st_size == 0:
                raise RuntimeError(f'Missing pixel-mask visualization: {mask_overview}')
            print(f'[{name}] {mask_overview}', flush=True)

    if config.get('output', {}).get('compact', False):
        baseline = resolve(config['output']['comparison_baseline'])
        if config['motion'].get('initialization_method') == 'hybrid':
            write_hybrid_report(output_root, baseline)
            return
        subprocess.run([sys.executable, str(HERE / 'compare_masked_feedback.py'),
                        '--group', str(output_root), '--baseline', str(baseline)],
                       cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

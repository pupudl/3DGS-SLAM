"""Summarize the original, restored early and preserved current initial motions."""
from pathlib import Path
import json

import cv2
import numpy as np
import yaml

from pair_utils import ROOT, discover, resolve, stereo_io
from visualize_sparse_hypotheses import draw_points, load_inputs


HERE = Path(__file__).resolve().parent
OUTPUTS = ROOT / 'experiments/render_guided_multi_frame/render_guided_two_frame/outputs'
PAIRS = ((2433, 2435), (2437, 2439), (2445, 2447), (2463, 2465))
GROUPS = (
    ('Original 3D-3D', 'full_feedback_four_pairs_ransac8_depth_fit_weighted'),
    ('Early hybrid: multiple background models', 'full_feedback_four_pairs_hybrid_multibg_early'),
    ('Current hybrid', 'full_feedback_four_pairs_hybrid_initial'),
)


def load(group, first, second):
    result = next((OUTPUTS / group).glob(f'*_{first:010d}_{second:010d}/*/summary.json')).parent
    summary = json.loads((result / 'summary.json').read_text())
    cfg = yaml.safe_load((result / 'config.yaml').read_text())
    with np.load(result / 'pairwise_state.npz') as saved:
        matches = (saved['anchors_initial'].copy() if 'anchors_initial' in saved else
                   saved['anchors_final'][:summary['initial_3d_anchors']].copy())
        if 'initial_motion_labels' in saved:
            labels = saved['initial_motion_labels'].copy()
        else:
            _, K, anchors = load_inputs(cfg, matches)
            models, labels, supports = discover(anchors, cfg['motion'], cfg.get('seed', 7))
            assert supports == summary['initial_model_supports']
            np.testing.assert_allclose(np.stack(models), saved['H_sparse'], atol=1e-7, rtol=1e-7)
    image_root = resolve(cfg['dataset']['data_root']) / 'data_2d_raw' / cfg['dataset']['sequence']
    images = [stereo_io.read_image(image_root / 'image_00/data_rect' / f'{frame:010d}.png')
              for frame in (first, second)]
    return result, summary, matches, labels, images


def panel(image, title, detail):
    image = cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA)
    header = np.full((64, 704, 3), 255, np.uint8)
    cv2.putText(header, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(header, detail, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .50, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack((header, image))


def main():
    group = OUTPUTS / GROUPS[1][1]
    comparisons = group / 'comparisons'
    comparisons.mkdir(exist_ok=True)
    records = []
    for first, second in PAIRS:
        top, bottom, record = [], [], {'frames': [first, second], 'methods': []}
        hybrid_matches = None
        for title, name in GROUPS:
            result, summary, matches, labels, images = load(name, first, second)
            if name != GROUPS[0][1]:
                if hybrid_matches is not None:
                    np.testing.assert_array_equal(matches, hybrid_matches)
                hybrid_matches = matches
            supports = summary['initial_model_supports']
            unknown = int(np.count_nonzero(labels < 0))
            detail = f'K={len(supports)}; matches={len(matches)}; unassigned={unknown}'
            top.append(panel(draw_points(images[0], matches[:, :2], labels), title, detail))
            bottom.append(panel(draw_points(images[1], matches[:, 2:], labels),
                                'Frame 2: assigned feature correspondences', f'supports={supports}'))
            record['methods'].append(dict(group=name, initial_models=len(supports),
                supports=supports, unassigned=unknown, initial_matches=len(matches),
                final_models=summary['final_models'], result=str(result.relative_to(group.parent))))
        footer = np.full((44, 2112, 3), 255, np.uint8)
        cv2.putText(footer, 'Colors are local motion IDs; gray = unassigned. Original uses 50% of matches; both hybrids use 100%.',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 1, cv2.LINE_AA)
        target = comparisons / f'{first}_{second}_initial_comparison.png'
        assert cv2.imwrite(str(target), np.vstack((np.hstack(top), np.hstack(bottom), footer)))
        record['comparison'] = str(target.relative_to(group))
        records.append(record)
    (group / 'comparison.json').write_text(json.dumps(records, indent=2))
    lines = ['# 多背景早期版本：四组恢复复跑结果', '',
             '本目录单独保存多背景早期算法的复跑结果。当前留存版及其代码、配置和结果没有修改。'
             '初始匹配、深度范围、混合代价权重在早期版与当前版之间相同；两版初始锚点坐标已逐项核对一致。', '',
             '早期源码来自当时的临时文件，经已记录的配置读取及 E 尺度拟合修正恢复。'
             '第一组的 8 个初始候选支持数、306 个未归属点、721 个提案和选模收益与历史记录一致。'
             '这里只验证了初始发现的历史复现，后续反馈为本次重新运行，不声称完整历史工作区逐字节恢复。', '',
             '备份代码见 [运行与源码说明](../../backups/hybrid_multibg_early/README.md)，'
             '细节见 [恢复记录](RESTORATION.md) 和 [三版方法对比](METHOD_COMPARISON.md)。', '',
             '## 初始候选与最终数量', '',
             '| 帧对 | 原版初始 K | 多背景版初始 K | 当前版初始 K | 多背景版最终 K | 当前版最终 K | 多背景版支持数 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for record in records:
        old, early, current = record['methods']
        pair = '→'.join(map(str, record['frames']))
        lines.append(f"| {pair} | {old['initial_models']} | {early['initial_models']} | {current['initial_models']} | "
                     f"{early['final_models']} | {current['final_models']} | {early['supports']} |")
    lines += ['', '候选 K 不等于真实物体数。三种方法的支持定义不同，颜色仅代表各自运行内部编号，不能跨方法按同色认定同一物体。', '',
              '## 四组初始分组图', '',
              '每张图从左到右为原版、多背景早期版、当前版；上排第一帧，下排第二帧。灰点为未归属。'
              '原版匹配随机减半，两个混合版本使用相同的全量匹配，图中同时标出匹配数。', '',
              '| 帧对 | 三版初始匹配归属对比 |', '|---|---|']
    for record in records:
        lines.append(f"| {'→'.join(map(str, record['frames']))} | [查看]({record['comparison']}) |")
    lines += ['', '## 多背景版各阶段 overview', '',
              '| 帧对 | 初始候选 | 初始联合优化 | 补匹配 | 反馈前后 | 最终运动 |', '|---|---|---|---|---|---|']
    stages = ['sparse_motion_visualization', 'ownership_stage_comparison', 'feedback_match_visualization',
              'feedback_final_visualization', 'motion_visualization']
    for record in records:
        result = Path(record['methods'][1]['result']).relative_to(group.name)
        links = ' | '.join(f'[查看]({result / stage / "overview.png"})' for stage in stages)
        lines.append(f"| {'→'.join(map(str, record['frames']))} | {links} |")
    lines += ['', '共保留 20 张阶段 overview 和 4 张初始分组对比图。'
              'NPZ、配置、summary、cycles 和逐轮证据完整保留。未使用动态物体 mask。', '',
              '## 复跑', '', '```bash',
              'env CUDA_VISIBLE_DEVICES=3 /media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
              'experiments/render_guided_multi_frame/render_guided_two_frame/backups/hybrid_multibg_early/run_four_pairs.py',
              '/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
              'experiments/render_guided_multi_frame/render_guided_two_frame/backups/hybrid_multibg_early/compare_versions.py', '```', '']
    (group / 'README.md').write_text('\n'.join(lines))
    print('Wrote four comparison images and README:', group)


if __name__ == '__main__':
    main()

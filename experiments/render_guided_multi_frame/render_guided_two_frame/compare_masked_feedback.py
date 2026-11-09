"""Compare saved masked-repair runs with depth-fit RANSAC8 baselines on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from analyze_motion_partition import metrics, projected_labels
from pair_utils import resolve
from visualize_feedback_final import motion_lineage
from visualize_motion import source_map
from visualize_ownership_stages import caption, ownership_image, pixel_coordinates


def load_state(result):
    summary = json.loads((result / 'summary.json').read_text())
    cfg = yaml.safe_load((result / 'config.yaml').read_text())
    keys = ('H_sparse', 'H_joint', 'H_final', 'q_gaussian_joint', 'q_gaussian_final',
            'anchors_final', 'anchor_gaussian_indices_final', 'anchor_gaussian_xyz_final',
            'anchor_gaussian_uv_final')
    with np.load(result / 'pairwise_state.npz') as archive:
        state = {key: archive[key] for key in keys if key in archive}
    with np.load(result / 'score_maps.npz') as archive:
        maps = {key: archive[key] for key in ('final', 'coverage_final', 'problem_mask_first',
                                             'discovery_first', 'rendered_rgb_final')
                if key in archive}
    return summary, cfg, state, maps


def final_map(base, state):
    result = {**base, 'xyz': base['xyz'].copy(), 'uv': base['uv'].copy()}
    if 'anchor_gaussian_indices_final' in state:
        ids = state['anchor_gaussian_indices_final']
        result['xyz'][ids] = state['anchor_gaussian_xyz_final']
        result['uv'][ids] = state['anchor_gaussian_uv_final']
    return result


def small_caption(image, title, detail=''):
    return caption(cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA), title, detail)


def compare(baseline, result, output):
    old, old_cfg, old_state, old_maps = load_state(baseline)
    new, cfg, new_state, new_maps = load_state(result)
    initial = int(new['initial_3d_anchors'])
    if (old['initial_3d_anchors'] != initial or
            not np.array_equal(old_state['anchors_final'][:initial], new_state['anchors_final'][:initial]) or
            not np.allclose(old_state['H_sparse'], new_state['H_sparse'], atol=1e-7, rtol=1e-7)):
        raise ValueError(f'Initial sparse evidence or hypotheses changed for {result}')
    for key in ('motion', 'joint', 'gaussians', 'features', 'ownership', 'optimization', 'depth'):
        if old_cfg[key] != cfg[key]:
            raise ValueError(f'Unexpected non-feedback configuration change: {key}')
    base, K, first_image, second_image = source_map(cfg)
    common = old_maps['coverage_final'] & new_maps['coverage_final']
    mask = new_maps['problem_mask_first'].astype(bool)
    first, second = map(int, new['frames'])
    pair = f'{first}_{second}'
    output.mkdir(parents=True, exist_ok=True)
    row = dict(frames=new['frames'], initial_evidence_equal=True,
               feedback_mode=new.get('feedback_mode', 'patch'),
               config_file={'global_mask_repair':
                            'kitti360_full_feedback_four_pairs_ransac8_depth_fit_global_mask_repair.yaml',
                            'score_mask_repair':
                            'kitti360_full_feedback_four_pairs_ransac8_depth_fit_score_mask.yaml'}.get(
                            new.get('feedback_mode'),
                            'kitti360_full_feedback_four_pairs_ransac8_depth_fit_masked_repair.yaml'),
               joint_H_max_difference=float(np.max(np.abs(old_state['H_joint']-new_state['H_joint']))),
               joint_q_max_difference=float(np.max(np.abs(old_state['q_gaussian_joint']-
                                                          new_state['q_gaussian_joint']))),
               common_pixels=int(common.sum()), initial_problem_mask_pixels=int(mask.sum()),
               baseline=str(baseline), result=str(result), methods={})
    owners, renders, matches_images, diagnostic_labels = [], [], [], []
    mask_union = mask.copy()
    if new.get('feedback_mode') == 'score_mask_repair':
        for evidence in result.glob('feedback_evidence_cycle_*.npz'):
            with np.load(evidence) as archive:
                mask_union |= archive['problem_mask']
    for name, summary, state, maps, path in (
            ('baseline', old, old_state, old_maps, baseline),
            ('masked_repair', new, new_state, new_maps, result)):
        score = maps['final']
        def mean(region):
            selected = common & region
            return float(score[selected].mean()) if selected.any() else None
        accepted = [dict(cycle=cycle['cycle'], **proposal)
                    for cycle in summary['cycles'] for proposal in cycle['proposals']
                    if proposal.get('accepted')]
        info = dict(feedback_mode=summary.get('feedback_mode', 'patch'),
                    final_models=summary['final_models'], final_3d_anchors=summary['final_3d_anchors'],
                    score_common=mean(np.ones_like(common)),
                    score_initial_problem_mask=mean(mask), score_outside_initial_mask=mean(~mask),
                    pose_error=summary['pose_error_final'], elapsed_seconds=summary['elapsed_seconds'],
                    accepted_operations=accepted,
                    render_coverage_pixels=int(maps['coverage_final'].sum()),
                    cycles=[{key: cycle.get(key) for key in
                             ('cycle', 'problem_mask_pixels', 'active_pending_3d_anchors',
                              'retained_3d_anchors', 'pending_3d_anchors', 'selected_update',
                              'mask_improvement', 'outside_improvement', 'selection')}
                            for cycle in summary['cycles']])
        row['methods'][name] = info
        gmap = final_map(base, state)
        identities, _ = motion_lineage(len(summary['initial_model_supports']), summary['cycles'])
        labels = np.asarray(identities)[state['q_gaussian_final'].argmax(axis=1)]
        rows, cols = pixel_coordinates(gmap, first_image.shape[:2])
        ownership = ownership_image(first_image, rows, cols, labels)
        title = {'patch': 'Original patch feedback',
                 'masked_repair': 'Patch-limited mask feedback',
                 'global_mask_repair': 'Full-image pixel-mask feedback',
                 'score_mask_repair': 'Composite score + budget mask'}[info['feedback_mode']]
        owners.append(small_caption(ownership, title,
                                   f"{pair}: K={summary['final_models']}; colors follow initial component identities"))
        rendered = (cv2.cvtColor(maps['rendered_rgb_final'], cv2.COLOR_RGB2BGR)
                    if 'rendered_rgb_final' in maps else cv2.imread(str(path / 'render_final.png')))
        if rendered is None:
            raise ValueError(f'No saved final RGB rendering: {path}')
        renders.append(small_caption(rendered, 'Final frame-2 rendering',
                                     f"Same-pixel residual: {info['score_common']:.5f}"))
        drawing = second_image.copy()
        if name == 'masked_repair':
            drawing[mask_union] = (.65*drawing[mask_union]+.35*np.array([0, 255, 255])).astype(np.uint8)
        added = state['anchors_final'][initial:]
        for match in added:
            xy = tuple(np.rint(match[2:]).astype(int))
            cv2.circle(drawing, xy, 2, (0, 0, 255), -1, cv2.LINE_AA)
        matches_images.append(small_caption(drawing, 'Retained feedback matches',
                                           f"{len(added)} new points; red = matches" +
                                           ('; yellow = mask union across rounds' if name == 'masked_repair' else '')))
        diagnostic_labels.append(projected_labels(gmap['xyz'], state['H_final'],
                                                   state['q_gaussian_final'], K,
                                                   second_image.shape[:2], summary['reference_final']))
    if new.get('feedback_mode') == 'score_mask_repair':
        values = new_maps['discovery_first']
        heat = cv2.applyColorMap(np.rint(np.clip(values/.6, 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(heat, contours, -1, (255, 255, 0), 1)
        selected = new['cycles'][0]['selection']
        matches_images[0] = small_caption(heat, 'Composite score + selected mask (new)',
            f"Round 1: {mask.sum():,}/{selected['budget_pixels']:,} pixels; contour = selected")
    figure = np.vstack((np.hstack(owners), np.hstack(renders), np.hstack(matches_images)))
    if not cv2.imwrite(str(output / f'{pair}_comparison.png'), figure):
        raise OSError('Could not save comparison image')
    mask_root = (resolve(cfg['dataset']['data_root']) / 'data_2d_raw' /
                 cfg['dataset']['sequence'] / 'gt_dynamic_masks_2435_2463_stride2' /
                 'rigidmask_frontend_probe')
    masks = list(mask_root.glob(f'*_frame_{second:010d}_from_{first:010d}/dynamic/dynamic_mask.png'))
    row['evaluation_mask'] = str(masks[0]) if len(masks) == 1 else None
    if len(masks) == 1:
        truth = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE) > 127
        shared = (diagnostic_labels[0] >= 0) & (diagnostic_labels[1] >= 0) & common
        row['segmentation_common_pixels'] = int(shared.sum())
        for name, labels in zip(('baseline', 'masked_repair'), diagnostic_labels):
            row['methods'][name]['partition_diagnostic'] = metrics(labels, truth, shared)
    row['comparison_image'] = f'comparisons/{pair}_comparison.png'
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    group, baseline = resolve(args.group), resolve(args.baseline)
    rows = []
    for result in sorted(group.glob('*/*/summary.json')):
        path = result.parent
        original = list((baseline / path.parent.name).glob('*/summary.json'))
        if len(original) != 1:
            raise ValueError(f'Missing or ambiguous baseline: {path.parent.name}')
        row = compare(original[0].parent, path, group / 'comparisons')
        rows.append(row)
        print(row['frames'], {name: (v['score_common'], v['pose_error'])
                               for name, v in row['methods'].items()}, flush=True)
    if len(rows) != 4:
        raise ValueError(f'Expected four complete runs, found {len(rows)}')
    report = dict(baseline=str(baseline), group=str(group), pairs=rows,
                  metric_note='Final residuals use identical covered pixels across methods. '
                              'Partition metrics use nearest-depth projected Gaussian centers '
                              'on shared render/center coverage, not full rasterized segmentation. '
                              'Evaluation masks never enter motion estimation.')
    (group / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    score_mode = rows[0]['feedback_mode'] == 'score_mask_repair'
    global_mode = rows[0]['feedback_mode'] in ('global_mask_repair', 'score_mask_repair')
    mask_description = ('保留原综合评分 S=局部平均残差+0.25×按软归属权重平均的归属梯度/全图梯度尺度。'
                        '用无网格高斯邻域均值替代固定 patch 均值（sigma=3 像素），'
                        '归一化尺度仍取空间平均前投影梯度的 95% 分位数。'
                        '先过原 0.12 综合分门槛，再按综合分从高到低选择像素；'
                        '每轮面积预算 110,592 像素，对应原 12×96×96 的选区总面积，'
                        '没有固定方块、独立残差/梯度阈值或 OR 判定。'
                        '小连通域清理只删除像素，不膨胀突破评分或预算。'
                        '两帧完整图像匹配，目标关键点先限制到 mask 内再取 top-k；'
                        if score_mode else
                        '在全图直接按像素残差或归属梯度生成 mask，去除小连通噪声并轻微膨胀；'
                        '两帧完整图像提取与匹配，目标关键点先用 mask 筛选再取 top-k。'
                        '点间最小距离抑制重复证据，全流程不使用 patch、窗口排名、矩形裁剪或网格限额。'
                        if global_mode else
                        '问题 mask 内筛目标关键点（图像上下文保留、mask 内取 top-k）；空间网格限制新增证据密度；')
    lines = ['# RANSAC8 + depth fit：' + ('综合评分与预算 mask' if score_mode else
                                           '全图像素 mask' if global_mode else 'mask') +
             ' 补点与已有运动重新拟合', '',
             f'与 `{baseline.name}` 对照。四组共用一套新参数；'
             '未按帧对调参。脚本核对了初始 3D 匹配和稀疏 H 完全一致，非反馈配置不变。', '',
             '新流程：' + mask_description + '待判定点跨轮缓存；'
             '同证据比较继续优化、归属刷新、替换非参考运动和新增运动。'
             '已有运动替换保留身份；默认不替换参考运动。所有候选还需通过问题区、区外和覆盖率检查。', '',
             '候选在同一批经过质量与密度筛选的待判定证据上等预算比较；'
             '按选中候选的变换筛选误差 <0.5 m 的新点，'
             '并在实际保留点上对两方再作等预算验收。原有 split/merge/prune 保留，并增加 mask 验收。', '',
             '**这是组合流程的效果比较，不是单独证明某一项改动的消融。** '
             '候选比较和实际保留集合的二次验收均计入耗时。', '',
             '初始稀疏匹配、运动假设完全一致。初始联合优化存在数值差异：'
             f"H 最大绝对差 {max(r['joint_H_max_difference'] for r in rows):.2g}，"
             f"q 最大绝对差 {max(r['joint_q_max_difference'] for r in rows):.4f}；"
             '这部分在反馈开始之前产生，数值也保存在 comparison.json。', '',
             '## 最终结果', '',
             '残差在新旧最终结果的同一批覆盖像素上计算；每格为「旧 → 新」，越低越好。'
             '问题区固定为新流程第一轮 mask，区外使用其补集。', '',
             '| 帧对 | 最终运动数 | 最终锚点 | 同覆盖残差 | 问题区残差 | 区外残差 | 平移误差/m | 旋转误差/° |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    def change(a, b, digits=4):
        return f'{a:.{digits}f} → {b:.{digits}f}'
    for row in rows:
        a, b = row['methods']['baseline'], row['methods']['masked_repair']
        pair = '→'.join(map(str, row['frames']))
        lines.append(f"| {pair} | {a['final_models']} → {b['final_models']} | "
                     f"{a['final_3d_anchors']} → {b['final_3d_anchors']} | "
                     f"{change(a['score_common'], b['score_common'])} | "
                     f"{change(a['score_initial_problem_mask'], b['score_initial_problem_mask'])} | "
                     f"{change(a['score_outside_initial_mask'], b['score_outside_initial_mask'])} | "
                     f"{change(a['pose_error']['translation_m'], b['pose_error']['translation_m'])} | "
                     f"{change(a['pose_error']['rotation_deg'], b['pose_error']['rotation_deg'])} |")
    lines += ['', '## 运动划分诊断', '',
              '使用现有事后评估掩码，把非参考分量视作动态。采用高斯中心投影和最近深度，'
              '只比较共同覆盖像素；这是划分诊断，不是完整光栅化语义指标。2463→2465 无评估掩码。', '',
              '| 帧对 | Precision 旧→新 | Recall 旧→新 | IoU 旧→新 | 静态误报率 旧→新 |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        if not row['evaluation_mask']:
            continue
        a, b = [row['methods'][name]['partition_diagnostic']
                for name in ('baseline', 'masked_repair')]
        values = ' | '.join(change(a[key], b[key], 3) for key in
                            ('precision', 'recall', 'iou', 'static_false_positive_rate'))
        lines.append(f"| {'→'.join(map(str, row['frames']))} | {values} |")
    score_wins = sum(r['methods']['masked_repair']['score_common'] <
                     r['methods']['baseline']['score_common'] for r in rows)
    evaluated = [r for r in rows if r['evaluation_mask']]
    iou_wins = sum(r['methods']['masked_repair']['partition_diagnostic']['iou'] >
                   r['methods']['baseline']['partition_diagnostic']['iou'] for r in evaluated)
    lines += ['', '## 本轮观察', '',
              f'- {score_wins}/4 组同覆盖渲染残差下降；{iou_wins}/{len(evaluated)} 组划分诊断 IoU 提升。',
              '- 接受一次 repair 或 birth 表示通过几何/渲染检验，不保证动态划分更准确。']
    if global_mode:
        lines += ['', '## 实际全图 mask', '',
                  ('每个帧对保留各阶段 overview 和一张新旧对比图。对比图底排左侧为新版综合评分和实际选区轮廓，'
                   '右侧为各轮 mask 并集及保留匹配。原始 mask、匹配、渲染和状态保存在 NPZ 中，'
                   '重复分图和单分量图不默认保留。各阶段总览见 [OVERVIEWS.md](OVERVIEWS.md)。' if score_mode else
                   '每个运行的 `feedback_match_visualization/overview.png` 显示真实 mask、'
                   '残差/梯度触发图和匹配点，不画 patch。每轮独立保存原始匹配及其实际 mask，'
                   '可检查所有目标匹配点是否在所属轮次的 mask 内。'), '',
                  '| 帧对 | 第一轮 mask 像素 | 占有效深度区域 | 实际 mask 与匹配图 |',
                  '|---|---:|---:|---|']
        for row in rows:
            result = Path(row['result'])
            summary = json.loads((result / 'summary.json').read_text())
            first_cycle = summary['cycles'][0]
            relative = result.relative_to(group) / 'feedback_match_visualization/overview.png'
            lines.append(f"| {'→'.join(map(str, row['frames']))} | "
                         f"{first_cycle['problem_mask_pixels']:,} | "
                         f"{first_cycle['problem_mask_fraction']:.1%} | [查看]({relative}) |")
    lines += ['', '## 图像与接受的操作', '',
              '并排图：左为原版，右为新版；依次是首帧高斯归属、最终 RGB 渲染、保留的新匹配。'
              '归属颜色追踪初始运动身份，编号重排不会换色。红点是新匹配，黄色是各轮 mask 并集。'
              + ('底排左侧为新版综合评分与第一轮实际 mask 轮廓。' if score_mode else ''), '']
    for row in rows:
        a, b = row['methods']['baseline'], row['methods']['masked_repair']
        pair = '→'.join(map(str, row['frames']))
        operations = ', '.join(f"{p['kind']}" + (f" M{p['component']}" if 'component' in p else '') +
                               f" (cycle {p['cycle']})" for p in b['accepted_operations']) or '无结构操作'
        lines.append(f"- {pair}：[新旧并排图]({row['comparison_image']})；{operations}；"
                     f"耗时 {a['elapsed_seconds']:.1f} → {b['elapsed_seconds']:.1f} 秒。")
    lines += ['', '数值、每轮 mask/缓存数量和验收记录见 `comparison.json`、各运行的 '
              '`summary.json` / `cycles.json`。' +
              ('每轮 `feedback_evidence_cycle_XX.npz` 保存实际 mask、原始匹配、候选点和待判定点；'
               '精简模式不生成单独的 mask PNG。' if score_mode else
               '各轮 `problem_mask_cycle_XX.png` 与 `feedback_evidence_cycle_XX.npz` '
               '保存实际问题区、候选点和待判定点。'), '',
              '## 复现', '', '```bash',
              '/media/data/qiuyu_data/miniconda3/envs/lsgslam/bin/python '
              'experiments/render_guided_multi_frame/render_guided_two_frame/run_four_pairs.py --config '
              'experiments/render_guided_multi_frame/render_guided_two_frame/configs/' +
              rows[0]['config_file'], '```', '']
    (group / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

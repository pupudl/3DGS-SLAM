#!/usr/bin/env python3
"""Verify a completed object-window run and visualize ownership changes on CPU.

This reports consistency and label changes only. It does not treat changed
labels as correctness evidence and does not read ground-truth annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from visualize_objects import caption, object_color, read_image, resized


def counts(labels):
    ids, values = np.unique(np.asarray(labels, np.int64), return_counts=True)
    return {str(int(oid)): int(value) for oid, value in zip(ids, values)}


def initial_labels(saved):
    probabilities = saved['q_object_initial']
    object_ids = saved['evidence_object_ids']
    result = object_ids[probabilities.argmax(axis=1)].astype(np.int64)
    result[probabilities.sum(axis=1) <= 0] = -1
    return result


def hard_overlay(image, uv, labels):
    """Draw categorical centres opaquely; probability never fades the colour."""
    out = image.copy()
    uv = np.asarray(uv)
    finite = np.isfinite(uv).all(axis=1)
    xy = np.zeros_like(uv, dtype=np.int64)
    xy[finite] = np.rint(uv[finite]).astype(np.int64)
    h, w = image.shape[:2]
    valid = (finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) &
             (xy[:, 1] >= 0) & (xy[:, 1] < h))
    if not valid.any():
        return out
    ids, inverse = np.unique(labels[valid], return_inverse=True)
    palette = np.asarray([(145, 145, 145) if oid < 0 else object_color(int(oid))
                          for oid in ids], np.float32)
    pixels = xy[valid]
    out[pixels[:, 1], pixels[:, 0]] = palette[inverse].astype(np.uint8)
    return out


def _load(path, keys):
    with np.load(path) as archive:
        return {key: archive[key] for key in keys}


def summarize(result, width=960):
    result = Path(result).resolve()
    state = json.loads((result / 'sequence_state.json').read_text())
    cfg = state.get('config')
    if cfg is None:
        cfg = yaml.safe_load((result / 'config.yaml').read_text())
    requested_frames = list(map(int, state['frames']))
    expected_pairs = list(zip(requested_frames[:-1], requested_frames[1:]))
    pairs_by_frames = {(int(pair['first']), int(pair['second'])): pair for pair in state['pairs']}
    if set(expected_pairs) != set(pairs_by_frames):
        raise ValueError('Run is incomplete: sequence_state does not contain every requested pair')
    records, saved_states, panels = [], {}, []
    total_changed = total_known = total_unknown = 0
    all_object_ids = set()
    for first, second in expected_pairs:
        row = pairs_by_frames[(first, second)]
        path = Path(row['result_path'])
        directory = path if path.is_absolute() else result / path
        saved = _load(directory / 'object_state.npz', (
            'gaussian_ids', 'object_ids', 'object_probabilities', 'object_labels',
            'q_object_initial', 'evidence_object_ids', 'q_motion_initial',
            'window_frames', 'window_pairs', 'uv', 'object_track_ids', 'object_track_labels'))
        motion = _load(directory / 'pairwise_state.npz', (
            'q_gaussian_sparse', 'q_gaussian_joint', 'q_gaussian_final'))
        frozen = (np.array_equal(motion['q_gaussian_sparse'], motion['q_gaussian_joint']) and
                  np.array_equal(motion['q_gaussian_sparse'], motion['q_gaussian_final']))
        raw_export_equal = np.array_equal(motion['q_gaussian_sparse'], saved['q_motion_initial'])
        initial = initial_labels(saved)
        final = saved['object_labels']
        known = final >= 0
        changed = known & (initial != final)
        transition_pairs, transition_counts = np.unique(
            np.column_stack((initial[changed], final[changed])), axis=0, return_counts=True)
        transitions = [dict(initial_object=int(before), final_object=int(after), count=int(count))
                       for (before, after), count in zip(transition_pairs, transition_counts)]
        temporal = row.get('object_window')
        if temporal is None:
            temporal_path = directory / 'temporal_state.json'
            temporal = json.loads(temporal_path.read_text()).get('object_window', {})
        summary_path = directory / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        record = dict(
            first=first, second=second, gaussians=len(final),
            unique_gaussian_ids=len(np.unique(saved['gaussian_ids'])),
            frozen_motion_probabilities=frozen, raw_motion_export_equal=raw_export_equal,
            initial_object_distribution=counts(initial), final_object_distribution=counts(final),
            final_highconfidence=int(known.sum()), final_unknown=int((~known).sum()),
            highconfidence_changed_from_initial=int(changed.sum()), transitions=transitions,
            sparse_tracks=len(saved['object_track_ids']),
            highconfidence_tracks=int((saved['object_track_labels'] >= 0).sum()),
            inherited_3d_tracks=int(row.get('inherited_3d_tracks', 0)),
            links=temporal.get('links', {}), models=temporal.get('models', []),
            render_optimization=temporal.get('render_optimization'),
            elapsed_seconds=summary.get('elapsed_seconds'),
            archived_window_frames=saved['window_frames'].astype(int).tolist())
        records.append(record)
        saved_states[(first, second)] = saved
        all_object_ids.update(map(int, saved['object_ids']))
        total_changed += record['highconfidence_changed_from_initial']
        total_known += record['final_highconfidence']
        total_unknown += record['final_unknown']
        image = read_image(cfg, first)
        size = (width, max(1, round(image.shape[0]*width/image.shape[1])))
        before = caption(cv2.resize(hard_overlay(image, saved['uv'], initial), size,
                                   interpolation=cv2.INTER_NEAREST),
                         f'Frame {first}: INITIAL object argmax',
                         'Local-motion initialization mapped to global object IDs')
        after = caption(cv2.resize(hard_overlay(image, saved['uv'], final), size,
                                  interpolation=cv2.INTER_NEAREST),
                        f'Frame {first}: WINDOW object labels',
                        f'Accepted labels only; gray = uncertain; changed: '
                        f'{record["highconfidence_changed_from_initial"]}; unknown: {record["final_unknown"]}')
        panels.append(np.hstack((before, after)))
    latest = saved_states[expected_pairs[-1]]
    active_pairs = {tuple(map(int, pair)) for pair in latest['window_pairs']}
    active = sorted(set(saved_states) & active_pairs)
    shared_checks = []
    for index, pair_a in enumerate(active):
        a = saved_states[pair_a]
        for pair_b in active[index+1:]:
            b = saved_states[pair_b]
            shared, rows_a, rows_b = np.intersect1d(a['gaussian_ids'], b['gaussian_ids'],
                                                  return_indices=True)
            same_columns = np.array_equal(a['object_ids'], b['object_ids'])
            same_probability = (same_columns and np.array_equal(
                a['object_probabilities'][rows_a], b['object_probabilities'][rows_b]))
            same_labels = np.array_equal(a['object_labels'][rows_a], b['object_labels'][rows_b])
            shared_checks.append(dict(pair_a=list(pair_a), pair_b=list(pair_b),
                                      shared_gaussians=len(shared),
                                      global_columns_equal=bool(same_columns),
                                      posterior_exactly_equal=bool(same_probability),
                                      labels_exactly_equal=bool(same_labels)))
    checks = dict(
        all_requested_pairs_completed=len(records) == len(expected_pairs),
        motion_probabilities_frozen=all(row['frozen_motion_probabilities'] for row in records),
        original_motion_export_unchanged=all(row['raw_motion_export_equal'] for row in records),
        no_duplicate_gaussian_ids=all(row['gaussians'] == row['unique_gaussian_ids'] for row in records),
        active_window_archives_complete=active_pairs <= set(saved_states),
        active_shared_posteriors_equal=all(row['posterior_exactly_equal'] for row in shared_checks),
        active_shared_labels_equal=all(row['labels_exactly_equal'] for row in shared_checks),
        active_snapshot_window_frames_equal=all(np.array_equal(
            saved_states[pair]['window_frames'], latest['window_frames']) for pair in active))
    render_reports = [row['render_optimization'] for row in records if row['render_optimization']]
    if render_reports:
        checks.update(
            every_update_uses_gaussian_raster=len(render_reports) == len(records),
            render_gradients_reach_object_logits=all(
                row['render_passes'] > 0 and row['nonzero_gradient_steps'] > 0
                for row in render_reports),
            render_window_objective_decreased=all(
                row['final_losses']['total'] <= row['initial_losses']['total'] + 1e-6
                for row in render_reports),
            final_render_window_covers_all_frames=set(render_reports[-1]['frames']) ==
                set(map(int, latest['window_frames'])))
    verification = dict(
        result=str(result), frames=requested_frames, active_pairs=[list(pair) for pair in active],
        checks=checks, passed=all(checks.values()), pairs=records, shared_comparisons=shared_checks,
        shared_comparisons_with_overlap=sum(row['shared_gaussians'] > 0 for row in shared_checks),
        observation_totals=dict(highconfidence=total_known, unknown=total_unknown,
                                highconfidence_changed_from_initial=total_changed),
        scope='Counts are per-source-map observations, not deduplicated physical surfaces.',
        correctness_evaluated=False, ground_truth_used=False)
    (result / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    # Preserve one global palette across every before/after row.
    legend = np.full((64, 2*width, 3), 248, np.uint8)
    cv2.putText(legend, 'Same source RGB; Gaussian centres; gray = uncertain (no accepted label); no GT evaluation',
                (12, 21), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 30, 30), 1, cv2.LINE_AA)
    for index, oid in enumerate([-1] + sorted(all_object_ids)):
        x = 12 + 110*index
        if x + 105 >= legend.shape[1]:
            break
        color = (145, 145, 145) if oid < 0 else object_color(oid)
        cv2.rectangle(legend, (x, 37), (x+18, 53), color, -1)
        cv2.putText(legend, 'unknown' if oid < 0 else f'O{oid}', (x+24, 51),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (30, 30, 30), 1, cv2.LINE_AA)
    overview = np.vstack([legend, *panels])
    overview_path = result / 'before_after_overview.png'
    if not cv2.imwrite(str(overview_path), overview):
        raise OSError(overview_path)
    lines = ['# object 窗口运行验证', '',
             f'帧：`{requested_frames}`；共处理 {len(records)} 个帧对。', '',
             f'一致性检查：**{"全部通过" if verification["passed"] else "存在失败"}**。'
             '逐项结果见 [verification.json](verification.json)。', '',
             f'- 局部运动概率逐元素冻结检查：{"通过" if checks["motion_probabilities_frozen"] else "失败"}。',
             f'- 当前窗口共享高斯后验与标签一致性：'
             f'{"通过" if checks["active_shared_posteriors_equal"] and checks["active_shared_labels_equal"] else "失败"}。',
             '- object 初始化仅提供起点；下表统计最终有置信度且与初始化不同的归属。', '',
             '[初始化与窗口结果对照](before_after_overview.png) · '
             '[逐 object 多帧结果](objects/index.html)', '',
             '| 帧对 | 高斯数 | 延续 track 关联 | 几何关联 | 新高斯 | 最终有置信度 | 未确定 | 与初始化不同的有置信度归属 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        links = row['links']
        lines.append(f'| {row["first"]}→{row["second"]} | {row["gaussians"]} | '
                     f'{links.get("track_links", 0)} | {links.get("geometry_links", 0)} | '
                     f'{links.get("new_gaussians", 0)} | {row["final_highconfidence"]} | '
                     f'{row["final_unknown"]} | {row["highconfidence_changed_from_initial"]} |')
    if render_reports:
        lines += ['', '窗口内实际执行 Gaussian RGB、深度与覆盖率渲染损失的反向传播，'
                  '共同更新共享高斯 object logits。下表记录每次窗口更新的优化目标；'
                  '在各次 logits 优化内部，几何、颜色、输入的 object 运动及原两帧归属概率保持固定。'
                  '这不表示完整序列没有执行进入窗口前的 object 运动估计或重拟合。', '',
                  '| 窗口 | 初始目标 | 最终目标 | 非零梯度步数 | 渲染次数 |',
                  '|---|---:|---:|---:|---:|']
        for row in render_reports:
            lines.append(f'| {row["frames"]} | {row["initial_losses"]["total"]:.6f} | '
                         f'{row["final_losses"]["total"]:.6f} | '
                         f'{row["nonzero_gradient_steps"]} | {row["render_passes"]} |')
        if state.get('replay'):
            lines += ['', '本次复用已有帧对的高斯、运动和身份重新运行 object 优化；'
                      '没有重新执行两帧运动估计。旧结果不作为正确标签监督。']
    lines += ['', '图中左侧为只读局部运动初始化映射到全局 object 后的最大概率标签；'
              '右侧为窗口优化的有置信度标签，未确定高斯显示为灰色。两侧使用同一张源 RGB、'
              '同一组高斯中心和一致的全局 object 配色。左右的标签接受规则不同：'
              '灰色只表示未达到置信度，不能理解为已经判定原标签错误或已经纠正。', '',
              '**归属发生变化不代表已经证明物体分割正确。** 本次未使用 GT mask，'
              '验证的是概率冻结、窗口共享状态的一致性与实际产生的归属变化。'
              '计数按各源帧的高斯观测统计，同一物理表面可能出现在多行；'
              '图像是高斯中心标签叠加，不是完整高斯 raster 或融合地图。', '']
    if not verification['passed']:
        lines[6:6] = ['检查存在失败，具体失败项以 JSON 为准，以下描述仅说明检查目标。', '']
    (result / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--width', type=int, default=960)
    args = parser.parse_args()
    if args.width < 160:
        parser.error('--width must be at least 160')
    verification = summarize(args.result, args.width)
    print(json.dumps(dict(passed=verification['passed'],
                          verification=str(args.result / 'verification.json'),
                          report=str(args.result / 'RESULTS.md'),
                          overview=str(args.result / 'before_after_overview.png'))))
    raise SystemExit(0 if verification['passed'] else 1)


if __name__ == '__main__':
    main()

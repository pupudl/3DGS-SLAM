#!/usr/bin/env python3
"""CPU comparison of fixed-input object height/connectivity constraints.

Both solutions are measured against one graph/core selection reconstructed
from the result's original observation costs, source RGB-D and Gaussian IDs.
This script measures constraint violations and core retention, not accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE/'render_guided_two_frame'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from object_connectivity_constraints import build_connectivity_state, update_connectivity_targets
from object_height_constraints import estimate_ground_plane, height_penalty
from pair_utils import load_depth, resolve, stereo_io
from replay_object_window import stereo_confidence
from visualize_objects import pair_frames, resolved_path


KEYS = ('gaussian_ids', 'object_ids', 'object_probabilities', 'object_labels',
        'xyz', 'uv', 'color', 'scale', 'basis', 'K', 'H_object', 'object_model_valid',
        'evidence_object_ids', 'q_object_initial', 'q_motion_initial',
        'observation_costs', 'observation_weights', 'track_ids', 'window_pairs',
        'ground_plane_report_json')


def _archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in KEYS if key in archive.files}


def _same(a, b):
    return bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))


def _state(result):
    state = json.loads((result/'sequence_state.json').read_text())
    config = state.get('config')
    if config is None:
        config = yaml.safe_load((result/'config.yaml').read_text())
    pairs = {pair_frames(pair): pair for pair in state['pairs']}
    return state, config, pairs


def _source_reader(cfg):
    data = resolve(cfg['dataset']['data_root'])
    sequence = data/'data_2d_raw'/cfg['dataset']['sequence']
    K, baseline = stereo_io.read_calibration(data/'calibration/perspective.txt')
    cache = {}

    def read(frame):
        if frame not in cache:
            name = f'{int(frame):010d}.png'
            image = stereo_io.read_image(sequence/'image_00/data_rect'/name)
            right = stereo_io.read_image(sequence/'image_01/data_rect'/name)
            depth = load_depth(sequence, frame, image, right, K, baseline, cfg['depth'])
            sky = sequence/'sky_masks'/name
            if sky.is_file():
                depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
            confidence = stereo_confidence(image, right, depth, K, baseline,
                                            float(cfg['depth'].get('confidence_rgb_sigma', .15)))
            cache[frame] = image, confidence
        return cache[frame]
    return read


def _quality(confidence, uv):
    h, w = confidence.shape
    finite = np.isfinite(uv).all(1)
    xy = np.zeros_like(uv, np.int64)
    xy[finite] = np.rint(uv[finite]).astype(np.int64)
    valid = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    values = np.zeros(len(uv))
    values[valid] = np.clip(np.nan_to_num(confidence[xy[valid, 1], xy[valid, 0]], nan=0.), 0, 1)
    return values


def _height_metrics(probabilities, labels, object_ids, outside, penalty, reliability):
    foreground = np.asarray(object_ids) != 0
    mass = probabilities[:, foreground].sum(1)
    hard = labels > 0
    return dict(outside_nonbackground_soft_mass=float(mass[outside].sum()),
                weighted_outside_nonbackground_soft_mass=float((mass*reliability*outside).sum()),
                expected_height_penalty=float((mass*reliability*penalty).sum()),
                outside_hard_nonbackground=int(np.count_nonzero(outside & hard)),
                outside_hard_background=int(np.count_nonzero(outside & (labels == 0))),
                outside_unknown=int(np.count_nonzero(outside & (labels < 0))),
                by_object={str(int(oid)):dict(
                    outside_soft_mass=float(probabilities[outside, column].sum()),
                    outside_hard_count=int(np.count_nonzero(outside & (labels == oid))))
                    for column, oid in enumerate(object_ids) if oid != 0})


def _connectivity_metrics(graph, probabilities, labels):
    targets = update_connectivity_targets(graph, probabilities)
    high = float(graph['options'].get('high_probability', .35))
    result = dict(disconnected_high_soft_mass=0., halo_and_disconnected_soft_mass=0.,
                  expected_connectivity_penalty=0., disconnected_hard_nonbackground=0,
                  objects={}, shared_target_report=targets['report'])
    for column, oid in enumerate(graph['object_ids']):
        core = graph['core_by_object'].get(int(oid))
        if core is None or not len(core):
            result['objects'][str(int(oid))] = dict(
                action=graph['diagnostics']['objects'][str(int(oid))]['action'])
            continue
        penalized = targets['penalty'][:, column] > 0
        disconnected = penalized & graph['observed'][:, column] & (probabilities[:, column] >= high)
        row = dict(action='evaluate_same_reference_core',
                   disconnected_high_count=int(disconnected.sum()),
                   disconnected_high_soft_mass=float(probabilities[disconnected, column].sum()),
                   halo_and_disconnected_soft_mass=float(probabilities[penalized, column].sum()),
                   expected_connectivity_penalty=float((probabilities[:, column]*targets['penalty'][:, column]).sum()),
                   disconnected_hard_count=int(np.count_nonzero(disconnected & (labels == oid))))
        result['objects'][str(int(oid))] = row
        result['disconnected_high_soft_mass'] += row['disconnected_high_soft_mass']
        result['halo_and_disconnected_soft_mass'] += row['halo_and_disconnected_soft_mass']
        result['expected_connectivity_penalty'] += row['expected_connectivity_penalty']
        result['disconnected_hard_nonbackground'] += row['disconnected_hard_count']
    return result


def _probability_rows(archive, order, object_ids):
    local = {int(oid): column for column, oid in enumerate(archive['object_ids'])}
    if set(local) != set(map(int, object_ids)):
        raise ValueError('The compared solutions do not contain the same global object IDs')
    columns = [local[int(oid)] for oid in object_ids]
    return archive['object_probabilities'][order][:, columns]


def _overview(baseline, result, object_id):
    def image(directory):
        for name in (f'object{object_id}', f'object_{object_id:04d}'):
            path = directory/'objects'/name/'overview.png'
            if path.is_file():
                return cv2.imread(str(path))
        return None
    before, after = image(baseline), image(result)
    if before is None or after is None or before.shape != after.shape:
        return None
    # Each existing overview retains its original five frame columns and two
    # render/mask rows; only a case label is added before stacking both cases.
    panels = []
    for label, value in (('BASELINE: original two-row overview', before),
                         ('SPATIAL: same object / frame layout', after)):
        bar = np.full((34, value.shape[1], 3), 246, np.uint8)
        cv2.putText(bar, label, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (20, 20, 20), 1, cv2.LINE_AA)
        panels.extend((bar, value))
    path = result/f'object{object_id}_spatial_comparison.png'
    if not cv2.imwrite(str(path), np.vstack(panels)):
        raise OSError(path)
    return path.name


def compare(baseline, result, overview_object=3):
    baseline, result = Path(baseline).resolve(), Path(result).resolve()
    base_state, base_cfg, base_pairs = _state(baseline)
    new_state, cfg, new_pairs = _state(result)
    if set(base_pairs) != set(new_pairs):
        raise ValueError('The compared runs must contain exactly the same frame pairs')
    pairs = sorted(new_pairs)
    archives = {}
    original = {}
    paths = {}
    for pair in pairs:
        paths[pair] = resolved_path(new_pairs[pair]['result_path'], result)
        archives[pair] = _archive(paths[pair]/'object_state.npz')
        base_path = resolved_path(base_pairs[pair]['result_path'], baseline)/'object_state.npz'
        original[pair] = archives[pair] if baseline == result else _archive(base_path)
    latest = archives[pairs[-1]]
    active = {tuple(map(int, pair)) for pair in latest.get('window_pairs', np.asarray(pairs))}
    if not active <= set(pairs):
        raise ValueError('An active window archive is absent')
    object_ids = np.asarray(latest['object_ids'], np.int64)
    gids = np.unique(np.concatenate([archives[pair]['gaussian_ids'] for pair in sorted(active)]))
    p_base, p_new = (np.zeros((len(gids), len(object_ids))) for _ in range(2))
    labels_base, labels_new = np.full(len(gids), -1, np.int64), np.full(len(gids), -1, np.int64)
    seen_base, seen_new = np.zeros(len(gids), bool), np.zeros(len(gids), bool)
    height_options = dict(cfg.get('object_window', {}).get('height', {}))
    connection_options = dict(cfg.get('object_window', {}).get('connectivity', {}))
    reader = _source_reader(cfg)
    height_sum, weight_sum, allowed_sum = np.zeros(len(gids)), np.zeros(len(gids)), np.zeros(len(gids))
    window = SimpleNamespace(_active_gids=gids, object_ids=object_ids.tolist(), _active={})
    checks = dict(same_frame_pairs=True, gaussian_ids_equal=True, object_ids_equal=True,
                  source_geometry_equal=True, raw_motion_probabilities_equal=True,
                  initial_object_probabilities_equal=True, original_local_motions_equal=True,
                  object_motions_equal=True, object_model_validity_equal=True, raw_observation_costs_equal=True,
                  repeated_shared_posteriors_consistent=True)
    source_metrics = []
    for pair in pairs:
        new, old = archives[pair], original[pair]
        lookup = {int(gid): row for row, gid in enumerate(old['gaussian_ids'])}
        if set(lookup) != set(map(int, new['gaussian_ids'])):
            raise ValueError(f'Gaussian ID sets differ for frame pair {pair}')
        alignment = np.asarray([lookup[int(gid)] for gid in new['gaussian_ids']], np.int64)
        checks['gaussian_ids_equal'] &= set(lookup) == set(map(int, new['gaussian_ids']))
        checks['object_ids_equal'] &= set(map(int, old['object_ids'])) == set(map(int, new['object_ids']))
        for key in ('xyz', 'uv', 'color', 'scale'):
            checks['source_geometry_equal'] &= _same(old[key][alignment], new[key])
        checks['source_geometry_equal'] &= _same(old['K'], new['K'])
        if 'basis' in old or 'basis' in new:
            checks['source_geometry_equal'] &= ('basis' in old and 'basis' in new and
                                                _same(old['basis'][alignment], new['basis']))
        checks['raw_motion_probabilities_equal'] &= _same(old['q_motion_initial'][alignment], new['q_motion_initial'])
        checks['initial_object_probabilities_equal'] &= (
            _same(old['evidence_object_ids'], new['evidence_object_ids']) and
            _same(old['q_object_initial'][alignment], new['q_object_initial']))
        old_columns = [int(np.flatnonzero(old['object_ids'] == oid)[0]) for oid in new['object_ids']]
        checks['object_motions_equal'] &= _same(old['H_object'][old_columns], new['H_object'])
        old_valid = old.get('object_model_valid', np.isfinite(old['H_object']).all(axis=(1, 2)))
        new_valid = new.get('object_model_valid', np.isfinite(new['H_object']).all(axis=(1, 2)))
        checks['object_model_validity_equal'] &= _same(old_valid[old_columns], new_valid)
        checks['raw_observation_costs_equal'] &= (
            _same(old['observation_costs'][alignment], new['observation_costs']) and
            _same(old['observation_weights'][alignment], new['observation_weights']))
        base_dir = resolved_path(base_pairs[pair]['result_path'], baseline)
        if (base_dir/'gaussian_state.npz').is_file() and (paths[pair]/'gaussian_state.npz').is_file():
            with np.load(base_dir/'gaussian_state.npz') as a, np.load(paths[pair]/'gaussian_state.npz') as b:
                checks['original_local_motions_equal'] &= _same(a['H'], b['H'])
        else:
            checks['original_local_motions_equal'] = False
        old_probability = _probability_rows(old, alignment, object_ids)
        new_probability = _probability_rows(new, np.arange(len(new['gaussian_ids'])), object_ids)
        image, confidence = reader(pair[0])
        if 'ground_plane_report_json' in new:
            plane = json.loads(str(new['ground_plane_report_json'].item()))
            plane_source = 'result_saved_ground_plane_report_json'
        else:
            plane = estimate_ground_plane(new['xyz'], new['uv'], image.shape, confidence, height_options)
            plane_source = 'estimated_once_from_raw_source_stereo'
        penalty, reliability = height_penalty(new['xyz'], plane, height_options)
        reliability *= _quality(confidence, new['uv'])
        signed_height = new['xyz'] @ np.asarray(plane['normal'])+float(plane['offset'])
        lower = float(height_options.get('ground_min_m', -.2))
        upper = float(height_options.get('max_height_m', plane['camera_height']+
                                        float(height_options.get('above_camera_m', 2.))))
        known = bool(plane['valid']) & np.isfinite(signed_height) & (reliability > 0)
        outside = known & ((signed_height < lower) | (signed_height > upper))
        source_metrics.append(dict(
            frames=list(pair), observations=len(new['gaussian_ids']),
            plane_source=plane_source, plane=plane, height_band_m=[lower, upper],
            known_height_observations=int(known.sum()), outside_height_observations=int(outside.sum()),
            baseline=_height_metrics(old_probability, old['object_labels'][alignment], object_ids,
                                     outside, penalty, reliability),
            result=_height_metrics(new_probability, new['object_labels'], object_ids,
                                   outside, penalty, reliability)))
        if pair not in active:
            continue
        rows = np.searchsorted(gids, new['gaussian_ids'])
        for output, labels, seen, probability, hard in (
                (p_base, labels_base, seen_base, old_probability, old['object_labels'][alignment]),
                (p_new, labels_new, seen_new, new_probability, new['object_labels'])):
            existing = seen[rows]
            checks['repeated_shared_posteriors_consistent'] &= bool(
                np.allclose(output[rows[existing]], probability[existing], atol=1e-10, rtol=1e-10) and
                np.array_equal(labels[rows[existing]], hard[existing]))
            output[rows], labels[rows], seen[rows] = probability, hard, True
        weight = np.where(known, reliability, 0)
        height_sum[rows] += np.where(known, signed_height, 0)*weight
        weight_sum[rows] += weight
        allowed_sum[rows] += weight*((signed_height >= lower) & (signed_height <= upper))
        window._active[pair] = dict(
            first=pair[0], second=pair[1], gaussian_ids=new['gaussian_ids'],
            object_ids=new['evidence_object_ids'], costs=new['observation_costs'],
            weights=new['observation_weights'], track_ids=new.get('track_ids'),
            payload={key:new[key] for key in ('xyz', 'uv', 'color')},
            render_observations={'source':image})
    valid = weight_sum > 0
    height_info = dict(height=np.divide(height_sum, weight_sum, out=np.full(len(gids), np.nan), where=valid),
                       valid=valid, allowed=~valid | (allowed_sum >= .5*weight_sum))
    graph = build_connectivity_state(window, connection_options, height_info)
    connection_base = _connectivity_metrics(graph, p_base, labels_base)
    connection_new = _connectivity_metrics(graph, p_new, labels_new)
    cores = {}
    for oid, rows in graph['core_by_object'].items():
        column = int(np.flatnonzero(object_ids == oid)[0])
        selected_ids = gids[rows]
        accepted_before = labels_base[rows] == oid
        cores[str(oid)] = dict(
            count=len(rows), selection='same_result_raw_motion_cost_core_for_both_solutions',
            gaussian_ids_head=selected_ids[:1000].astype(int).tolist(),
            gaussian_ids_sha256=hashlib.sha256(selected_ids.tobytes()).hexdigest(),
            baseline=dict(mean_probability=float(p_base[rows, column].mean()),
                          soft_mass=float(p_base[rows, column].sum()),
                          accepted=int(accepted_before.sum()), unknown=int((labels_base[rows] < 0).sum())),
            result=dict(mean_probability=float(p_new[rows, column].mean()),
                        soft_mass=float(p_new[rows, column].sum()),
                        accepted=int((labels_new[rows] == oid).sum()), unknown=int((labels_new[rows] < 0).sum())),
            retained_previously_accepted=int(np.count_nonzero(accepted_before & (labels_new[rows] == oid))),
            previously_accepted_count=int(accepted_before.sum()))
    height_totals = {}
    for name in ('baseline', 'result'):
        height_totals[name] = {key:float(sum(row[name][key] for row in source_metrics))
                               for key in ('outside_nonbackground_soft_mass',
                                           'weighted_outside_nonbackground_soft_mass',
                                           'expected_height_penalty', 'outside_hard_nonbackground')}
    identity = bool(np.array_equal(p_base, p_new) and np.array_equal(labels_base, labels_new))
    report = dict(
        baseline=str(baseline), result=str(result), checks=checks,
        controlled_inputs_equal=bool(all(checks.values())), identical_solutions=identity,
        active_window_pairs=[list(pair) for pair in sorted(active)],
        active_unique_gaussians=len(gids), object_ids=object_ids.astype(int).tolist(),
        source_height_comparisons=source_metrics, height_observation_totals=height_totals,
        fixed_graph=graph['diagnostics'],
        graph_selection='built_once_from_result_saved_raw_costs_and_source_RGBD; reused_for_both_solutions',
        connectivity=dict(baseline=connection_base, result=connection_new), reliable_core_retention=cores,
        ground_truth_used=False, semantic_accuracy_evaluated=False,
        constraint_is_strict_single_connected_component=False,
        counting='Height totals count source-map observations; connectivity/core counts use unique active Gaussian IDs.')
    overview = _overview(baseline, result, overview_object)
    report['overview'] = overview
    (result/'spatial_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    before, after = height_totals['baseline'], height_totals['result']
    lines = ['# 空间约束对照评估', '',
             f'控制输入一致性：**{"全部通过" if report["controlled_inputs_equal"] else "存在差异"}**；'
             f'两组最终解是否完全相同：`{identity}`。', '',
             '同一源帧的高斯 ID 对齐后，高度使用新结果保存的地面平面；若归档中没有平面，'
             '仅从实际源 RGB-D 和双目置信度估计一次，两组共用。未使用 GT。', '',
             '| 指标 | 基线 | 空间约束结果 |', '|---|---:|---:|',
             f'| 高度越界 non-background 软质量 | {before["outside_nonbackground_soft_mass"]:.3f} | {after["outside_nonbackground_soft_mass"]:.3f} |',
             f'| 可靠性加权越界软质量 | {before["weighted_outside_nonbackground_soft_mass"]:.3f} | {after["weighted_outside_nonbackground_soft_mass"]:.3f} |',
             f'| 越界 non-background 硬标签观测数 | {int(before["outside_hard_nonbackground"])} | {int(after["outside_hard_nonbackground"])} |',
             f'| 断开高概率区域的 non-background 软质量 | {connection_base["disconnected_high_soft_mass"]:.3f} | {connection_new["disconnected_high_soft_mass"]:.3f} |',
             f'| 断开及边界 halo 的连接惩罚能量 | {connection_base["expected_connectivity_penalty"]:.3f} | {connection_new["expected_connectivity_penalty"]:.3f} |', '',
             '连接指标只建立一份图和可靠运动核心：使用新结果归档的原始观测代价、深度和高斯 ID；'
             '两组概率都对照同一核心评估，不分别挑选有利核心。无可靠核心的 object 跳过连接指标。', '',
             '| 可靠运动核心 | 高斯数 | 基线平均概率 | 新平均概率 | 基线/新接受标签数 |',
             '|---|---:|---:|---:|---:|']
    for oid, core in cores.items():
        lines.append(f'| O{oid} | {core["count"]} | {core["baseline"]["mean_probability"]:.4f} | '
                     f'{core["result"]["mean_probability"]:.4f} | '
                     f'{core["baseline"]["accepted"]} / {core["result"]["accepted"]} |')
    if overview:
        lines += ['', f'[Object {overview_object} 原双排布局对照]({overview})']
    lines += ['', '**指标下降只说明更符合所施加的空间约束，不是分割准确率提升的证明。** '
              '连接约束是软约束，不保证严格单连通；硬标签减少也可能来自转为未确定。'
              '可靠运动核心用于检查原有物体证据是否保留，并不是 GT 车辆区域。'
              '高度总量按各源帧观测计数，连接指标按窗口内唯一高斯计数。', '',
              '[完整检查、逐帧统计及固定核心信息](spatial_comparison.json)', '']
    (result/'SPATIAL_COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--overview-object', type=int, default=3)
    args = parser.parse_args()
    report = compare(args.baseline, args.result, args.overview_object)
    print(json.dumps(dict(controlled_inputs_equal=report['controlled_inputs_equal'],
                          identical_solutions=report['identical_solutions'],
                          graph_nodes=report['fixed_graph']['nodes'], graph_edges=report['fixed_graph']['edges'],
                          height=report['height_observation_totals'],
                          output=str(args.result/'SPATIAL_COMPARISON.md'))))
    raise SystemExit(0 if report['controlled_inputs_equal'] else 1)


if __name__ == '__main__':
    main()

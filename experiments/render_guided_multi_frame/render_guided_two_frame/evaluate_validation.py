"""Evaluate ablations on fixed evaluation-only dynamic masks and shared pixels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STANDARD = ('single_rigid', 'multi_no_feedback', 'full_feedback')


def load_result(path: Path):
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    maps = np.load(path / 'score_maps.npz')
    image = cv2.imread(str(path / 'render_final.png'), cv2.IMREAD_COLOR)
    if image is None or 'rendered_depth_final' not in maps:
        raise RuntimeError(f'Incomplete validation output: {path}')
    state = np.load(path / 'pairwise_state.npz')
    return {'summary': summary, 'maps': maps, 'image': image,
            'anchors': state['anchors_final'],
            'q_anchor_final': state['q_anchor_final']}


def valid_depth_mask(sequence_root: Path, frame: int, shape: tuple[int, int], cfg: dict):
    source = cfg['depth']
    depth = np.load(sequence_root / source['subdir'] / f'{frame:010d}.npy')
    if depth.shape != shape:
        raise ValueError(f'Depth shape {depth.shape} != {shape}')
    valid = np.isfinite(depth) & (depth >= source['min_m']) & (depth <= source['max_m'])
    sky_path = sequence_root / 'sky_masks' / f'{frame:010d}.png'
    if sky_path.exists():
        sky = cv2.imread(str(sky_path), cv2.IMREAD_GRAYSCALE)
        if sky is None or sky.shape != shape:
            raise ValueError(f'Invalid sky mask: {sky_path}')
        valid &= sky <= 127
    return depth, valid


def metrics(result: dict, target: np.ndarray, depth: np.ndarray,
            region: np.ndarray, observed: np.ndarray, common: np.ndarray) -> dict:
    maps = result['maps']
    coverage = maps['coverage_final'] & observed & region
    observed_region = observed & region
    shared = common & region
    diff = (result['image'].astype(np.float32)-target.astype(np.float32))/255.0
    abs_rgb = np.abs(diff).mean(axis=2)
    mse_rgb = (diff*diff).mean(axis=2)
    score = maps['final']
    rendered_depth = maps['rendered_depth_final']
    count = int(shared.sum())
    return {'observed_pixels': int(observed_region.sum()),
            'covered_pixels': int(coverage.sum()),
            'coverage_fraction': float(coverage.sum()/max(int(observed_region.sum()), 1)),
            'common_pixels': count,
            'rgb_l1_common': float(abs_rgb[shared].mean()) if count else None,
            'rgb_psnr_common_db': float(-10*np.log10(max(float(mse_rgb[shared].mean()), 1e-12))) if count else None,
            'rgb_l1_all_observed': float(abs_rgb[observed_region].mean()) if observed_region.any() else None,
            'residual_score_common': float(score[shared].mean()) if count else None,
            'depth_abs_common_m': float(np.abs(rendered_depth[shared]-depth[shared]).mean()) if count else None}


def anchors_in_mask(anchors: np.ndarray, mask: np.ndarray) -> int:
    if len(anchors) == 0:
        return 0
    h, w = mask.shape
    uv = np.rint(anchors[:, 2:]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return int(mask[uv[inside, 1], uv[inside, 0]].sum())


def comparison_image(pair_report: Path, target: np.ndarray,
                     dynamic: np.ndarray, results: dict):
    pair_report.mkdir(parents=True, exist_ok=True)
    mask_u8 = dynamic.astype(np.uint8)*255
    cv2.imwrite(str(pair_report / 'evaluation_dynamic_mask.png'), mask_u8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    items = [('Target (red: evaluation mask)', target)]
    items += [(name, results[name]['image']) for name in STANDARD]
    rows = []
    for label, image in items:
        shown = image.copy()
        cv2.drawContours(shown, contours, -1, (0, 0, 255), 2)
        h, w = shown.shape[:2]
        bar = np.full((35, w, 3), 255, np.uint8)
        cv2.putText(bar, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (15, 15, 15), 2, cv2.LINE_AA)
        rows += [bar, shown]
    cv2.imwrite(str(pair_report / 'comparison.png'), np.vstack(rows))


def fmt(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, default=HERE / 'outputs/validation_20261007')
    args = parser.parse_args()
    group = args.group.resolve()
    manifest = json.loads((group / 'manifest.json').read_text(encoding='utf-8'))
    sequence = manifest['sequence']
    sequence_root = ROOT / 'data/kitti360/data_2d_raw' / sequence
    records, holdout_rows, ownership_rows = [], [], []
    for pair in manifest['pairs']:
        first, second = pair['first'], pair['second']
        pair_dir = group / 'pairs' / pair['pair_key']
        results = {name: load_result(pair_dir / name) for name in pair['variants']}
        target = cv2.imread(str(sequence_root / 'image_00/data_rect' / f'{second:010d}.png'))
        dynamic_u8 = cv2.imread(pair['evaluation_mask'], cv2.IMREAD_GRAYSCALE)
        if target is None or dynamic_u8 is None or dynamic_u8.shape != target.shape[:2]:
            raise RuntimeError(f'Missing target/mask or mismatched shape for {first}->{second}')
        dynamic = dynamic_u8 > 127
        cfg = yaml.safe_load((pair_dir / 'full_feedback/config.yaml').read_text())
        depth, observed = valid_depth_mask(sequence_root, second, dynamic.shape, cfg)
        common = observed.copy()
        for name in STANDARD:
            common &= results[name]['maps']['coverage_final']
        initial_count = results['single_rigid']['summary']['initial_3d_anchors']
        initial_matches = results['single_rigid']['anchors'][:initial_count]
        for name in STANDARD:
            result = results[name]
            if result['summary']['initial_3d_anchors'] != initial_count or not np.allclose(result['anchors'][:initial_count], initial_matches):
                raise RuntimeError(f'Initial anchors differ across ablations: {first}->{second} {name}')
            uv = np.rint(initial_matches[:, 2:]).astype(int)
            h, w = dynamic.shape
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            truth = np.zeros(initial_count, dtype=bool)
            truth[inside] = dynamic[uv[inside, 1], uv[inside, 0]]
            predicted = (result['q_anchor_final'][:initial_count].argmax(axis=1)
                         != result['summary']['reference_final'])
            tp = int((truth & predicted).sum())
            fp = int((~truth & predicted).sum())
            fn = int((truth & ~predicted).sum())
            ownership_rows.append({'pair': f'{first}->{second}', 'variant': name,
                                   'initial_anchors': initial_count,
                                   'mask_dynamic_anchors': int(truth.sum()),
                                   'predicted_nonreference_anchors': int(predicted.sum()),
                                   'true_positive': tp, 'false_positive': fp,
                                   'false_negative': fn,
                                   'precision': tp/(tp+fp) if tp+fp else 0.0,
                                   'recall': tp/(tp+fn) if tp+fn else 0.0,
                                   'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0})
            for region_name, region in (('dynamic', dynamic), ('static', ~dynamic)):
                row = {'pair': f'{first}->{second}', 'variant': name, 'region': region_name,
                       **metrics(result, target, depth, region, observed, common),
                       'final_models': result['summary']['final_models'],
                       'final_3d_anchors': result['summary']['final_3d_anchors'],
                       'anchors_in_dynamic_mask': anchors_in_mask(result['anchors'], dynamic),
                       'camera_rotation_error_deg': result['summary']['pose_error_final']['rotation_deg'],
                       'camera_translation_error_m': result['summary']['pose_error_final']['translation_m'],
                       'accepted_structure_proposals': sum(
                           int(p.get('accepted', False))
                           for cycle in result['summary']['cycles'] for p in cycle['proposals'])}
                records.append(row)
        pair_report = group / 'reports' / f'{first:010d}_{second:010d}'
        comparison_image(pair_report, target, dynamic, results)
        if pair['holdout_roi'] is not None:
            x0, y0, x1, y1 = pair['holdout_roi']
            object_mask = np.zeros_like(dynamic)
            object_mask[y0:y1, x0:x1] = dynamic[y0:y1, x0:x1]
            hcommon = observed.copy()
            for name in ('holdout_no_feedback', 'holdout_full_feedback'):
                hcommon &= results[name]['maps']['coverage_final']
            for name in ('holdout_no_feedback', 'holdout_full_feedback'):
                result = results[name]
                holdout_rows.append({
                    'pair': f'{first}->{second}', 'variant': name,
                    'holdout_roi': pair['holdout_roi'],
                    'removed_initial_matches': result['summary']['holdout_evaluation']['initial_matches_removed'],
                    'novel_3d_candidates_in_roi': result['summary']['holdout_evaluation']['new_3d_anchors_in_holdout'],
                    'retained_anchors_on_mask': anchors_in_mask(result['anchors'], object_mask),
                    'selected_patch_hits': result['summary']['holdout_evaluation']['selected_patch_hits'],
                    **metrics(result, target, depth, object_mask, observed, hcommon)})
    report_dir = group / 'reports'
    report_dir.mkdir(exist_ok=True)
    if records:
        with (group / 'evaluation.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    aggregates = {}
    for region in ('dynamic', 'static'):
        aggregates[region] = {}
        for name in STANDARD:
            selected = [row for row in records if row['region'] == region and row['variant'] == name]
            common_pixels = sum(row['common_pixels'] for row in selected)
            observed_pixels = sum(row['observed_pixels'] for row in selected)
            aggregates[region][name] = {
                'common_pixels': common_pixels,
                'rgb_l1_common': sum(row['rgb_l1_common']*row['common_pixels'] for row in selected)/common_pixels,
                'residual_score_common': sum(row['residual_score_common']*row['common_pixels'] for row in selected)/common_pixels,
                'rgb_l1_all_observed': sum(row['rgb_l1_all_observed']*row['observed_pixels'] for row in selected)/observed_pixels,
                'coverage_fraction': sum(row['covered_pixels'] for row in selected)/observed_pixels,
            }
    with (group / 'ownership_diagnostic.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ownership_rows[0]))
        writer.writeheader(); writer.writerows(ownership_rows)
    (group / 'evaluation.json').write_text(json.dumps({'aggregate': aggregates, 'standard': records,
                                                       'holdout': holdout_rows,
                                                       'ownership_diagnostic': ownership_rows}, indent=2), encoding='utf-8')
    lines = ['# 两帧方法验证', '',
             f'数据：{sequence}，{len(manifest["pairs"])} 个帧对。动态掩码仅用于事后评估和受控移除初始匹配；估计过程不读取掩码。',
             '', '方案：single_rigid（单刚体）、multi_no_feedback（多刚体，关闭发现反馈）、full_feedback（完整方法）。',
             '各方法的 RGB、深度和残差在三组共同覆盖的相同像素上比较；覆盖率及所有有效深度像素的 RGB 误差另列，防止空洞造成虚假改善。',
             '', '## 跨帧对加权汇总', '',
             '| 区域 | 方法 | 覆盖率 | 共同像素 RGB L1 ↓ | 全部有效像素 RGB L1 ↓ | 残差 ↓ |',
             '|---|---|---:|---:|---:|---:|']
    for region in ('dynamic', 'static'):
        for name in STANDARD:
            row = aggregates[region][name]
            lines.append(f"| {region} | {name} | {100*row['coverage_fraction']:.1f}% | {fmt(row['rgb_l1_common'])} | {fmt(row['rgb_l1_all_observed'])} | {fmt(row['residual_score_common'])} |")
    wins = {}
    for baseline in ('single_rigid', 'multi_no_feedback'):
        wins[baseline] = sum(
            next(x for x in records if x['pair']==f"{pair['first']}->{pair['second']}" and x['region']=='dynamic' and x['variant']=='full_feedback')['rgb_l1_common']
            < next(x for x in records if x['pair']==f"{pair['first']}->{pair['second']}" and x['region']=='dynamic' and x['variant']==baseline)['rgb_l1_common']
            for pair in manifest['pairs'])
    holdout_pairs = sorted({row['pair'] for row in holdout_rows})
    holdout_recovered = sum(
        next(x for x in holdout_rows if x['pair']==key and x['variant']=='holdout_full_feedback')['retained_anchors_on_mask'] > 0
        for key in holdout_pairs)
    lines += ['', '## 目前能得出的结论', '',
              f'多刚体方法相对单刚体，在 {wins["single_rigid"]}/{len(manifest["pairs"])} 组动态区域降低 RGB 误差。',
              f'完整反馈相对多刚体无反馈，仅在 {wins["multi_no_feedback"]}/{len(manifest["pairs"])} 组动态区域降低 RGB 误差；加权动态残差与静态区域 RGB 误差没有同步改善。',
              f'受控漏检在 {holdout_recovered}/{len(holdout_pairs)} 组恢复了动态掩码内最终保留的锚点。非参考运动的硬归属有大量静态误报；当前证据支持多运动建模改善渲染，但尚不能证明运动分解或渲染反馈稳定有效。',
              '', '## 动态区域', '',
             '| 帧对 | 方法 | 覆盖率 | 共同像素 RGB L1 ↓ | 全部有效像素 RGB L1 ↓ | 残差 ↓ | 深度误差 m ↓ |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in records:
        if row['region'] != 'dynamic': continue
        lines.append(f"| {row['pair']} | {row['variant']} | {100*row['coverage_fraction']:.1f}% | {fmt(row['rgb_l1_common'])} | {fmt(row['rgb_l1_all_observed'])} | {fmt(row['residual_score_common'])} | {fmt(row['depth_abs_common_m'], 3)} |")
    lines += ['', '## 静态区域', '',
              '| 帧对 | 方法 | 覆盖率 | 共同像素 RGB L1 ↓ | 全部有效像素 RGB L1 ↓ | 残差 ↓ |',
              '|---|---|---:|---:|---:|---:|']
    for row in records:
        if row['region'] != 'static': continue
        lines.append(f"| {row['pair']} | {row['variant']} | {100*row['coverage_fraction']:.1f}% | {fmt(row['rgb_l1_common'])} | {fmt(row['rgb_l1_all_observed'])} | {fmt(row['residual_score_common'])} |")
    lines += ['', '## 位姿与结构', '',
              '| 帧对 | 方法 | 模型数 | 动态掩码内保留锚点 | 相机旋转误差 ° ↓ | 相机平移误差 m ↓ |',
              '|---|---|---:|---:|---:|---:|']
    for row in records:
        if row['region'] != 'dynamic': continue
        lines.append(f"| {row['pair']} | {row['variant']} | {row['final_models']} | {row['anchors_in_dynamic_mask']} | {fmt(row['camera_rotation_error_deg'], 3)} | {fmt(row['camera_translation_error_m'], 3)} |")
    lines += ['', '## 运动归属诊断', '',
              '在三组完全相同的初始锚点上，按 argmax(q_anchor) 是否等于参考运动做二分类；这是硬归属诊断，不等同于最终的连续软归属。',
              '', '| 帧对 | 方法 | 掩码内动态锚点 | 判为非参考运动 | Precision ↑ | Recall ↑ | F1 ↑ |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in ownership_rows:
        lines.append(f"| {row['pair']} | {row['variant']} | {row['mask_dynamic_anchors']} | {row['predicted_nonreference_anchors']} | {fmt(row['precision'], 3)} | {fmt(row['recall'], 3)} | {fmt(row['f1'], 3)} |")
    lines += ['', '## 受控漏检', '',
              '移除动态掩码最大连通区域外接框内的初始匹配；框只用于构造漏检，方法并不知道物体掩码。', '',
              '| 帧对 | 方法 | 移除初始匹配 | 框内新 3D 候选 | 动态掩码内最终保留锚点 | 目标区域 RGB L1 ↓ |',
              '|---|---|---:|---:|---:|---:|']
    for row in holdout_rows:
        lines.append(f"| {row['pair']} | {row['variant']} | {row['removed_initial_matches']} | {row['novel_3d_candidates_in_roi']} | {row['retained_anchors_on_mask']} | {fmt(row['rgb_l1_common'])} |")
    lines += ['', '## 按帧对查看', '',
              '| 帧对 | 动态掩码 | 四行图像对比 | 完整反馈渲染 | 单刚体归属 | 多刚体无反馈归属 | 完整反馈归属 |',
              '|---|---|---|---|---|---|---|']
    for pair in manifest['pairs']:
        first, second = pair['first'], pair['second']
        report_rel = f'reports/{first:010d}_{second:010d}'
        full_rel = f"pairs/{pair['pair_key']}/full_feedback/render_final.png"
        motion_links = []
        for variant in STANDARD:
            motion_rel = f"pairs/{pair['pair_key']}/{variant}/motion_visualization"
            motion_links.append(
                f'[overview]({motion_rel}/overview.png) · [parameters]({motion_rel}/motions.json)'
                if (group / motion_rel / 'overview.png').is_file() else '—')
        lines.append(f'| {first}->{second} | [mask]({report_rel}/evaluation_dynamic_mask.png) | [comparison]({report_rel}/comparison.png) | [render]({full_rel}) | ' + ' | '.join(motion_links) + ' |')
    lines += ['', '## 文件结构', '',
              '- pairs/<帧对>/<方法>/：原始 summary、渲染图、残差图、深度和状态。',
              '- configs/、logs/：逐运行配置与日志。',
              '- reports/<帧对>/comparison.png：目标图、三组结果及红色评估区域。',
              '- pairs/<帧对>/<方法>/motion_visualization/：各方法的第一帧及第二帧运动归属、分运动渲染、相对刚体运动参数。颜色表示估计的运动编号，不是物体语义类别或真值；不同方法的同色编号不保证对应同一运动。',
              '- evaluation.csv、evaluation.json：逐帧对、逐区域的机器可读指标；ownership_diagnostic.csv 保存硬归属诊断。',
              '', '限制：这三个帧对来自同一序列；只有相机位姿 GT，现有物体位姿文件是算法估计结果，未作为独立 GT 计分。']
    (group / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote {group / "report.md"}')
    for region in ('dynamic', 'static'):
        for baseline in ('single_rigid', 'multi_no_feedback'):
            wins = 0
            for pair in manifest['pairs']:
                key = f"{pair['first']}->{pair['second']}"
                full = next(x for x in records if x['pair']==key and x['region']==region and x['variant']=='full_feedback')
                base = next(x for x in records if x['pair']==key and x['region']==region and x['variant']==baseline)
                wins += full['rgb_l1_common'] < base['rgb_l1_common']
            print(f'{region} full_feedback RGB wins vs {baseline}: {wins}/{len(manifest["pairs"])}')


if __name__ == '__main__':
    main()

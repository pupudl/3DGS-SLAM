#!/usr/bin/env python3
"""Sequential tracks and persistent objects around the two-frame estimator."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAIR_DIR = HERE / 'render_guided_two_frame'
sys.path.insert(0, str(PAIR_DIR))

from feature_tracks import FeatureTrackStore
from object_state import ObjectRegistry


def configure_object_window(cfg):
    """Keep local motion responsibilities immutable in object-window mode.

    Feedback may add observations or repair poses, but may not resize or reset
    the local probability matrix. Persistent object births still come from the
    next pair's independent motion discovery.
    """
    cfg = deepcopy(cfg)
    if cfg.get('object_window', {}).get('enabled', False):
        cfg.setdefault('joint', {})['freeze_motion_ownership'] = True
        for operation in ('birth', 'split', 'merge', 'prune'):
            cfg.setdefault('discovery', {})[f'enable_{operation}'] = False
    return cfg


def merge_config(base, updates):
    result = deepcopy(base)
    for key, value in updates.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else deepcopy(value))
    return result


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop('base_config', None)
    if parent:
        parent = (path.parent / parent).resolve()
        cfg = merge_config(load_config(parent), cfg)
    return cfg


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


class SequencePairContext:
    """Only these hooks connect persistent state to a pairwise optimization."""

    def __init__(self, first, second, tracks, objects, output_root, object_backend=None):
        self.first, self.second = first, second
        self.tracks, self.objects = tracks, objects
        self.output_root = output_root
        self.object_backend = object_backend
        self.initial_report = {}
        self.prior_tracks = set(objects.to_dict().get('track_objects', {}))

    def match_initial(self, image0, image1, extractor, matcher):
        return self.tracks.match_pair(self.first, self.second, image0, image1,
                                      extractor, matcher)

    def match_feedback(self, image0, image1, extractor, matcher, mask):
        return self.tracks.match_mask(self.first, self.second, image0, image1,
                                      extractor, matcher, mask)

    def record_matches(self, matches):
        return self.tracks.register_matches(self.first, self.second, matches)

    def initialize(self, anchors, K, models, labels, cfg):
        ids = self.record_matches(anchors['matches'])
        models, labels, q, self.initial_report = self.objects.prepare_pair(
            self.first, self.second, anchors, K, cfg['motion'], models, labels, ids)
        return models, labels, q, self.initial_report

    def finish(self, anchors, gmap, state, K, output, cfg, observations=None):
        if self.object_backend is not None:
            required = {'source', 'target', 'source_depth', 'target_depth',
                        'source_confidence', 'target_confidence'}
            if observations is None or required - observations.keys():
                raise ValueError('Object-window optimization requires the raw pair observations')
        from hybrid_motion import _core_labels
        ids = self.record_matches(anchors['matches'])
        _, labels = _core_labels(anchors, list(state['H']), K, cfg['motion']['hybrid'])
        mapping = self.objects.finish_pair(
            self.first, self.second, state['H'], state['q_anchor'], ids,
            int(state['reference']), str(output.relative_to(self.output_root)), labels=labels)
        columns = np.asarray([mapping[str(k)] for k in range(len(state['H']))], np.int64)
        np.savez_compressed(output / 'pair_tracks.npz', matches=anchors['matches'],
                            track_ids=ids, labels=labels, q_anchor=state['q_anchor'],
                            local_to_object=columns)
        np.savez_compressed(output / 'gaussian_state.npz',
                            xyz=gmap['xyz'], uv=gmap['uv'], color=gmap['color'],
                            scale=gmap['scale'], q=state['q'], H=state['H'], K=K,
                            local_to_object=columns)
        window_report = None
        if self.object_backend is not None:
            window_report = self.object_backend.process(
                self.first, self.second, gmap, anchors, ids, mapping, state, K,
                output=output, registry=self.objects, **observations)
        row = dict(first=self.first, second=self.second,
                   result_path=str(output.relative_to(self.output_root)),
                   local_to_object=mapping, reference=int(state['reference']),
                   inherited_3d_tracks=sum(str(int(t)) in self.prior_tracks for t in ids if t >= 0),
                   retained_3d_tracks=int(np.count_nonzero(ids >= 0)),
                   core_3d_tracks=int(np.count_nonzero(labels >= 0)),
                   initialization=self.initial_report)
        if window_report is not None:
            row['object_window'] = window_report
        (output / 'temporal_state.json').write_text(json.dumps(row, indent=2))
        self.final_report = row
        return row


def stage_overviews(result, log_path):
    scripts = (
        ('visualize_sparse_hypotheses', 'sparse_motion_visualization'),
        ('visualize_ownership_stages', 'ownership_stage_comparison'),
        ('visualize_feedback_matches', 'feedback_match_visualization'),
        ('visualize_feedback_final', 'feedback_final_visualization'),
        ('visualize_motion', 'motion_visualization'),
    )
    with log_path.open('w') as log:
        for script, directory in scripts:
            with tempfile.TemporaryDirectory(prefix='multi_frame_overview_') as temporary:
                subprocess.run([sys.executable, str(PAIR_DIR / f'{script}.py'),
                                '--result', str(result), '--output', temporary],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                destination = result / directory
                destination.mkdir(exist_ok=True)
                for source in [Path(temporary) / 'overview.png', *Path(temporary).glob('*.json')]:
                    shutil.copy2(source, destination / source.name)


def write_report(output, state, track_summary):
    window = state.get('object_window')
    lines = ['# 多帧 track 与持续 object 实验', '',
             '共享帧的特征、描述子和编号在帧对之间复用；补点建立的 track 继续参与后续匹配。'
             '延续 track 提供 object 初始归属，运动使用当前帧对匹配重新拟合，'
             '随后完整执行两帧联合优化和渲染反馈。未使用动态物体标注。', '',
             f"处理帧：`{state['frames']}`。共 {track_summary['tracks']} 条二维 track，"
             f"其中 {track_summary['tracks_with_three_or_more_frames']} 条覆盖至少三帧。", '',
             '[按 object 选择查看多帧结果](objects/index.html)', '',
             '每个 object 的总览固定两排，每列一帧；N 帧共 2N 张图。上排严格使用保存的最终 '
             '`object_labels == object_id` 的高斯进行实际 CUDA RGB 渲染，并按所有 object 共同计算的'
             '最近有效高斯可见性裁剪；下排使用相同可见区域，在原图上叠加该 object 的颜色，'
             '每个像素最多显示一个 object，无覆盖区域保留原图。'
             '未确定标签不纳入任何 object；重新渲染需要 CUDA，不重新优化。', '',
             '| 帧对 | 最终运动 | 继承的三维 track | 最终核心 track | object ID（按局部候选顺序） |',
             '|---|---:|---:|---:|---|']
    for row in state['pairs']:
        lines.append(f"| {row['first']}→{row['second']} | {len(row['local_to_object'])} | "
                     f"{row['inherited_3d_tracks']} | {row['core_3d_tracks']} | "
                     f"{list(row['local_to_object'].values())} |")
    lines += ['', '### 跨帧身份关联', '',
              '先用当前核心 track 关联已有 object。尚未关联的候选还可以使用延续灰点的相对软偏好'
              '补充身份线索；已建立的核心关联保持不变，灰点仍保持灰色，不因此成为几何核心点。']
    for event in state.get('events', []):
        if event['type'] == 'identity_from_soft_tracks':
            first, second = event['frames']
            lines.append(f"\n{first}→{second} 的局部候选 M{event['local_model']} 通过 "
                         f"{len(event['support_track_ids'])} 条延续灰点 track，关联回 "
                         f"object {event['object_id']}。")
    lines += ['', '### 每一步的 overview', '',
              '| 帧对 | 初始候选 | 联合优化 | 补匹配 | 反馈前后 | 最终运动 |',
              '|---|---|---|---|---|---|']
    directories = ('sparse_motion_visualization', 'ownership_stage_comparison',
                   'feedback_match_visualization', 'feedback_final_visualization',
                   'motion_visualization')
    for row in state['pairs']:
        links = ' | '.join(f"[查看]({row['result_path']}/{d}/overview.png)" for d in directories)
        lines.append(f"| {row['first']}→{row['second']} | {links} |")
    lines += ['', '### 留存状态', '',
              '`tracks/` 保存特征描述子、二维轨迹和出现帧；`sequence_state.json` 保存持续 object ID、'
              '历史映射及拆分/合并事件；各帧对额外保存完整高斯状态和 track 到 object 的当时归属。', '',
              '当前为顺序帧对更新。高斯按帧保存，尚未融合成持续的物体地图；'
              'object 表示持续关联的运动分组，不保证与真实物体一一对应。'
              '末帧使用保存的 object 运动变换高斯后进行实际 CUDA 高斯渲染。', '']
    if window is not None:
        lines[2] = ('相邻帧运动概率仅用于初始化并保持只读；高斯 object 属性由窗口内的 RGB、'
                    '双目深度和 track 观测联合优化。窗口内各帧共享的高斯 ID 使用同一份 object '
                    '参数，最新帧没有额外权重；track 继承标签不当作重复观测。')
        render_enabled = bool(state.get('config', {}).get('object_window', {}).get('render', {}).get('enabled', False))
        objective_note = (
            '窗口内的实际 Gaussian RGB、深度与覆盖率渲染损失共同反向传播到共享 object logits；'
            '中心观测和 track 误差作为辅助项。每个窗口帧等权，梯度累计完成后共同更新。'
            '每个帧对先根据当前观测选择/重拟合 object 运动；进入本轮 logits 优化后，'
            '高斯几何、颜色和这些运动固定，尚未融合为一份物体几何地图。'
            if render_enabled else
            '此配置明确关闭渲染优化，仅使用高斯中心的光度、深度和匹配误差；'
            '几何固定，运动按帧对拟合。未使用整窗可微 raster 损失或物体几何融合。')
        lines += ['### object 优化窗口', '',
                  f"最后窗口图像：`{window['window_frames']}`；共享高斯总数："
                  f"{window['total_gaussians']}，当前窗口高斯：{window['active_gaussians']}。", '',
                  '`object_state.npz` 保存优化后的 object 概率/标签、只读的局部运动初始化、'
                  '逐帧实际观测代价及对应窗口。每次更新重写所有窗口内结果，滑出窗口后不再修改。'
                  '`gaussian_state.npz` 与原阶段 overview 保留局部运动初始化；按 object 查看优先读取'
                  '窗口结果。', '',
                  objective_note, '']
        spatial = window.get('render_optimization', {}).get('spatial_constraints')
        if spatial:
            lines += ['### 高度与核心连通约束', '',
                      '各源帧在相机坐标中独立估计可信地面，再按共享高斯 ID 汇总相对高度；'
                      '高度范围外的非背景概率受到软惩罚，O0 保留。非背景 object 使用真实多帧'
                      '运动证据选择可靠核心，定期更新与核心断开的高概率区域惩罚，和渲染损失'
                      '共同反传。O0、无可靠核心及无观测区域不被强制单连通。', '',
                      '这些是软约束，不保证每帧可见 mask 只有一个连通块；约束违反量下降'
                      '不代表已证明真实物体分割准确率提高。', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def run_sequence(cfg, frames, output):
    import torch
    from strict_two_frame import run
    from visualize_objects import visualize

    output = resolve(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    cfg = configure_object_window(cfg)
    cfg['motion']['initialization_method'] = 'hybrid'
    cfg['output']['compact'] = True
    cfg['output_dir'] = str(output / 'pairs')
    cfg['run_tag'] = 'multi_frame'
    cfg['sequence']['frames'] = frames
    cfg['sequence']['output_dir'] = str(output)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    tracks, objects = FeatureTrackStore(cfg), ObjectRegistry(cfg)
    object_backend = None
    if cfg.get('object_window', {}).get('enabled', False):
        from gaussian_object_backend import SequenceObjectBackend
        object_backend = SequenceObjectBackend(cfg)
    state = dict(frames=frames, sequence=cfg['dataset']['sequence'], dataset=cfg['dataset'],
                 config=cfg, tracking_dir='tracks', pairs=[], objects={})
    for first, second in zip(frames[:-1], frames[1:]):
        pair_cfg = deepcopy(cfg)
        pair_cfg['frames'] = dict(first=first, second=second)
        context = SequencePairContext(first, second, tracks, objects, output, object_backend)
        print(f'[{first}→{second}] Estimate with persistent tracks...', flush=True)
        with (logs / f'{first}_{second}_estimate.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = run(pair_cfg, pair_context=context)
        state['pairs'].append(context.final_report)
        object_state = objects.to_dict()
        state.update({k: v for k, v in object_state.items() if k != 'pairs'})
        if object_backend is not None:
            state['object_window'] = object_backend.window.to_dict()
        track_summary = tracks.save(output / 'tracks')
        (output / 'sequence_state.json').write_text(json.dumps(state, indent=2))
        result = output / context.final_report['result_path']
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{first}→{second}] {summary['final_models']} motions; "
              f"{context.final_report['inherited_3d_tracks']} inherited tracks. Save stage overviews...",
              flush=True)
        stage_overviews(result, logs / f'{first}_{second}_visualize.log')
    visualize(output)
    write_report(output, state, track_summary)
    snapshot = output / 'source_snapshot'
    for source_dir in (HERE, PAIR_DIR):
        destination = snapshot / source_dir.name
        destination.mkdir(parents=True, exist_ok=True)
        for source in source_dir.glob('*.py'):
            shutil.copy2(source, destination / source.name)
    shutil.copy2(output / 'config.yaml', snapshot / 'config.yaml')
    print(f'Complete: {output / "README.md"}', flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/kitti360_short_sequence.yaml')
    parser.add_argument('--frames', nargs='+', type=int, help='Explicit ordered frame IDs')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int, help='Inclusive final frame')
    parser.add_argument('--step', type=int, default=2)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    frames = (args.frames if args.frames is not None else
              list(range(args.start, args.end + 1, args.step)) if args.start is not None else
              list(cfg['sequence']['frames']))
    if len(frames) < 2 or any(a >= b for a, b in zip(frames[:-1], frames[1:])):
        parser.error('Choose at least two strictly increasing frame IDs')
    run_sequence(cfg, frames, args.output or cfg['sequence']['output_dir'])


if __name__ == '__main__':
    main()

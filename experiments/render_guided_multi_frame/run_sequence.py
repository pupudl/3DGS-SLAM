#!/usr/bin/env python3
"""Sequential tracks and persistent objects around the two-frame estimator."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VISUALIZATION_DIR = HERE / 'tools' / 'visualization'
sys.path.insert(0, str(HERE))

from tracking.feature_tracks import FeatureTrackStore
from tracking.object_state import ObjectRegistry


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

    def __init__(self, first, second, tracks, objects, output_root, object_backend=None, map_backend=None,
                 feature_models=None):
        self.first, self.second = first, second
        self.tracks, self.objects = tracks, objects
        self.output_root = output_root
        self.object_backend = object_backend
        self.map_backend = map_backend
        self._feature_models = {} if feature_models is None else feature_models
        self.initial_report = {}
        self.prior_tracks = set(objects.to_dict().get('track_objects', {}))

    def feature_models(self, factory):
        """Load immutable inference networks once per sequence, lazily."""
        if 'models' not in self._feature_models:
            self._feature_models['models'] = factory()
        return self._feature_models['models']

    def match_initial(self, image0, image1, extractor, matcher,
                      source_mask=None, target_mask=None):
        return self.tracks.match_pair(self.first, self.second, image0, image1,
                                      extractor, matcher, source_mask=source_mask,
                                      target_mask=target_mask)

    def match_feedback(self, image0, image1, extractor, matcher, mask, source_mask=None):
        return self.tracks.match_mask(self.first, self.second, image0, image1,
                                      extractor, matcher, mask, source_mask=source_mask)

    def record_matches(self, matches):
        return self.tracks.register_matches(self.first, self.second, matches)

    def initialize(self, anchors, K, models, labels, cfg):
        ids = self.record_matches(anchors['matches'])
        models, labels, q, self.initial_report = self.objects.prepare_pair(
            self.first, self.second, anchors, K, cfg['motion'], models, labels, ids)
        return models, labels, q, self.initial_report

    def finish(self, anchors, gmap, state, K, output, cfg, observations=None):
        started = time.perf_counter()
        timings = {}
        if self.object_backend is not None:
            required = {'source', 'target', 'source_depth', 'target_depth',
                        'source_confidence', 'target_confidence'}
            if observations is None or required - observations.keys():
                raise ValueError('Object-window optimization requires the raw pair observations')
        from frontend.hybrid_motion import _core_labels
        ids = self.record_matches(anchors['matches'])
        _, labels = _core_labels(anchors, list(state['H']), K, cfg['motion']['hybrid'])
        mapping = self.objects.finish_pair(
            self.first, self.second, state['H'], state['q_anchor'], ids,
            int(state['reference']), str(output.relative_to(self.output_root)), labels=labels,
            anchors=anchors, K=K, motion_cfg=cfg['motion'])
        columns = np.asarray([mapping[str(k)] for k in range(len(state['H']))], np.int64)
        timings['identity_seconds'] = time.perf_counter()-started
        started = time.perf_counter()
        np.savez_compressed(output / 'pair_tracks.npz', matches=anchors['matches'],
                            track_ids=ids, labels=labels, q_anchor=state['q_anchor'],
                            local_to_object=columns)
        np.savez_compressed(output / 'gaussian_state.npz',
                            xyz=gmap['xyz'], uv=gmap['uv'], color=gmap['color'],
                            scale=gmap['scale'], q=state['q'], H=state['H'], K=K,
                            local_to_object=columns)
        timings['pair_archive_seconds'] = time.perf_counter()-started
        window_report = None
        started = time.perf_counter()
        if self.object_backend is not None:
            window_report = self.object_backend.process(
                self.first, self.second, gmap, anchors, ids, mapping, state, K,
                output=output, registry=self.objects, **observations)
        timings['object_backend_seconds'] = time.perf_counter()-started
        row = dict(first=self.first, second=self.second,
                   result_path=str(output.relative_to(self.output_root)),
                   local_to_object=mapping, reference=int(state['reference']),
                   inherited_3d_tracks=sum(str(int(t)) in self.prior_tracks for t in ids if t >= 0),
                   retained_3d_tracks=int(np.count_nonzero(ids >= 0)),
                   core_3d_tracks=int(np.count_nonzero(labels >= 0)),
                   initialization=self.initial_report)
        if window_report is not None:
            row['object_window'] = window_report
        if self.map_backend is not None:
            started = time.perf_counter()
            # Ownership is solved first on immutable observations, then reliable
            # corrections are synchronized to persistent map IDs before fitting.
            snapshot = self.object_backend.window._active[(self.first, self.second)]
            posterior = self.object_backend.window.get_state(snapshot['gaussian_ids'])
            row['gaussian_map'] = self.map_backend.process(
                self.first, self.second, gmap, posterior, snapshot['object_ids'],
                snapshot['payload']['H_object'], snapshot['payload']['object_model_valid'], K,
                observations, self.objects, snapshot['track_ids'],
                presence=window_report.get('lifecycle'),
                observation_ids=snapshot['gaussian_ids'], ownership_window=self.object_backend.window)
            timings['map_backend_seconds'] = time.perf_counter()-started
        row['timings'] = timings
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
                subprocess.run([sys.executable, str(VISUALIZATION_DIR / f'{script}.py'),
                                '--result', str(result), '--output', temporary],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                destination = result / directory
                destination.mkdir(exist_ok=True)
                for source in [Path(temporary) / 'overview.png', *Path(temporary).glob('*.json')]:
                    shutil.copy2(source, destination / source.name)


def write_report(output, state, track_summary):
    if state.get('config', {}).get('sequence', {}).get('streaming', False):
        window = state.get('object_window', {})
        lines = ['# 序列 object 窗口运行', '',
                 f"处理 {len(state['frames'])} 帧、{len(state['pairs'])} 个帧对；"
                 f"共 {track_summary['tracks']} 条 track、{track_summary['observations']} 次二维观测。", '',
                 f"归属阶段窗口 {state['config']['object_window']['window_frames']} 帧，"
                 f"每次推进 {state['config']['object_window'].get('window_step', 1)} 个采样帧；"
                 f"最后保留 {window.get('retained_gaussians', 0)} 个高斯，"
                 f"累计创建 {window.get('total_gaussians', 0)} 个高斯 ID。", '',
                 '归属阶段窗口外帧对结果冻结到磁盘，退出窗口的观测归属参数和特征缓存被释放。'
                 '仍在窗口中的高斯保留已退出帧对的中心/track 充分统计量；'
                 '旧像素渲染损失不进行精确边缘化，上一轮后验仅提供渲染求解的热启动。', '',
                 '`object_lifecycle.jsonl` 保存完整生命周期事件，`pairs.jsonl` 保存帧对记录；'
                 '`sequence_state.json` 保存紧凑索引与最终 object 区间。'
                 '`tracks/` 保存增量观测、磁盘索引及结束时导出的兼容 NPZ。', '',
                 '| Object | 状态 | 首次可靠观测 | 最后可靠观测 | 归档确认帧 | 原因 |',
                 '|---|---|---:|---:|---:|---|']
        for oid, record in sorted(state['objects'].items(), key=lambda item: int(item[0])):
            lines.append(f"| {oid} | {record.get('status')} | {record.get('first_observed_frame')} | "
                         f"{record.get('last_observed_frame')} | {record.get('retired_at_frame')} | "
                         f"{record.get('retirement_reason') or ''} |")
        if (output / 'objects/index.html').is_file():
            lines += ['', '[逐 object、逐帧查看 RGB 和 mask](objects/index.html)']
        if state.get('gaussian_map', {}).get('enabled'):
            mapping = state['gaussian_map']
            submaps = mapping.get('submaps_enabled', False)
            map_link = ('gaussian_map/submaps/manifest.json' if submaps else
                        'gaussian_map/map/manifest.json')
            total = mapping.get('global_stored_gaussians', mapping['total_map_gaussians'])
            lines += ['', '### 持久高斯地图联合优化', '',
                      f"存活地图高斯 {total}，最后活动批次 "
                      f"{mapping['active_map_gaussians']}。背景存世界坐标，车辆存各自规范坐标。", '',
                      '每次几何优化固定当前可靠 object 归属、相机和物体运动，联合优化位置、RGB、'
                      '三轴尺度、四元数和不透明度。地图与 Adam 状态跨窗口保存；'
                      '原始观测和 object 初始化保持只读。相机与物体位姿暂不参与地图联合优化。', '',
                      f'[地图检查点]({map_link})']
            map_options = state['config'].get('gaussian_map', {})
            if submaps:
                size = map_options['submaps'].get('frames', 50)
                lines += ['', f'固定每个完整子图 {size} 张处理图像、相邻重叠 1 张，'
                          '不按点数提前封存。归档子图的完整地图释放出内存；'
                          '全局相机坐标与 object 身份跨子图延续。', '',
                          '[全局分块场景](gaussian_map/global_scene.json)']
            if map_options.get('snapshot_scope') == 'full_visible_map':
                lines += ['', '逐帧快照包含完整可见地图，与参与本轮训练的点数预算独立。']
            if map_options.get('ownership_correction', {}).get('enabled'):
                lines += ['', '已启用观测归属到地图的修正：可靠窗口证据经过确认后，迁移稳定地图 ID，'
                          '同时变换规范坐标及协方差，保留迁移日志。']
            if map_options.get('refinement', {}).get('enabled'):
                lines += ['', '已启用受限 clone/split/prune：依据实际渲染梯度和可见观测细化，'
                          '保护冻结物体及 track 锚点；拓扑变化后继续优化新高斯。']
            if (output / 'gaussian_map/visualization/index.html').is_file():
                lines += ['', '[学习后的地图可视化](gaussian_map/visualization/index.html)']
        else:
            lines += ['', '本运行使用局部帧对几何及 object 归属优化，尚未融合世界坐标物体地图。']
        lines += ['', '归档 ID 不复用，也不凭外观自动重识别已经归档的物体。', '']
        (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')
        return
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


def prune_empty_object_records(output, state):
    """Keep primary result objects only when a final saved Gaussian owns them."""
    supported = set()
    for pair in state['pairs']:
        directory = Path(pair['result_path'])
        if not directory.is_absolute():
            directory = Path(output) / directory
        with np.load(directory / 'object_state.npz') as archive:
            labels = archive['object_labels']
            supported.update(map(int, np.unique(labels[labels >= 0])))
    # A persistent map can still own Gaussians after the corresponding raw
    # observations have been corrected or left the ownership lag.
    manifests = [Path(output) / 'gaussian_map/map/manifest.json',
                 *Path(output).glob('gaussian_map/submaps/submap_*/map/manifest.json')]
    for path in manifests:
        if path.is_file():
            records = json.loads(path.read_text())['objects']
            supported.update(int(oid) for oid, record in records.items() if record['gaussian_count'] > 0)
    state['discarded_objects'] = {oid: record for oid, record in state['objects'].items()
                                  if int(oid) not in supported}
    state['objects'] = {oid: record for oid, record in state['objects'].items() if int(oid) in supported}


def snapshot_sources(output, source_root=HERE):
    """Freeze provenance before estimation, even if files change during a run."""
    snapshot = Path(output) / 'source_snapshot'
    source_files = [source_root / 'run_sequence.py']
    for package in ('frontend', 'tracking', 'ownership', 'mapping', 'tools'):
        source_files.extend((source_root / package).rglob('*.py'))
    for source in source_files:
        destination = snapshot / source.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(Path(output) / 'config.yaml', snapshot / 'config.yaml')
    if Path(source_root).resolve() == HERE:
        dependencies = []
        for name in ('superpoint.py', 'lightglue.py'):
            source = ROOT / 'sp_lg' / name
            destination = Path(output) / 'source_dependencies' / 'sp_lg' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            dependencies.append(dict(path=str(source.relative_to(ROOT)),
                sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
        (Path(output) / 'source_dependencies/manifest.json').write_text(
            json.dumps(dependencies, indent=2))


def run_sequence(cfg, frames, output):
    import torch
    from frontend.strict_two_frame import run
    from tools.visualization.visualize_objects import visualize

    cfg = configure_object_window(cfg)
    streaming = bool(cfg.get('sequence', {}).get('streaming', False))
    if streaming and not cfg.get('object_window', {}).get('enabled', False):
        raise ValueError('Streaming requires object_window.enabled: true')
    if streaming and not cfg.get('object_lifecycle', {}).get('enabled', True):
        raise ValueError('Streaming requires object_lifecycle.enabled: true')
    map_enabled = bool(cfg.get('gaussian_map', {}).get('enabled', False))
    if map_enabled and not streaming:
        raise ValueError('Persistent Gaussian mapping requires sequence.streaming: true')
    output = resolve(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new, empty output directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    cfg['motion']['initialization_method'] = 'hybrid'
    cfg['output']['compact'] = True
    cfg['output_dir'] = str(output / 'pairs')
    cfg['run_tag'] = 'multi_frame'
    cfg['sequence']['frames'] = frames
    cfg['sequence']['output_dir'] = str(output)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    snapshot_sources(output)
    tracks, objects = FeatureTrackStore(cfg), ObjectRegistry(cfg)
    if streaming:
        objects.set_event_log(output / 'object_lifecycle.jsonl')
    object_backend = None
    if cfg.get('object_window', {}).get('enabled', False):
        from ownership.gaussian_object_backend import SequenceObjectBackend
        object_backend = SequenceObjectBackend(cfg)
    map_backend = None
    if map_enabled:
        if cfg['gaussian_map'].get('submaps', {}).get('enabled', False):
            from mapping.gaussian_submaps import SubmapGaussianMapBackend
            map_backend = SubmapGaussianMapBackend(cfg, output, frames)
        else:
            from mapping.gaussian_map_backend import GaussianMapBackend
            map_backend = GaussianMapBackend(cfg, output)
    state = dict(frames=frames, sequence=cfg['dataset']['sequence'], dataset=cfg['dataset'],
                 config=cfg, tracking_dir='tracks', pairs=[], objects={}, status='running')
    feature_models = {}
    for first, second in zip(frames[:-1], frames[1:]):
        pair_started = time.perf_counter()
        pair_cfg = deepcopy(cfg)
        pair_cfg['frames'] = dict(first=first, second=second)
        context = SequencePairContext(first, second, tracks, objects, output, object_backend, map_backend,
                                      feature_models=feature_models)
        print(f'[{first}→{second}] Estimate with persistent tracks...', flush=True)
        with (logs / f'{first}_{second}_estimate.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                summary = run(pair_cfg, pair_context=context)
        estimate_seconds = time.perf_counter()-pair_started
        row = context.final_report
        if streaming:
            with (output / 'pairs.jsonl').open('a') as journal:
                journal.write(json.dumps(row) + '\n')
            # Full diagnostics remain in temporal_state.json / pairs.jsonl.
            state['pairs'].append({key: row[key] for key in (
                'first', 'second', 'result_path', 'local_to_object', 'reference',
                'inherited_3d_tracks', 'retained_3d_tracks', 'core_3d_tracks')})
        else:
            state['pairs'].append(row)
        track_summary = tracks.save(output / 'tracks')
        if streaming:
            objects.retain_tracks(tracks.current_track_ids)
            object_backend.retain_tracks(tracks.current_track_ids)
            if map_backend is not None:
                map_backend.retain_tracks(tracks.current_track_ids)
        object_state = objects.to_dict()
        state.update({k: v for k, v in object_state.items() if k != 'pairs'})
        if object_backend is not None:
            state['object_window'] = object_backend.window.to_dict()
        if map_backend is not None:
            state['gaussian_map'] = map_backend.to_dict()
        state['last_completed_frame'] = second
        write_sequence_state(output, state)
        result = output / context.final_report['result_path']
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{first}→{second}] {summary['final_models']} motions; "
              f"{context.final_report['inherited_3d_tracks']} inherited tracks.",
              flush=True)
        if cfg['sequence'].get('stage_overviews', not streaming):
            stage_overviews(result, logs / f'{first}_{second}_visualize.log')
        timing = dict(first=first, second=second, estimate_seconds=estimate_seconds,
                      pair_total_seconds=time.perf_counter()-pair_started,
                      final_sequence_products_included=False,
                      **row.get('timings', {}))
        with (output / 'timings.jsonl').open('a') as journal:
            journal.write(json.dumps(timing)+'\n')
    if streaming:
        track_summary = tracks.save(output / 'tracks', final=True)
        objects.finalize_sequence()
        state.update({k: v for k, v in objects.to_dict().items() if k != 'pairs'})
    if map_backend is not None:
        state['gaussian_map'] = map_backend.finalize(objects)
        if state['gaussian_map'].get('submaps_enabled'):
            from tools.export_gaussian_submaps import export_submaps
            export_submaps(output)
    if cfg.get('object_window', {}).get('prune_empty_objects', False):
        prune_empty_object_records(output, state)
    if streaming:
        (output / 'object_intervals.json').write_text(json.dumps(state['objects'], indent=2))
    state['status'] = 'complete'
    state['track_summary'] = track_summary
    write_sequence_state(output, state)
    if cfg['sequence'].get('visualize_objects', True):
        visualize(output)
    write_report(output, state, track_summary)
    if cfg['sequence'].get('visualize_localization', False):
        from tools.visualization.visualize_localization import visualize_localization
        state['localization'] = visualize_localization(output, state=state)
        write_sequence_state(output, state)
    print(f'Complete: {output / "README.md"}', flush=True)
    return state


def write_sequence_state(output, state):
    """Leave the last complete frame's index readable if a write is interrupted."""
    destination = output / 'sequence_state.json'
    temporary = output / 'sequence_state.tmp.json'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(destination)


def sequence_frames(cfg, frames=None, start=None, end=None, step=None):
    options = cfg.get('sequence', {})
    if frames is not None and any(value is not None for value in (start, end, step)):
        raise ValueError('--frames cannot be combined with --start, --end or --stride')
    if step is not None:
        if step <= 0:
            raise ValueError('--stride must be positive')
        if start is None and options.get('frames') is not None:
            raise ValueError('--stride requires --start/--end when the config has explicit frames')
    if frames is not None:
        result = list(frames)
    else:
        if (start is None) != (end is None):
            raise ValueError('--start and --end must be supplied together')
        if start is None and options.get('frames') is not None:
            result = list(options['frames'])
        else:
            start = options.get('start') if start is None else start
            end = options.get('end') if end is None else end
            stride = int(options.get('step', 2) if step is None else step)
            if start is None or end is None or stride <= 0:
                raise ValueError('Choose start/end frames and a positive step')
            result = list(range(int(start), int(end) + 1, stride))
    if len(result) < 2 or any(frame < 0 for frame in result) or any(
            a >= b for a, b in zip(result[:-1], result[1:])):
        raise ValueError('Choose at least two nonnegative, strictly increasing frame IDs')
    return result


def default_sequence_output(sequence, frames):
    """Name a run by its actual input, including irregular frame selections."""
    intervals = {second-first for first, second in zip(frames[:-1], frames[1:])}
    if len(intervals) == 1:
        sampling = f's{intervals.pop()}'
    else:
        sampling = 'frames' + hashlib.sha256(json.dumps(frames).encode()).hexdigest()[:10]
    return HERE / 'outputs' / f'{sequence}_{frames[0]:04d}_{frames[-1]:04d}_{sampling}'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/kitti360_sequence.yaml',
                        help='Shared method config (default: configs/kitti360_sequence.yaml)')
    parser.add_argument('--sequence', help='Dataset sequence name, e.g. 2013_05_28_drive_0002_sync')
    parser.add_argument('--frames', nargs='+', type=int, help='Explicit ordered frame IDs')
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int, help='Inclusive final frame')
    parser.add_argument('--step', '--stride', dest='step', type=int,
                        help='Dataset frame-number interval; independent of window advance')
    parser.add_argument('--window-frames', type=int, help='Sampled images per ownership window')
    parser.add_argument('--window-step', type=int, help='Sampled images advanced between ownership windows')
    parser.add_argument('--output', type=Path, help='New output directory; defaults to a name based on the selected input')
    parser.add_argument('--dry-run', action='store_true', help='Show selected sequence, frames and output without running estimation')
    parser.add_argument('--streaming', action='store_true', help='Enable bounded long-sequence windows and object lifecycles')
    parser.add_argument('--joint-map', action='store_true', help='Enable persistent joint Gaussian geometry optimization')
    parser.add_argument('--submap-frames', '--submap-step', dest='submap_frames', type=int,
                        help='Enable strict fixed-size submaps; number of sampled images per submap, overlap one')
    parser.add_argument('--max-active-gaussians', type=int,
                        help='Maximum map Gaussians in one geometry solve; full visible snapshots remain independent')
    parser.add_argument('--no-visualize', action='store_true', help='Skip final object rendering')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.sequence is not None:
        cfg.setdefault('dataset', {})['sequence'] = args.sequence
    sequence = cfg.get('dataset', {}).get('sequence')
    if not isinstance(sequence, str) or not sequence or sequence in ('.', '..') or Path(sequence).name != sequence:
        parser.error('--sequence must be a dataset sequence name without directory components')
    if args.step is not None:
        cfg.setdefault('sequence', {})['step'] = args.step
    if args.window_frames is not None:
        cfg.setdefault('object_window', {})['window_frames'] = args.window_frames
    if args.window_step is not None:
        cfg.setdefault('object_window', {})['window_step'] = args.window_step
    if args.streaming:
        cfg.setdefault('sequence', {})['streaming'] = True
    if args.joint_map:
        defaults = load_config(HERE / 'configs/kitti360_joint_gaussian_map.yaml')['gaussian_map']
        cfg['gaussian_map'] = merge_config(defaults, cfg.get('gaussian_map', {}))
        cfg['gaussian_map']['enabled'] = True
        cfg.setdefault('sequence', {})['streaming'] = True
    if args.submap_frames is not None:
        if args.submap_frames < 2:
            parser.error('--submap-frames must be at least 2')
        defaults = load_config(HERE / 'configs/kitti360_joint_gaussian_submaps.yaml')['gaussian_map']
        cfg['gaussian_map'] = merge_config(defaults, cfg.get('gaussian_map', {}))
        cfg['gaussian_map']['enabled'] = True
        cfg['gaussian_map']['submaps'] = dict(cfg['gaussian_map'].get('submaps', {}),
                                            enabled=True, frames=args.submap_frames, overlap_frames=1)
        cfg.setdefault('sequence', {})['streaming'] = True
    if args.max_active_gaussians is not None:
        if args.max_active_gaussians < 1:
            parser.error('--max-active-gaussians must be positive')
        cfg.setdefault('gaussian_map', {})['max_active_gaussians'] = args.max_active_gaussians
    if args.no_visualize:
        cfg.setdefault('sequence', {})['visualize_objects'] = False
    try:
        frames = sequence_frames(cfg, args.frames, args.start, args.end, args.step)
    except ValueError as error:
        parser.error(str(error))
    window = cfg.get('object_window', {})
    if window.get('enabled', False):
        window_frames, window_step = int(window.get('window_frames', 5)), int(window.get('window_step', 1))
        if window_frames < 2 or not 1 <= window_step < window_frames:
            parser.error('Ownership windows require --window-frames >= 2 and 1 <= --window-step < --window-frames')
    input_overridden = any(value is not None for value in (
        args.sequence, args.frames, args.start, args.end, args.step))
    output = args.output
    if output is None and not input_overridden:
        output = cfg.get('sequence', {}).get('output_dir')
    if output is None:
        output = default_sequence_output(sequence, frames)
    print(f'Sequence: {sequence}', flush=True)
    frame_summary = str(frames) if len(frames) <= 20 else f'{frames[:8]} ... {frames[-8:]}'
    print(f'Frames ({len(frames)}): {frame_summary}', flush=True)
    print(f'Output: {resolve(output)}', flush=True)
    if args.dry_run:
        return
    run_sequence(cfg, frames, output)


if __name__ == '__main__':
    main()

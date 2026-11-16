"""Plot saved background odometry in 2D and report SE(3)-aligned ATE RMSE.

No estimator or GPU is invoked. Object0's motion maps static scene points from
the first camera to the second; camera-to-world therefore accumulates inv(H0).
Ground truth is read only for this post-estimation visualization, without scale
fitting. Invalid background intervals split trajectories rather than invent poses.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[2]
ROOT = EXP.parents[1]


def rigid(matrix):
    matrix = np.asarray(matrix)
    return (matrix.shape == (4, 4) and np.isfinite(matrix).all() and
            np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-5) and
            np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-3) and
            abs(np.linalg.det(matrix[:3, :3])-1) < 1e-3)


def background_motion(directory, pair):
    """Prefer the actual O0 backend model, even after posterior column pruning."""
    path = Path(directory) / 'object_state.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            for prefix, ids_key, valid_key in [
                ('evidence_H_object', 'evidence_object_ids', 'evidence_object_model_valid'),
                ('H_object', 'object_ids', 'object_model_valid'),
            ]:
                if prefix not in saved.files or ids_key not in saved.files:
                    continue
                ids, models = saved[ids_key], saved[prefix]
                found = np.flatnonzero(ids == 0)
                if len(found) != 1:
                    return None, 'object0_missing_or_ambiguous'
                c = int(found[0])
                if valid_key in saved.files and not bool(saved[valid_key][c]):
                    return None, 'object0_model_invalid'
                return (models[c].copy(), f'object_state.npz:{prefix}[object0]') if rigid(models[c]) else (
                    None, 'object0_transform_invalid')
    # Compatibility with runs predating the persistent object backend. A
    # local optimizer reference is accepted only if explicitly mapped to O0.
    path = Path(directory) / 'gaussian_state.npz'
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            if {'H', 'local_to_object'} <= set(saved.files):
                found = np.flatnonzero(saved['local_to_object'] == 0)
                reference = int(pair.get('reference', -1))
                c = reference if reference in found else (int(found[0]) if len(found) == 1 else -1)
                if c >= 0 and rigid(saved['H'][c]):
                    return saved['H'][c].copy(), 'gaussian_state.npz:H[explicit_object0_mapping]'
    return None, 'background_transform_unavailable'


def compose_trajectory(pairs, motions):
    """Return contiguous camera-to-segment-start trajectories and interval records."""
    segments, intervals = [], []
    current = None
    for pair, (motion, source) in zip(pairs, motions):
        first, second = int(pair['first']), int(pair['second'])
        if second <= first:
            raise ValueError('Localization intervals must have increasing frame IDs')
        row = dict(first=first, second=second, valid=motion is not None, source=source,
                   translation_error_m=None, rotation_error_deg=None)
        intervals.append(row)
        if motion is None:
            current = None
            continue
        if not rigid(motion):
            raise ValueError('Localization requires rigid background motions')
        if current is None or current['frames'][-1] != first:
            current = dict(segment=len(segments), frames=[first], poses=[np.eye(4)])
            segments.append(current)
        current['frames'].append(second)
        current['poses'].append(current['poses'][-1] @ np.linalg.inv(motion))
        row['segment'] = current['segment']
        row['H_background'] = np.asarray(motion).tolist()
    return segments, intervals


def read_ground_truth(path, requested):
    """cam0_to_world.txt already describes the rectified cam0 in this dataset."""
    result = {}
    if not path.is_file():
        return result
    requested = set(map(int, requested))
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            values = line.split()
            if not values or values[0].startswith('#'):
                continue
            if len(values) != 17:
                raise ValueError(f'Expected frame + 16 pose values: {path}:{number}')
            frame = int(values[0])
            if frame not in requested:
                continue
            pose = np.asarray(values[1:], float).reshape(4, 4)
            if not rigid(pose):
                raise ValueError(f'Invalid ground-truth pose: {path}:{number}')
            # Remove only text-rounding nonorthogonality, without fitting to
            # estimates or changing the camera position or metric scale.
            u, _, vt = np.linalg.svd(pose[:3, :3])
            pose[:3, :3] = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
            result[frame] = pose
    return result


def rotation_error(rotation):
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2, -1, 1))))


def align_camera_centers(estimated, reference):
    """Least-squares rigid alignment, with neither scaling nor reflection."""
    estimated, reference = np.asarray(estimated, float), np.asarray(reference, float)
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 3:
        raise ValueError('ATE alignment needs matching N x 3 camera centers')
    if not len(estimated) or not np.isfinite([estimated, reference]).all():
        raise ValueError('ATE alignment needs finite, nonempty camera centers')
    source_mean, target_mean = estimated.mean(axis=0), reference.mean(axis=0)
    source, target = estimated-source_mean, reference-target_mean
    u, _, vt = np.linalg.svd(source.T @ target)
    transform = np.eye(4)
    transform[:3, :3] = vt.T @ np.diag([1., 1., np.linalg.det(vt.T @ u.T)]) @ u.T
    transform[:3, 3] = target_mean-transform[:3, :3] @ source_mean
    return transform


def compare_ground_truth(segments, intervals, truth):
    rows = []
    for segment in segments:
        frames, poses = segment['frames'], np.asarray(segment['poses'])
        common = [i for i, frame in enumerate(frames) if frame in truth]
        anchor = common[0] if common else 0
        # Normalizing the GT coordinate frame is solely for readable X/Z axes.
        # ATE uses all matching centers to fit one SE(3), not the first pose.
        reference = np.linalg.inv(truth[frames[anchor]]) if common else None
        gt_poses = {i: reference @ truth[frames[i]] for i in common}
        transform = (align_camera_centers(poses[common, :3, 3],
                     np.asarray([gt_poses[i][:3, 3] for i in common])) if common else np.eye(4))
        estimated = transform @ poses
        segment['ate_alignment'] = transform.tolist() if common else None
        segment['matched_ground_truth_frames'] = len(common)
        for index, frame in enumerate(frames):
            gt = gt_poses.get(index)
            rows.append(dict(frame=frame, segment=segment['segment'],
                coordinate_reference_frame=frames[anchor], camera_to_segment_start=poses[index],
                estimated_aligned=estimated[index], ground_truth_aligned=gt,
                position_error_m=(float(np.linalg.norm(estimated[index, :3, 3]-gt[:3, 3]))
                                  if gt is not None else None),
                orientation_error_deg=(rotation_error(gt[:3, :3].T @ estimated[index, :3, :3])
                                       if gt is not None else None)))
    for pair in intervals:
        if not pair['valid'] or pair['first'] not in truth or pair['second'] not in truth:
            continue
        expected = np.linalg.inv(truth[pair['second']]) @ truth[pair['first']]
        error = np.linalg.inv(expected) @ np.asarray(pair['H_background'])
        pair.update(translation_error_m=float(np.linalg.norm(error[:3, 3])),
                    rotation_error_deg=rotation_error(error[:3, :3]))
    return rows


def _rmse(values):
    values = [v for v in values if v is not None]
    return float(np.sqrt(np.mean(np.square(values)))) if values else None


def _plot(output, rows, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, top = plt.subplots(figsize=(7.5, 6))
    for sid in sorted({r['segment'] for r in rows}):
        group = [r for r in rows if r['segment'] == sid]
        positions = np.asarray([r['estimated_aligned'][:3, 3] for r in group])
        label = f'Estimated segment {sid}' if metrics['segments'] > 1 else 'Estimated'
        top.plot(positions[:, 0], positions[:, 2], '.-', color='tab:orange', label=label)
        gt = np.asarray([r['ground_truth_aligned'][:3, 3] for r in group
                         if r['ground_truth_aligned'] is not None]).reshape(-1, 3)
        if len(gt):
            top.plot(gt[:, 0], gt[:, 2], '--', color='black', label='Ground truth' if sid == 0 else None)
        top.scatter(positions[0, 0], positions[0, 2], color='green', marker='o', zorder=4)
        top.scatter(positions[-1, 0], positions[-1, 2], color='red', marker='x', zorder=4)
    if rows:
        top.legend(fontsize=9, loc='lower right')
    else:
        top.text(.5, .5, 'No valid background transforms', ha='center', transform=top.transAxes)
    top.set(xlabel='X / right (m)', ylabel='Z / forward (m)',
            title=f"{metrics['sequence']}\n2D trajectory | {metrics['estimated_frames']} frames")
    top.axis('equal')
    top.grid(alpha=.3)
    ate = metrics['ate_rmse_m']
    if ate is not None:
        annotation = f'ATE RMSE = {ate:.4f} m'
    elif metrics['ate_rmse_on_matched_segments_m'] is not None:
        annotation = f"Partial ATE RMSE = {metrics['ate_rmse_on_matched_segments_m']:.4f} m"
    else:
        annotation = 'ATE RMSE = unavailable (no matching GT)'
    annotation += ('\nSE(3) alignment, scale = 1; 3D translation'
                   f"\nGT coverage: {metrics['matched_ground_truth_frames']}/{metrics['intended_frames']} frames")
    top.text(.03, .97, annotation, transform=top.transAxes, va='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='.8', alpha=.9))
    fig.tight_layout()
    fig.savefig(output/'trajectory.png', dpi=160)
    plt.close(fig)
    # Remove the error chart made by the previous version of this same tool.
    (output/'errors.png').unlink(missing_ok=True)


def visualize_localization(result, output=None, state=None):
    result = Path(result).expanduser().resolve()
    state = json.loads((result/'sequence_state.json').read_text()) if state is None else state
    output = result/'localization' if output is None else Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    motions = []
    for pair in state['pairs']:
        directory = Path(pair['result_path'])
        motions.append(background_motion(directory if directory.is_absolute() else result/directory, pair))
    segments, intervals = compose_trajectory(state['pairs'], motions)
    dataset = state.get('config', {}).get('dataset', state.get('dataset', {}))
    sequence = dataset.get('sequence', state.get('sequence', 'unknown'))
    data_root = Path(dataset.get('data_root', 'data/kitti360')).expanduser()
    data_root = data_root if data_root.is_absolute() else ROOT/data_root
    gt_path = data_root/'data_poses'/sequence/'cam0_to_world.txt'
    frames = [frame for segment in segments for frame in segment['frames']]
    truth = read_ground_truth(gt_path, frames)
    rows = compare_ground_truth(segments, intervals, truth)
    ate_rmse = _rmse([r['position_error_m'] for r in rows])
    matched = sum(r['ground_truth_aligned'] is not None for r in rows)
    full = (state.get('status') == 'complete' and len(intervals) == len(state['frames'])-1 and
            all(p['valid'] for p in intervals) and len(segments) == 1 and
            frames == list(map(int, state['frames'])))
    metrics = dict(sequence=sequence, run_status=state.get('status'),
        intended_frames=len(state['frames']), estimated_frames=len(rows),
        completed_intervals=len(intervals), valid_intervals=sum(p['valid'] for p in intervals),
        segments=len(segments), full_trajectory_available=full,
        ground_truth_file=str(gt_path), ground_truth_file_exists=gt_path.is_file(),
        ground_truth_camera='rectified cam0; cam0_to_world.txt used directly',
        matched_ground_truth_frames=matched,
        alignment='SE(3) least-squares camera-center alignment per contiguous segment; scale fixed at 1',
        ate_translation_dimensions=3, visualization_axes='XZ in first matched GT camera coordinates',
        ate_rmse_on_matched_segments_m=ate_rmse,
        ate_rmse_m=ate_rmse if full and matched == len(state['frames']) else None,
        segment_ate=[dict(segment=s['segment'],
                          matched_frames=s['matched_ground_truth_frames'],
                          ate_rmse_m=_rmse([r['position_error_m'] for r in rows if r['segment'] == s['segment']]),
                          alignment_transform=s['ate_alignment']) for s in segments],
        relative_translation_rmse_m=_rmse([p['translation_error_m'] for p in intervals]),
        relative_rotation_rmse_deg=_rmse([p['rotation_error_deg'] for p in intervals]),
        evaluated_relative_intervals=sum(p['translation_error_m'] is not None for p in intervals),
        invalid_intervals=[{k: p[k] for k in ('first', 'second', 'source')} for p in intervals if not p['valid']],
        scope='Saved O0 odometry only; no loop closure/global pose refinement. '
              'Sampled-interval errors are not official KITTI segment drift scores.',
        ground_truth_used_for_estimation=False)
    with (output/'errors.csv').open('w', newline='') as stream:
        fields = ['frame', 'segment', 'coordinate_reference_frame', 'position_error_m', 'orientation_error_deg']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with (output/'relative_errors.csv').open('w', newline='') as stream:
        fields = ['first', 'second', 'valid', 'source', 'translation_error_m', 'rotation_error_deg']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(intervals)
    np.savez_compressed(output/'trajectory.npz', frames=np.asarray(frames, np.int64),
        segment_ids=np.asarray([r['segment'] for r in rows], np.int64),
        camera_to_segment_start=np.asarray([r['camera_to_segment_start'] for r in rows]).reshape(-1, 4, 4),
        estimated_aligned=np.asarray([r['estimated_aligned'] for r in rows]).reshape(-1, 4, 4),
        ground_truth_aligned=np.asarray([np.full((4, 4), np.nan) if r['ground_truth_aligned'] is None
                                       else r['ground_truth_aligned'] for r in rows]).reshape(-1, 4, 4))
    _plot(output, rows, metrics)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    # Both automatically completed runs and explicit postprocessing get links.
    report = result/'README.md'
    if report.is_file() and output == result/'localization':
        start, end = '<!-- localization:start -->', '<!-- localization:end -->'
        text = report.read_text()
        score = f'{metrics["ate_rmse_m"]:.4f} m' if metrics['ate_rmse_m'] is not None else '无完整序列评分'
        links = (f'{start}\n\n### 定位效果\n\n[二维轨迹及ATE](localization/trajectory.png) · '
                 '[指标及有效区间](localization/metrics.json)\n\n'
                 f'ATE RMSE：{score}。全轨迹SE(3)对齐，不拟合尺度，ATE使用三维平移误差；'
                 '无效背景运动处断开轨迹。\n\n'+end)
        if start in text and end in text:
            a, b = text.index(start), text.index(end)+len(end)
            text = text[:a]+links+text[b:]
        else:
            text = text.rstrip()+'\n\n'+links+'\n'
        report.write_text(text)
    return metrics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--result', required=True, type=Path, help='Existing sequence output directory')
    parser.add_argument('--output', type=Path, help='Default: <result>/localization')
    args = parser.parse_args()
    metrics = visualize_localization(args.result, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

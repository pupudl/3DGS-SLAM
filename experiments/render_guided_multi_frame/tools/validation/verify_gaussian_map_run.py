#!/usr/bin/env python3
"""Sequential, read-only consistency audit for persistent Gaussian map runs.

This checks saved parameters, ownership, lifetimes, fixed poses and resource
budgets. It does not measure reconstruction/segmentation accuracy or establish
equivalence between the CPU fixture renderer and the native CUDA renderer.
Only main() writes an atomic JSON report; verify() never changes the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import argparse
import json
import os
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parents[2]

WIDTHS = dict(xyz=3, colors=3, log_scales=3, rotations=4, opacity_logits=1)
STATUSES = {'active', 'missing', 'archived', 'retired'}


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, (float, np.floating)) or bool(np.isfinite(value))


def _ids(value, count=None):
    value = np.asarray(value)
    return bool(value.ndim == 1 and np.issubdtype(value.dtype, np.integer) and
                (count is None or len(value) == count) and np.all(value >= 0) and
                len(np.unique(value)) == len(value))


def _parameters(archive, count):
    for name, width in WIDTHS.items():
        value = archive[name]
        if value.shape != (count, width) or not np.isfinite(value).all():
            return False
    if np.any((archive['colors'] < 0) | (archive['colors'] > 1)):
        return False
    norms = np.linalg.norm(archive['rotations'].astype(np.float64), axis=1)
    if not np.allclose(norms, 1., atol=2e-4):
        return False
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(archive['log_scales'])
        opacity = np.exp(-np.logaddexp(0., -archive['opacity_logits']))
    return bool(np.isfinite(scales).all() and np.all(scales > 0) and
                np.isfinite(opacity).all() and np.all((opacity >= 0) & (opacity <= 1)))


def _rigid(value):
    value = np.asarray(value, np.float64)
    return bool(value.shape == (4, 4) and np.isfinite(value).all() and
                np.allclose(value[3], [0, 0, 0, 1], atol=1e-5) and
                np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3) and
                np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-3))


def _interval(record):
    first, last = record.get('first_observed_frame'), record.get('last_observed_frame')
    if first is None or last is None:
        return None
    return (int(first), int(last)) if first <= last else None


def _rewind_owners(owners, events, position, revision):
    """Consume each journal event once while snapshots descend by revision."""
    while position < len(events) and events[position]['revision'] > revision:
        event = events[position]
        ids = np.asarray(event['map_ids'], np.int64)
        if event['kind'] in ('reassign', 'remove'):
            owners[ids] = event['old_object_ids']
        else:
            owners[ids] = -1  # child did not exist before its append revision
        position += 1
    return position


def _optimization_report(value):
    return bool(isinstance(value, dict) and _finite_tree(value) and
                value.get('optimized_parameters') == list(WIDTHS) and
                value.get('iterations_completed', -1) >= 0 and
                value.get('track_count', -1) >= 0 and value.get('track_3d_count', -1) >= 0 and
                value.get('track_3d_count', 0) <= value.get('track_count', 0) and
                value.get('ownership_optimized') is False and
                value.get('camera_poses_optimized') is False and value.get('object_poses_optimized') is False)


def verify(result):
    """Return an audit report while loading at most one geometry block/view.

    One int64 map-ID-to-owner array persists on CPU. Image pixels and Gaussian
    parameter histories are not cached. Errors are recorded as failed checks.
    """
    from tools.visualization.visualize_gaussian_map import prepare_snapshot

    result = Path(result).resolve()
    if (result / 'gaussian_map/submaps/manifest.json').is_file():
        from tools.validation.verify_gaussian_submaps import verify_submaps
        return verify_submaps(result)
    errors, checks = [], {}

    def check(name, value, detail=None):
        checks[name] = checks.get(name, True) and bool(value)
        if not value and detail:
            errors.append(dict(check=name, detail=str(detail)))

    try:
        state = json.loads((result / 'sequence_state.json').read_text())
        cfg = state.get('config', {})
        map_cfg = cfg.get('gaussian_map', {})
        frames = list(map(int, state.get('frames', [])))
        pairs = state.get('pairs', [])
        objects = state.get('objects', {})
        check('complete', state.get('status') == 'complete')
        check('mapping_enabled', map_cfg.get('enabled') is True)
        expected_pairs = list(zip(frames[:-1], frames[1:]))
        check('ordered_sequence', len(frames) >= 2 and frames == sorted(set(frames)) and
              [(int(p['first']), int(p['second'])) for p in pairs] == expected_pairs)
        summary = state.get('gaussian_map', {})
        check('mapping_finalized', summary.get('enabled') is True and summary.get('finalized') is True)
        manifest = json.loads((result / 'gaussian_map/map/manifest.json').read_text())
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, checks={**checks, 'required_run_artifacts': False},
                    errors=[dict(check='required_run_artifacts', detail=str(error))],
                    scope='saved-state consistency; accuracy and CUDA equivalence not evaluated')

    metadata = manifest.get('objects', {})
    try:
        if not isinstance(metadata, dict) or any(
                not isinstance(row, dict) or not isinstance(row.get('gaussian_count'), int) or
                isinstance(row['gaussian_count'], bool) or row['gaussian_count'] < 0
                for row in metadata.values()):
            raise ValueError('Object metadata requires nonnegative integer Gaussian counts')
        live_count = sum(row['gaussian_count'] for row in metadata.values())
        pruned = np.asarray(manifest.get('pruned_ids', []))
        if pruned.ndim != 1 or (len(pruned) and not _ids(pruned)):
            raise ValueError('pruned_ids must contain unique nonnegative integer IDs')
        pruned = pruned.astype(np.int64, copy=True)
        allocation_count = live_count + len(pruned)
    except (ValueError, TypeError, KeyError) as error:
        return dict(passed=False, checks={**checks, 'checkpoint_manifest': False},
                    errors=[dict(check='checkpoint_manifest', detail=str(error))],
                    scope='saved-state consistency; accuracy and CUDA equivalence not evaluated')
    check('checkpoint_format', manifest.get('format_version') in (1, 2))
    check('checkpoint_contiguous_ids', isinstance(manifest.get('next_id'), int) and
          not isinstance(manifest['next_id'], bool) and manifest['next_id'] == allocation_count and
          manifest.get('live_count', live_count) == live_count and
          manifest.get('allocated_count', allocation_count) == allocation_count and
          bool(np.all(pruned < allocation_count)))
    # Allocate from actual block/pruned counts, rather than an unchecked next_id.
    owners = np.full(max(0, allocation_count), -1, np.int64)
    adam_max_steps = {name: 0 for name in WIDTHS}
    object_records = []
    for key, row in metadata.items():
        try:
            oid, count = int(key), int(row['gaussian_count'])
            filename = row['filename']
            if oid < 0 or count < 0 or Path(filename).name != filename:
                raise ValueError('Invalid object block metadata or filename')
            life = objects.get(str(oid), {})
            interval = _interval(life)
            check('checkpoint_lifetimes', interval is not None and row.get('status') in STATUSES and
                  row.get('status') == life.get('status'), f'Object {oid}: lifetime/status disagreement')
            check('checkpoint_parameters', True)
            check('checkpoint_adam', True)
            with np.load(result / 'gaussian_map/map' / filename, allow_pickle=False) as archive:
                ids = archive['map_ids']
                valid_ids = _ids(ids, count) and bool(np.all(ids < len(owners)))
                if valid_ids:
                    valid_ids = bool(np.all(owners[ids] == -1))
                check('checkpoint_contiguous_ids', valid_ids, f'Object {oid}: duplicate/out-of-range map IDs')
                if valid_ids:
                    owners[ids] = oid
                check('checkpoint_parameters', _parameters(archive, count), f'Object {oid}: invalid learned parameters')
                for name in ('initial_xyz', 'initial_log_scales', 'association_colors'):
                    if name in archive.files:
                        value = archive[name]
                        check('checkpoint_parameters', value.shape == (count, 3) and np.isfinite(value).all(),
                              f'Object {oid}: invalid {name}')
                for name in ('observation_count', 'last_seen_frame'):
                    value = archive[name]
                    valid = value.shape == (count,) and np.issubdtype(value.dtype, np.integer)
                    if name == 'observation_count':
                        valid = valid and bool(np.all(value >= 0))
                    elif interval is not None:
                        # A correction may migrate geometry measured while the
                        # surface still belonged to an earlier background ID.
                        migrated = set()
                        for event in manifest.get('ownership_events', []):
                            when = event.get('frame_id')
                            if event.get('new_object_id') == oid and isinstance(when, int) and interval[0] <= when <= interval[1]:
                                migrated.update(event.get('map_ids', []))
                        prebirth = np.isin(ids, list(migrated)) & (value >= min(frames))
                        valid = valid and bool(np.all(((value >= interval[0]) | prebirth) & (value <= interval[1])))
                    check('checkpoint_lifetimes', valid, f'Object {oid}: invalid {name}')
                for name, width in WIDTHS.items():
                    for moment, size in (('exp_avg', width), ('exp_avg_sq', width), ('row_steps', 1)):
                        value = archive[f'adam_{name}_{moment}']
                        valid = value.shape == (count, size) and bool(np.isfinite(value).all())
                        if moment in ('exp_avg_sq', 'row_steps'):
                            valid = valid and bool(np.all(value >= 0))
                        if moment == 'row_steps':
                            valid = valid and np.issubdtype(value.dtype, np.integer)
                            if valid and value.size:
                                adam_max_steps[name] = max(adam_max_steps[name], int(value.max()))
                        check('checkpoint_adam', valid, f'Object {oid}: invalid Adam {name}.{moment}')
            object_records.append(dict(object_id=oid, gaussians=count, status=row.get('status')))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            check('checkpoint_parameters', False, f'Object {key}: {error}')
    check('checkpoint_contiguous_ids', np.array_equal(np.flatnonzero(owners < 0), np.sort(pruned)),
          'Allocated map IDs must partition into live rows and explicit pruned IDs')
    events, checkpoint_revision = [], manifest.get('revision', 0)
    try:
        from mapping.persistent_gaussian_map import PersistentGaussianMap
        ownership_events, topology_events, checkpoint_revision = PersistentGaussianMap.validate_journal(manifest, owners)
        events = sorted(ownership_events + topology_events, key=lambda row: row['revision'], reverse=True)
        check('ownership_and_topology_journal', True)
    except (ValueError, TypeError, KeyError, IndexError) as error:
        check('ownership_and_topology_journal', False, error)
        checkpoint_revision = 0
    check('checkpoint_summary_counts', summary.get('total_map_gaussians') == live_count and
          summary.get('allocated_map_gaussians', allocation_count) == allocation_count and
          summary.get('map', {}).get('gaussian_count') == live_count and
          summary.get('map', {}).get('allocated_count', allocation_count) == allocation_count and
          summary.get('map', {}).get('object_count') == len(metadata))

    # Trajectory metadata is small (poses only); geometry remains sequential.
    trajectory, frame_poses = [], {}
    try:
        with (result / 'gaussian_map/trajectory.jsonl').open() as stream:
            for line in stream:
                if line.strip():
                    trajectory.append(json.loads(line))
        check('trajectory_pair_order', [(r['first'], r['second']) for r in trajectory] == expected_pairs)
        for index, row in enumerate(trajectory):
            first, second = int(row['first']), int(row['second'])
            a, b = np.asarray(row['camera_to_world_first']), np.asarray(row['camera_to_world_second'])
            pa, pb = row['object_to_world_first'], row['object_to_world_second']
            check('fixed_trajectory', _rigid(a) and _rigid(b) and
                  all(_rigid(pose) for pose in (*pa.values(), *pb.values())))
            if index == 0:
                check('fixed_trajectory', np.allclose(a, np.eye(4), atol=1e-5))
            elif index <= len(trajectory)-1:
                check('fixed_trajectory', np.allclose(a, trajectory[index-1]['camera_to_world_second'], atol=1e-5))
            frame_poses[first] = (a, pa)
            frame_poses.setdefault(second, (b, pb))
    except (OSError, ValueError, KeyError, TypeError) as error:
        check('fixed_trajectory', False, error)

    observed_frames = {}
    previous_total, previous_allocated, report_records = 0, 0, []
    nonzero_groups = set()
    window_size = int(map_cfg.get('window_frames', cfg.get('object_window', {}).get('window_frames', 5)))
    active_budget = int(map_cfg.get('max_active_gaussians', 100000))
    sample_budget = int(map_cfg.get('max_samples_per_frame', 30000))
    insertion_budget = int(map_cfg.get('max_new_gaussians_per_frame', 20000))
    for index, pair in enumerate(pairs):
        try:
            directory = Path(pair['result_path'])
            if not directory.is_absolute():
                directory = result / directory
            temporal = json.loads((directory / 'temporal_state.json').read_text())
            report = temporal['gaussian_map']
            optimization = report['optimization']
            refinement = report.get('refinement', {})
            post_optimization = refinement.get('post_optimization')
            check('fixed_ownership_and_poses', report.get('fixed_object_ownership') is True and
                  report.get('camera_poses_optimized') is False and report.get('object_poses_optimized') is False and
                  optimization.get('ownership_optimized') is False and
                  optimization.get('camera_poses_optimized') is False and optimization.get('object_poses_optimized') is False)
            window = report['window_frames']
            expected_window = frames[max(0, index + 2-window_size):index+2]
            check('bounded_active_window', window == expected_window and len(window) <= window_size)
            total, inserted = report['total_map_gaussians'], report['inserted_gaussians']
            refined_inserted, removed = refinement.get('inserted', 0), refinement.get('removed', 0)
            counts = (total, inserted, refined_inserted, removed)
            valid_counts = all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts)
            valid_counts = valid_counts and total == previous_total + inserted + refined_inserted - removed and inserted <= insertion_budget
            allocated = report.get('allocated_map_gaussians', previous_allocated + inserted + refined_inserted)
            valid_counts = valid_counts and allocated == previous_allocated + inserted + refined_inserted and total <= allocated
            check('append_only_map_counts', valid_counts)
            check('map_topology_counts', valid_counts)
            previous_total = total
            previous_allocated = allocated
            check('bounded_active_gaussians', 0 <= report['active_map_gaussians'] <= active_budget and
                  report['active_map_gaussians'] <= total and report.get('deferred_visible_gaussians', -1) >= 0 and
                  0 <= report['sampled_observations'] <= sample_budget)
            check('optimization_reports', _optimization_report(optimization))
            if post_optimization is not None:
                check('optimization_reports', _optimization_report(post_optimization))
            for stage in [optimization] + ([post_optimization] if post_optimization is not None else []):
                for group, gradient in stage.get('gradients', {}).items():
                    if gradient.get('nonzero_steps', 0) > 0:
                        nonzero_groups.add(group)
            report_revision = report.get('map_revision', 0)
            check('snapshot_revisions', isinstance(report_revision, int) and not isinstance(report_revision, bool) and
                  0 <= report_revision <= checkpoint_revision)
            life = temporal['object_window']['lifecycle']
            for endpoint in ('source', 'target'):
                observation = life[endpoint]
                frame = int(observation['frame'])
                for oid in observation['observed_object_ids']:
                    observed_frames.setdefault(int(oid), set()).add(frame)
            if index < len(trajectory):
                pose_row = trajectory[index]
                with np.load(directory / 'object_state.npz', allow_pickle=False) as saved:
                    ids = saved['object_ids']
                    models, valid = saved['H_object'], saved['object_model_valid']
                    bg = np.flatnonzero(ids == 0)
                    a, b = np.asarray(pose_row['camera_to_world_first']), np.asarray(pose_row['camera_to_world_second'])
                    check('fixed_trajectory', len(bg) == 1 and bool(valid[bg[0]]) and
                          np.allclose(b, a @ np.linalg.inv(models[bg[0]]), atol=2e-4))
                    source, target = pose_row['object_to_world_first'], pose_row['object_to_world_second']
                    for key, target_pose in target.items():
                        oid = int(key)
                        if oid == 0:
                            check('fixed_trajectory', np.allclose(target_pose, np.eye(4), atol=1e-5))
                            continue
                        column = np.flatnonzero(ids == oid)
                        check('fixed_trajectory', key in source and len(column) == 1 and bool(valid[column[0]]) and
                              np.allclose(target_pose, b @ models[column[0]] @ np.linalg.inv(a) @
                                          np.asarray(source[key]), atol=2e-4))
            report_records.append(dict(first=int(pair['first']), second=int(pair['second']),
                                       active_gaussians=report['active_map_gaussians'],
                                       total_gaussians=total, allocated_gaussians=allocated,
                                       iterations=optimization['iterations_completed'] + (post_optimization or {}).get('iterations_completed', 0),
                                       main_iterations=optimization['iterations_completed'],
                                       post_iterations=(post_optimization or {}).get('iterations_completed', 0),
                                       tracks=optimization.get('track_count', 0)))
        except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            check('optimization_reports', False, f'Pair {pair.get("first")}->{pair.get("second")}: {error}')
    check('append_only_map_counts', previous_total == live_count and previous_allocated == allocation_count)
    check('map_topology_counts', previous_total == live_count and previous_allocated == allocation_count)
    for oid, row in objects.items():
        if not row.get('confirmed', True):
            continue
        interval = _interval(row)
        observed = observed_frames.get(int(oid), set())
        check('observed_lifecycle_intervals', bool(observed) and interval == (min(observed), max(observed)),
              f'Object {oid}: final interval disagrees with observation history')
        if row.get('status') in ('retired', 'archived') and row.get('end_frame') is not None:
            check('observed_lifecycle_intervals', row['end_frame'] == row.get('last_observed_frame'))

    snapshots = sorted((result / 'gaussian_map/frames').glob('frame_*.npz'))
    check('snapshots_complete', [p.name for p in snapshots] == [f'frame_{frame:010d}.npz' for frame in frames])
    snapshot_records = []
    snapshot_order = []
    for path in snapshots:
        try:
            # ZIP member access loads this scalar only, leaving geometry on
            # disk until its turn. Descending revisions permit one in-place
            # owner table and one reverse pass over the whole event journal.
            with np.load(path, allow_pickle=False) as archive:
                raw = np.asarray(archive['map_revision']) if 'map_revision' in archive.files else np.asarray(0)
                valid = (raw.shape == () and np.issubdtype(raw.dtype, np.integer) and
                         0 <= int(raw) <= checkpoint_revision and
                         ('map_revision' in archive.files or not events))
            snapshot_order.append((int(raw) if valid else checkpoint_revision + 1, path))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            check('snapshot_revisions', False, f'{path.name}: {error}')
            snapshot_order.append((checkpoint_revision + 1, path))
    event_position = 0
    for _, path in sorted(snapshot_order, key=lambda item: (item[0], item[1].name), reverse=True):
        try:
            with np.load(path, allow_pickle=False) as archive:
                snapshot = {key: archive[key] for key in archive.files}
            batch, view, geometry = prepare_snapshot(snapshot)
            frame = int(np.asarray(snapshot['frame']).item())
            ids, labels = snapshot['map_ids'], snapshot['object_labels']
            check('snapshots_parameters', _parameters(snapshot, len(ids)) and _ids(snapshot['object_ids']) and
                  np.issubdtype(np.asarray(labels).dtype, np.integer) and
                  frame in frames and path.name == f'frame_{frame:010d}.npz' and len(view['rows']) == len(ids))
            routed = _ids(ids) and bool(np.all(ids < len(owners)))
            raw_revision = np.asarray(snapshot.get('map_revision', 0))
            valid_revision = (raw_revision.shape == () and np.issubdtype(raw_revision.dtype, np.integer) and
                              0 <= int(raw_revision) <= checkpoint_revision and
                              ('map_revision' in snapshot or not events))
            check('snapshot_revisions', valid_revision, path.name)
            if routed and valid_revision:
                event_position = _rewind_owners(owners, events, event_position, int(raw_revision))
                historical = owners[ids]
            else:
                historical = None
            check('immutable_map_ownership', historical is not None and bool(np.array_equal(historical, labels)), path.name)
            # Rendering snapshots may contain every visible map row while
            # training uses a bounded selection. These are separate budgets.
            scope = str(np.asarray(snapshot.get('snapshot_scope', '')).item())
            check('bounded_active_gaussians', scope == 'full_visible_map' or len(ids) <= active_budget)
            for oid in snapshot['object_ids']:
                oid = int(oid)
                interval = _interval(objects.get(str(oid), {}))
                check('no_ghost_objects', interval is not None and interval[0] <= frame <= interval[1] and
                      frame in observed_frames.get(oid, set()), f'{path.name}: Object {oid}')
            if frame not in frame_poses:
                check('snapshot_fixed_poses', False, f'No recorded pose for frame {frame}')
            else:
                camera, poses = frame_poses[frame]
                for oid, transform in zip(snapshot['object_ids'], snapshot['object_to_camera']):
                    key = str(int(oid))
                    check('snapshot_fixed_poses', key in poses and
                          np.allclose(transform, np.linalg.inv(camera) @ poses[key], atol=2e-4),
                          f'{path.name}: Object {oid} pose disagrees with trajectory')
            snapshot_records.append(dict(frame=frame, gaussians=len(ids),
                                         map_revision=int(raw_revision) if valid_revision else None,
                                         object_ids=list(map(int, snapshot['object_ids']))))
            del snapshot, batch, view, geometry
        except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            check('snapshots_parameters', False, f'{path.name}: {error}')
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(passed=all(checks.values()), checks=checks, errors=errors,
                frames=len(frames), pairs=len(pairs), gaussians=live_count,
                total_map_gaussians=live_count, allocated_map_gaussians=allocation_count,
                pruned_map_gaussians=len(pruned), map_revision=checkpoint_revision,
                objects=object_records, snapshots=sorted(snapshot_records, key=lambda row: row['frame']), optimization=report_records,
                nonzero_gradient_groups=sorted(nonzero_groups), adam_max_steps=adam_max_steps,
                geometry_memory='one object checkpoint or one frame snapshot; compact CPU ownership routing and journal metadata',
                scope='saved-state consistency; reconstruction/segmentation accuracy and CUDA equivalence not evaluated',
                ground_truth_used=False, dataset=cfg.get('dataset'),
                replay_declared=bool(state.get('replay', False) or (result / 'replay_provenance.json').is_file()))


def _atomic_report(path, report):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.gaussian_map_verification.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.result)
    path = args.result / 'gaussian_map_verification.json'
    _atomic_report(path, report)
    print(json.dumps(dict(passed=report['passed'], checks=report['checks'], report=str(path)), indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

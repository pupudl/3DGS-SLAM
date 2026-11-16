"""Read-only consistency audit for disk-backed, fixed image-count submaps."""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import json

import numpy as np


def verify_submaps(result):
    """Audit one checkpoint block/frame at a time, never concatenate maps.

    Object lifecycles belong to the global registry. A chunk archived while a
    vehicle is active is valid even when that global object retires much later.
    """
    from mapping.gaussian_submaps import partition_submaps
    from mapping.persistent_gaussian_map import PersistentGaussianMap
    from tools.validation.verify_gaussian_map_run import (WIDTHS, STATUSES, _ids, _interval,
                                         _optimization_report, _parameters,
                                         _rewind_owners, _rigid)
    from tools.visualization.visualize_gaussian_map import discover_map_snapshots, prepare_snapshot

    root = Path(result).resolve()
    checks, errors = {}, []

    def check(name, value, detail=None):
        checks[name] = checks.get(name, True) and bool(value)
        if not value:
            errors.append(dict(check=name, detail=str(detail or 'Condition failed')))

    def path(relative):
        value = (root / relative).resolve()
        if not value.is_relative_to(root):
            raise ValueError('Submap artifact path leaves the result directory')
        return value

    try:
        state = json.loads((root / 'sequence_state.json').read_text())
        manifest = json.loads((root / 'gaussian_map/submaps/manifest.json').read_text())
        frames = list(map(int, state['frames']))
        pairs = state['pairs']
        cfg = state['config']
        map_cfg = cfg['gaussian_map']
        objects = state.get('objects', {})
        entries = manifest['submaps']
        ranges = partition_submaps(frames, manifest['frames_per_submap'], manifest['overlap_frames'])
        expected_pairs = list(zip(frames[:-1], frames[1:]))
        check('complete', state.get('status') == 'complete' and manifest.get('finalized') is True)
        check('mapping_enabled', map_cfg.get('enabled') is True)
        check('ordered_sequence', [(int(p['first']), int(p['second'])) for p in pairs] == expected_pairs)
        check('strict_submap_partition', manifest.get('format') == 'fixed_gaussian_submaps' and
              manifest.get('format_version') == 1 and manifest['requested_frame_ids'] == frames and
              len(entries) == len(ranges) == manifest['planned_submaps'] and
              manifest['processed_pairs'] == len(expected_pairs))
        check('global_coordinates', manifest.get('coordinate_system') == 'background_world;dynamic_object_canonical' and
              manifest.get('global_object_ids') is True and manifest.get('object_lifecycle_resets_at_boundary') is False and
              np.allclose(manifest['world_from_submap'], np.eye(4)))
        check('archived_geometry_released', manifest.get('resident_submaps') == 0)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(passed=False, checks={**checks, 'required_run_artifacts': False},
                    errors=errors+[dict(check='required_run_artifacts', detail=str(error))])

    poses, trajectory, pair_poses = {}, [], {}
    observed = {}
    pair_reports = {}
    active_budget = int(map_cfg.get('max_active_gaussians', 100000))
    window_size = int(map_cfg.get('window_frames', cfg.get('object_window', {}).get('window_frames', 5)))
    try:
        with (root / 'gaussian_map/trajectory.jsonl').open() as stream:
            trajectory = [json.loads(line) for line in stream if line.strip()]
        check('trajectory_pair_order', [(r['first'], r['second']) for r in trajectory] == expected_pairs)
        previous = None
        for row in trajectory:
            first, second = int(row['first']), int(row['second'])
            pair_poses[(first, second)] = row
            a, b = np.asarray(row['camera_to_world_first']), np.asarray(row['camera_to_world_second'])
            pa, pb = row['object_to_world_first'], row['object_to_world_second']
            check('fixed_trajectory', _rigid(a) and _rigid(b) and all(_rigid(T) for T in (*pa.values(), *pb.values())))
            if previous is None:
                check('fixed_trajectory', np.allclose(a, np.eye(4), atol=1e-5))
            else:
                check('camera_boundary_continuity', np.allclose(a, previous['camera_to_world_second'], atol=2e-4))
                if previous.get('submap_id') != row.get('submap_id'):
                    old = previous['object_to_world_second']
                    check('object_boundary_continuity', all(np.allclose(pa[key], old[key], atol=2e-4)
                          for key in pa.keys() & old.keys()))
            previous = row
            poses[(int(row['submap_id']), first)] = (a, pa)
            poses.setdefault((int(row['submap_id']), second), (b, pb))
        for index, pair in enumerate(pairs):
            directory = path(pair['result_path'])
            temporal = json.loads((directory / 'temporal_state.json').read_text())
            report = temporal['gaussian_map']
            pair_reports[(int(pair['first']), int(pair['second']))] = report
            check('bounded_active_gaussians', 0 <= report['active_map_gaussians'] <= active_budget and
                  report['active_map_gaussians'] <= report['total_map_gaussians'] and
                  report.get('deferred_visible_gaussians', -1) >= 0)
            sid = int(report['submap_id'])
            local_frames = ranges[sid]['frame_ids']
            stop = local_frames.index(int(pair['second']))+1
            check('bounded_active_window', report['window_frames'] == local_frames[max(0, stop-window_size):stop])
            check('optimization_reports', _optimization_report(report['optimization']))
            post = report.get('refinement', {}).get('post_optimization')
            if post is not None:
                check('optimization_reports', _optimization_report(post))
            check('fixed_ownership_and_poses', report.get('fixed_object_ownership') is True and
                  report.get('camera_poses_optimized') is False and report.get('object_poses_optimized') is False)
            if index < len(trajectory):
                pose_row = trajectory[index]
                a = np.asarray(pose_row['camera_to_world_first'])
                b = np.asarray(pose_row['camera_to_world_second'])
                with np.load(directory / 'object_state.npz', allow_pickle=False) as saved:
                    ids, models, valid = saved['object_ids'], saved['H_object'], saved['object_model_valid']
                    background = np.flatnonzero(ids == 0)
                    check('pair_motion_trajectory', len(background) == 1 and bool(valid[background[0]]) and
                          np.allclose(b, a @ np.linalg.inv(models[background[0]]), atol=2e-4))
                    source, target = pose_row['object_to_world_first'], pose_row['object_to_world_second']
                    for key, target_pose in target.items():
                        if int(key) == 0:
                            check('pair_motion_trajectory', np.allclose(target_pose, np.eye(4), atol=1e-5))
                            continue
                        column = np.flatnonzero(ids == int(key))
                        check('pair_motion_trajectory', key in source and len(column) == 1 and bool(valid[column[0]]) and
                              np.allclose(target_pose, b @ models[column[0]] @ np.linalg.inv(a) @
                                          np.asarray(source[key]), atol=2e-4))
            life = temporal['object_window']['lifecycle']
            for endpoint in ('source', 'target'):
                for oid in life[endpoint]['observed_object_ids']:
                    observed.setdefault(int(oid), set()).add(int(life[endpoint]['frame']))
    except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        check('trajectory_and_pair_reports', False, error)
    for oid, record in objects.items():
        if not record.get('confirmed', True):
            continue
        evidence = observed.get(int(oid), set())
        check('observed_lifecycle_intervals', bool(evidence) and _interval(record) == (min(evidence), max(evidence)),
              f'Object {oid}: global lifecycle disagrees with measured frames')

    total_live, total_allocated, snapshot_records, chunk_records = 0, 0, [], []
    for position, entry in enumerate(entries):
        try:
            sid = int(entry['submap_id'])
            expected = ranges[position]
            check('strict_submap_partition', sid == position and all(entry.get(key) == value for key, value in expected.items()) and
                  entry.get('state') == 'closed' and entry['processed_pairs'] == entry['image_count']-1,
                  f'Submap {sid}: fixed frame boundary changed')
            check('global_coordinates', np.allclose(entry['world_from_submap'], np.eye(4)) and
                  entry.get('map_id_namespace') == f'submap_{sid:06d}')
            checkpoint_path = path(entry['checkpoint'])
            checkpoint = json.loads(checkpoint_path.read_text())
            metadata = checkpoint['objects']
            live = sum(int(row['gaussian_count']) for row in metadata.values())
            pruned = np.asarray(checkpoint.get('pruned_ids', []), np.int64)
            allocation = live+len(pruned)
            check('checkpoint_contiguous_ids', checkpoint.get('next_id') == allocation and
                  (not len(pruned) or _ids(pruned)) and bool(np.all(pruned < allocation)))
            owners = np.full(allocation, -1, np.int64)
            for key, row in metadata.items():
                oid, count = int(key), int(row['gaussian_count'])
                filename = row['filename']
                if Path(filename).name != filename:
                    raise ValueError('Invalid checkpoint object filename')
                check('checkpoint_status', row.get('status') in STATUSES and str(oid) in objects and
                      row.get('status') == entry.get('object_status_at_close', {}).get(str(oid)))
                with np.load(checkpoint_path.parent / filename, allow_pickle=False) as block:
                    ids = block['map_ids']
                    valid = _ids(ids, count) and bool(np.all(ids < allocation))
                    if valid:
                        valid = bool(np.all(owners[ids] < 0))
                    check('checkpoint_contiguous_ids', valid)
                    if valid:
                        owners[ids] = oid
                    check('checkpoint_parameters', _parameters(block, count))
                    for name in ('initial_xyz', 'initial_log_scales', 'association_colors'):
                        check('checkpoint_parameters', block[name].shape == (count, 3) and np.isfinite(block[name]).all())
                    for name, width in WIDTHS.items():
                        for moment, size in (('exp_avg', width), ('exp_avg_sq', width), ('row_steps', 1)):
                            value = block[f'adam_{name}_{moment}']
                            valid = value.shape == (count, size) and np.isfinite(value).all()
                            if moment in ('exp_avg_sq', 'row_steps'):
                                valid = valid and bool(np.all(value >= 0))
                            if moment == 'row_steps':
                                valid = valid and np.issubdtype(value.dtype, np.integer)
                            check('checkpoint_adam', valid)
            check('checkpoint_contiguous_ids', np.array_equal(np.flatnonzero(owners < 0), np.sort(pruned)))
            ownership, topology, revision = PersistentGaussianMap.validate_journal(checkpoint, owners)
            events = sorted(ownership+topology, key=lambda row: row['revision'], reverse=True)
            check('ownership_and_topology_journal', True)
            check('checkpoint_summary_counts', live == entry['live_gaussians'] and allocation == entry['allocated_gaussians'])
            seed_path = path(entry['directory']) / 'boundary_seed.json'
            seed_count = 0
            if sid:
                seed = json.loads(seed_path.read_text())
                old, new = np.asarray(seed['source_map_ids']), np.asarray(seed['destination_map_ids'])
                seed_count = len(new)
                check('boundary_seed_provenance', seed['source_submap_id'] == sid-1 and
                      seed['destination_submap_id'] == sid and seed['frame'] == entry['frame_ids'][0] and
                      seed['geometry'] == 'same_global_owner_canonical_coordinates' and
                      len(old) == len(new) == seed['tracked_anchor_count'] and
                      (not len(new) or (_ids(old) and _ids(new))) and bool(np.all(new < allocation)) and
                      bool(np.all(old < entries[sid-1]['allocated_gaussians'])))
            previous_live = previous_allocated = seed_count
            for first, second in zip(entry['frame_ids'][:-1], entry['frame_ids'][1:]):
                report = pair_reports[(first, second)]
                refinement = report.get('refinement', {})
                inserted = int(report['inserted_gaussians'])
                added, removed = int(refinement.get('inserted', 0)), int(refinement.get('removed', 0))
                check('map_topology_counts', report['total_map_gaussians'] == previous_live+inserted+added-removed and
                      report.get('allocated_map_gaussians', previous_allocated+inserted+added) == previous_allocated+inserted+added)
                previous_live = report['total_map_gaussians']
                previous_allocated = report.get('allocated_map_gaussians', previous_allocated+inserted+added)
            check('map_topology_counts', previous_live == live and previous_allocated == allocation)
            with path(entry['trajectory']).open() as stream:
                local_trajectory = [json.loads(line) for line in stream if line.strip()]
            check('trajectory_pair_order', [(row['first'], row['second']) for row in local_trajectory] ==
                  list(zip(entry['frame_ids'][:-1], entry['frame_ids'][1:])))
            for row in local_trajectory:
                global_row = pair_poses[(row['first'], row['second'])]
                for endpoint in ('first', 'second'):
                    camera = np.asarray(global_row['camera_to_world_'+endpoint])
                    owner_poses = global_row['object_to_world_'+endpoint]
                    recorded = row['object_to_world_'+endpoint]
                    check('local_global_trajectory_agreement',
                          np.allclose(row['camera_to_world_'+endpoint], camera, atol=2e-4) and
                          recorded.keys() == owner_poses.keys() and
                          all(np.allclose(T, owner_poses[key], atol=2e-4) for key, T in recorded.items()))
            snapshots = sorted(path(entry['snapshots']).glob('frame_*.npz'))
            check('snapshots_complete', [p.name for p in snapshots] == [f'frame_{frame:010d}.npz' for frame in entry['frame_ids']])
            order = []
            for snapshot_path in snapshots:
                with np.load(snapshot_path, allow_pickle=False) as saved:
                    raw = np.asarray(saved.get('map_revision', 0))
                    valid = raw.shape == () and np.issubdtype(raw.dtype, np.integer) and 0 <= int(raw) <= revision
                    check('snapshot_revisions', valid)
                    order.append((int(raw) if valid else revision+1, snapshot_path))
            event_position = 0
            for snapshot_revision, snapshot_path in sorted(order, key=lambda item: (item[0], item[1].name), reverse=True):
                with np.load(snapshot_path, allow_pickle=False) as saved:
                    snapshot = {key: saved[key] for key in saved.files}
                batch, view, _ = prepare_snapshot(snapshot)
                frame = int(snapshot['frame'])
                ids = snapshot['map_ids']
                check('snapshots_parameters', _parameters(snapshot, len(ids)) and frame in entry['frame_ids'] and len(view['rows']) == len(ids))
                valid = _ids(ids) and bool(np.all(ids < allocation)) and snapshot_revision <= revision
                if valid:
                    event_position = _rewind_owners(owners, events, event_position, snapshot_revision)
                    valid = np.array_equal(owners[ids], snapshot['object_labels'])
                check('immutable_map_ownership', valid)
                scope = str(np.asarray(snapshot.get('snapshot_scope', '')).item())
                check('snapshot_training_budget_separated', scope == 'full_visible_map' or len(ids) <= active_budget)
                camera, object_poses = poses[(sid, frame)]
                for oid, T in zip(snapshot['object_ids'], snapshot['object_to_camera']):
                    key = str(int(oid))
                    interval = _interval(objects.get(key, {}))
                    check('no_ghost_objects', interval is not None and interval[0] <= frame <= interval[1] and frame in observed.get(int(oid), set()))
                    check('snapshot_fixed_poses', key in object_poses and np.allclose(T, np.linalg.inv(camera) @ object_poses[key], atol=2e-4))
                snapshot_records.append(dict(submap_id=sid, frame=frame, gaussians=len(ids), snapshot_scope=scope))
                del snapshot, batch, view
            total_live += live
            total_allocated += allocation
            chunk_records.append(dict(submap_id=sid, frames=entry['image_count'], gaussians=live, allocated=allocation))
        except (OSError, ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
            check('submap_artifacts', False, f'Submap {entry.get("submap_id")}: {error}')
    check('checkpoint_summary_counts', manifest.get('global_stored_gaussians') == total_live)
    try:
        chosen = discover_map_snapshots(root)
        check('shared_frame_selected_once', [p.name for p in chosen] == [f'frame_{frame:010d}.npz' for frame in frames])
    except (OSError, ValueError, KeyError, TypeError) as error:
        check('shared_frame_selected_once', False, error)
    return dict(passed=all(checks.values()), checks={key: bool(value) for key, value in checks.items()},
                errors=errors, frames=len(frames), pairs=len(pairs), submaps=chunk_records,
                gaussians=total_live, total_map_gaussians=total_live, allocated_map_gaussians=total_allocated,
                snapshots=sorted(snapshot_records, key=lambda row: (row['frame'], row['submap_id'])),
                geometry_memory='one archived object block or frame; one current submap ownership routing table',
                scope='saved-state consistency; reconstruction/segmentation accuracy and CUDA equivalence not evaluated',
                ground_truth_used=False, dataset=cfg.get('dataset'))

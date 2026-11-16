"""Saved map audit rejects identity, geometry, lifetime and budget corruption."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

from copy import deepcopy
import json
import subprocess
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
from mapping.persistent_gaussian_map import PersistentGaussianMap
from tools.validation.verify_gaussian_map_run import WIDTHS, verify


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


class GaussianMapAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = PersistentGaussianMap()
        for frame in (10, 12, 14):
            self.store.integrate(0, [[0., 0., 3.]], [[0., 1., 0.]], frame_id=frame)
            if frame <= 12:
                self.store.integrate(1, [[0., 0., 2.]], [[1., 0., 0.]], frame_id=frame)
        self.store.set_status(0, 'active', last_seen_frame=14)
        self.store.set_status(1, 'retired', last_seen_frame=12)
        self.store.save(self.root / 'gaussian_map/map')
        self.cfg = dict(sequence=dict(streaming=True), object_window=dict(window_frames=3),
                        gaussian_map=dict(enabled=True, window_frames=3, max_active_gaussians=10,
                                          max_samples_per_frame=10, max_new_gaussians_per_frame=10,
                                          optimization=dict(backend='torch_reference', iterations=0)))
        self.state = dict(status='complete', frames=[10, 12, 14], config=self.cfg,
                          objects={'0': dict(status='active', confirmed=True,
                                             first_observed_frame=10, last_observed_frame=14),
                                   '1': dict(status='retired', confirmed=True, end_frame=12,
                                             first_observed_frame=10, last_observed_frame=12)},
                          pairs=[], gaussian_map=dict(enabled=True, finalized=True,
                                                      total_map_gaussians=2, map=self.store.to_dict()))
        poses = {'0': np.eye(4).tolist(), '1': np.eye(4).tolist()}
        trajectory = []
        for index, (first, second) in enumerate(((10, 12), (12, 14))):
            relative = f'pairs/{first}_{second}'
            self.state['pairs'].append(dict(first=first, second=second, result_path=relative))
            directory = self.root / relative
            directory.mkdir(parents=True)
            optimization = dict(optimized_parameters=list(WIDTHS), iterations_completed=0,
                                ownership_optimized=False, camera_poses_optimized=False,
                                object_poses_optimized=False, track_count=0, track_3d_count=0,
                                gradients={key: dict(max_abs=0., max_norm=0., nonzero_steps=0) for key in WIDTHS},
                                initial_losses={}, final_losses={})
            mapping = dict(fixed_object_ownership=True, camera_poses_optimized=False,
                           object_poses_optimized=False, window_frames=[10, 12] if index == 0 else [10, 12, 14],
                           total_map_gaussians=2, inserted_gaussians=2 if index == 0 else 0,
                           active_map_gaussians=2, deferred_visible_gaussians=0,
                           sampled_observations=2, optimization=optimization)
            life = dict(source=dict(frame=first, observed_object_ids=[0, 1]),
                        target=dict(frame=second, observed_object_ids=[0, 1] if second == 12 else [0]))
            (directory / 'temporal_state.json').write_text(json.dumps(
                dict(gaussian_map=mapping, object_window=dict(lifecycle=life))))
            np.savez_compressed(directory / 'object_state.npz', object_ids=np.array([0, 1]),
                                H_object=np.repeat(np.eye(4)[None], 2, axis=0),
                                object_model_valid=np.ones(2, bool))
            trajectory.append(dict(first=first, second=second,
                                   camera_to_world_first=np.eye(4).tolist(), camera_to_world_second=np.eye(4).tolist(),
                                   object_to_world_first=deepcopy(poses),
                                   object_to_world_second=deepcopy(poses) if index == 0 else {'0': poses['0']}))
        (self.root / 'gaussian_map/trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trajectory))
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))
        frames = self.root / 'gaussian_map/frames'
        frames.mkdir()
        self.K = np.array([[16., 0., 8.5], [0., 16., 8.5], [0., 0., 1.]], np.float32)
        for frame in (10, 12, 14):
            ids = np.array([0, 1] if frame < 14 else [0], np.int64)
            batch = self.store.select(ids, include_optimizer=False)
            selected = {key: batch[key] for key in (*WIDTHS, 'map_ids')}
            selected.update(frame=np.array(frame), object_labels=batch['object_ids'],
                            object_ids=np.unique(batch['object_ids']), K=self.K,
                            object_to_camera=np.repeat(np.eye(4)[None], len(ids), axis=0))
            np.savez_compressed(frames / f'frame_{frame:010d}.npz', **selected)

    def snapshot_path(self, frame=14):
        return self.root / f'gaussian_map/frames/frame_{frame:010d}.npz'

    def block_path(self, oid=1):
        manifest = json.loads((self.root / 'gaussian_map/map/manifest.json').read_text())
        return self.root / 'gaussian_map/map' / manifest['objects'][str(oid)]['filename']

    def save_final_map(self):
        self.store.save(self.root / 'gaussian_map/map')
        self.state['gaussian_map'].update(total_map_gaussians=len(self.store),
                                          allocated_map_gaussians=self.store.next_id,
                                          map=self.store.to_dict())
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))

    def set_snapshot_revision(self, frame, revision, ids=None):
        path = self.snapshot_path(frame)
        state = read_npz(path)
        state['map_revision'] = np.array(revision, np.int64)
        if ids is not None:
            batch = self.store.select(ids, include_optimizer=False)
            state.update({key: batch[key] for key in (*WIDTHS, 'map_ids')})
            state.update(object_labels=batch['object_ids'], object_ids=np.unique(batch['object_ids']))
            state['object_to_camera'] = np.repeat(np.eye(4)[None], len(state['object_ids']), axis=0)
        np.savez_compressed(path, **state)

    def update_pair(self, index, **changes):
        pair = self.state['pairs'][index]
        path = self.root / pair['result_path'] / 'temporal_state.json'
        temporal = json.loads(path.read_text())
        temporal['gaussian_map'].update(changes)
        path.write_text(json.dumps(temporal))

    def test_valid_audit_is_read_only_and_reports_scope(self):
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob('*') if path.is_file()}
        report = verify(self.root)
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['gaussians'], 2)
        self.assertFalse(report['ground_truth_used'])
        self.assertIn('accuracy', report['scope'])
        after = {path: path.stat().st_mtime_ns for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.root / 'gaussian_map_verification.json').exists())

    def test_snapshot_cannot_change_immutable_owner(self):
        path = self.snapshot_path(12)
        state = read_npz(path)
        state['object_labels'][:] = state['object_labels'][::-1]
        np.savez_compressed(path, **state)
        self.assertFalse(verify(self.root)['checks']['immutable_map_ownership'])

    def test_retired_owner_cannot_reappear_after_last_observation(self):
        state = read_npz(self.snapshot_path(12))
        state['frame'] = np.array(14)
        np.savez_compressed(self.snapshot_path(14), **state)
        report = verify(self.root)
        self.assertFalse(report['checks']['no_ghost_objects'])
        self.assertFalse(report['passed'])

    def test_nonunit_quaternion_and_negative_adam_moment_are_rejected(self):
        path = self.block_path()
        block = read_npz(path)
        block['rotations'][0] = [2., 0., 0., 0.]
        block['adam_xyz_exp_avg_sq'][0, 0] = -1.
        np.savez_compressed(path, **block)
        report = verify(self.root)
        self.assertFalse(report['checks']['checkpoint_parameters'])
        self.assertFalse(report['checks']['checkpoint_adam'])

    def test_duplicate_ids_across_object_checkpoints_are_rejected(self):
        path = self.block_path()
        block = read_npz(path)
        block['map_ids'][0] = 0
        np.savez_compressed(path, **block)
        self.assertFalse(verify(self.root)['checks']['checkpoint_contiguous_ids'])

    def test_nonfinite_snapshot_and_missing_snapshot_are_rejected(self):
        path = self.snapshot_path(12)
        state = read_npz(path)
        state['xyz'][0, 0] = float('nan')
        np.savez_compressed(path, **state)
        self.snapshot_path(14).unlink()
        report = verify(self.root)
        self.assertFalse(report['checks']['snapshots_parameters'])
        self.assertFalse(report['checks']['snapshots_complete'])

    def test_active_map_budget_and_pose_contract_are_checked(self):
        path = self.root / 'pairs/10_12/temporal_state.json'
        temporal = json.loads(path.read_text())
        temporal['gaussian_map']['active_map_gaussians'] = 11
        temporal['gaussian_map']['camera_poses_optimized'] = True
        path.write_text(json.dumps(temporal))
        report = verify(self.root)
        self.assertFalse(report['checks']['bounded_active_gaussians'])
        self.assertFalse(report['checks']['fixed_ownership_and_poses'])

    def test_recorded_camera_motion_must_match_background_pair_model(self):
        path = self.root / 'pairs/10_12/object_state.npz'
        saved = read_npz(path)
        saved['H_object'][0, 0, 3] = 1.
        np.savez_compressed(path, **saved)
        self.assertFalse(verify(self.root)['checks']['fixed_trajectory'])

    def test_version_one_checkpoint_and_legacy_reports_remain_valid(self):
        path = self.root / 'gaussian_map/map/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['format_version'] = 1
        for key in ('live_count', 'allocated_count', 'pruned_ids', 'revision', 'next_revision',
                    'ownership_events', 'topology_events'):
            manifest.pop(key)
        path.write_text(json.dumps(manifest))
        self.assertTrue(verify(self.root)['passed'])

    def test_historical_snapshots_survive_later_ownership_change_and_prune(self):
        self.store.reassign([1], 0, np.eye(4), frame_id=12)
        self.store.remove([0], frame_id=14)
        self.save_final_map()
        for frame in (10, 12):
            self.set_snapshot_revision(frame, 0)
        self.set_snapshot_revision(14, 2, ids=[1])
        self.update_pair(0, map_revision=0, allocated_map_gaussians=2)
        self.update_pair(1, map_revision=2, allocated_map_gaussians=2,
                         total_map_gaussians=1, active_map_gaussians=1,
                         refinement=dict(inserted=0, removed=1))
        report = verify(self.root)
        self.assertTrue(report['passed'], report)
        self.assertEqual((report['total_map_gaussians'], report['allocated_map_gaussians'],
                          report['pruned_map_gaussians']), (1, 2, 1))
        # A snapshot after removal cannot still use the deleted map ID.
        self.set_snapshot_revision(10, 2)
        self.assertFalse(verify(self.root)['checks']['immutable_map_ownership'])

    def test_split_children_and_post_refinement_optimization_are_audited(self):
        children = self.store.append_from_parent([0], frame_id=14, reason='split')
        self.store.remove([0], frame_id=14, reason='split_parent')
        self.save_final_map()
        for frame in (10, 12):
            self.set_snapshot_revision(frame, 0)
        self.set_snapshot_revision(14, 2, ids=children)
        path = self.root / 'pairs/12_14/temporal_state.json'
        post = json.loads(path.read_text())['gaussian_map']['optimization']
        post['iterations_completed'] = 3
        self.update_pair(0, map_revision=0, allocated_map_gaussians=2)
        self.update_pair(1, map_revision=2, allocated_map_gaussians=3,
                         refinement=dict(inserted=1, removed=1, post_optimization=post))
        report = verify(self.root)
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['optimization'][-1]['iterations'], 3)
        self.assertEqual(report['optimization'][-1]['post_iterations'], 3)
        # A child cannot be rendered in a snapshot predating its allocation.
        self.set_snapshot_revision(10, 0, ids=children)
        self.assertFalse(verify(self.root)['checks']['immutable_map_ownership'])

    def test_future_or_missing_snapshot_revision_is_rejected_after_transfer(self):
        self.store.reassign([1], 0, np.eye(4), frame_id=12)
        self.save_final_map()
        for frame in (10, 12, 14):
            self.set_snapshot_revision(frame, 0)
        self.set_snapshot_revision(14, 2)
        self.assertFalse(verify(self.root)['checks']['snapshot_revisions'])
        self.set_snapshot_revision(14, 0)
        path = self.snapshot_path(12)
        state = read_npz(path)
        state.pop('map_revision')
        np.savez_compressed(path, **state)
        self.assertFalse(verify(self.root)['checks']['snapshot_revisions'])

    def test_tampered_transfer_journal_and_topology_count_are_rejected(self):
        self.store.reassign([1], 0, np.eye(4), frame_id=12)
        self.save_final_map()
        for frame in (10, 12, 14):
            self.set_snapshot_revision(frame, 0)
        path = self.root / 'gaussian_map/map/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['ownership_events'][0]['transforms'][0][0][0] = 2.
        path.write_text(json.dumps(manifest))
        report = verify(self.root)
        self.assertFalse(report['checks']['ownership_and_topology_journal'])
        self.update_pair(1, refinement=dict(inserted=0, removed=1))
        self.assertFalse(verify(self.root)['checks']['map_topology_counts'])

    def test_pruned_partition_and_post_optimization_pose_contract_are_checked(self):
        path = self.root / 'gaussian_map/map/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['pruned_ids'] = [0]
        path.write_text(json.dumps(manifest))
        report = verify(self.root)
        self.assertFalse(report['checks']['checkpoint_contiguous_ids'])
        temporal_path = self.root / 'pairs/12_14/temporal_state.json'
        post = json.loads(temporal_path.read_text())['gaussian_map']['optimization']
        post['object_poses_optimized'] = True
        self.update_pair(1, refinement=dict(inserted=0, removed=0, post_optimization=post))
        self.assertFalse(verify(self.root)['checks']['optimization_reports'])

    def test_cli_writes_atomic_report_and_returns_failure_for_corrupt_state(self):
        command = [sys.executable, str(HERE / 'tools/validation/verify_gaussian_map_run.py'), '--result', str(self.root)]
        success = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr + success.stdout)
        report_path = self.root / 'gaussian_map_verification.json'
        self.assertTrue(json.loads(report_path.read_text())['passed'])
        self.state['status'] = 'running'
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))
        failure = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failure.returncode, 1, failure.stderr + failure.stdout)
        self.assertFalse(json.loads(report_path.read_text())['passed'])
        self.assertEqual(list(self.root.glob('.gaussian_map_verification.*.tmp')), [])


if __name__ == '__main__':
    unittest.main()

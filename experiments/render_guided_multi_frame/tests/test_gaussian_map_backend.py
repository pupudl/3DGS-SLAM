"""Canonical-map and lifecycle checks with known camera/object rigid motions."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parents[1]
from mapping.gaussian_map_backend import GaussianMapBackend, transform_points
from mapping.persistent_gaussian_map import PersistentGaussianMap


def pose(translation=(0., 0., 0.), angles=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix()
    result[:3, 3] = translation
    return result


class GaussianMapBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.K = np.asarray([[20., 0., 20.], [0., 21., 16.], [0., 0., 1.]])
        self.background = np.asarray([[-1., -.8, 6.], [1., -.8, 6.],
                                      [-1., .8, 6.5], [1., .8, 6.5]])
        self.local = np.asarray([[-.3, -.2, -.1], [.3, -.2, .1],
                                 [-.3, .2, .1], [.3, .2, -.1]])
        self.colors = np.asarray([[.8, .2, .2], [.2, .8, .2], [.2, .2, .8], [.8, .8, .2]], np.float32)
        self.scales = np.tile([.08, .05, .03], (4, 1))
        self.canonical_basis = Rotation.from_euler('xyz', [.15, -.2, .1]).as_matrix()
        self.cfg = dict(gaussian_map=dict(
            enabled=True, window_frames=3, backend='torch_reference', iterations=0,
            downsample=1, observation_stride=1, initial_scale_multiplier=1.,
            max_active_gaussians=100, max_samples_per_frame=100,
            max_new_gaussians_per_frame=100, checkpoint_interval=2,
            association_distance=.025, color_distance=.2, freeze_when_missing=True))
        self.backend = GaussianMapBackend(self.cfg, self.root)

    def camera(self, frame):
        return pose((.07 * frame, .015 * frame, .01 * frame),
                    (.003 * frame, .018 * frame, -.005 * frame))

    def owner_pose(self, oid, frame):
        if oid == 0:
            return np.eye(4)
        birth = 0 if oid == 1 else 5
        age = frame - birth
        return pose((.15 * age + (0. if oid == 1 else .8), .03 * age, 5. + .02 * age),
                    (.01 * age, -.035 * age, .02 * age))

    def frame(self, frame, owners):
        camera_inv = np.linalg.inv(self.camera(frame))
        xyz, colors, scales, bases, labels, tracks = [], [], [], [], [], []
        for oid in [0] + list(owners):
            local = self.background if oid == 0 else self.local
            owner_to_camera = camera_inv @ self.owner_pose(oid, frame)
            xyz.append(transform_points(local, owner_to_camera))
            colors.append(self.colors)
            scales.append(self.scales)
            bases.append(np.repeat((owner_to_camera[:3, :3] @ self.canonical_basis)[None], 4, axis=0))
            labels.extend([oid] * 4)
            tracks.extend(np.arange(4) + 100 * oid)
        xyz = np.concatenate(xyz)
        uv = (xyz @ self.K.T)[:, :2] / xyz[:, 2:]
        gmap = dict(xyz=xyz, uv=uv, color=np.concatenate(colors), scale=np.concatenate(scales),
                    basis=np.concatenate(bases))
        return gmap, np.asarray(labels, np.int64), np.asarray(tracks, np.int64)

    def pair(self, first, source_owners=(1,), target_owners=(1,), statuses=None,
             backend=None, modify=None):
        backend = self.backend if backend is None else backend
        second = first + 1
        gmap, labels, tracks = self.frame(first, source_owners)
        if modify is not None:
            modify(gmap, labels, tracks)
        object_ids = np.asarray(sorted({0} | set(source_owners) | set(target_owners)), np.int64)
        camera0, camera1 = self.camera(first), self.camera(second)
        models = np.asarray([np.linalg.inv(camera1) @ self.owner_pose(int(oid), second) @
                             np.linalg.inv(self.owner_pose(int(oid), first)) @ camera0
                             for oid in object_ids])
        probabilities = np.asarray([[float(oid == label) for oid in object_ids] for label in labels])
        posterior = dict(object_labels=labels, object_probabilities=probabilities)
        image = np.zeros((33, 41, 3), np.uint8)
        depth = np.full((33, 41), 6., np.float32)
        observations = dict(source=image, target=image.copy(), source_depth=depth,
                            target_depth=depth.copy(), source_confidence=np.ones_like(depth),
                            target_confidence=np.ones_like(depth))
        statuses = {} if statuses is None else statuses
        known = set(backend.store.object_ids) | set(map(int, object_ids))
        registry = SimpleNamespace(objects={oid: dict(status=statuses.get(oid, 'active'),
                                                       last_observed_frame=first if oid else second)
                                             for oid in known})
        presence = dict(source=dict(observed_object_ids=[0] + list(source_owners)),
                        target=dict(observed_object_ids=[0] + list(target_owners)))
        report = backend.process(first, second, gmap, posterior, object_ids, models,
                                 np.ones(len(object_ids), bool), self.K,
                                 observations, registry, tracks, presence=presence)
        return report, gmap, labels, tracks

    def snapshot(self, frame):
        with np.load(self.backend.frames_dir / f'frame_{frame:010d}.npz', allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def test_zero_iteration_migration_matches_camera_points_and_covariance(self):
        _, gmap, labels, _ = self.pair(0)
        source = self.snapshot(0)
        target = self.snapshot(1)
        for oid in (0, 1):
            row = source['object_labels'] == oid
            pose0 = source['object_to_camera'][np.flatnonzero(source['object_ids'] == oid)[0]]
            pose1 = target['object_to_camera'][np.flatnonzero(target['object_ids'] == oid)[0]]
            np.testing.assert_allclose(transform_points(source['xyz'][row], pose0),
                                       gmap['xyz'][labels == oid], atol=1e-6)
            expected_target = transform_points(
                self.background if oid == 0 else self.local,
                np.linalg.inv(self.camera(1)) @ self.owner_pose(oid, 1))
            np.testing.assert_allclose(transform_points(target['xyz'][target['object_labels'] == oid], pose1),
                                       expected_target, atol=1e-6)
            rotations = Rotation.from_quat(source['rotations'][row][:, [1, 2, 3, 0]]).as_matrix()
            scaled = rotations * np.exp(source['log_scales'][row])[:, None, :]
            covariance = pose0[:3, :3] @ (scaled @ scaled.transpose(0, 2, 1)) @ pose0[:3, :3].T
            measured = gmap['basis'][labels == oid] * gmap['scale'][labels == oid][:, None, :]
            np.testing.assert_allclose(covariance, measured @ measured.transpose(0, 2, 1), atol=1e-7)

    def test_noncommuting_camera_and_vehicle_motions_fuse_one_map_across_windows(self):
        for frame in range(9):
            report, gmap, labels, _ = self.pair(frame)
            self.assertEqual(report['total_map_gaussians'], 8)
            self.assertEqual(report['inserted_gaussians'], 8 if frame == 0 else 0)
            self.assertEqual(report['matched_observations'], 0 if frame == 0 else 8)
            self.assertLessEqual(len(report['window_frames']), 3)
            np.testing.assert_allclose(self.backend._camera_to_world, self.camera(frame + 1), atol=1e-8)
            np.testing.assert_allclose(self.backend._object_to_world[1], self.owner_pose(1, frame + 1), atol=1e-8)
            np.testing.assert_allclose(self.backend.store.export_object(0)['xyz'], self.background, atol=1e-6)
            np.testing.assert_allclose(self.backend.store.export_object(1)['xyz'], self.local, atol=1e-6)
            # Measured sparse RGB-D anchors remain source-camera measurements.
            measurements = self.backend._views[frame]['measured_tracks']
            np.testing.assert_array_equal(measurements['xyz'], gmap['xyz'])
        self.assertEqual(report['window_frames'], [7, 8, 9])
        self.assertTrue((self.backend.frames_dir / 'frame_0000000000.npz').is_file())

    def test_missing_retired_vehicle_stops_current_rendering_and_keeps_archived_map(self):
        self.pair(0)
        car_ids = self.backend.store.ids_for_object(1)
        before = self.backend.store.select(car_ids)
        for state in before['optimizer_state']['params'].values():
            state['exp_avg'][:] = .3
            state['exp_avg_sq'][:] = .1
            state['row_steps'][:] = 7
        self.backend.store.commit(car_ids, {}, before['optimizer_state'])
        self.backend.cfg['iterations'] = 2
        self.pair(1, target_owners=(), statuses={1: 'missing'})
        self.assertNotIn(1, self.snapshot(2)['object_ids'])
        self.assertIn(1, self.snapshot(1)['object_ids'])
        frozen = self.backend.store.select(car_ids)
        for name in PARAMETER_NAMES:
            np.testing.assert_array_equal(frozen[name], before[name])
            np.testing.assert_array_equal(frozen['optimizer_state']['params'][name]['row_steps'],
                                          before['optimizer_state']['params'][name]['row_steps'])
        self.backend.cfg['iterations'] = 0
        for frame in range(2, 7):
            report, _, _, _ = self.pair(frame, source_owners=(), target_owners=(), statuses={1: 'retired'})
            self.assertNotIn(1, self.snapshot(frame + 1)['object_ids'])
        batch, _, _ = self.backend._active_batch([])
        self.assertNotIn(1, batch['object_ids'])
        self.assertEqual(report['total_map_gaussians'], 8)
        np.testing.assert_array_equal(self.backend.store.export_object(1)['xyz'], before['xyz'])
        self.backend.store.save(self.root / 'checkpoint')
        loaded = PersistentGaussianMap.load(self.root / 'checkpoint')
        self.assertEqual(loaded.get_status(1), 'retired')
        np.testing.assert_array_equal(loaded.ids_for_object(1), car_ids)
        np.testing.assert_array_equal(loaded.export_object(1)['xyz'], before['xyz'])

    def test_new_vehicle_gets_distinct_ids_after_old_vehicle_window_eviction(self):
        self.pair(0)
        old = set(self.backend.store.ids_for_object(1))
        self.pair(1, target_owners=(), statuses={1: 'missing'})
        for frame in range(2, 5):
            self.pair(frame, source_owners=(), target_owners=(), statuses={1: 'retired'})
        report, _, _, _ = self.pair(5, source_owners=(2,), target_owners=(2,), statuses={1: 'retired'})
        self.assertEqual(report['inserted_gaussians'], 4)
        self.assertEqual(report['total_map_gaussians'], 12)
        new = set(self.backend.store.ids_for_object(2))
        self.assertFalse(new & old)
        self.assertGreater(min(new), max(old))
        self.assertEqual(set(self.snapshot(6)['object_ids']), {0, 2})

    def test_return_after_pose_gap_recovers_from_reliable_canonical_tracks(self):
        self.pair(0)
        old_ids = self.backend.store.ids_for_object(1)
        self.pair(1, target_owners=(), statuses={1: 'missing'})
        self.pair(2, source_owners=(), target_owners=(1,), statuses={1: 'active'})
        self.assertNotIn(1, self.backend._object_to_world)
        report, _, _, _ = self.pair(3)
        self.assertEqual(report['skipped_objects'], [])
        self.assertEqual(report['inserted_gaussians'], 0)
        np.testing.assert_array_equal(self.backend.store.ids_for_object(1), old_ids)
        np.testing.assert_allclose(self.backend._object_to_world[1], self.owner_pose(1, 4), atol=1e-6)

    def test_gap_without_surviving_tracks_skips_stale_object_pose(self):
        self.pair(0)
        self.pair(1, target_owners=(), statuses={1: 'missing'})
        self.backend.retain_tracks(np.arange(4))
        self.pair(2, source_owners=(), target_owners=(1,), statuses={1: 'active'})
        report, _, _, _ = self.pair(3)
        self.assertEqual([row['object_id'] for row in report['skipped_objects']], [1])
        self.assertEqual(report['inserted_gaussians'], 0)
        self.assertNotIn(1, self.snapshot(4)['object_ids'])

    def test_reassigned_tracks_cannot_recover_a_different_current_owner(self):
        self.pair(0)
        self.pair(1, target_owners=(), statuses={1: 'missing'})
        self.pair(2, source_owners=(), target_owners=(1, 2), statuses={1: 'active'})

        def reassigned_tracks(gmap, labels, tracks):
            tracks[labels == 1] = np.arange(4) + 900
            tracks[labels == 2] = np.arange(4) + 100

        report, _, _, _ = self.pair(3, source_owners=(1, 2), target_owners=(1, 2), modify=reassigned_tracks)
        self.assertEqual([row['object_id'] for row in report['skipped_objects']], [1])
        self.assertNotIn(1, self.snapshot(4)['object_ids'])
        self.assertEqual(len(self.backend.store.ids_for_object(1)), 4)

    def test_shared_insertion_and_active_budgets_and_track_cache_are_bounded(self):
        cfg = dict(gaussian_map=dict(self.cfg['gaussian_map'], max_new_gaussians_per_frame=5,
                                    max_active_gaussians=3))
        backend = GaussianMapBackend(cfg, self.root / 'bounded')
        report, _, _, _ = self.pair(0, backend=backend)
        self.assertEqual(report['inserted_gaussians'], 5)
        self.assertEqual(report['insertion_budget_skipped'], 3)
        self.assertEqual(report['active_map_gaussians'], 3)
        self.assertEqual(report['deferred_visible_gaussians'], 2)
        self.assertEqual(len(backend.store.ids_for_object(1)), 4)
        self.assertEqual(len(backend.store.ids_for_object(0)), 1)
        backend.retain_tracks([100, 101])
        self.assertEqual(set(backend._track_landmarks), {100, 101})


PARAMETER_NAMES = ('xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits')


if __name__ == '__main__':
    unittest.main()

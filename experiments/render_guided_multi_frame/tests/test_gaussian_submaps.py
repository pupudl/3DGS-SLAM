"""Strict submap partitioning, boundary coordinates and resident-map lifetime."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import copy
import gc
import json
import tempfile
from types import SimpleNamespace
import unittest
import weakref

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
from mapping.gaussian_submaps import SubmapGaussianMapBackend, partition_submaps
from mapping.gaussian_map_backend import GaussianMapBackend, transform_points
from mapping.persistent_gaussian_map import PersistentGaussianMap, PARAMETER_WIDTHS


class _CheapBackend:
    """No rendering; exercises manager storage/pose/lifetime with actual maps."""
    def __init__(self, cfg, output_root, map_output=None):
        self.output = Path(map_output)
        self.frames_dir = self.output / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.store = PersistentGaussianMap(cfg)
        self.ownership = SimpleNamespace(links={}, last_frames={}, pending={})
        self._camera_to_world = None
        self._pose_frame = None
        self._object_to_world = {0: np.eye(4)}
        self._track_landmarks = {}
        self._views = {}
        self._updates = 0
        self._last_report = {}

    def process(self, first, second, registry):
        source = np.eye(4) if self._camera_to_world is None else self._camera_to_world.copy()
        target = source.copy()
        target[0, 3] += (second-first)*.01
        points = np.asarray([[first*.1, 0., 5.]], np.float32)
        result = self.store.integrate(0, points, np.asarray([[.1, .2, .3]]), frame_id=first)
        self._track_landmarks = {int(first): (0, int(result['map_ids'][0]))}
        self._camera_to_world = target
        self._pose_frame = int(second)
        self._views = {int(frame): dict(object_to_camera={0: np.linalg.inv(camera)})
                       for frame, camera in ((first, source), (second, target))}
        self._updates += 1
        self._last_report = dict(enabled=True, updates=self._updates, total_map_gaussians=len(self.store),
                                 window_frames=[int(first), int(second)])
        return dict(self._last_report)

    def retain_tracks(self, ids):
        ids = set(map(int, ids))
        self._track_landmarks = {track: value for track, value in self._track_landmarks.items() if track in ids}

    def finalize(self, registry):
        for oid in self.store.object_ids:
            self.store.set_status(oid, registry.objects[oid]['status'])
        self.store.save(self.output / 'map')
        return dict(self._last_report, finalized=True)


class GaussianSubmapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = SimpleNamespace(objects={0: dict(status='active'), 1: dict(status='active')})

    def manager(self, ids, frames=50, **options):
        cfg = dict(gaussian_map=dict(submaps=dict(frames=frames, overlap_frames=1, **options)))
        return SubmapGaussianMapBackend(cfg, self.root, ids, backend_factory=_CheapBackend)

    def test_strict_fifty_image_ranges_and_single_shared_frame(self):
        ranges = partition_submaps(range(148), frames=50)
        self.assertEqual([item['frame_ids'] for item in ranges],
                         [list(range(0, 50)), list(range(49, 99)), list(range(98, 148))])
        self.assertTrue(all(item['image_count'] == 50 and not item['partial'] for item in ranges))
        for left, right in zip(ranges, ranges[1:]):
            self.assertEqual(set(left['frame_ids']) & set(right['frame_ids']), {left['frame_ids'][-1]})

    def test_no_singleton_tail_at_exact_boundary_and_partial_is_explicit(self):
        self.assertEqual(len(partition_submaps(range(50))), 1)
        self.assertEqual(len(partition_submaps(range(99))), 2)
        self.assertEqual([item['image_count'] for item in partition_submaps(range(100))], [50, 50, 2])
        self.assertTrue(partition_submaps(range(100))[-1]['partial'])

    def test_sampling_interval_does_not_change_processed_image_count(self):
        ids = list(range(2433, 2632, 2))
        ranges = partition_submaps(ids)
        self.assertEqual([item['image_count'] for item in ranges], [50, 50, 2])
        self.assertEqual(ranges[1]['frame_ids'][0], ids[49])
        for bad in ([1, 1], [2, 1], [-1, 0], [0., 1.]):
            with self.assertRaises(ValueError):
                partition_submaps(bad)
        for count in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                partition_submaps(range(10), frames=count)
        with self.assertRaises(ValueError):
            partition_submaps(range(10), overlap_frames=2)

    def test_capacity_raised_for_full_length_and_never_causes_early_close(self):
        cfg = dict(gaussian_map=dict(max_new_gaussians_per_frame=80000,
                                    submaps=dict(frames=50, overlap_frames=1),
                                    refinement=dict(enabled=True, max_total_gaussians=1, max_new_per_update=3000)))
        manager = SubmapGaussianMapBackend(cfg, self.root, range(100), backend_factory=_CheapBackend)
        self.assertGreaterEqual(manager._effective_cfg()['gaussian_map']['refinement']['max_total_gaussians'],
                                49*(80000+3000)+manager.max_boundary_landmarks)
        manager.process(0, 1, registry=self.registry)
        self.assertEqual(manager.to_dict()['submap_id'], 0)
        self.assertFalse(manager.to_dict()['submap_closed'])
        self.assertEqual(cfg['gaussian_map']['refinement']['max_total_gaussians'], 1)

    def test_archived_geometry_is_released_and_only_sparse_boundary_remains(self):
        manager = self.manager(range(5), frames=3)
        manager.process(0, 1, registry=self.registry)
        reference = weakref.ref(manager.store)
        manager.process(1, 2, registry=self.registry)
        gc.collect()
        self.assertIsNone(reference())
        self.assertIsNone(manager.backend)
        self.assertLessEqual(len(manager._boundary['batch']['map_ids']), 1)
        self.assertEqual(manager.to_dict()['resident_submaps'], 0)
        self.assertEqual(manager._entries[0]['image_count'], 3)
        self.assertEqual(manager._entries[0]['object_status_at_close']['0'], 'active')
        manager.retain_tracks([])
        self.assertEqual(len(manager._boundary['batch']['map_ids']), 0)
        manager.process(2, 3, registry=self.registry)
        self.assertEqual(manager.to_dict()['submap_id'], 1)
        np.testing.assert_allclose(manager.backend._camera_to_world[0, 3], .03)

    def test_thousand_images_have_one_live_submap_and_global_pose_continuity(self):
        manager = self.manager(range(1000))
        for frame in range(999):
            report = manager.process(frame, frame+1, registry=self.registry)
            manager.retain_tracks([frame])
            self.assertLessEqual(report['resident_submaps'], 1)
        result = manager.finalize(self.registry)
        manifest = json.loads(manager.manifest_path.read_text())
        self.assertEqual(len(manifest['submaps']), 21)
        self.assertTrue(all(entry['image_count'] == 50 for entry in manifest['submaps'][:-1]))
        self.assertTrue(manifest['submaps'][-1]['partial'])
        self.assertEqual(manifest['processed_pairs'], 999)
        self.assertTrue(manifest['finalized'])
        self.assertEqual(result['resident_submaps'], 0)
        self.assertIsNone(manager.backend)
        lines = (manager.output / 'trajectory.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 999)
        last = json.loads(lines[-1])
        self.assertAlmostEqual(last['camera_to_world_second'][0][3], 9.99, places=9)
        self.assertEqual(self.registry.objects[1]['status'], 'active')
        self.assertTrue(all((self.root / entry['checkpoint']).is_file() for entry in manifest['submaps']))

    def test_pair_order_and_premature_finalize_fail_before_discarding_state(self):
        manager = self.manager(range(4), frames=3)
        with self.assertRaises(ValueError):
            manager.process(1, 2, registry=self.registry)
        manager.process(0, 1, registry=self.registry)
        with self.assertRaises(RuntimeError):
            manager.finalize(self.registry)
        manager.process(1, 2, registry=self.registry)
        manager.process(2, 3, registry=self.registry)
        manager.finalize(self.registry)
        with self.assertRaises(RuntimeError):
            manager.process(3, 4, registry=self.registry)


class BoundaryImportTests(unittest.TestCase):
    def batch(self):
        source = PersistentGaussianMap(dict(association_distance=.01))
        ids = source.integrate(7, np.asarray([[0., 0., 2.], [.1, 0., 2.]]),
                               np.asarray([[.1, .2, .3], [.3, .4, .5]]), frame_id=4)['map_ids']
        batch = source.select(ids)
        # Distinct learned rows may become geometrically identical. Exact seed
        # import must preserve both instead of invoking integration deduplication.
        quaternions = Rotation.from_euler('xyz', [[.2, -.3, .4], [.1, .5, -.7]]).as_quat()[:, [3, 0, 1, 2]]
        changed = dict(xyz=np.repeat([[.05, 0., 2.]], 2, axis=0), colors=np.full((2, 3), .6),
                       rotations=quaternions)
        for state in batch['optimizer_state']['params'].values():
            state['exp_avg'][:] = .3
            state['exp_avg_sq'][:] = .4
            state['row_steps'][:] = 17
        source.commit(ids, changed, batch['optimizer_state'])
        return source.select(ids)

    def test_exact_seed_preserves_raw_anchors_moments_and_distinct_rows(self):
        destination = PersistentGaussianMap(dict(association_distance=10.))
        destination.integrate(0, [[0., 0., 5.]], [[.2, .2, .2]])
        original = self.batch()
        untouched = copy.deepcopy(original)
        ids = destination.import_rows(original, 7)
        np.testing.assert_array_equal(ids, [1, 2])
        saved = destination.select(ids)
        for key in (*PARAMETER_WIDTHS, 'initial_xyz', 'initial_log_scales', 'association_colors',
                    'observation_count', 'last_seen_frame'):
            np.testing.assert_array_equal(saved[key], original[key])
            np.testing.assert_array_equal(original[key], untouched[key])
        for name, state in original['optimizer_state']['params'].items():
            for key, value in state.items():
                np.testing.assert_array_equal(saved['optimizer_state']['params'][name][key], value)
        original['xyz'][:] = 100
        self.assertLess(destination.select(ids)['xyz'].max(), 10)

    def test_invalid_import_leaves_destination_unchanged(self):
        destination = PersistentGaussianMap()
        batch = self.batch()
        malformed = copy.deepcopy(batch)
        malformed['optimizer_state']['params']['xyz']['exp_avg_sq'][0, 0] = -1
        for rows, owner in ((malformed, 7), (batch, 8)):
            with self.assertRaises(ValueError):
                destination.import_rows(rows, owner)
            self.assertEqual(destination.next_id, 0)
            self.assertEqual(len(destination), 0)


class RealBoundaryCoordinateTests(unittest.TestCase):
    def test_missing_known_owner_without_pose_is_not_reborn_in_a_new_canonical_frame(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cfg = dict(gaussian_map=dict(submaps=dict(frames=3, overlap_frames=1)))
        manager = SubmapGaussianMapBackend(cfg, root, range(5))
        manager._open()
        old = manager.backend
        old.store.integrate(7, [[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                            np.full((3, 3), .5), frame_id=1)
        old.store.set_status(7, 'missing', last_seen_frame=1)
        old._camera_to_world = np.eye(4)
        old._pose_frame = 2
        old._object_to_world = {0: np.eye(4)}
        boundary = manager._capture_boundary(old)
        new = GaussianMapBackend(cfg, root, map_output=root/'new')
        manager._restore_boundary(new, boundary)
        self.assertIn(7, new.store.object_ids)
        self.assertEqual(len(new.store.ids_for_object(7)), 0)
        gmap = dict(xyz=np.asarray([[0., 0., 5.], [.1, 0., 5.], [0., .1, 5.]]))
        _, _, source, target, skipped = new._poses(
            2, 3, gmap, np.asarray([7, 7, 7]), np.asarray([-1, -1, -1]),
            np.asarray([0, 7]), np.repeat(np.eye(4)[None], 2, axis=0),
            np.ones(2, bool), {0, 7}, {0, 7})
        self.assertNotIn(7, source)
        self.assertNotIn(7, target)
        self.assertEqual(skipped[0]['object_id'], 7)

    def test_noncommuting_camera_and_owner_motion_cross_boundary_without_reset(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        cfg = dict(gaussian_map=dict(enabled=True, backend='torch_reference', iterations=0,
                                    window_frames=3, downsample=1, observation_stride=1,
                                    initial_scale_multiplier=1., max_active_gaussians=100,
                                    max_samples_per_frame=100, max_new_gaussians_per_frame=100,
                                    association_distance=.02, checkpoint_interval=2,
                                    submaps=dict(frames=4, overlap_frames=1)))
        manager = SubmapGaussianMapBackend(cfg, root, range(7))
        K = np.asarray([[12., 0., 12.], [0., 13., 10.], [0., 0., 1.]])
        bg = np.asarray([[-1., -.4, 6.], [1., -.4, 6.], [-1., .4, 6.], [1., .4, 6.]])
        car = np.asarray([[-.3, -.2, 0.], [.3, -.2, 0.], [-.3, .2, 0.], [.3, .2, 0.]])
        image = np.zeros((21, 25, 3), np.uint8)
        depth = np.full((21, 25), 6., np.float32)
        measurements = dict(source=image, target=image, source_depth=depth, target_depth=depth,
                            source_confidence=np.ones_like(depth), target_confidence=np.ones_like(depth))

        def pose(frame, owner=False):
            value = np.eye(4)
            value[:3, :3] = Rotation.from_euler('xyz', np.asarray([.009, .018, -.006])*frame*
                                              (-1 if owner else 1)).as_matrix()
            value[:3, 3] = ([.12*frame, .02*frame, 5.] if owner else [.05*frame, -.01*frame, .005*frame])
            return value

        registry = SimpleNamespace(objects={0: dict(status='active'), 1: dict(status='active')})
        previous_owner_pose = None
        for first in range(6):
            second = first+1
            C0, C1 = pose(first), pose(second)
            owners = [0] if first < 2 else [0, 1]
            xyz, labels, bases, tracks = [], [], [], []
            for oid in owners:
                owner_pose = np.eye(4) if oid == 0 else pose(first, owner=True)
                T = np.linalg.inv(C0) @ owner_pose
                xyz.append(transform_points(bg if oid == 0 else car, T))
                bases.append(np.repeat(T[None, :3, :3], 4, axis=0))
                labels.extend([oid]*4)
                tracks.extend(np.arange(4)+100*oid)
            xyz = np.concatenate(xyz)
            labels = np.asarray(labels)
            ids = np.asarray(owners)
            gmap = dict(xyz=xyz, uv=(xyz@K.T)[:, :2]/xyz[:, 2:],
                        color=np.full((len(xyz), 3), .5), scale=np.full((len(xyz), 3), .05),
                        basis=np.concatenate(bases))
            models = np.asarray([np.linalg.inv(C1) @ (np.eye(4) if oid == 0 else pose(second, True)) @
                                 np.linalg.inv(np.eye(4) if oid == 0 else pose(first, True)) @ C0 for oid in ids])
            posterior = dict(object_labels=labels,
                             object_probabilities=(labels[:, None] == ids[None]).astype(float))
            presence = dict(source=dict(observed_object_ids=owners),
                            target=dict(observed_object_ids=[0] if second < 2 else [0, 1]))
            report = manager.process(first, second, gmap, posterior, ids, models,
                                     np.ones(len(ids), bool), K, measurements, registry,
                                     np.asarray(tracks), presence=presence)
            manager.retain_tracks(tracks)
            if first == 2:
                previous_owner_pose = manager._boundary['object_to_world'][1].copy()
            if first == 3:
                new_source = manager.backend._views[first]['object_to_camera'][1]
                np.testing.assert_allclose(C0 @ new_source, previous_owner_pose, atol=1e-8)
                np.testing.assert_allclose(manager.backend._camera_to_world, C1, atol=1e-8)
                self.assertEqual(report['submap_id'], 1)
        manager.finalize(registry)
        old = np.load(root/'gaussian_map/submaps/submap_000000/frames/frame_0000000003.npz')
        new = np.load(root/'gaussian_map/submaps/submap_000001/frames/frame_0000000003.npz')
        self.addCleanup(old.close)
        self.addCleanup(new.close)
        for oid in (0, 1):
            before = old['object_labels'] == oid
            after = new['object_labels'] == oid
            np.testing.assert_allclose(old['xyz'][before], new['xyz'][after], atol=1e-6)
            np.testing.assert_allclose(old['rotations'][before], new['rotations'][after], atol=1e-6)
        self.assertEqual(registry.objects[1]['status'], 'active')


if __name__ == '__main__':
    unittest.main()

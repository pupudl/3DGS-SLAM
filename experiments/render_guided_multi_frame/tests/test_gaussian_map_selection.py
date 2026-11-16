"""Coverage, refinement reserve and full-map snapshot regression checks."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
from mapping.gaussian_map_backend import GaussianMapBackend
from mapping.gaussian_map_selection import select_active_gaussians
from mapping.gaussian_map_optimizer import optimize_gaussian_map


class ActiveSelectionTests(unittest.TestCase):
    def test_all_visible_fast_path_keeps_every_owner_track_and_occluder(self):
        rows, report = select_active_gaussians(
            np.array([7, 9, 15]), np.array([0, 3, 3]),
            np.array([[0, 2], [1, -1], [1, 3]]), np.ones((3, 2)), 10,
            track_ids=[9, 9], priority_ids=[15], recent_ids=[7, 9], frozen=[False, True, True])
        np.testing.assert_array_equal(rows, [0, 1, 2])
        self.assertEqual(report['selection_mode'], 'all_visible')
        self.assertEqual(report['frozen_selected'], 2)
        self.assertEqual(report['track_selected'], 1)
        self.assertEqual(report['selected_view_tiles'], 4)

    def test_spatial_round_robin_uses_every_tile_instead_of_uniform_ids(self):
        ids = np.arange(80)
        tiles = (ids//10)[:, None]
        rows, report = select_active_gaussians(
            ids, np.zeros(80, np.int64), tiles, np.ones((80, 1)), 8,
            options=dict(track_fraction=0, dynamic_fraction=0, recent_fraction=0,
                         frozen_fraction=0, fairness_fraction=1))
        self.assertEqual(len(np.unique(tiles[rows])), 8)
        self.assertEqual(report['selected_view_tiles'], 8)
        self.assertEqual(report['deferred_visible_gaussians'], 72)

    def test_deferred_points_rotate_even_in_one_dense_tile(self):
        ids = np.arange(20)
        last, seen = {}, set()
        for update in range(5):
            rows, _ = select_active_gaussians(
                ids, np.zeros(20, np.int64), np.zeros((20, 1), np.int64),
                np.ones((20, 1)), 4, update=update, last_selected=last,
                options=dict(track_fraction=0, dynamic_fraction=0, recent_fraction=0,
                             frozen_fraction=0, fairness_fraction=1))
            seen.update(ids[rows])
            last.update({int(mid): update for mid in ids[rows]})
        self.assertEqual(seen, set(ids))

    def test_children_tracks_dynamic_owner_and_frozen_occluder_have_reservations(self):
        ids = np.arange(100)
        owners = np.zeros(100, np.int64)
        owners[-5:] = 7
        frozen = np.zeros(100, bool)
        frozen[80:85] = True
        rows, report = select_active_gaussians(
            ids, owners, (ids//5)[:, None], np.ones((100, 1)), 20,
            priority_ids=[98, 99], track_ids=[90, 91], recent_ids=[50, 51],
            frozen=frozen)
        selected = set(ids[rows])
        self.assertTrue({98, 99, 90, 91}.issubset(selected))
        self.assertTrue(any(frozen[rows]))
        self.assertGreater(report['per_object']['7']['selected'], 0)
        self.assertEqual(report['priority_selected'], 2)

    def test_multiple_views_and_depth_quality_avoid_hidden_uniform_choice(self):
        ids = np.arange(8)
        tiles = np.array([[0, -1], [0, -1], [1, -1], [1, -1],
                          [-1, 2], [-1, 2], [-1, 3], [-1, 3]])
        quality = np.where(tiles >= 0, 1., 0.)
        quality[::2] *= .01
        rows, report = select_active_gaussians(
            ids, np.zeros(8, np.int64), tiles, quality, 4,
            options=dict(track_fraction=0, dynamic_fraction=0, recent_fraction=0,
                         frozen_fraction=0, fairness_fraction=0))
        np.testing.assert_array_equal(ids[rows], [1, 3, 5, 7])
        self.assertEqual(report['selected_view_tiles'], 4)


class FullMapBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.cfg = dict(gaussian_map=dict(
            max_active_gaussians=4, observation_stride=1,
            active_selection=dict(enabled=True, tile_size_px=8),
            snapshot_scope='full_visible_map', backend='torch_reference', iterations=1,
            downsample=1, association_distance=.005,
            refinement=dict(enabled=True, warmup_pairs=0, interval=1,
                            max_new_per_update=2, max_children_per_object=2,
                            max_splits_per_update=1, max_clones_per_update=0,
                            max_prunes_per_update=0, min_observation_count=1,
                            min_visible_steps=1, gradient_threshold=.001,
                            max_growth_fraction=1., split_radius_px=1.,
                            max_total_gaussians=100)))
        self.backend = GaussianMapBackend(self.cfg, self.path, map_output=self.path/'direct_map')
        self.K = np.array([[20., 0, 16.], [0, 20., 16.], [0, 0, 1.]])
        xyz = np.array([[-.6, -.3, 4], [-.3, -.3, 4], [0, -.3, 4],
                        [.3, -.3, 4], [.6, -.3, 4], [0, .3, 4]], np.float32)
        result = self.backend.store.integrate(0, xyz, np.full((6, 3), .3),
                                             scales=np.full((6, 3), .15), frame_id=0)
        self.ids = result['map_ids']
        self.image = np.full((33, 33, 3), 180, np.uint8)
        self.depth = np.full((33, 33), 4., np.float32)
        self.backend._view(0, self.image, self.depth, np.ones_like(self.depth),
                           np.eye(4), {0: np.eye(4)}, self.K)
        self.backend._view(1, self.image, self.depth, np.ones_like(self.depth),
                           np.eye(4), {0: np.eye(4)}, self.K)

    def test_full_snapshot_exceeds_training_cap_without_stale_object_ghosts(self):
        car = self.backend.store.integrate(7, np.array([[0, .5, 4], [.3, .5, 4]]),
                                          np.full((2, 3), .5), scales=np.full((2, 3), .1), frame_id=0)
        self.backend.store.set_status(7, 'missing')
        self.backend._views[0]['object_to_camera'][7] = np.eye(4)
        batch, views, deferred = self.backend._active_batch(self.ids)
        self.assertEqual(len(batch['map_ids']), 4)
        self.assertEqual(deferred, 4)
        car_rows = batch['object_ids'] == 7
        self.assertTrue(np.any(car_rows))
        self.assertFalse(np.any(batch['trainable_mask'][car_rows]))
        counts = self.backend._write_frames(batch, views)
        self.assertEqual(counts, {'0': 8, '1': 6})
        self.assertEqual(self.backend.output, self.path/'direct_map')
        with np.load(self.backend.frames_dir/'frame_0000000000.npz') as snapshot:
            self.assertEqual(len(snapshot['map_ids']), 8)
            self.assertEqual(int(snapshot['optimized_in_last_solve'].sum()), 4)
            self.assertEqual(set(snapshot['object_ids']), {0, 7})
        with np.load(self.backend.frames_dir/'frame_0000000001.npz') as snapshot:
            self.assertEqual(set(snapshot['object_ids']), {0})
            self.assertFalse(np.any(np.isin(snapshot['map_ids'], car['map_ids'])))

    def test_full_snapshot_keeps_large_splat_crossing_image_edge(self):
        crossing = self.backend.store.integrate(0, [[-5., 0., 4.]], [[.4, .4, .4]],
                                               scales=[[1., 1., .1]], frame_id=0)['map_ids'][0]
        self.backend._write_frames(self.backend.store.select(self.ids), [])
        with np.load(self.backend.frames_dir/'frame_0000000000.npz') as snapshot:
            self.assertIn(crossing, snapshot['map_ids'])

    def test_full_snapshot_writes_empty_frame_instead_of_old_geometry(self):
        self.backend._views[1]['object_to_camera'] = {}
        batch, views, _ = self.backend._active_batch(self.ids)
        self.backend._write_frames(batch, views)
        with np.load(self.backend.frames_dir/'frame_0000000001.npz') as snapshot:
            self.assertEqual(len(snapshot['map_ids']), 0)
            self.assertEqual(snapshot['object_to_camera'].shape, (0, 4, 4))

    def test_saturated_map_reserves_growth_and_children_receive_adam_updates(self):
        self.backend._track_landmarks[101] = (0, int(self.ids[0]))
        reserve = self.backend._refinement_reserve()
        self.assertEqual(reserve, 1)
        batch, _, _ = self.backend._active_batch(self.ids, reserve=reserve)
        self.assertEqual(len(batch['map_ids']), 3)
        n = len(batch['map_ids'])
        stats = dict(map_ids=batch['map_ids'], screen_gradient_sum=np.ones(n),
                     screen_gradient_count=np.ones(n), max_radius_px=np.full(n, 5.),
                     visible_steps=np.ones(n))
        report = self.backend._refine(batch, dict(refinement_stats=stats), 0)
        self.assertEqual(report['split'], 1)
        children = report['_child_ids']
        self.assertEqual(len(children), 2)
        after, views, _ = self.backend._active_batch(self.ids, priority_ids=children)
        self.assertTrue(set(children).issubset(after['map_ids']))
        self.assertIn(self.ids[0], after['map_ids'])
        optimized = optimize_gaussian_map(after, views, self.backend.cfg)
        self.backend.store.commit(after['map_ids'], optimized['parameters'], optimized['optimizer_state'])
        learned = self.backend.store.select(children)
        steps = learned['optimizer_state']['params']['colors']['row_steps']
        self.assertTrue(np.all(steps > 0))
        self.assertEqual(self.backend._selection_report['priority_selected'], 2)

    def test_process_reports_full_render_coverage_separately_from_training(self):
        raw = self.backend.store.select(self.ids, include_optimizer=False)
        xyz = raw['xyz']
        gmap = dict(xyz=xyz.copy(), uv=(xyz @ self.K.T)[:, :2]/xyz[:, 2:],
                    color=raw['colors'].copy(), scale=np.exp(raw['log_scales']))
        observations = dict(source=self.image, target=self.image,
                            source_depth=self.depth, target_depth=self.depth,
                            source_confidence=np.ones_like(self.depth),
                            target_confidence=np.ones_like(self.depth))
        registry = SimpleNamespace(objects={0: dict(status='active', last_observed_frame=1)})
        self.backend.cfg['collect_refinement_stats'] = True
        report = self.backend.process(
            0, 1, gmap, dict(object_labels=np.zeros(6, np.int64),
                            object_probabilities=np.ones((6, 1))),
            np.array([0]), np.eye(4)[None], np.ones(1, bool), self.K,
            observations, registry, np.full(6, -1, np.int64),
            presence=dict(source=dict(observed_object_ids=[0]), target=dict(observed_object_ids=[0])))
        self.assertEqual(report['snapshot_scope'], 'full_visible_map')
        self.assertEqual(report['active_selection']['refinement_reserved'], 1)
        self.assertLessEqual(report['active_map_gaussians'], 4)
        self.assertGreater(report['snapshot_gaussians_per_frame']['0'], report['active_map_gaussians'])
        self.assertGreater(report['optimized_trainable_gaussians'], 0)


if __name__ == '__main__':
    unittest.main()

"""Real reference-renderer solves across persistent-map topology changes."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parents[1]
from mapping.gaussian_map_backend import GaussianMapBackend
from mapping.gaussian_map_optimizer import PARAMETER_WIDTHS, render_map_view
from mapping.persistent_gaussian_map import PersistentGaussianMap

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class GaussianMapRefinementBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.K = np.array([[45., 0., 20.5], [0., 45., 12.5], [0., 0., 1.]], np.float32)
        self.cfg = dict(gaussian_map=dict(
            enabled=True, window_frames=4, observation_stride=1, initial_scale_multiplier=1.,
            max_active_gaussians=12, max_samples_per_frame=20, max_new_gaussians_per_frame=0,
            checkpoint_interval=1, association_distance=.03, color_distance=.1,
            freeze_when_missing=True,
            optimization=dict(backend='torch_reference', downsample=1, collect_refinement_stats=True,
                stages=[dict(downsample=2, iterations=1), dict(downsample=1, iterations=2)],
                lr_xyz=0., lr_log_scales=0., lr_rotations=0., lr_opacity_logits=0., lr_colors=.002,
                min_optimization_alpha=.001, xyz_prior_weight=0., scale_prior_weight=0.),
            refinement=dict(enabled=True, warmup_pairs=0, interval=1, refine_iterations=3,
                min_observation_count=2, min_visible_steps=1, prune_min_observation_count=2,
                gradient_threshold=1e-12, split_radius_px=7., prune_opacity_threshold=.02,
                max_growth_fraction=1., min_points_per_object=0, max_new_per_update=3,
                max_splits_per_update=1, max_clones_per_update=1, max_prunes_per_update=1,
                max_children_per_object=3, max_total_gaussians=30)))
        self.backend = GaussianMapBackend(self.cfg, self.root)
        self.xyz = np.array([[-.9, 0., 3.], [-.45, 0., 3.], [0., 0., 3.],
                             [.45, 0., 3.], [.9, 0., 3.]], np.float32)
        self.scales = np.array([[.22, .16, .08], [.07, .06, .04], [.07, .06, .04],
                                [.22, .16, .08], [.07, .06, .04]], np.float32)
        self.colors = np.array([[.8, .25, .2], [.2, .8, .25], [.25, .2, .8],
                                [.7, .6, .2], [.2, .6, .7]], np.float32)
        for frame in range(4):
            result = self.backend.store.integrate(0, self.xyz, self.colors, scales=self.scales,
                opacity=np.array([.7, .7, .015, .7, .7]), frame_id=frame)
        self.background_ids = result['map_ids']
        self.car_xyz = np.array([[-.4, .55, 3.], [.4, .55, 3.]], np.float32)
        for frame in range(4):
            result = self.backend.store.integrate(1, self.car_xyz, [[.8, .2, .6], [.2, .8, .6]],
                scales=np.full((2, 3), .23), opacity=[.7, .015], frame_id=frame)
        self.car_ids = result['map_ids']
        self.backend.store.set_status(1, 'missing')
        self.truth = self.backend.store.select(np.r_[self.background_ids, self.car_ids])
        for ids in (self.background_ids, self.car_ids):
            block = self.backend.store.select(ids)
            self.backend.store.commit(ids, dict(colors=block['colors']*.6))
        # A historical view retains the missing object as a frozen occluder.
        old = self.observation(np.r_[self.background_ids, self.car_ids])
        self.backend._view(0, old['image'], old['depth'], old['confidence'], np.eye(4),
                           {0: np.eye(4), 1: np.eye(4)}, self.K)
        self.registry = SimpleNamespace(objects={0: dict(status='active', last_observed_frame=2),
                                                  1: dict(status='missing', last_observed_frame=0)})

    def observation(self, ids):
        rows = np.flatnonzero(np.isin(self.truth['map_ids'], ids))
        view = dict(frame=0, rows=rows, transforms=np.repeat(np.eye(4)[None], len(rows), 0),
                    K=self.K, image_shape=(25, 41))
        result = render_map_view(self.truth, view, dict(backend='torch_reference'))
        return dict(image=np.rint(np.clip(result['rgb'].numpy().transpose(1, 2, 0)[:, :, ::-1], 0, 1)*255).astype(np.uint8),
                    depth=result['depth'].numpy(), confidence=(result['alpha'].numpy() > .001).astype(np.float32))

    def process(self, extra=False):
        xyz, colors, scales = self.xyz.copy(), self.colors.copy(), self.scales.copy()
        if extra:
            xyz = np.r_[xyz, [[1.15, 0., 3.]]]
            colors = np.r_[colors, [[.2, .2, .2]]]
            scales = np.r_[scales, [[.07, .06, .04]]]
        uv = (xyz @ self.K.T)[:, :2]/xyz[:, 2:]
        gmap = dict(xyz=xyz, uv=uv, color=colors, scale=scales)
        labels = np.zeros(len(xyz), np.int64)
        posterior = dict(object_labels=labels, object_probabilities=np.ones((len(xyz), 1)))
        tracks = np.full(len(xyz), -1, np.int64)
        tracks[3] = 123  # Protect a large, high-gradient sparse landmark.
        observed = self.observation(self.background_ids)
        observations = dict(source=observed['image'], target=observed['image'].copy(),
            source_depth=observed['depth'], target_depth=observed['depth'].copy(),
            source_confidence=observed['confidence'], target_confidence=observed['confidence'].copy())
        return self.backend.process(1, 2, gmap, posterior, np.array([0]), np.eye(4)[None],
            np.array([True]), self.K, observations, self.registry, tracks,
            presence=dict(source=dict(observed_object_ids=[0]), target=dict(observed_object_ids=[0])))

    def test_split_clone_prune_retrain_children_and_checkpoint_revision(self):
        frozen = self.backend.store.select(self.car_ids)
        protected = int(self.background_ids[3])
        with patch.object(self.backend.store, 'append_from_parent', wraps=self.backend.store.append_from_parent) as append:
            with patch.object(self.backend.store, 'remove', wraps=self.backend.store.remove) as remove:
                report = self.process()
        refinement = report['refinement']
        self.assertTrue(refinement['changed'])
        self.assertEqual((refinement['split'], refinement['cloned'], refinement['pruned']), (1, 1, 1))
        self.assertEqual(refinement['inserted'], 3)
        self.assertEqual(refinement['removed'], 2)
        self.assertEqual(refinement['post_optimization']['iterations_completed'], 3)
        self.assertEqual(refinement['post_optimization']['downsample'], 1)
        self.assertEqual(append.call_count, 1)
        self.assertEqual(remove.call_count, 1)
        parent_ids = np.asarray(append.call_args.args[0])
        self.assertNotIn(protected, parent_ids)
        removed_ids = np.asarray(remove.call_args.args[0])
        self.assertNotIn(protected, removed_ids)
        self.assertIn(int(self.background_ids[2]), removed_ids)
        self.assertTrue(self.backend.store.is_live(np.array([protected]))[0])
        children = np.arange(7, 10, dtype=np.int64)
        learned = self.backend.store.select(children)
        self.assertTrue(np.all(learned['optimizer_state']['params']['colors']['row_steps'] == 3))
        self.assertTrue(np.any(learned['optimizer_state']['params']['colors']['exp_avg'] != 0))
        self.assertEqual(len(self.backend.store), 8)
        self.assertEqual(self.backend.store.next_id, 10)
        self.assertEqual(self.backend.store.revision, 2)
        self.assertEqual(self.backend._track_landmarks[123], (0, protected))
        self.assertLessEqual(report['active_map_gaussians'], self.backend.max_active)
        for name in PARAMETER_WIDTHS:
            after = self.backend.store.select(self.car_ids)
            np.testing.assert_array_equal(after[name], frozen[name])
            np.testing.assert_array_equal(after['optimizer_state']['params'][name]['row_steps'],
                                          frozen['optimizer_state']['params'][name]['row_steps'])
        loaded = PersistentGaussianMap.load(self.backend.output/'map')
        self.assertEqual(loaded.revision, self.backend.store.revision)
        self.assertEqual(loaded.next_id, 10)
        self.assertEqual(len(loaded), 8)
        for oid in self.backend.store.object_ids:
            actual, expected = loaded.export_object(oid, include_optimizer=True), self.backend.store.export_object(oid, include_optimizer=True)
            for name in ('map_ids', *PARAMETER_WIDTHS):
                np.testing.assert_array_equal(actual[name], expected[name])
            np.testing.assert_array_equal(actual['optimizer_state']['params']['colors']['row_steps'],
                                          expected['optimizer_state']['params']['colors']['row_steps'])
        for frame in (0, 1, 2):
            with np.load(self.backend.frames_dir/f'frame_{frame:010d}.npz', allow_pickle=False) as snapshot:
                self.assertEqual(int(snapshot['map_revision']), loaded.revision)
                self.assertFalse(np.any(np.isin(snapshot['map_ids'], removed_ids)))
                self.assertTrue(set(children) <= set(snapshot['map_ids']))
                if frame:
                    self.assertNotIn(1, snapshot['object_ids'])

    def test_global_insertion_and_active_growth_budgets_hold_with_real_solve(self):
        self.backend.max_new = 2
        self.backend.max_active = 7
        self.backend.cfg['refinement']['max_total_gaussians'] = 7
        report = self.process(extra=True)
        self.assertEqual(report['inserted_gaussians'], 0)
        self.assertEqual(report['insertion_budget_skipped'], 1)
        self.assertLessEqual(report['active_map_gaussians'], 7)
        self.assertLessEqual(len(self.backend.store), 7)
        self.assertLessEqual(report['refinement']['plan']['total_count_after'], 7)
        self.assertLessEqual(report['refinement']['plan']['active_count_after'], 7)

    def test_disabled_refinement_keeps_topology_and_legacy_adam_flow(self):
        self.backend.cfg['refinement']['enabled'] = False
        report = self.process()
        self.assertFalse(report['refinement']['changed'])
        self.assertEqual(self.backend.store.revision, 0)
        self.assertEqual(len(self.backend.store), 7)
        self.assertEqual(self.backend.store.next_id, 7)
        self.assertEqual(report['optimization']['iterations_completed'], 3)
        self.assertTrue(self.backend.store.is_live(self.background_ids).all())


if __name__ == '__main__':
    unittest.main()

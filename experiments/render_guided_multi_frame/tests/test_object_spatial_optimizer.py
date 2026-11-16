"""Spatial constraints must update shared logits, rather than filter exports."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('spatial_optimizer_object_window', HERE / 'ownership/object_window.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectWindow = _module.ObjectWindow


def plane(offset=1.5, valid=True):
    return dict(valid=valid, reason='fixed_measured_plane' if valid else 'insufficient_ground_support',
                normal=[0., -1., 0.], offset=offset, camera_height=offset,
                confidence=1. if valid else 0., fit_rmse_m=0.)


class SpatialObservationTests(unittest.TestCase):
    def snapshot(self, first, y, quality=1.):
        return dict(first=first, second=first+1, gaussian_ids=np.array([7]),
                    payload=dict(xyz=np.array([[0., y, 5.]]), uv=np.array([[4., 4.]])),
                    render_observations=dict(source=np.zeros((8, 8, 3), np.uint8),
                                             source_confidence=np.full((8, 8), quality)))

    def test_shared_height_uses_each_camera_plane_and_reuses_cached_measurements(self):
        from ownership.object_spatial_constraints import prepare_spatial_constraints
        snapshots = {(0, 1): self.snapshot(0, 0., .5),
                     (1, 2): self.snapshot(1, .3, 1.)}
        window = SimpleNamespace(_active_gids=np.array([7]), _active=snapshots)
        cfg = dict(height=dict(enabled=True, soft_margin_m=.3))
        with patch('ownership.object_height_constraints.estimate_ground_plane', side_effect=[plane(1.5), plane(1.8)]) as fit:
            result = prepare_spatial_constraints(window, cfg)
            second = prepare_spatial_constraints(window, cfg)
        self.assertEqual(fit.call_count, 2)
        np.testing.assert_allclose(result['height_info']['height'], [1.5])
        np.testing.assert_array_equal(result['height_cost'], [0.])
        self.assertGreater(result['height_reliability'][0], 0)
        np.testing.assert_array_equal(second['height_reliability'], result['height_reliability'])
        self.assertEqual(result['report']['height']['valid_planes'], 2)
        for snap in snapshots.values():
            self.assertTrue(bool(snap['payload']['ground_plane_valid']))
            self.assertTrue(json.loads(str(snap['payload']['ground_plane_report_json']))['valid'])

    def test_invalid_ground_plane_yields_no_spatial_height_evidence(self):
        from ownership.object_spatial_constraints import prepare_spatial_constraints
        window = SimpleNamespace(_active_gids=np.array([7]), _active={(0, 1): self.snapshot(0, -10.)})
        with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane(0., False)):
            result = prepare_spatial_constraints(window, dict(height=dict(enabled=True)))
        np.testing.assert_array_equal(result['height_cost'], [0.])
        np.testing.assert_array_equal(result['height_reliability'], [0.])
        self.assertEqual(result['report']['height']['valid_planes'], 0)
        self.assertEqual(result['report']['height']['planes']['0']['reason'], 'insufficient_ground_support')

    def test_backend_measured_plane_payload_is_reused_before_refitting(self):
        from ownership.object_spatial_constraints import prepare_spatial_constraints
        snap = self.snapshot(0, -4.)
        measured = plane()
        snap['payload'].update(ground_plane_normal=np.array(measured['normal']),
                               ground_plane_offset=np.asarray(measured['offset']),
                               ground_plane_confidence=np.asarray(measured['confidence']),
                               ground_plane_valid=np.asarray(True),
                               ground_plane_report_json=np.asarray(json.dumps(measured)))
        window = SimpleNamespace(_active_gids=np.array([7]), _active={(0, 1): snap})
        with patch('ownership.object_height_constraints.estimate_ground_plane', side_effect=AssertionError('Already measured')):
            result = prepare_spatial_constraints(window, dict(height=dict(enabled=True)))
        self.assertGreater(result['height_cost'][0], 0.)
        self.assertEqual(result['report']['height']['valid_planes'], 1)

    def test_hard_height_uses_any_observation_without_majority_or_quality_gate(self):
        from ownership.object_spatial_constraints import prepare_spatial_constraints
        snapshots = {(0, 1): self.snapshot(0, -.6, quality=0.),
                     (1, 2): self.snapshot(1, 0.), (2, 3): self.snapshot(2, .1)}
        window = SimpleNamespace(_active_gids=np.array([7]), _active=snapshots)
        cfg = dict(height=dict(enabled=True, above_camera_m=.5, hard_upper_bound=True))
        with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane()):
            result = prepare_spatial_constraints(window, cfg)
        np.testing.assert_array_equal(result['hard_background_mask'], [True])
        np.testing.assert_array_equal(result['height_info']['allowed'], [False])
        self.assertEqual(result['report']['height']['hard_background_gaussians'], 1)

    def test_hard_height_writes_o0_probabilities_labels_and_tracks_without_fit_support(self):
        cfg = dict(height=dict(enabled=True, above_camera_m=.5, hard_upper_bound=True))
        window = ObjectWindow(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / str(i) for i in range(2)]
            for first, y in enumerate([0., -.6]):
                snap = self.snapshot(first, y)
                with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane()):
                    state = window.update(first, first+1, np.array([7]), np.array([3, 0]),
                                          np.array([[.99, .01]]), np.full((1, 2), np.nan),
                                          payload=snap['payload'], render_observations=snap['render_observations'],
                                          track_ids=np.array([9]), output_dir=outputs[first])
            np.testing.assert_array_equal(state['object_probabilities'], [[0., 1.]])
            np.testing.assert_array_equal(state['object_labels'], [0])
            for output in outputs:
                with np.load(output / 'object_state.npz') as saved:
                    np.testing.assert_array_equal(saved['object_probabilities'], [[0., 1.]])
                    np.testing.assert_array_equal(saved['object_labels'], [0])
                    np.testing.assert_array_equal(saved['object_track_labels'], [0])
                    np.testing.assert_array_equal(saved['object_track_probabilities'], [[0., 1.]])
                    np.testing.assert_array_equal(saved['height_forced_background'], [True])
                    np.testing.assert_array_equal(saved['q_object_initial'], [[.99, .01]])

    def test_missing_core_fallback_rewrites_shared_state_and_preserves_supported_objects(self):
        cfg = dict(sequence=dict(streaming=True, frames=[0, 1, 2]), object_window=dict(
            window_frames=7, window_step=5, connectivity=dict(enabled=True,
            no_core_to_background=True, core_min_frames=2, core_untracked_min_frames=2, core_min_points=3)))
        window = ObjectWindow(cfg)
        uv = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [11, 0]], float)
        payload = dict(xyz=np.column_stack((uv*.1, np.full(5, 5.))), uv=uv)
        q = np.tile([.01, .09, .9], (5, 1))
        costs = np.array([[.9, .05, .9]]*3+[[.9, .9, .05]]*2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for first in (0, 1):
                state = window.update(first, first+1, np.arange(5)+10, np.array([0, 1, 2]), q, costs,
                                      payload=payload, track_ids=np.arange(5)+500, output_dir=root / str(first))
            self.assertTrue((state['object_labels'][:3] == 1).all())
            np.testing.assert_array_equal(state['object_probabilities'][3:], [[1., 0., 0.]]*2)
            np.testing.assert_array_equal(state['object_labels'][3:], [0, 0])
            np.testing.assert_array_equal(state['object_probabilities'][:, 2], 0.)
            for first in (0, 1):
                with np.load(root / str(first) / 'object_state.npz') as saved:
                    np.testing.assert_array_equal(saved['object_probabilities'], state['object_probabilities'])
                    np.testing.assert_array_equal(saved['connectivity_forced_background'], [False]*3+[True]*2)
                    np.testing.assert_array_equal(saved['height_forced_background'], False)
                    np.testing.assert_array_equal(saved['connectivity_disabled_object_ids'], [2])
                    np.testing.assert_array_equal(saved['object_track_labels'], [1, 1, 1, 0, 0])
                    np.testing.assert_array_equal(saved['q_object_initial'], q)
                    self.assertTrue(np.isfinite(saved['object_logits']).all())


@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA raster evaluation is required')
class ObjectSpatialOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diff_gaussian_rasterization import GaussianRasterizer
        from ownership.object_render_optimizer import render_gaussian_view
        from utils.recon_helpers import setup_camera
        cls.raster_class = GaussianRasterizer
        cls.render_view = staticmethod(render_gaussian_view)
        cls.setup_camera = staticmethod(setup_camera)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.h = self.w = 48
        self.K = np.array([[20., 0., 24.], [0., 20., 24.], [0., 0., 1.]])
        self.raster = self.raster_class(raster_settings=self.setup_camera(
            self.w, self.h, self.K, np.eye(4), depth_threshold=1.))

    def config(self):
        return dict(object_window=dict(
            enabled=True, window_frames=5, prior_floor=1e-12,
            render=dict(enabled=True, iterations=50, lr=.18, downsample=1, verbose=False,
                        gaussian_opacity=.7, rgb_weight=0., depth_weight=0., coverage_weight=0.,
                        centre_weight=0., prior_weight=0., entropy_weight=0.)))

    def geometry(self, xyz, uv=None):
        xyz = np.asarray(xyz, float)
        if uv is None:
            uv = xyz[:, :2]/xyz[:, 2:]*self.K.diagonal()[:2]+self.K[:2, 2]
        models = np.repeat(np.eye(4)[None], 2, axis=0)
        models[1, 0, 3] = .2
        return dict(xyz=xyz, uv=np.asarray(uv, float), color=np.tile([.8, .4, .2], (len(xyz), 1)),
                    scale=np.full((len(xyz), 3), .15), K=self.K.copy(), H_object=models,
                    object_model_valid=np.ones(2, bool))

    def observations(self, gmap):
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device='cuda').contiguous()
        q = np.tile([1., 0.], (len(gmap['xyz']), 1))
        with torch.no_grad():
            rendered = self.render_view(
                torch, self.raster, tensor(gmap['xyz']), tensor(gmap['color']), tensor(gmap['scale']),
                tensor(q), tensor(gmap['H_object']), opacity=.7)
        rgb = rendered['rgb'].permute(1, 2, 0).cpu().numpy()
        bgr = np.ascontiguousarray(np.rint(np.clip(rgb[:, :, ::-1], 0, 1)*255).astype(np.uint8))
        depth = np.full((self.h, self.w), gmap['xyz'][0, 2], np.float32)
        confidence = np.ones((self.h, self.w), np.float32)
        return dict(source=bgr, target=bgr.copy(), source_depth=depth,
                    target_depth=depth.copy(), source_confidence=confidence,
                    target_confidence=confidence.copy())

    def update(self, window, first, gmap, q, costs, *, order=None, tracked=False):
        n = len(gmap['xyz'])
        order = np.arange(n) if order is None else np.asarray(order)
        payload = {key: (value[order].copy() if key in {'xyz', 'uv', 'color', 'scale'} else value.copy())
                   for key, value in gmap.items()}
        tracks = np.r_[np.arange(3)+500, np.full(n-3, -1)] if tracked else np.full(n, -1)
        motion = np.arange(1, n*3+1, dtype=np.float32).reshape(n, 3)
        motion /= motion.sum(axis=1, keepdims=True)
        return window.update(
            first, first+1, (np.arange(n)+100)[order], np.array([0, 1]),
            np.asarray(q)[order], np.asarray(costs)[order], payload=payload,
            track_ids=tracks[order], q_motion_initial=motion[order],
            output_dir=self.output / str(first), render_observations=self.observations(gmap))

    def test_height_loss_changes_only_outside_nonbackground_mass_and_exempts_background(self):
        cfg = self.config()
        cfg['object_window']['height'] = dict(enabled=True, weight=1., ground_min_m=-.2,
                                               above_camera_m=2., soft_margin_m=.3)
        window = ObjectWindow(cfg)
        gmap = self.geometry([[0., -4., 8.], [.25, 0., 8.], [.6, -4., 8.]])
        q = np.array([[.1, .9], [.2, .8], [1., 0.]])
        original_geometry = {key: value.copy() for key, value in gmap.items()}
        with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane()):
            state = self.update(window, 0, gmap, q, np.full((3, 2), .5))
        report = state['report']['render_optimization']
        self.assertGreater(report['nonzero_gradient_steps'], 0)
        self.assertGreater(report['render_passes'], 0)
        self.assertLess(report['final_losses']['height'], report['initial_losses']['height']*.2)
        self.assertLess(state['object_probabilities'][0, 1], .2)
        np.testing.assert_allclose(state['object_probabilities'][1], q[1], atol=1e-6)
        np.testing.assert_allclose(state['object_probabilities'][2], q[2], atol=1e-8)
        self.assertEqual(report['spatial_constraints']['height']['outside_band_gaussians'], 2)
        self.assertEqual(report['spatial_constraints']['height']['background_object_id'], 0)
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            np.testing.assert_array_equal(saved['q_object_initial'], q)
        for key, value in original_geometry.items():
            np.testing.assert_array_equal(gmap[key], value)

    def test_hard_height_blocks_foreground_render_floor_and_invalid_background_fallback(self):
        import ownership.object_render_optimizer as render_module
        cfg = self.config()
        cfg['object_window']['render'].update(iterations=2, rgb_weight=1., min_branch_probability=.1)
        cfg['object_window']['height'] = dict(enabled=True, weight=0., above_camera_m=.5,
                                               hard_upper_bound=True)
        original = render_module.render_gaussian_view
        for background_valid in (True, False):
            with self.subTest(background_valid=background_valid):
                window = ObjectWindow(cfg)
                gmap = self.geometry([[0., -.6, 8.], [.25, -.5, 8.]])
                gmap['object_model_valid'][0] = background_valid
                rendered_probabilities = []

                def capture(*args, **kwargs):
                    rendered_probabilities.append(args[5].detach().cpu().numpy())
                    return original(*args, **kwargs)

                with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane()), \
                        patch.object(render_module, 'render_gaussian_view', side_effect=capture):
                    state = self.update(window, 0, gmap, [[.01, .99], [.2, .8]], np.full((2, 2), np.nan))
                np.testing.assert_array_equal(state['object_probabilities'][0], [1., 0.])
                self.assertEqual(state['object_labels'][0], 0)
                self.assertGreater(state['object_probabilities'][1, 1], 0.)
                self.assertGreater(len(rendered_probabilities), 0)
                for probabilities in rendered_probabilities:
                    np.testing.assert_array_equal(probabilities[0], [1., 0.] if background_valid else [0.])

    def test_invalid_plane_does_not_force_foreground_to_background(self):
        cfg = self.config()
        cfg['object_window']['render'].update(iterations=1)
        cfg['object_window']['height'] = dict(enabled=True, above_camera_m=.5, hard_upper_bound=True)
        window = ObjectWindow(cfg)
        gmap = self.geometry([[0., -4., 8.]])
        with patch('ownership.object_height_constraints.estimate_ground_plane', return_value=plane(valid=False)):
            state = self.update(window, 0, gmap, [[.01, .99]], [[.9, .1]])
        np.testing.assert_allclose(state['object_probabilities'], [[.01, .99]], atol=1e-6)
        self.assertEqual(state['object_labels'][0], 1)

    def test_no_core_fallback_blocks_render_exploration_with_or_without_a_background_pose(self):
        import ownership.object_render_optimizer as render_module
        cfg = self.config()
        cfg['object_window']['render'].update(iterations=2, rgb_weight=1., min_branch_probability=.1)
        cfg['object_window']['connectivity'] = dict(enabled=True, weight=0., seed_weight=0.,
                                                    no_core_to_background=True)
        original = render_module.render_gaussian_view
        for background_valid in (True, False):
            with self.subTest(background_valid=background_valid):
                window = ObjectWindow(cfg)
                gmap = self.geometry([[0., 0., 8.], [.25, 0., 8.]])
                gmap['object_model_valid'][0] = background_valid
                rendered_probabilities = []

                def capture(*args, **kwargs):
                    rendered_probabilities.append(args[5].detach().cpu().numpy())
                    return original(*args, **kwargs)

                with patch.object(render_module, 'render_gaussian_view', side_effect=capture):
                    state = self.update(window, 0, gmap, [[.01, .99]]*2, [[.9, .05]]*2)
                np.testing.assert_array_equal(state['object_probabilities'], [[1., 0.]]*2)
                np.testing.assert_array_equal(state['object_labels'], [0, 0])
                self.assertGreater(len(rendered_probabilities), 0)
                for probabilities in rendered_probabilities:
                    np.testing.assert_array_equal(probabilities, [[1., 0.]]*2 if background_valid else [[0.]]*2)
                with np.load(self.output / '0/object_state.npz') as saved:
                    np.testing.assert_array_equal(saved['connectivity_forced_background'], True)
                    np.testing.assert_array_equal(saved['height_forced_background'], False)

    def test_region_completion_trains_uncertain_body_and_saves_per_frame_targets(self):
        cfg = self.config()
        cfg['object_window']['connectivity'] = dict(
            enabled=True, weight=0., seed_weight=0., core_min_frames=2, core_min_points=3)
        cfg['object_window']['region_completion'] = dict(enabled=True, weight=1.)
        window = ObjectWindow(cfg)
        uv = np.column_stack((np.arange(10, 20), np.full(10, 16)))
        xyz = np.column_stack(((uv-self.K[:2, 2])*5/self.K.diagonal()[:2], np.full(10, 5.)))
        gmap = self.geometry(xyz, uv)
        costs = np.array([[.9, .05]]*3+[[.5, .45]]*5+[[.05, .9]]*2)
        q = np.array([[.2, .8]]*3+[[.8, .2]]*7)
        self.update(window, 0, gmap, q, costs, tracked=True)
        self.update(window, 1, gmap, q, costs, tracked=True)
        state = window.get_state(np.arange(10)+100)
        report = state['report']['render_optimization']
        self.assertGreater(report['weights']['region'], 0)
        self.assertGreater(report['initial_losses']['region'], report['final_losses']['region'])
        self.assertTrue(np.all(state['object_probabilities'][3:8, 1] > .9))
        self.assertTrue(np.all(state['object_probabilities'][8:, 1] < .3))
        self.assertEqual(report['spatial_constraints']['connectivity']['graph']['objects']['1']['core_count'], 3)
        with np.load(self.output/'1/object_state.npz') as saved:
            np.testing.assert_array_equal(saved['region_completion_gaussian_ids'], np.arange(100, 108))
            np.testing.assert_array_equal(saved['region_completion_object_ids'], 1)
            np.testing.assert_array_equal(saved['q_object_initial'], q)

    def test_connectivity_loss_keeps_reliable_core_and_removes_larger_island_with_shared_history(self):
        cfg = self.config()
        cfg['object_window']['connectivity'] = dict(
            enabled=True, weight=1., seed_weight=.1, update_interval=5,
            core_min_frames=2, core_min_points=3, edge_distance_m=.3)
        window = ObjectWindow(cfg)
        x = np.r_[np.arange(8, 11), np.arange(28, 40)]
        uv = np.column_stack((x, np.full(len(x), 16)))
        xyz = np.column_stack(((uv-self.K[:2, 2])*5/self.K.diagonal()[:2], np.full(len(x), 5.)))
        gmap = self.geometry(xyz, uv)
        n = len(x)
        costs = np.vstack((np.tile([1., .05], (3, 1)), np.tile([.05, .9], (n-3, 1))))
        q = np.tile([.01, .99], (n, 1))
        self.update(window, 0, gmap, q, costs, tracked=True)
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            old_q = saved['q_motion_initial'].copy()
            before = saved['object_probabilities'].copy()
        self.update(window, 1, gmap, q, costs, order=np.arange(n)[::-1], tracked=True)
        state = window.get_state(np.arange(n)+100)
        report = state['report']['render_optimization']
        probability = state['object_probabilities']
        self.assertTrue(np.all(probability[:3, 1] > .9))
        self.assertTrue(np.all(probability[3:, 1] < .35))
        self.assertGreater(report['nonzero_gradient_steps'], 0)
        self.assertGreater(report['render_passes'], 0)
        self.assertEqual(report['spatial_constraints']['connectivity']['graph']['objects']['1']['core_count'], 3)
        self.assertEqual(report['loss_comparison'], 'same_final_detached_connectivity_targets')
        self.assertGreater(report['initial_losses_before_target_refresh']['connectivity'], 0)
        # Once every island row has crossed below the high-probability gate,
        # final targets are empty. Both ends must then have zero connectivity
        # loss under those SAME targets; target removal is not credited as a
        # reduction of a fixed objective.
        self.assertEqual(report['spatial_constraints']['connectivity']['final_targets']['penalized_entries'], 0)
        self.assertAlmostEqual(report['initial_losses']['connectivity'], 0.)
        self.assertAlmostEqual(report['final_losses']['connectivity'], 0.)
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            np.testing.assert_array_equal(saved['q_motion_initial'], old_q)
            self.assertFalse(np.allclose(saved['object_probabilities'], before))
            np.testing.assert_allclose(saved['object_probabilities'], probability)
            np.testing.assert_array_equal(saved['object_labels'], state['object_labels'])


if __name__ == '__main__':
    unittest.main()

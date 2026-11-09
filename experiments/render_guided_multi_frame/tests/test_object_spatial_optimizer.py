"""Spatial constraints must update shared logits, rather than filter exports."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location('spatial_optimizer_object_window', HERE / 'object_window.py')
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
        from object_spatial_constraints import prepare_spatial_constraints
        snapshots = {(0, 1): self.snapshot(0, 0., .5),
                     (1, 2): self.snapshot(1, .3, 1.)}
        window = SimpleNamespace(_active_gids=np.array([7]), _active=snapshots)
        cfg = dict(height=dict(enabled=True, soft_margin_m=.3))
        with patch('object_height_constraints.estimate_ground_plane', side_effect=[plane(1.5), plane(1.8)]) as fit:
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
        from object_spatial_constraints import prepare_spatial_constraints
        window = SimpleNamespace(_active_gids=np.array([7]), _active={(0, 1): self.snapshot(0, -10.)})
        with patch('object_height_constraints.estimate_ground_plane', return_value=plane(0., False)):
            result = prepare_spatial_constraints(window, dict(height=dict(enabled=True)))
        np.testing.assert_array_equal(result['height_cost'], [0.])
        np.testing.assert_array_equal(result['height_reliability'], [0.])
        self.assertEqual(result['report']['height']['valid_planes'], 0)
        self.assertEqual(result['report']['height']['planes']['0']['reason'], 'insufficient_ground_support')

    def test_backend_measured_plane_payload_is_reused_before_refitting(self):
        from object_spatial_constraints import prepare_spatial_constraints
        snap = self.snapshot(0, -4.)
        measured = plane()
        snap['payload'].update(ground_plane_normal=np.array(measured['normal']),
                               ground_plane_offset=np.asarray(measured['offset']),
                               ground_plane_confidence=np.asarray(measured['confidence']),
                               ground_plane_valid=np.asarray(True),
                               ground_plane_report_json=np.asarray(json.dumps(measured)))
        window = SimpleNamespace(_active_gids=np.array([7]), _active={(0, 1): snap})
        with patch('object_height_constraints.estimate_ground_plane', side_effect=AssertionError('Already measured')):
            result = prepare_spatial_constraints(window, dict(height=dict(enabled=True)))
        self.assertGreater(result['height_cost'][0], 0.)
        self.assertEqual(result['report']['height']['valid_planes'], 1)


@unittest.skipUnless(torch.cuda.is_available(), 'Actual CUDA raster evaluation is required')
class ObjectSpatialOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diff_gaussian_rasterization import GaussianRasterizer
        from object_render_optimizer import render_gaussian_view
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
        with patch('object_height_constraints.estimate_ground_plane', return_value=plane()):
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

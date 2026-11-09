"""Real CUDA regression checks for shared object logits and image rendering.

These tests deliberately disable centre costs and ownership regularization.
They therefore fail if object updates do not receive actual RGB/depth raster
gradients. CPU-only environments skip them; no mock renderer is substituted.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("render_test_object_window", HERE / "object_window.py")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectWindow = _module.ObjectWindow


class ObjectRenderWeightTests(unittest.TestCase):
    def test_bidirectional_views_give_every_frame_the_same_total_weight(self):
        from object_render_optimizer import build_render_views
        models = np.repeat(np.eye(4)[None], 2, axis=0)
        observations = dict(source=np.zeros((2, 2, 3), np.uint8),
                            target=np.zeros((2, 2, 3), np.uint8),
                            source_depth=np.ones((2, 2)), target_depth=np.ones((2, 2)),
                            source_confidence=np.ones((2, 2)),
                            target_confidence=np.ones((2, 2)))
        snapshots = {}
        for first in range(5):
            snapshots[(first, first+1)] = dict(
                first=first, second=first+1, gaussian_ids=np.array([7]),
                object_ids=np.array([0, 1]), render_observations=observations,
                payload=dict(H_object=models, object_model_valid=np.ones(2, bool), K=np.eye(3)))
        window = SimpleNamespace(object_ids=[0, 1], _active_gids=np.array([7]), _active=snapshots)
        views = build_render_views(window)
        self.assertEqual(sum(view['direction'] == 'forward' for view in views), 5)
        self.assertEqual(sum(view['direction'] == 'backward' for view in views), 4)
        frame_weights = {}
        for view in views:
            frame_weights[view['frame']] = frame_weights.get(view['frame'], 0.) + view['frame_weight']
        self.assertEqual(set(frame_weights), set(range(6)))
        for weight in frame_weights.values():
            self.assertAlmostEqual(weight, 1/6)
        self.assertAlmostEqual(sum(frame_weights.values()), 1.)
        window._active = dict(reversed(list(snapshots.items())))
        reversed_views = build_render_views(window)
        self.assertEqual([(view['frame'], view['direction'], view['frame_weight']) for view in views],
                         [(view['frame'], view['direction'], view['frame_weight']) for view in reversed_views])


@unittest.skipUnless(torch.cuda.is_available(), "Actual differentiable Gaussian rasterization needs CUDA")
class ObjectRenderOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diff_gaussian_rasterization import GaussianRasterizer
        from object_render_optimizer import render_gaussian_view, optimize_render_window
        from utils.recon_helpers import setup_camera
        cls.raster_class = GaussianRasterizer
        cls.render_view = staticmethod(render_gaussian_view)
        cls.optimize_window = staticmethod(optimize_render_window)
        cls.setup_camera = staticmethod(setup_camera)

    def setUp(self):
        torch.manual_seed(17)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.h = self.w = 48
        self.K = np.array([[48., 0., 24.], [0., 48., 24.], [0., 0., 1.]])
        self.raster = self.raster_class(raster_settings=self.setup_camera(
            self.w, self.h, self.K, np.eye(4), depth_threshold=1.))

    def configuration(self, *, depth_only=False, window_frames=6, iterations=45):
        return dict(object_window=dict(
            enabled=True, window_frames=window_frames, prior_weight=.05,
            temperature=.25, min_probability=.6, min_margin=.1,
            render=dict(enabled=True, iterations=iterations, lr=.18, downsample=1,
                        rgb_weight=0. if depth_only else 1.,
                        depth_weight=1. if depth_only else 0., coverage_weight=0.,
                        centre_weight=0., prior_weight=0., entropy_weight=0.,
                        gaussian_opacity=.7)))

    def scene(self, *, depth_only=False):
        xyz = (np.array([[0., 0., 3.]]) if depth_only else
               np.array([[-.18, -.15, 3.], [.18, -.15, 3.],
                         [-.18, .15, 3.], [.18, .15, 3.]]))
        n = len(xyz)
        models = np.repeat(np.eye(4)[None], 2, axis=0)
        models[1, 2 if depth_only else 0, 3] = 1.2 if depth_only else .65
        uv = xyz[:, :2]/xyz[:, 2:] * self.K.diagonal()[:2] + self.K[:2, 2]
        return dict(xyz=xyz, uv=uv, color=np.tile([.8, .35, .15], (n, 1)),
                    scale=np.full((n, 3), .22 if depth_only else .11), K=self.K.copy(),
                    H_object=models, object_model_valid=np.ones(2, bool))

    def observation(self, scene, winner):
        n = len(scene['xyz'])
        q = np.zeros((n, 2), np.float32)
        q[:, winner] = 1.
        tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device='cuda').contiguous()
        with torch.no_grad():
            target = self.render_view(
                torch, self.raster, tensor(scene['xyz']), tensor(scene['color']),
                tensor(scene['scale']), tensor(q), tensor(scene['H_object']), opacity=.7)
            source = self.render_view(
                torch, self.raster, tensor(scene['xyz']), tensor(scene['color']),
                tensor(scene['scale']), tensor(q), tensor(np.repeat(np.eye(4)[None], 2, axis=0)),
                opacity=.7)

        def arrays(view):
            rgb = view['rgb'].detach().permute(1, 2, 0).cpu().numpy()
            alpha = view['alpha'].detach().cpu().numpy()
            depth = view['depth'].detach().cpu().numpy()
            valid = alpha > .02
            bgr = np.rint(np.clip(rgb[:, :, ::-1], 0, 1)*255).astype(np.uint8)
            return np.ascontiguousarray(bgr), np.where(valid, depth, np.nan), valid.astype(np.float32)

        source_rgb, source_depth, source_confidence = arrays(source)
        target_rgb, target_depth, target_confidence = arrays(target)
        return dict(source=source_rgb, target=target_rgb, source_depth=source_depth,
                    target_depth=target_depth, source_confidence=source_confidence,
                    target_confidence=target_confidence)

    def update(self, window, first, scene, observation, q_initial, *, order=None):
        n = len(scene['xyz'])
        order = np.arange(n) if order is None else np.asarray(order)
        gaussian_ids = np.array([30, 7, 90, 11][:n], np.int64)[order]
        payload = {key: (value[order].copy() if key in {'xyz', 'uv', 'color', 'scale'} else value.copy())
                   for key, value in scene.items()}
        # The frozen local motion matrix intentionally has a different number
        # of columns from the persistent object attribute.
        motion_q = np.tile([.2, .3, .5], (n, 1)).astype(np.float32)[order]
        return window.update(
            first, first+1, gaussian_ids, np.array([0, 1], np.int64),
            np.asarray(q_initial, float)[order], np.full((n, 2), .5),
            output_dir=self.output / str(first), payload=payload,
            q_motion_initial=motion_q, render_observations=observation)

    def render_report(self, state):
        report = state['report']['render_optimization']
        for stage in ('initial_losses', 'final_losses'):
            self.assertTrue(np.isfinite(report[stage]['total']))
        return report

    def test_rgb_render_loss_corrects_objects_without_updating_old_motion_q(self):
        scene = self.scene()
        n = len(scene['xyz'])
        window = ObjectWindow(self.configuration())
        original_geometry = {key: value.copy() for key, value in scene.items()}
        state = self.update(window, 0, scene, self.observation(scene, 0),
                            np.tile([.1, .9], (n, 1)))
        report = self.render_report(state)
        self.assertLess(report['final_losses']['rgb'], report['initial_losses']['rgb']*.65)
        self.assertGreater(report['nonzero_gradient_steps'], 0)
        self.assertTrue(np.isfinite(state['object_probabilities']).all())
        self.assertTrue(np.all(state['object_probabilities'][:, 0] > .75))
        self.assertTrue(np.all(state['object_labels'] == 0))
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            np.testing.assert_array_equal(saved['q_motion_initial'], np.tile([.2, .3, .5], (n, 1)).astype(np.float32))
        for key, original in original_geometry.items():
            np.testing.assert_array_equal(scene[key], original)

    def test_depth_color_pass_alone_supplies_object_logit_gradients(self):
        scene = self.scene(depth_only=True)
        window = ObjectWindow(self.configuration(depth_only=True))
        state = self.update(window, 0, scene, self.observation(scene, 0), [[.1, .9]])
        report = self.render_report(state)
        self.assertLess(report['final_losses']['depth'], report['initial_losses']['depth']*.6)
        self.assertGreater(report['nonzero_gradient_steps'], 0)
        self.assertGreater(state['object_probabilities'][0, 0], .75)

    def test_probability_below_native_alpha_cutoff_recovers_with_soft_branch_floor(self):
        scene = self.scene()
        scene['H_object'][1, 0, 3] = 10.  # The wrong branch is entirely outside the image.
        n = len(scene['xyz'])
        cfg = self.configuration(iterations=60)
        window = ObjectWindow(cfg)
        # .005 * .7 < 1/255: without the differentiable branch floor the
        # correct branch contributes no pixels and receives no RGB gradient.
        self.assertLess(.005*.7, 1/255)
        state = self.update(window, 0, scene, self.observation(scene, 0),
                            np.tile([.005, .995], (n, 1)))
        report = self.render_report(state)
        self.assertGreater(report['nonzero_gradient_steps'], 0)
        self.assertLess(report['final_losses']['rgb'], report['initial_losses']['rgb']*.65)
        self.assertTrue(np.all(state['object_probabilities'][:, 0] > .75))

    def test_shared_ids_rewrite_earlier_snapshot_without_reinitializing_or_changing_q(self):
        scene = self.scene()
        n = len(scene['xyz'])
        window = ObjectWindow(self.configuration())
        unobserved = self.observation(scene, 0)
        for key in ('source_confidence', 'target_confidence'):
            unobserved[key] = np.zeros_like(unobserved[key])
        first = self.update(window, 0, scene, unobserved, np.tile([.1, .9], (n, 1)))
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            old_q = saved['q_motion_initial'].copy()
            initial_attributes = saved['object_probabilities'].copy()
        second = self.update(window, 1, scene, self.observation(scene, 0),
                             np.tile([.01, .99], (n, 1)), order=[2, 0, 3, 1])
        self.assertTrue(np.all(second['object_probabilities'][:, 0] > .75))
        with np.load(self.output / '0' / 'object_state.npz') as saved:
            np.testing.assert_array_equal(saved['q_motion_initial'], old_q)
            self.assertFalse(np.allclose(saved['object_probabilities'], initial_attributes))
            lookup = {int(gid): row for row, gid in enumerate(second['gaussian_ids'])}
            matching = [lookup[int(gid)] for gid in saved['gaussian_ids']]
            np.testing.assert_allclose(saved['object_probabilities'], second['object_probabilities'][matching])
            np.testing.assert_array_equal(saved['object_labels'], second['object_labels'][matching])
            np.testing.assert_array_equal(saved['window_frames'], [0, 1, 2])
        np.testing.assert_array_equal(first['gaussian_ids'], [30, 7, 90, 11])

    def test_four_historical_views_outweigh_one_bad_latest_view_with_equal_frame_weights(self):
        scene = self.scene()
        n = len(scene['xyz'])
        cfg = self.configuration(window_frames=10, iterations=35)
        window = ObjectWindow(cfg)
        for first in (0, 2, 4, 6, 8):
            last = first == 8
            state = self.update(window, first, scene, self.observation(scene, 0 if last else 1),
                                np.tile([.99, .01] if last else [.2, .8], (n, 1)))
        self.assertTrue(np.all(state['object_probabilities'][:, 1] > .7))
        self.assertTrue(np.all(state['object_labels'] == 1))
        # Accumulate an identical objective in opposite snapshot order, from
        # identical logits. A per-view optimizer step or extra newest-frame
        # coefficient would make the two results disagree.
        reference, reversed_window = deepcopy(window), deepcopy(window)
        initial = np.full_like(window._render_initial_probabilities, .5)
        reference._render_initial_probabilities = initial.copy()
        reversed_window._render_initial_probabilities = initial.copy()
        reversed_window._active = dict(reversed(list(reversed_window._active.items())))
        left = self.optimize_window(reference, cfg['object_window'])
        right = self.optimize_window(reversed_window, cfg['object_window'])
        np.testing.assert_allclose(left['probabilities'], right['probabilities'], atol=5e-4, rtol=5e-4)
        self.assertTrue(np.all(np.asarray(left['probabilities'])[:, 1] > .7))


if __name__ == '__main__':
    unittest.main()

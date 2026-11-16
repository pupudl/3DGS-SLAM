"""Meaningful numerical and multi-view checks for persistent map optimization."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import copy
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np


PATH = Path(__file__).resolve().parents[1] / 'mapping/gaussian_map_optimizer.py'
SPEC = importlib.util.spec_from_file_location('gaussian_map_optimizer', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
try:
    import torch
except ImportError:
    torch = None


def batch_fixture():
    return dict(map_ids=np.array([40, 70, 90], np.int64),
                xyz=np.array([[-.19, -.09, 2.2], [.16, .11, 2.65], [.1, -.15, 3.3]], np.float32),
                colors=np.array([[.8, .3, .2], [.2, .7, .9], [.8, .8, .1]], np.float32),
                log_scales=np.log(np.array([[.15, .065, .09], [.08, .16, .06], [.1, .05, .09]], np.float32)),
                rotations=np.array([[.94, .1, .2, .25], [.97, -.11, .2, .04], [1., 0., 0., 0.]], np.float32),
                opacity_logits=np.array([[1.4], [1.1], [.9]], np.float32))


def view_fixture(rows=None, frame=0, dx=0., shape=(17, 19)):
    rows = np.array([0, 1, 2] if rows is None else rows, np.int64)
    transform = np.repeat(np.eye(4, dtype=np.float32)[None], len(rows), 0)
    transform[:, 0, 3] = dx
    # Per-owner transforms really differ, rather than duplicating every point
    # into every object model as in soft object assignment optimization.
    if len(rows) > 1:
        transform[1, 1, 3] = .03
    h, w = shape
    return dict(frame=frame, rows=rows, transforms=transform,
                K=np.array([[30., 0., (w-1)/2], [0., 29., (h-1)/2], [0., 0., 1.]], np.float32),
                image_shape=shape)


def observed_view(batch, view, backend='torch_reference'):
    rendered = module.render_map_view(batch, view, dict(backend=backend))
    value = dict(view)
    rgb = rendered['rgb'].detach().cpu().numpy().transpose(1, 2, 0)
    value['image'] = np.rint(np.clip(rgb[:, :, ::-1], 0, 1)*255).astype(np.uint8)
    value['depth'] = rendered['depth'].detach().cpu().numpy().copy()
    value['confidence'] = (rendered['alpha'].detach().cpu().numpy() > .08).astype(np.float32)
    return value


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class GaussianMapOptimizerTests(unittest.TestCase):
    def test_all_five_parameter_groups_match_numerical_derivatives(self):
        batch = batch_fixture()
        parameters = {name: torch.tensor(batch[name], dtype=torch.float64, requires_grad=True)
                      for name in module.PARAMETER_WIDTHS}
        view = view_fixture(shape=(11, 13))
        rows = torch.tensor(view['rows'])
        transforms = torch.tensor(view['transforms'], dtype=torch.float64)
        renderer = module.TorchReferenceRenderer()
        weight = torch.arange(11*13, dtype=torch.float64).reshape(11, 13)/143.

        def objective(values):
            xyz, covariance = module.camera_geometry(values, rows, transforms)
            rendered = renderer(xyz, covariance, values['colors'][rows],
                                values['opacity_logits'][rows].sigmoid(), view['K'], (11, 13))
            return ((rendered['rgb']*weight).square().sum() +
                    .025*(rendered['depth']*rendered['alpha']*weight).sum() +
                    .1*(rendered['alpha']*weight).sum())

        objective(parameters).backward()
        epsilon = 1e-5
        for name, index in dict(xyz=(0, 0), colors=(0, 1), log_scales=(0, 0),
                                rotations=(0, 3), opacity_logits=(0, 0)).items():
            analytical = float(parameters[name].grad[index])
            self.assertGreater(abs(analytical), 1e-5, name)
            plus = {key: value.detach().clone() for key, value in parameters.items()}
            minus = {key: value.detach().clone() for key, value in parameters.items()}
            plus[name][index] += epsilon
            minus[name][index] -= epsilon
            numerical = float((objective(plus)-objective(minus))/(2*epsilon))
            self.assertAlmostEqual(analytical, numerical, delta=max(1e-6, abs(numerical)*1e-4), msg=name)

    def test_multiview_map_fit_improves_loss_and_preserves_observations(self):
        truth = batch_fixture()
        views = [observed_view(truth, view_fixture(frame=i, dx=dx))
                 for i, dx in enumerate((-.13, .0, .15))]
        original_views = copy.deepcopy(views)
        initial = copy.deepcopy(truth)
        initial['xyz'] += np.array([.025, -.018, .09], np.float32)
        initial['colors'] = initial['colors']*.72
        initial['log_scales'] += .15
        initial['rotations'][:, 3] += .08
        initial['opacity_logits'] -= .45
        original_batch = copy.deepcopy(initial)
        result = module.optimize_gaussian_map(initial, views, dict(
            backend='torch_reference', iterations=65, downsample=1, lr_xyz=.006,
            lr_colors=.025, lr_log_scales=.015, lr_rotations=.01, lr_opacity_logits=.025,
            coverage_weight=.001, xyz_prior_weight=.0001, scale_prior_weight=.0001))
        report = result['report']
        self.assertLess(report['final_losses']['total'], report['initial_losses']['total']*.45)
        self.assertLess(report['final_losses']['depth'], report['initial_losses']['depth'])
        self.assertLess(report['final_losses']['rgb'], report['initial_losses']['rgb'])
        for name in module.PARAMETER_WIDTHS:
            self.assertGreater(report['gradients'][name]['max_abs'], 0., name)
            self.assertGreater(np.max(np.abs(result['parameters'][name]-initial[name])), 1e-6, name)
            np.testing.assert_array_equal(initial[name], original_batch[name])
        np.testing.assert_allclose(np.linalg.norm(result['parameters']['rotations'], axis=1), 1., atol=1e-6)
        self.assertTrue(np.all(np.exp(result['parameters']['log_scales']) > 0))
        for actual, expected in zip(views, original_views):
            for key in ('image', 'depth', 'confidence', 'transforms'):
                np.testing.assert_array_equal(actual[key], expected[key])
        self.assertFalse(report['ownership_optimized'])
        self.assertFalse(report['camera_poses_optimized'])
        self.assertFalse(report['object_poses_optimized'])

    def test_adam_state_follows_map_ids_and_dormant_rows_do_not_drift(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture(rows=[0]))
        config = dict(backend='torch_reference', iterations=2, downsample=1)
        first = module.optimize_gaussian_map(batch, [view], config)
        state = first['optimizer_state']
        # Simulate momentum retained from a much earlier active window.
        for name in module.PARAMETER_WIDTHS:
            state['params'][name]['exp_avg'][2] = .15
            state['params'][name]['exp_avg_sq'][2] = .04
            state['params'][name]['row_steps'][2] = 17
        reorder = np.array([2, 0, 1])
        next_batch = {name: first['parameters'][name][reorder].copy() for name in module.PARAMETER_WIDTHS}
        next_batch['map_ids'] = batch['map_ids'][reorder].copy()
        next_view = dict(view, rows=np.array([1]))
        result = module.optimize_gaussian_map(next_batch, [next_view], config, optimizer_state=state)
        np.testing.assert_array_equal(result['optimizer_state']['map_ids'], [90, 40, 70])
        for name in module.PARAMETER_WIDTHS:
            np.testing.assert_array_equal(result['parameters'][name][0], next_batch[name][0])
            np.testing.assert_array_equal(result['parameters'][name][2], next_batch[name][2])
            np.testing.assert_array_equal(result['optimizer_state']['params'][name]['row_steps'].ravel(), [17, 4, 0])
            np.testing.assert_array_equal(result['optimizer_state']['params'][name]['exp_avg'][0],
                                          state['params'][name]['exp_avg'][2])

    def test_renderer_composites_near_and_far_owners_in_one_scene(self):
        batch = batch_fixture()
        batch['xyz'][:2] = [[0., 0., 2.], [0., 0., 4.]]
        batch['colors'][:2] = [[1., 0., 0.], [0., 0., 1.]]
        batch['log_scales'][:2] = np.log(.12)
        batch['opacity_logits'][:2] = np.log(9.)
        view = view_fixture(rows=[0, 1], shape=(9, 9))
        view['transforms'][:] = np.eye(4)
        # Native pixel-center convention is K projection minus half a pixel.
        view['K'][:2, 2] = 4.5
        rendered = module.render_map_view(batch, view, dict(backend='torch_reference'))
        pixel = rendered['rgb'][:, 4, 4].numpy()
        np.testing.assert_allclose(pixel, [.9, 0., .09], atol=1e-6)
        self.assertAlmostEqual(float(rendered['alpha'][4, 4]), .99, places=6)
        self.assertAlmostEqual(float(rendered['depth'][4, 4]), (2*.9+4*.09)/.99, places=5)

    def test_frozen_owners_still_occlude_without_updates_or_adam_steps(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        batch['colors'] *= .8
        batch['trainable_mask'] = np.array([False, True, False])
        result = module.optimize_gaussian_map(batch, [view], dict(
            backend='torch_reference', iterations=3, downsample=1))
        for name in module.PARAMETER_WIDTHS:
            np.testing.assert_array_equal(result['parameters'][name][[0, 2]], batch[name][[0, 2]])
            np.testing.assert_array_equal(result['optimizer_state']['params'][name]['row_steps'].ravel(), [0, 3, 0])
        self.assertEqual(result['report']['active_gaussian_count'], 1)
        self.assertEqual(result['report']['frozen_gaussian_count'], 2)
        batch['trainable_mask'][:] = False
        all_frozen = module.optimize_gaussian_map(batch, [view], dict(backend='torch_reference'))
        self.assertEqual(all_frozen['report']['reason'], 'no_trainable_gaussians')

    def test_sparse_track_loss_has_correct_reprojection_gradient_and_pixel_scaling(self):
        batch = batch_fixture()
        raw = view_fixture(rows=[0, 1])
        measured = batch['xyz'][:2].astype(np.float64) + raw['transforms'][:, :3, 3]
        projected = measured @ raw['K'].T
        uv = projected[:, :2]/projected[:, 2:]
        initial = batch['xyz'].astype(np.float64)
        initial[0] += [.025, -.035, .06]
        xyz = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        view = dict(rows=torch.tensor(raw['rows']), transforms=torch.tensor(raw['transforms'], dtype=torch.float64),
                    K=raw['K'], track_indices=torch.tensor([0, 1]),
                    track_uv=torch.tensor(uv, dtype=torch.float64),
                    track_xyz=torch.tensor(measured, dtype=torch.float64),
                    track_weights=torch.tensor([1., .7], dtype=torch.float64))
        loss = module.sparse_track_loss(dict(xyz=xyz), view)
        loss.backward()
        analytical = float(xyz.grad[0, 1])
        self.assertGreater(abs(analytical), 1e-4)
        positive, negative = xyz.detach().clone(), xyz.detach().clone()
        positive[0, 1] += 1e-6
        negative[0, 1] -= 1e-6
        numerical = float((module.sparse_track_loss(dict(xyz=positive), view)-
                           module.sparse_track_loss(dict(xyz=negative), view))/(2e-6))
        self.assertAlmostEqual(analytical, numerical, delta=abs(numerical)*1e-5)
        half = dict(view, K=raw['K']*np.array([[.5], [.5], [1.]], np.float32),
                    track_uv=view['track_uv']*.5, track_pixel_scale=[.5, .5])
        self.assertAlmostEqual(float(module.sparse_track_loss(dict(xyz=xyz), half)), float(loss), places=12)

    def test_track_only_measured_points_pin_geometry_without_rgbd_evidence(self):
        truth = batch_fixture()
        view = observed_view(truth, view_fixture(rows=[0, 1]))
        measured = truth['xyz'][:2].copy()+view['transforms'][:, :3, 3]
        projected = measured @ view['K'].T
        view.update(track_indices=np.array([0, 1]), track_uv=projected[:, :2]/projected[:, 2:],
                    track_xyz=measured, track_weights=np.array([1., .7], np.float32))
        view['confidence'][:] = 0
        initial = copy.deepcopy(truth)
        initial['xyz'][:2] += [.08, -.045, .12]
        original = copy.deepcopy(view)
        # Immutable source measurements can be views into read-only archives.
        for key in ('track_indices', 'track_uv', 'track_xyz', 'track_weights'):
            view[key].setflags(write=False)
        result = module.optimize_gaussian_map(initial, [view], dict(
            backend='torch_reference', downsample=2, iterations=65, lr_xyz=.006,
            rgb_weight=0., depth_weight=0., coverage_weight=0., track_weight=1.,
            xyz_prior_weight=0., scale_prior_weight=0.))
        self.assertLess(result['report']['final_losses']['track'], result['report']['initial_losses']['track']*.08)
        before = np.linalg.norm(initial['xyz'][:2]-truth['xyz'][:2], axis=1)
        after = np.linalg.norm(result['parameters']['xyz'][:2]-truth['xyz'][:2], axis=1)
        self.assertTrue(np.all(after < before*.1))
        self.assertEqual(result['report']['track_count'], 2)
        self.assertEqual(result['report']['track_3d_count'], 2)
        np.testing.assert_array_equal(result['optimizer_state']['params']['xyz']['row_steps'].ravel(), [65, 65, 0])
        for name in set(module.PARAMETER_WIDTHS)-{'xyz'}:
            np.testing.assert_array_equal(result['parameters'][name], initial[name])
            self.assertFalse(result['optimizer_state']['params'][name]['row_steps'].any())
        for key in ('track_indices', 'track_uv', 'track_xyz', 'track_weights'):
            np.testing.assert_array_equal(view[key], original[key])

    def test_invalid_track_measurements_raise(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        view.update(track_indices=np.array([0]), track_uv=np.zeros((1, 2)), track_weights=np.array([-1.]))
        with self.assertRaisesRegex(ValueError, 'Sparse track'):
            module.optimize_gaussian_map(batch, [view], dict(backend='torch_reference', iterations=1))
        view['track_weights'][:] = 1
        view['track_xyz'] = np.array([[0., 0., -2.]])
        with self.assertRaisesRegex(ValueError, 'track_xyz'):
            module.optimize_gaussian_map(batch, [view], dict(backend='torch_reference', iterations=1))

    def test_cuda_is_required_unless_reference_explicitly_selected(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'requires native CUDA'):
                module.optimize_gaussian_map(batch, [view], dict(iterations=1))

    def test_reference_renderer_guard_blocks_production_sized_inputs(self):
        renderer = module.TorchReferenceRenderer()
        with self.assertRaisesRegex(ValueError, 'tiny fixtures'):
            renderer(torch.zeros((300, 3)), torch.eye(3).repeat(300, 1, 1),
                     torch.zeros((300, 3)), torch.ones((300, 1)), np.eye(3), (17, 19))

    def test_explicit_reference_renderer_injection_selects_cpu(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        with patch.object(torch.cuda, 'is_available', return_value=False):
            result = module.optimize_gaussian_map(batch, [view], dict(iterations=1, downsample=1),
                                                  renderer=module.TorchReferenceRenderer())
        self.assertEqual(result['report']['backend'], 'torch_reference')
        self.assertEqual(result['report']['iterations_completed'], 1)

    def test_zero_iterations_returns_unchanged_cpu_state_without_cuda(self):
        batch = batch_fixture()
        result = module.optimize_gaussian_map(batch, [view_fixture()], dict(iterations=0))
        for name in module.PARAMETER_WIDTHS:
            np.testing.assert_array_equal(result['parameters'][name], batch[name])
            self.assertFalse(result['optimizer_state']['params'][name]['row_steps'].any())
        self.assertEqual(result['report']['reason'], 'no_iterations')

    def test_no_supported_depth_leaves_map_and_momentum_untouched(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        view['confidence'][:] = 0
        result = module.optimize_gaussian_map(batch, [view], dict(backend='torch_reference', iterations=3))
        self.assertEqual(result['report']['reason'], 'no_supported_measurements')
        for name in module.PARAMETER_WIDTHS:
            np.testing.assert_array_equal(result['parameters'][name], batch[name])
            self.assertFalse(result['optimizer_state']['params'][name]['row_steps'].any())

    def test_unknown_backend_invalid_geometry_and_duplicate_rows_raise(self):
        batch = batch_fixture()
        view = observed_view(batch, view_fixture())
        with self.assertRaisesRegex(ValueError, 'backend'):
            module.optimize_gaussian_map(batch, [view], dict(backend='automatic'))
        invalid = copy.deepcopy(batch)
        invalid['rotations'][0] = 0
        with self.assertRaisesRegex(ValueError, 'quaternions'):
            module.optimize_gaussian_map(invalid, [view], dict(iterations=0))
        view['rows'] = np.array([0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'uniquely index'):
            module.optimize_gaussian_map(batch, [view], dict(backend='torch_reference'))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'Native CUDA is unavailable')
class NativeGaussianMapTests(unittest.TestCase):
    def test_native_raster_all_geometry_gradients_and_joint_optimization(self):
        try:
            import diff_gaussian_rasterization  # noqa: F401
        except ImportError:
            self.skipTest('Native CUDA rasterizer is not installed')
        batch = batch_fixture()
        view = observed_view(batch, view_fixture(), backend='cuda')
        batch['xyz'] += [.015, -.025, .05]
        batch['colors'] *= .8
        batch['log_scales'] += .1
        batch['rotations'][:, 3] += .07
        batch['opacity_logits'] -= .2
        result = module.optimize_gaussian_map(batch, [view], dict(
            backend='cuda', iterations=12, downsample=1, lr_xyz=.002, lr_colors=.02,
            lr_log_scales=.01, lr_rotations=.005, coverage_weight=0.))
        self.assertLess(result['report']['final_losses']['total'], result['report']['initial_losses']['total'])
        for name in module.PARAMETER_WIDTHS:
            self.assertGreater(result['report']['gradients'][name]['max_abs'], 0., name)


if __name__ == '__main__':
    unittest.main()

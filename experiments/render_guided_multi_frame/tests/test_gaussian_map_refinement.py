"""Evidence, lifecycle, and budget checks for adaptive map topology."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import copy
import importlib.util
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'mapping'/(name+'.py'))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


refinement = load('gaussian_map_refinement')
optimizer = load('gaussian_map_optimizer')
try:
    import torch
except ImportError:
    torch = None


def fixture(n=10):
    batch = dict(map_ids=np.arange(10, 10+n, dtype=np.int64), object_ids=np.repeat([2, 5], n//2),
                 xyz=np.stack((np.linspace(-.25, .25, n), np.zeros(n), np.full(n, 2.)), 1).astype(np.float32),
                 colors=np.repeat([[.7, .3, .2]], n, 0).astype(np.float32),
                 log_scales=np.repeat(np.log([[.1, .06, .04]]), n, 0).astype(np.float32),
                 rotations=np.repeat([[1., 0., 0., 0.]], n, 0).astype(np.float32),
                 opacity_logits=np.ones((n, 1), np.float32), observation_count=np.full(n, 4),
                 trainable_mask=np.ones(n, bool))
    stats = dict(map_ids=batch['map_ids'].copy(), screen_gradient_sum=np.full(n, .01),
                 screen_gradient_count=np.full(n, 10), max_radius_px=np.full(n, 10.),
                 visible_steps=np.full(n, 10), gradient_units='loss_per_original_image_pixel')
    return batch, stats


class RefinementPlannerTests(unittest.TestCase):
    def test_split_uses_owned_rotated_axis_and_never_mutates_input(self):
        batch, stats = fixture()
        original = copy.deepcopy(batch)
        batch['rotations'][0] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        rotated_original = copy.deepcopy(batch)
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(max_growth_fraction=1.,
            max_new_per_update=2, max_splits_per_update=1))
        np.testing.assert_array_equal(plan['split_ids'], [10])
        child = plan['children']
        np.testing.assert_array_equal(child['parent_ids'], [10, 10])
        np.testing.assert_array_equal(child['object_ids'], [2, 2])
        np.testing.assert_allclose(child['xyz'].mean(0), batch['xyz'][0], atol=1e-7)
        np.testing.assert_allclose(np.abs(child['xyz'][:, 1]), [.05, .05], atol=1e-7)
        np.testing.assert_allclose(child['xyz'][:, 0], batch['xyz'][0, 0], atol=1e-7)
        self.assertTrue(np.all(child['log_scales'] < batch['log_scales'][0]))
        alpha_parent = 1/(1+np.exp(-batch['opacity_logits'][0, 0]))
        alpha_child = 1/(1+np.exp(-child['opacity_logits'][0, 0]))
        self.assertAlmostEqual(1-(1-alpha_child)**2, alpha_parent, places=6)
        for name in original:
            np.testing.assert_array_equal(batch[name], rotated_original[name])

    def test_zero_gradient_and_unobserved_points_never_split(self):
        batch, stats = fixture()
        stats['screen_gradient_sum'][:] = 0
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(max_growth_fraction=1.))
        self.assertEqual(plan['report']['child_count'], 0)
        stats['screen_gradient_sum'][:] = 1
        stats['visible_steps'][:] = 0
        batch['opacity_logits'][:] = -10
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(min_points_per_object=0))
        self.assertEqual(plan['report']['child_count'], 0)
        self.assertEqual(plan['report']['pruned_count'], 0)

    def test_frozen_owners_protected_landmarks_and_young_points_are_preserved(self):
        batch, stats = fixture()
        batch['trainable_mask'][:5] = False
        batch['refinement_protected_mask'] = np.zeros(10, bool)
        batch['refinement_protected_mask'][5] = True
        batch['observation_count'][6:] = 1
        batch['opacity_logits'][5:] = -10
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(max_growth_fraction=1., min_points_per_object=0))
        self.assertEqual(plan['report']['child_count'], 0)
        self.assertEqual(plan['report']['pruned_count'], 0)
        self.assertEqual(plan['report']['frozen_count'], 5)
        self.assertEqual(plan['report']['protected_count'], 1)

    def test_active_global_child_and_owner_budgets_use_net_live_growth(self):
        batch, stats = fixture()
        config = dict(max_growth_fraction=1., max_new_per_update=5, max_children_per_object=2,
                      max_active_gaussians=12, max_total_gaussians=102)
        plan = refinement.plan_gaussian_refinement(batch, stats, config, total_map_count=100)
        self.assertEqual(plan['report']['split_count'], 2)
        self.assertEqual(plan['report']['child_count'], 4)
        self.assertEqual(plan['report']['active_count_after'], 12)
        self.assertEqual(plan['report']['total_count_after'], 102)
        config['max_total_gaussians'] = 100
        plan = refinement.plan_gaussian_refinement(batch, stats, config, total_map_count=100)
        self.assertEqual(plan['report']['child_count'], 0)

    def test_prune_low_alpha_visible_points_preserves_owner_support(self):
        batch, stats = fixture()
        batch['opacity_logits'][:] = -10
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(min_points_per_object=4))
        self.assertEqual(plan['report']['pruned_count'], 2)
        np.testing.assert_array_equal(plan['prune_ids'], [10, 15])
        self.assertEqual(plan['report']['net_growth'], -2)
        self.assertEqual(plan['report']['child_count'], 0)

    def test_small_footprint_clones_and_id_alignment_survives_reordering(self):
        batch, stats = fixture()
        stats['max_radius_px'][:] = 3
        order = np.arange(9, -1, -1)
        stats = {key: value[order] if isinstance(value, np.ndarray) else value for key, value in stats.items()}
        plan = refinement.plan_gaussian_refinement(batch, stats, dict(max_growth_fraction=1.,
            max_new_per_update=1))
        np.testing.assert_array_equal(plan['clone_ids'], [10])
        self.assertFalse(len(plan['split_ids']))
        np.testing.assert_array_equal(plan['children']['log_scales'][0], batch['log_scales'][0])
        self.assertLess(plan['children']['opacity_logits'][0, 0], batch['opacity_logits'][0, 0])

    def test_native_ndc_and_depth_attenuation_normalize_to_reference_pixel_gradient(self):
        true_derivative = np.array([[.004, -.006], [.002, .008]])
        shape, pixel_scale, weight = (25, 41), np.array([.5, .5]), .2
        # Mirrors installed CUDA backward and Python raster wrapper exactly.
        attenuation = np.array([.25, 1.])
        native_surrogate = true_derivative*np.array([shape[1]/2., shape[0]/2.])
        native_surrogate *= attenuation[:, None]*weight
        normalized_native = optimizer._screen_gradient_original_pixels(native_surrogate, 'ndc',
            shape, pixel_scale, weight, attenuation)
        normalized_reference = optimizer._screen_gradient_original_pixels(true_derivative*weight,
            'render_pixels', shape, pixel_scale, weight)
        np.testing.assert_allclose(normalized_native, true_derivative*pixel_scale, atol=1e-14)
        np.testing.assert_allclose(normalized_native, normalized_reference, atol=1e-14)

    def test_bad_statistics_and_unbounded_settings_rejected(self):
        batch, stats = fixture()
        stats['screen_gradient_sum'][0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            refinement.plan_gaussian_refinement(batch, stats, {})
        batch, stats = fixture()
        with self.assertRaisesRegex(ValueError, 'gradient'):
            refinement.plan_gaussian_refinement(batch, stats, dict(gradient_threshold=0))
        with self.assertRaisesRegex(ValueError, 'total_map_count'):
            refinement.plan_gaussian_refinement(batch, stats, {}, total_map_count=9)


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class MultiscaleRefinementStatsTests(unittest.TestCase):
    def scene(self):
        batch, _ = fixture(2)
        view = dict(frame=0, rows=np.arange(2), transforms=np.repeat(np.eye(4)[None], 2, 0),
                    K=np.array([[30., 0., 9.5], [0., 29., 8.5], [0., 0., 1.]], np.float32), image_shape=(17, 19))
        result = optimizer.render_map_view(batch, view, dict(backend='torch_reference'))
        view.update(image=np.rint(result['rgb'].numpy().transpose(1, 2, 0)[:, :, ::-1]*255).astype(np.uint8),
                    depth=result['depth'].numpy(), confidence=(result['alpha'].numpy() > .05).astype(np.float32))
        batch['colors'] *= .65
        batch['xyz'][:, 0] += .035
        return batch, view

    def test_multiscale_carries_adam_and_matches_sequential_stages(self):
        batch, view = self.scene()
        original = copy.deepcopy(batch)
        config = dict(backend='torch_reference', collect_refinement_stats=True,
                      stages=[dict(downsample=2, iterations=2), dict(downsample=1, iterations=3)])
        actual = optimizer.optimize_gaussian_map(batch, [view], config)
        first = optimizer.optimize_gaussian_map(batch, [view], dict(backend='torch_reference', iterations=2, downsample=2))
        next_batch = dict(batch, **first['parameters'])
        expected = optimizer.optimize_gaussian_map(next_batch, [view], dict(backend='torch_reference', iterations=3,
            downsample=1), optimizer_state=first['optimizer_state'])
        for name in optimizer.PARAMETER_WIDTHS:
            np.testing.assert_allclose(actual['parameters'][name], expected['parameters'][name], atol=1e-7)
            np.testing.assert_array_equal(actual['optimizer_state']['params'][name]['row_steps'], [[5], [5]])
            np.testing.assert_array_equal(batch[name], original[name])
        self.assertEqual(actual['report']['downsample_schedule'], [2, 1])
        self.assertEqual(actual['report']['iterations_completed'], 5)
        stats = actual['refinement_stats']
        self.assertTrue(np.all(stats['screen_gradient_count'] == 5))
        self.assertTrue(np.all(stats['screen_gradient_sum'] > 0))
        self.assertTrue(np.all(stats['max_radius_px'] > 0))
        self.assertEqual(stats['gradient_units'], 'loss_per_original_image_pixel')

    def test_stats_exclude_frozen_owners_and_track_only_evidence(self):
        batch, view = self.scene()
        batch['trainable_mask'][0] = False
        output = optimizer.optimize_gaussian_map(batch, [view], dict(backend='torch_reference',
            iterations=2, downsample=1, collect_refinement_stats=True))
        self.assertEqual(output['refinement_stats']['visible_steps'][0], 0)
        self.assertEqual(output['refinement_stats']['screen_gradient_sum'][0], 0)
        self.assertEqual(output['refinement_stats']['visible_steps'][1], 2)

    def test_refinement_gradient_threshold_is_independent_of_window_length(self):
        batch, view = self.scene()
        config = dict(backend='torch_reference', iterations=1, downsample=1,
                      collect_refinement_stats=True,
                      learning_rates={name: 0. for name in optimizer.PARAMETER_WIDTHS})
        one = optimizer.optimize_gaussian_map(batch, [view], config)['refinement_stats']
        another_view = dict(view, frame=1)
        two = optimizer.optimize_gaussian_map(batch, [view, another_view], config)['refinement_stats']
        np.testing.assert_allclose(one['screen_gradient_sum']/one['screen_gradient_count'],
                                   two['screen_gradient_sum']/two['screen_gradient_count'], rtol=1e-6)

    def test_occluded_point_does_not_receive_refinement_visibility(self):
        batch, view = self.scene()
        batch['xyz'][:] = [[0., 0., 2.], [0., 0., 8.]]
        batch['log_scales'][1] = batch['log_scales'][0]+np.log(4.)
        batch['opacity_logits'][0] = 6.
        rendered = optimizer.render_map_view(batch, view, dict(backend='torch_reference'))
        view.update(image=np.rint(rendered['rgb'].numpy().transpose(1, 2, 0)[:, :, ::-1]*255).astype(np.uint8),
                    depth=rendered['depth'].numpy(), confidence=(rendered['alpha'].numpy() > .05).astype(np.float32))
        batch['colors'] *= .8
        output = optimizer.optimize_gaussian_map(batch, [view], dict(backend='torch_reference',
            iterations=2, downsample=1, collect_refinement_stats=True))
        self.assertGreater(output['refinement_stats']['visible_steps'][0], 0)
        self.assertEqual(output['refinement_stats']['visible_steps'][1], 0)
        self.assertEqual(output['refinement_stats']['screen_gradient_count'][1], 0)

    def test_ssim_is_zero_at_identity_and_has_structural_gradients(self):
        torch.manual_seed(10)
        image = torch.rand((3, 11, 13), requires_grad=True)
        target = image.detach().clone()
        self.assertAlmostEqual(float(optimizer.structural_dissimilarity(image, target).mean()), 0., places=7)
        target = target.roll(1, -1)
        loss = optimizer.structural_dissimilarity(image, target).mean()
        self.assertGreater(float(loss), .1)
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(float(image.grad.norm()), 0)
        batch, view = self.scene()
        output = optimizer.optimize_gaussian_map(batch, [view], dict(backend='torch_reference',
            iterations=2, downsample=1, ssim_weight=.2))
        self.assertGreater(output['report']['initial_losses']['ssim'], 0)
        self.assertTrue(np.isfinite(output['report']['final_losses']['total']))


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'Native CUDA is unavailable')
class NativeRefinementTests(unittest.TestCase):
    def test_native_screen_gradient_statistics_are_collected(self):
        batch, view = MultiscaleRefinementStatsTests().scene()
        try:
            output = optimizer.optimize_gaussian_map(batch, [view], dict(backend='cuda', iterations=2,
                downsample=1, collect_refinement_stats=True))
        except (ImportError, RuntimeError) as error:
            if 'diff_gaussian_rasterization' in str(error):
                self.skipTest('Native CUDA rasterizer is unavailable')
            raise
        self.assertTrue(np.all(output['refinement_stats']['screen_gradient_count'] > 0))
        self.assertTrue(np.all(output['refinement_stats']['screen_gradient_sum'] > 0))


if __name__ == '__main__':
    unittest.main()

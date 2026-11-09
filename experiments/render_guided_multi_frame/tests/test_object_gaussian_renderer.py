"""CPU coverage of accepted-only selection and CUDA render preparation."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / 'object_gaussian_renderer.py'
SPEC = importlib.util.spec_from_file_location('object_gaussian_renderer', MODULE)
renderer_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer_module)


class ObjectGaussianRendererTests(unittest.TestCase):
    def gaussian(self):
        models = np.repeat(np.eye(4)[None], 3, axis=0)
        models[0, :3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
        models[0, :3, 3] = [1., 2., 3.]
        models[1, :3, 3] = [100., 0., 0.]
        return dict(xyz=np.array([[1., 0., 2.], [0., 1., 3.], [2., 1., 4.], [3., 1., 5.]]),
                    color=np.array([[1., .1, .2], [.1, 1., .2], [.1, .2, 1.], [.5, .5, .5]]),
                    scale=np.array([.1, .2, .3, .4]),
                    object_ids=np.array([7, 0, 12]),
                    object_labels=np.array([7, -1, 0, 12]),
                    object_probabilities=np.array([[.1, .1, .8], [.99, .005, .005],
                                                   [.1, .1, .8], [.1, .8, .1]]),
                    H_object=models, object_model_valid=np.array([True, True, True]),
                    K=np.eye(3))

    def test_final_labels_alone_select_object_and_exclude_uncertain_even_with_high_probability(self):
        gmap = self.gaussian()
        prepared = renderer_module.prepare_object_gaussians(gmap, 7)
        np.testing.assert_array_equal(prepared['indices'], [0])
        np.testing.assert_array_equal(prepared['xyz'], gmap['xyz'][[0]])
        np.testing.assert_allclose(prepared['color'], gmap['color'][[0]])
        np.testing.assert_allclose(prepared['scale'], [[.1, .1, .1]])
        # The O7 argmax point is uncertain; an accepted O7 point whose argmax
        # is another object is still selected by its saved final hard label.
        self.assertEqual(prepared['label_selected_count'], 1)
        self.assertTrue(prepared['model_valid'])
        self.assertIsNone(prepared['cov6'])

    def test_terminal_uses_global_object_column_and_preserves_original_arrays(self):
        gmap = self.gaussian()
        original = gmap['xyz'].copy()
        prepared = renderer_module.prepare_object_gaussians(gmap, 7, terminal=True)
        np.testing.assert_allclose(prepared['xyz'], [[1., 3., 5.]])
        np.testing.assert_array_equal(gmap['xyz'], original)
        np.testing.assert_array_equal(prepared['indices'], [0])
        self.assertTrue(prepared['terminal'])

    def test_missing_invalid_or_absent_terminal_object_produces_empty_geometry(self):
        gmap = self.gaussian()
        gmap['object_model_valid'][0] = False
        invalid = renderer_module.prepare_object_gaussians(gmap, 7, terminal=True)
        self.assertFalse(invalid['model_valid'])
        self.assertEqual(invalid['xyz'].shape, (0, 3))
        self.assertEqual(invalid['label_selected_count'], 1)
        del gmap['H_object']
        missing = renderer_module.prepare_object_gaussians(gmap, 0, terminal=True)
        self.assertEqual(missing['reason'], 'missing_object_motion')
        absent = renderer_module.prepare_object_gaussians(gmap, 99)
        self.assertEqual(absent['reason'], 'no_accepted_gaussians')
        self.assertEqual(absent['indices'].shape, (0,))

    def test_empty_render_requires_no_cuda_and_returns_black_rgb_false_mask_and_json_metadata(self):
        gmap = self.gaussian()
        gmap['object_labels'][:] = -1
        renderer = renderer_module.GaussianObjectRenderer()
        with patch.object(renderer, '_load_cuda', side_effect=AssertionError('CUDA must not load')):
            rgb, silhouette, metadata = renderer.render(gmap, 7, (7, 11, 3), {})
        self.assertEqual(rgb.shape, (7, 11, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertFalse(rgb.any())
        self.assertEqual(silhouette.dtype, bool)
        self.assertFalse(silhouette.any())
        self.assertEqual(metadata['selected_gaussians'], 0)
        self.assertFalse(metadata['probability_weighted_opacity'])
        json.dumps(metadata)

    def test_scalar_and_single_column_scales_expand_to_identity_quaternion_scales(self):
        gmap = self.gaussian()
        for scale in (np.array([.1, .2, .3, .4]), np.array([[.1], [.2], [.3], [.4]]),
                      np.array([[.1]*3, [.2]*3, [.3]*3, [.4]*3])):
            gmap['scale'] = scale
            selected = renderer_module.prepare_object_gaussians(gmap, 0)
            np.testing.assert_allclose(selected['scale'], [[.3, .3, .3]])
            self.assertIsNone(selected['cov6'])

    def test_anisotropic_geometry_requires_orientation_and_rotates_covariance(self):
        gmap = self.gaussian()
        gmap['scale'] = np.array([[.1, .2, .3], [.2]*3, [.3]*3, [.4]*3])
        with self.assertRaisesRegex(ValueError, 'basis or covariance'):
            renderer_module.prepare_object_gaussians(gmap, 7)
        gmap['basis'] = np.repeat(np.eye(3)[None], 4, axis=0)
        prepared = renderer_module.prepare_object_gaussians(gmap, 7, terminal=True)
        np.testing.assert_allclose(prepared['cov'][0], np.diag([.04, .01, .09]), atol=1e-7)
        np.testing.assert_allclose(prepared['cov6'], [[.04, 0., 0., .01, 0., .09]], atol=1e-7)
        np.testing.assert_allclose(prepared['basis'][0], gmap['H_object'][0, :3, :3])
        del gmap['basis']
        gmap['cov6'] = np.tile([.01, 0., 0., .04, 0., .09], (4, 1))
        packed = renderer_module.prepare_object_gaussians(gmap, 7, terminal=True)
        np.testing.assert_allclose(packed['cov6'], prepared['cov6'], atol=1e-7)

    def test_missing_final_labels_never_falls_back_to_probabilities(self):
        gmap = self.gaussian()
        del gmap['object_labels']
        prepared = renderer_module.prepare_object_gaussians(gmap, 7)
        self.assertEqual(prepared['reason'], 'missing_final_object_labels')
        self.assertEqual(len(prepared['indices']), 0)


if __name__ == '__main__':
    unittest.main()

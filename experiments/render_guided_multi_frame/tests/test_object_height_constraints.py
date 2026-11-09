"""CPU checks for tilted-camera ground and bounded soft height constraints."""
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / 'object_height_constraints.py'
SPEC = importlib.util.spec_from_file_location('object_height_constraints', MODULE)
height_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(height_module)


class ObjectHeightConstraintsTests(unittest.TestCase):
    def road(self, count=2400, *, normal=None, height=1.55, seed=17):
        rng = np.random.default_rng(seed)
        normal = np.array([.04, -.993, -.11]) if normal is None else np.asarray(normal, float)
        normal /= np.linalg.norm(normal)
        z = rng.uniform(5.5, 27., count)
        u = rng.uniform(55., 585., count)
        x = (u-320.)*z/500.
        y = -(height+normal[0]*x+normal[2]*z)/normal[1]
        y += rng.normal(0, .012, count)
        xyz = np.column_stack((x, y, z))
        uv = np.column_stack((u, 500.*y/z+154.))
        return xyz, uv, normal

    def test_pitched_camera_recovers_up_and_true_camera_height_not_camera_y(self):
        xyz, uv, expected = self.road()
        result = height_module.estimate_ground_plane(xyz, uv, (240, 640), np.full(len(xyz), .95))
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(result['normal'], expected, atol=.008)
        self.assertAlmostEqual(result['camera_height'], 1.55, delta=.03)
        # At z≈7m the camera-Y road coordinate is ≈.8m; using Y alone
        # would underestimate vertical camera height by about half a metre.
        self.assertLess(np.median(xyz[np.abs(xyz[:, 2]-7.) < .5, 1]), 1.)
        self.assertGreater(result['depth_span_m'], 8.)
        self.assertGreaterEqual(result['occupied_depth_bins'], 3)
        json.dumps(result, allow_nan=False)

    def test_large_vertical_wall_cannot_replace_visible_ground(self):
        road, road_uv, expected = self.road(count=2000)
        rng = np.random.default_rng(21)
        wall = np.column_stack((np.full(3800, 4.), rng.uniform(-2., 2., 3800),
                                 rng.uniform(5., 28., 3800)))
        # Wall observations deliberately occupy much of the lower image.
        wall_uv = np.column_stack((rng.uniform(50, 590, len(wall)), rng.uniform(135, 235, len(wall))))
        xyz, uv = np.vstack((road, wall)), np.vstack((road_uv, wall_uv))
        result = height_module.estimate_ground_plane(xyz, uv, (240, 640))
        self.assertTrue(result['valid'], result)
        np.testing.assert_allclose(result['normal'], expected, atol=.02)
        self.assertAlmostEqual(result['camera_height'], 1.55, delta=.08)
        wall_only = height_module.estimate_ground_plane(wall, wall_uv, (240, 640))
        self.assertFalse(wall_only['valid'])

    def test_short_vehicle_surface_is_rejected_by_depth_coverage(self):
        rng = np.random.default_rng(9)
        xyz = np.column_stack((rng.uniform(-2, 2, 1800), np.full(1800, .65),
                               rng.uniform(4., 6., 1800)))
        uv = np.column_stack((rng.uniform(50, 590, len(xyz)), rng.uniform(145, 235, len(xyz))))
        result = height_module.estimate_ground_plane(xyz, uv, (240, 640))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'insufficient_spatial_or_depth_coverage')

    def test_unreliable_or_sparse_depth_disables_constraint_without_inventing_plane(self):
        xyz, uv, _ = self.road()
        unreliable = height_module.estimate_ground_plane(xyz, uv, (240, 640), np.zeros(len(xyz)))
        self.assertFalse(unreliable['valid'])
        penalty, reliability = height_module.height_penalty(xyz, unreliable)
        self.assertFalse(penalty.any())
        self.assertFalse(reliability.any())
        sparse = height_module.estimate_ground_plane(xyz[:20], uv[:20], (240, 640))
        self.assertFalse(sparse['valid'])
        json.dumps(unreliable, allow_nan=False)

    def test_optional_calibrated_up_accepts_large_pitch_without_assuming_y_gravity(self):
        angle = np.deg2rad(31.)
        normal = [0., -np.cos(angle), -np.sin(angle)]
        xyz, _, expected = self.road(normal=normal)
        # A calibrated camera can be steeply pitched; lower-image pixel
        # support is supplied independently to isolate the orientation prior.
        rng = np.random.default_rng(27)
        uv = np.column_stack((rng.uniform(50, 590, len(xyz)), rng.uniform(135, 235, len(xyz))))
        ordinary = height_module.estimate_ground_plane(xyz, uv, (240, 640))
        self.assertFalse(ordinary['valid'])
        calibrated = height_module.estimate_ground_plane(xyz, uv, (240, 640),
                                                        options={'config_up': normal})
        self.assertTrue(calibrated['valid'], calibrated)
        np.testing.assert_allclose(calibrated['normal'], expected, atol=.01)

    def test_feasible_car_zero_penalty_high_structure_outside_band_and_cap(self):
        plane = dict(valid=True, normal=[0., -1., 0.], offset=1.55,
                     camera_height=1.55, confidence=.9, fit_rmse_m=.03)
        heights = np.array([0., .5, 1.5, 2.5, 3.55, 5., 20.])
        xyz = np.column_stack((np.zeros(len(heights)), 1.55-heights, np.full(len(heights), 10.)))
        penalty, reliability = height_module.height_penalty(xyz, plane)
        np.testing.assert_array_equal(penalty[:5], 0.)
        self.assertGreater(penalty[5], 1.)
        self.assertGreater(penalty[6], penalty[5])
        self.assertTrue((penalty < 3.).all())
        self.assertTrue((reliability > .5).all())
        upper_override, _ = height_module.height_penalty(xyz, plane, {'max_height_m': 2.})
        self.assertGreater(upper_override[3], 0.)
        narrower_margin, _ = height_module.height_penalty(xyz, plane, {'soft_margin_m': .1})
        self.assertGreater(narrower_margin[5], penalty[5])

    def test_height_band_uses_fitted_normal_and_invalid_points_never_contribute(self):
        plane = dict(valid=True, normal=[0., -.8, -.6], offset=1.5,
                     camera_height=1.5, confidence=.8)
        xyz = np.array([[0., -6.25, 10.], [0., -13.75, 10.],
                        [np.nan, 0., 5.], [0., 0., -1.], [0., -1e7, 1e7]])
        penalty, reliability = height_module.height_penalty(xyz, plane)
        self.assertEqual(penalty[0], 0.)  # Height .5m despite camera Y=-6.25.
        self.assertGreater(penalty[1], 0.)
        np.testing.assert_array_equal(penalty[2:4], 0.)
        np.testing.assert_array_equal(reliability[2:], 0.)
        self.assertTrue(np.isfinite(penalty).all())
        self.assertLessEqual(penalty[-1], 3.)

    def test_image_confidence_input_and_sample_budget_keep_reliable_plane(self):
        xyz, uv, expected = self.road(count=6000)
        result = height_module.estimate_ground_plane(
            xyz, uv, (240, 640, 3), np.ones((240, 640)), options={'max_points': 1000})
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['sampled_points'], 1000)
        np.testing.assert_allclose(result['normal'], expected, atol=.015)


if __name__ == '__main__':
    unittest.main()

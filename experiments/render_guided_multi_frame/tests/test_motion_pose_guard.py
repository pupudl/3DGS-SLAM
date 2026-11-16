"""Synthetic CPU checks for preserving each motion's fixed sparse evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend.hybrid_motion import motion_costs
from frontend.motion_pose_guard import guard


K = np.array([[500., 0., 640.], [0., 500., 360.], [0., 0., 1.]])
CFG = {'min_support': 6, 'reprojection_threshold_px': 3.,
       'depth_sigma_m': .15, 'depth_sigma_relative': .015,
       'depth_weight': .05, 'depth_cost_cap': 4.,
       'outlier_cost': 1., 'core_margin': .1}


def project(xyz):
    return xyz[:, :2] / xyz[:, 2:] * 500. + K[:2, 2]


def scene(counts=(300, 8)):
    rng = np.random.default_rng(123)
    models = np.tile(np.eye(4), (len(counts), 1, 1))
    motions = [(-.2, 0., -.3), (.7, .05, -.1), (-.1, .7, .1)]
    first, second = [], []
    for i, n in enumerate(counts):
        models[i, :3, 3] = motions[i]
        xyz = rng.uniform([-2., -1., 14.], [2., 1., 22.], (n, 3))
        first.append(xyz)
        second.append(xyz + models[i, :3, 3])
    x0, x1 = np.concatenate(first), np.concatenate(second)
    return {'x0': x0, 'x1': x1,
            'matches': np.column_stack((project(x0), project(x1))),
            'confidence': np.ones(len(x0))}, models


class MotionPoseGuardTests(unittest.TestCase):
    def test_small_motion_cannot_be_diluted_by_large_background(self):
        anchors, initial = scene()
        proposed = initial.copy()
        proposed[1, 0, 3] += .7
        global_loss = np.minimum(motion_costs(anchors, proposed, K, CFG), 1.).min(axis=1).mean()
        self.assertLess(global_loss, .1)  # Global average would hide this loss.
        result = guard(anchors, initial, proposed, K, CFG)
        np.testing.assert_array_equal(result['accepted_mask'], [True, False])
        np.testing.assert_array_equal(result['models'][1], initial[1])
        self.assertEqual(result['report']['models'][1]['initial_core_count'], 8)
        self.assertIn('insufficient_retained_support', result['report']['models'][1]['fallback_reasons'])
        np.testing.assert_array_equal(proposed[1, 0, 3], initial[1, 0, 3] + .7)

    def test_reasonable_improvement_is_accepted(self):
        anchors, exact = scene()
        initial = exact.copy()
        initial[1, 0, 3] += .025
        result = guard(anchors, initial, exact, K, CFG)
        self.assertTrue(result['accepted_mask'].all())
        np.testing.assert_array_equal(result['models'], exact)
        report = result['report']['models'][1]
        self.assertLess(report['proposed_clipped_cost'], report['initial_clipped_cost'])
        self.assertEqual(report['retained_support_fraction'], 1.)

    def test_fewer_than_minimum_core_points_does_not_lock_pose(self):
        anchors, initial = scene((30, 5))
        proposed = initial.copy()
        proposed[1, 0, 3] += .7
        result = guard(anchors, initial, proposed, K, CFG)
        self.assertTrue(result['accepted_mask'][1])
        np.testing.assert_array_equal(result['models'][1], proposed[1])
        self.assertEqual(result['report']['models'][1]['reason'], 'insufficient_initial_core')

    def test_model_identity_permutation_is_equivariant(self):
        anchors, initial = scene((30, 8, 9))
        proposed = initial.copy()
        proposed[1, 0, 3] += .7
        baseline = guard(anchors, initial, proposed, K, CFG)
        order = np.array([2, 0, 1])
        permuted = guard(anchors, initial[order], proposed[order], K, CFG)
        np.testing.assert_array_equal(permuted['models'], baseline['models'][order])
        np.testing.assert_array_equal(permuted['accepted_mask'], baseline['accepted_mask'][order])
        self.assertEqual([r['initial_core_count'] for r in permuted['report']['models']], [9, 30, 8])

    def test_outliers_do_not_increase_initial_core(self):
        anchors, initial = scene((30, 5))
        outlier = {k: v[-10:].copy() for k, v in anchors.items()}
        outlier['x1'] += [9., 4., 3.]
        outlier['matches'][:, 2:] = project(outlier['x1'])
        anchors = {k: np.concatenate((v, outlier[k])) for k, v in anchors.items()}
        proposed = initial.copy()
        proposed[1, 0, 3] += .7
        result = guard(anchors, initial, proposed, K, CFG)
        self.assertEqual(result['report']['fixed_core_count'], 35)
        self.assertEqual(result['report']['models'][1]['initial_core_count'], 5)
        self.assertTrue(result['accepted_mask'][1])

    def test_cost_guard_rejects_deterioration_even_with_all_support_retained(self):
        anchors, initial = scene((8,))
        proposed = initial.copy()
        proposed[0, 0, 3] += .06
        result = guard(anchors, initial, proposed, K, CFG)
        report = result['report']['models'][0]
        self.assertEqual(report['retained_support_fraction'], 1.)
        self.assertFalse(result['accepted_mask'][0])
        self.assertEqual(report['fallback_reasons'], ['clipped_cost_increased'])

    def test_metric_support_cannot_be_replaced_by_capped_depth_cost(self):
        x0 = np.column_stack((np.linspace(-.01, .01, 8), np.zeros(8), np.full(8, 20.)))
        anchors = {'x0': x0, 'x1': x0.copy(),
                   'matches': np.column_stack((project(x0), project(x0)))}
        initial = np.eye(4)[None]
        proposed = initial.copy()
        proposed[0, 2, 3] = 1.
        self.assertTrue(np.all(motion_costs(anchors, proposed, K, CFG) < 1))
        result = guard(anchors, initial, proposed, K, CFG)
        self.assertEqual(result['report']['models'][0]['proposed_support_mass'], 0.)
        self.assertFalse(result['accepted_mask'][0])

    def test_confidence_mass_is_used_instead_of_raw_point_count(self):
        anchors, initial = scene((8,))
        # The same pose change hurts near points more. Most measured support
        # mass is placed on four distant points that remain good.
        anchors['x0'][:4, 2] = 3.
        anchors['x0'][4:, 2] = 40.
        anchors['x1'] = anchors['x0'] + initial[0, :3, 3]
        anchors['matches'] = np.column_stack((project(anchors['x0']), project(anchors['x1'])))
        anchors['confidence'] = np.array([.01]*4 + [1.]*4)
        proposed = initial.copy()
        proposed[0, 0, 3] += .07
        result = guard(anchors, initial, proposed, K, CFG)
        report = result['report']['models'][0]
        self.assertGreater(report['retained_support_fraction'], .98)
        self.assertTrue(result['accepted_mask'][0])

    def test_ambiguous_aliases_are_not_core(self):
        anchors, initial = scene((8,))
        initial = np.repeat(initial, 2, axis=0)
        proposed = initial.copy()
        proposed[0, 0, 3] += .7
        result = guard(anchors, initial, proposed, K, CFG)
        self.assertEqual(result['report']['fixed_core_count'], 0)
        self.assertTrue(result['accepted_mask'].all())

    def test_empty_and_zero_confidence_are_safe(self):
        anchors, initial = scene((8,))
        empty = {k: v[:0] for k, v in anchors.items()}
        result = guard(empty, initial, initial, K, CFG)
        self.assertTrue(result['accepted_mask'].all())
        no_models = guard(empty, [], [], K, CFG)
        self.assertEqual(no_models['models'].shape, (0, 4, 4))
        anchors['confidence'][:] = 0
        result = guard(anchors, initial, initial, K, CFG)
        self.assertEqual(result['report']['models'][0]['reason'], 'zero_initial_support_mass')


if __name__ == '__main__':
    unittest.main()

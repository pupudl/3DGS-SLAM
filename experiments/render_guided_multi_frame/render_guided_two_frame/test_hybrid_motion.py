"""CPU regressions for motion purity, minority support and depth-noise handling.

The image correspondences are projected from metric scenes, while the measured
stereo depths are perturbed independently. Labels remain hidden from discovery.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from hybrid_motion import _merge_aliases, discover_hybrid, motion_costs, motion_probabilities


K = np.array([[500., 0., 640.], [0., 500., 360.], [0., 0., 1.]])
CONFIG = {'min_support': 6, 'max_components': 4,
          'hybrid': {'local_seeds': 48, 'global_rigid_proposals': 128,
                     'competition_iterations': 4}}


def application_config():
    path = Path(__file__).parent / 'configs/kitti360_full_feedback_four_pairs_hybrid_initial.yaml'
    with path.open() as stream:
        return yaml.safe_load(stream)['motion']


def transform(translation, rotation=(0., .01, 0.)):
    motion = np.eye(4)
    motion[:3, :3] = cv2.Rodrigues(np.asarray(rotation, float))[0]
    motion[:3, 3] = translation
    return motion


def project(points):
    return points[:, :2] / points[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]


def backproject(pixels, depths):
    return np.column_stack(((pixels-K[:2, 2]) / [K[0, 0], K[1, 1]] * depths[:, None], depths))


def scene(counts=(160, 8), models=None, depth_noise=(.025, .025), outliers=0, seed=12):
    """Broad background and compact, nonplanar minority object without masks."""
    if models is None:
        models = [transform((-.22, .02, -.35)), transform((.85, .06, -.10))]
    rng = np.random.default_rng(seed)
    pixels0, pixels1, depths0, depths1, labels = [], [], [], [], []
    for index, (count, motion) in enumerate(zip(counts, models)):
        if index == 0:
            uv = rng.uniform([70, 60], [1210, 660], (count, 2))
            depth = rng.uniform(12, 35, count)
        else:
            uv = rng.uniform([510+100*(index-1), 290], [600+100*(index-1), 420], (count, 2))
            depth = rng.uniform(12, 19, count)
        xyz0 = backproject(uv, depth)
        xyz1 = xyz0 @ motion[:3, :3].T + motion[:3, 3]
        pixels0.append(uv)
        pixels1.append(project(xyz1))
        depths0.append(depth + rng.normal(0, depth_noise[0], count))
        depths1.append(xyz1[:, 2] + rng.normal(0, depth_noise[1], count))
        labels.extend([index]*count)
    if outliers:
        pixels0.append(rng.uniform([70, 60], [1210, 660], (outliers, 2)))
        pixels1.append(rng.uniform([70, 60], [1210, 660], (outliers, 2)))
        depths0.append(rng.uniform(12, 35, outliers))
        depths1.append(rng.uniform(12, 35, outliers))
        labels.extend([-1]*outliers)
    uv0, uv1 = np.concatenate(pixels0), np.concatenate(pixels1)
    anchors = {'matches': np.column_stack((uv0, uv1)),
               'x0': backproject(uv0, np.concatenate(depths0)),
               'x1': backproject(uv1, np.concatenate(depths1)),
               'confidence': np.ones(len(uv0))}
    return anchors, np.asarray(labels), models


def rigid_fit(a, b):
    """Unweighted 3D--3D baseline, independent of the hybrid implementation."""
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a-ac).T @ (b-bc))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    motion = np.eye(4)
    motion[:3, :3] = vt.T @ correction @ u.T
    motion[:3, 3] = bc-motion[:3, :3]@ac
    return motion


class HybridMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.minority_scene = scene(outliers=20)
        cls.minority_result = discover_hybrid(cls.minority_scene[0], K, CONFIG, seed=7)

    def test_eight_point_object_survives_without_borrowing_background(self):
        _, truth, _ = self.minority_scene
        models, labels, supports, report = self.minority_result
        self.assertEqual(len(models), 2)
        background = int(np.argmax(np.bincount(labels[truth == 0][labels[truth == 0] >= 0])))
        object_labels = labels[truth == 1]
        assigned = object_labels[object_labels >= 0]
        self.assertGreaterEqual(len(assigned), 7)
        object_id = int(np.argmax(np.bincount(assigned)))
        self.assertNotEqual(background, object_id)
        self.assertGreaterEqual(np.mean(labels[truth == 0] == background), .95)
        self.assertGreaterEqual(np.mean(object_labels == object_id), .875)
        self.assertGreaterEqual(np.mean(truth[labels == object_id] == 1), .90)
        self.assertEqual(supports, [int(np.sum(labels == i)) for i in range(len(models))])
        self.assertEqual(report['reference_index'], 0)
        self.assertEqual(background, 0)

    def test_unexplained_points_remain_unknown(self):
        anchors, truth, _ = self.minority_scene
        models, labels, _, _ = self.minority_result
        self.assertTrue(np.all(labels[truth < 0] == -1))
        probability, reliability = motion_probabilities(motion_costs(anchors, models, K, CONFIG['hybrid']), {})
        np.testing.assert_allclose(probability.sum(1), 1., atol=1e-12)
        self.assertTrue(np.all(reliability[truth < 0] < .01))

    def test_same_essential_geometry_distinct_metric_motions_remain_separate(self):
        # Equal R and parallel t have the same E up to scale. Pixel geometry
        # plus metric depths, rather than an E-only assignment, must split them.
        motions = [transform((.3, 0., .1)), transform((1.05, 0., .35))]
        anchors, truth, _ = scene((140, 10), models=motions, seed=19)
        models, labels, _, _ = discover_hybrid(anchors, K, CONFIG, seed=7)
        self.assertEqual(len(models), 2)
        bg = int(np.argmax(np.bincount(labels[truth == 0][labels[truth == 0] >= 0])))
        obj = int(np.argmax(np.bincount(labels[truth == 1][labels[truth == 1] >= 0])))
        self.assertNotEqual(bg, obj)
        self.assertGreaterEqual(np.mean(labels[truth == 0] == bg), .95)
        self.assertGreaterEqual(np.mean(labels[truth == 1] == obj), .90)
        self.assertGreater(np.linalg.norm(models[bg][:3, 3]-models[obj][:3, 3]), .5)

    def test_no_duplicate_model_for_two_regions_with_identical_motion(self):
        motion = transform((.3, -.05, -.1))
        anchors, _, _ = scene((120, 10), models=[motion, motion], seed=21)
        models, labels, _, _ = discover_hybrid(anchors, K, CONFIG, seed=7)
        self.assertEqual(len(models), 1)
        self.assertGreaterEqual(np.mean(labels == 0), .98)

    def test_two_minority_objects_and_background_remain_pure_with_depth_noise(self):
        motions = [transform((-.22, .02, -.35)), transform((.85, .06, -.10)),
                   transform((.2, -.8, 0.))]
        anchors, truth, _ = scene((160, 8, 10), models=motions,
                                  depth_noise=(.2, .2), outliers=20)
        models, labels, _, _ = discover_hybrid(anchors, K, CONFIG, seed=7)
        self.assertEqual(len(models), 3)
        winners = []
        for identity in range(3):
            group = labels[truth == identity]
            assigned = group[group >= 0]
            self.assertGreater(len(assigned), 0)
            winner = int(np.argmax(np.bincount(assigned)))
            winners.append(winner)
            # Inconsistent noisy depths may abstain; accepted core points must
            # stay pure, rather than being forced into a different motion.
            self.assertGreaterEqual(np.mean(group == winner), .80)
            self.assertTrue(np.all((group == winner) | (group < 0)))
            self.assertEqual(np.mean(truth[labels == winner] == identity), 1.)
        self.assertEqual(len(set(winners)), 3)
        self.assertTrue(np.all(labels[truth < 0] == -1))

    def test_application_config_does_not_split_noisy_background(self):
        anchors, truth, _ = scene((160, 10), depth_noise=(.4, .4), outliers=20)
        config = application_config()
        self.assertEqual(config['hybrid']['local_seeds'], 64)
        self.assertEqual(config['hybrid']['neighborhood_sizes'], [6, 12, 24, 48])
        models, labels, supports, _ = discover_hybrid(anchors, K, config, seed=7)
        self.assertEqual(len(models), 2)
        self.assertGreaterEqual(supports[0], 120)
        self.assertGreaterEqual(supports[1], 8)
        self.assertTrue(np.all((labels[truth == 0] == 0) | (labels[truth == 0] == -1)))
        self.assertTrue(np.all((labels[truth == 1] == 1) | (labels[truth == 1] == -1)))
        np.testing.assert_array_equal(labels[truth < 0], -1)

    def test_hybrid_six_point_override_preserves_exact_small_object(self):
        # Legacy min_support=20 remains for legacy/feedback fitting. The hybrid
        # initializer must honor its own minimum=6 without borrowing background.
        config = application_config()
        self.assertEqual(config['min_support'], 20)
        self.assertEqual(config['hybrid']['min_support'], 6)
        anchors, truth, _ = scene((160, 6), depth_noise=(0., 0.), outliers=20)
        models, labels, supports, report = discover_hybrid(anchors, K, config, seed=7)
        self.assertEqual(len(models), 2)
        self.assertEqual(report['min_support'], 6)
        self.assertEqual(supports, [160, 6])
        np.testing.assert_array_equal(labels[truth == 0], 0)
        np.testing.assert_array_equal(labels[truth == 1], 1)
        np.testing.assert_array_equal(labels[truth < 0], -1)

    def test_duplicate_background_poses_merge_without_erasing_small_object(self):
        anchors, truth, motions = scene((160, 8), depth_noise=(.2, .2))
        alias = motions[0].copy()
        alias[:3, 3] += [.015, 0., .01]
        models, _, merges = _merge_aliases(anchors, [motions[0], alias, motions[1]],
                                         ['rigid', 'pnp', 'essential'], K,
                                         application_config()['hybrid'], 6)
        self.assertEqual(len(models), 2)
        self.assertEqual(len(merges), 1)
        costs = motion_costs(anchors, models, K, application_config()['hybrid'])
        labels = costs.argmin(1)
        self.assertTrue(np.all(labels[truth == 0] == 0))
        self.assertTrue(np.all(labels[truth == 1] == 1))

    def test_correct_pixels_with_absurd_depths_do_not_become_metric_core(self):
        anchors, _, _ = scene((120,), models=[np.eye(4)], depth_noise=(.04, .04))
        rng = np.random.default_rng(5)
        uv = rng.uniform([100, 100], [1100, 600], (8, 2))
        xyz = backproject(uv, rng.uniform(15, 25, 8))
        bad = {'matches': np.column_stack((uv, uv)), 'x0': xyz,
               'x1': xyz*3., 'confidence': np.ones(8)}
        anchors = {key: np.concatenate((value, bad[key])) for key, value in anchors.items()}
        config = application_config()
        # Both images have exactly matching pixels, so capped soft scoring alone
        # treats these rays as plausible despite the threefold depth mismatch.
        cost = motion_costs(anchors, [np.eye(4)], K, config['hybrid'])
        self.assertTrue(np.all(cost[-8:, 0] < config['hybrid']['outlier_cost']))
        models, labels, supports, _ = discover_hybrid(anchors, K, config, seed=7)
        self.assertEqual(len(models), 1)
        self.assertEqual(supports, [120])
        np.testing.assert_array_equal(labels[:120], 0)
        np.testing.assert_array_equal(labels[-8:], -1)

    def test_precise_pixels_reduce_bias_from_noisy_stereo_depth(self):
        anchors, _, _ = scene((150,), models=[transform((.5, -.1, .25), (0., .03, 0.))],
                              depth_noise=(.12, .8), seed=17)
        config = {**CONFIG, 'max_components': 1}
        models, _, _, _ = discover_hybrid(anchors, K, config, seed=7)
        baseline = rigid_fit(anchors['x0'], anchors['x1'])
        uv1 = anchors['matches'][:, 2:]
        def median_pixel_error(motion):
            predicted = project(anchors['x0'] @ motion[:3, :3].T + motion[:3, 3])
            return float(np.median(np.linalg.norm(predicted-uv1, axis=1)))
        self.assertLess(median_pixel_error(models[0]), .75*median_pixel_error(baseline))
        self.assertLess(median_pixel_error(models[0]), 1.)

    def test_same_seed_reproduces_models_and_assignments(self):
        repeated = discover_hybrid(self.minority_scene[0], K, CONFIG, seed=7)
        np.testing.assert_array_equal(repeated[1], self.minority_result[1])
        np.testing.assert_allclose(np.stack(repeated[0]), np.stack(self.minority_result[0]), atol=1e-10)
        self.assertEqual(repeated[3]['proposal_attempts'], self.minority_result[3]['proposal_attempts'])

    def test_numpy_and_differentiable_costs_agree_on_cpu(self):
        import torch
        from joint_pair import hybrid_motion_costs_torch
        anchors, _, motions = scene((20, 8), depth_noise=(.2, .4))
        # Include a candidate with points behind the target camera.
        invalid = np.eye(4)
        invalid[2, 3] = -100.
        motions.append(invalid)
        cfg = {'reprojection_threshold_px': 2.5, 'depth_sigma_m': .25,
               'depth_sigma_relative': .02, 'depth_weight': .3, 'depth_cost_cap': 3.}
        tensors = [torch.tensor(x, dtype=torch.float64) for x in
                   (anchors['x0'], anchors['x1'], anchors['matches'][:, :2],
                    anchors['matches'][:, 2:], np.stack(motions), K)]
        tensors[4].requires_grad_(True)
        cost = hybrid_motion_costs_torch(*tensors, cfg)
        np.testing.assert_allclose(cost.detach().numpy(), motion_costs(anchors, motions, K, cfg),
                                   rtol=1e-8, atol=1e-8)
        cost[:, :2].sum().backward()
        self.assertTrue(torch.isfinite(tensors[4].grad).all())


if __name__ == '__main__':
    unittest.main()

"""Regressions for terminal, gray-only recovery of swallowed hypotheses."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hybrid_motion import _recover_unassigned, _refit, discover_hybrid
from test_hybrid_motion import K, application_config, backproject, scene, transform


class GrayRecoveryTests(unittest.TestCase):
    def recover(self, anchors, motions, labels, prior=None, maximum=8, mode=None):
        cfg = application_config()['hybrid']
        if mode is not None:
            cfg['gray_recovery_mode'] = mode
        return _recover_unassigned(
            anchors, [motions[0]], labels, ['pnp'], prior or motions,
            ['pnp']*len(prior or motions), K, cfg, 6, maximum)

    def test_default_mode_matches_explicit_all_unassigned(self):
        anchors, truth, motions = scene((100, 12), outliers=8)
        labels = np.where(truth == 0, 0, -1)
        legacy = self.recover(anchors, motions, labels)
        explicit = self.recover(anchors, motions, labels, mode='all_unassigned')
        np.testing.assert_array_equal(legacy[0], explicit[0])
        np.testing.assert_array_equal(legacy[1], explicit[1])
        self.assertEqual(legacy[2:], explicit[2:])

    def test_unexplained_mode_filters_yellow_points_before_refitting(self):
        anchors, truth, motions = scene((100, 12))
        labels = np.where(truth == 0, 0, -1)
        labels[90:100] = -1  # Background observations already explained well.
        with patch('hybrid_motion._refit', wraps=_refit) as fit:
            models, final, _, report = self.recover(
                anchors, motions, labels, mode='unexplained_only')
        self.assertEqual(len(models), 2)
        self.assertEqual(report['excluded_yellow_gray_points'], 10)
        self.assertEqual(report['eligible_gray_points'], 12)
        fit.assert_called_once()
        np.testing.assert_array_equal(fit.call_args.args[2], np.flatnonzero(truth == 1))
        np.testing.assert_array_equal(final[90:100], -1)
        np.testing.assert_array_equal(final[truth == 1], 1)
        np.testing.assert_array_equal(final[:90], 0)

    def test_small_motion_already_explained_is_excluded_only_in_strict_mode(self):
        motions = [transform((0., 0., 0.)), transform((.065, 0., 0.))]
        anchors, truth, motions = scene((100, 12), models=motions, depth_noise=(0., 0.))
        labels = np.where(truth == 0, 0, -1)
        all_gray = self.recover(anchors, motions, labels, mode='all_unassigned')
        strict = self.recover(anchors, motions, labels, mode='unexplained_only')
        self.assertEqual(len(all_gray[0]), 2)
        self.assertEqual(len(strict[0]), 1)
        self.assertEqual(strict[3]['excluded_yellow_gray_points'], 12)
        np.testing.assert_array_equal(strict[1], labels)

    def test_exact_threshold_and_final_acceptance_use_same_selection(self):
        # All nine gray points fit the recovered model perfectly. Only six
        # have baseline cost >= 1; the three below 1 must never get labels.
        anchors = dict(matches=np.zeros((9,4)), x0=np.zeros((9,3)),
                       x1=np.zeros((9,3)), confidence=np.ones(9))
        old, recovered = np.eye(4), np.eye(4)
        recovered[0,3] = 1.
        baseline_cost = np.array([.999]*3+[1.]*3+[1.001]*3)

        def costs(anchors, models, K, cfg):
            return np.column_stack([baseline_cost if H[0,3] == 0 else np.zeros(9)
                                    for H in models])

        with patch('hybrid_motion.motion_costs', side_effect=costs), \
             patch('hybrid_motion._metric_consistency',
                   side_effect=lambda a,h,c: np.ones((9,len(h)),bool)), \
             patch('hybrid_motion._refit', side_effect=lambda a,h,i,k,c: h.copy()) as fit:
            models, labels, _, report = self.recover(
                anchors, [old], np.full(9,-1), prior=[recovered], mode='unexplained_only')
        self.assertEqual(len(models), 2)
        np.testing.assert_array_equal(fit.call_args.args[2], np.arange(3,9))
        np.testing.assert_array_equal(labels[:3], -1)
        np.testing.assert_array_equal(labels[3:], 1)
        self.assertEqual(report['excluded_yellow_gray_points'], 3)

    def test_invalid_mode_is_rejected(self):
        anchors, truth, motions = scene((100, 12))
        with self.assertRaisesRegex(ValueError, 'gray_recovery_mode'):
            self.recover(anchors, motions, np.where(truth == 0,0,-1), mode='typo')

    def test_recovers_missing_motion_without_duplicate_or_outlier_assignment(self):
        anchors, truth, motions = scene((100, 12), outliers=10)
        labels = np.where(truth == 0, 0, -1)
        alias = motions[1].copy()
        alias[:3, 3] += [.003, 0., 0.]
        models, final, _, report = self.recover(
            anchors, motions, labels, [motions[0], motions[1], alias])
        self.assertEqual(len(models), 2)
        np.testing.assert_array_equal(final[truth == 0], 0)
        np.testing.assert_array_equal(final[truth == 1], 1)
        np.testing.assert_array_equal(final[truth < 0], -1)
        np.testing.assert_array_equal(models[0], motions[0])
        self.assertEqual(report['recovered_points'], 12)
        self.assertEqual(report['passes'], 1)

    def test_recovery_does_not_change_previously_assigned_points(self):
        anchors, truth, motions = scene((100, 12))
        labels = np.where(truth == 0, 0, -1)
        labels[np.flatnonzero(truth == 1)[0]] = 0
        _, final, _, report = self.recover(anchors, motions, labels)
        np.testing.assert_array_equal(final[labels >= 0], labels[labels >= 0])
        self.assertEqual(report['recovered_points'], 11)

    def test_five_points_and_full_component_budget_do_not_spawn_motion(self):
        for count, maximum in [(5, 8), (12, 1)]:
            with self.subTest(count=count, maximum=maximum):
                anchors, truth, motions = scene((100, count))
                labels = np.where(truth == 0, 0, -1)
                models, final, _, _ = self.recover(
                    anchors, motions, labels, maximum=maximum)
                self.assertEqual(len(models), 1)
                np.testing.assert_array_equal(final, labels)

    def test_capped_depth_outliers_cannot_be_restored(self):
        anchors, _, motions = scene((100,), models=[np.eye(4)])
        rng = np.random.default_rng(17)
        uv = rng.uniform([100, 100], [1000, 600], (8, 2))
        xyz = backproject(uv, np.full(8, 20.))
        bad = dict(matches=np.column_stack((uv, uv)), x0=xyz, x1=xyz*3,
                   confidence=np.ones(8))
        anchors = {k: np.concatenate((v, bad[k])) for k,v in anchors.items()}
        labels = np.r_[np.zeros(100, int), np.full(8, -1)]
        models, final, _, _ = self.recover(anchors, motions, labels)
        self.assertEqual(len(models), 1)
        np.testing.assert_array_equal(final[-8:], -1)

    def test_old_alias_must_improve_over_current_motion(self):
        anchors, _, motions = scene((100,))
        labels = np.r_[np.zeros(90, int), np.full(10, -1)]
        alias = motions[0].copy()
        alias[:3, 3] += [.02, 0., 0.]
        models, final, _, _ = self.recover(
            anchors, motions, labels, [motions[0], alias])
        self.assertEqual(len(models), 1)
        np.testing.assert_array_equal(final, labels)

    def test_full_initializer_recovers_motion_after_forced_merge(self):
        anchors, truth, _ = scene((120, 12), outliers=10)
        cfg = application_config()
        cfg['hybrid']['local_seeds'] = 32
        cfg['hybrid']['global_rigid_proposals'] = 64

        def swallow(anchors, models, sources, K, cfg, minimum):
            return [models[0].copy()], [sources[0]], []

        with patch('hybrid_motion._merge_aliases', side_effect=swallow):
            baseline = discover_hybrid(anchors, K, cfg, seed=7)
            enabled = deepcopy(cfg)
            enabled['hybrid']['recover_unassigned_after_merge'] = True
            restored = discover_hybrid(anchors, K, enabled, seed=7)
        old_models, old_labels = baseline[:2]
        models, labels, _, report = restored
        np.testing.assert_array_equal(models[:len(old_models)], old_models)
        np.testing.assert_array_equal(labels[old_labels >= 0], old_labels[old_labels >= 0])
        self.assertTrue(np.all(old_labels[truth == 1] < 0))
        self.assertGreaterEqual(np.count_nonzero(labels[truth == 1] >= 0), 10)
        self.assertTrue(report['gray_recovery']['existing_models_unchanged'])


if __name__ == '__main__':
    unittest.main()

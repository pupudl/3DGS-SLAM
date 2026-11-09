"""Regression checks for masked collection, retryable evidence and motion repair."""
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
import yaml

from masked_feedback import (PendingMatches, balanced_indices, fit_proposals,
                             global_problem_mask, minimum_errors, problem_mask, region_gate,
                             score_guided_mask, spaced_indices, warm_ownership)
from pair_utils import full_image_masked_matches, masked_features
from visualize_feedback_final import motion_lineage


class MaskedFeedbackTests(unittest.TestCase):
    def test_composite_selects_joint_moderate_evidence_without_separate_thresholds(self):
        residual = np.full((10, 10), .06, np.float32)
        counts = np.ones_like(residual)
        gradients = counts*.30
        config = dict(score_smoothing_sigma_px=0, normalize_gradient=False,
                      gradient_weight=.25, min_score=.12, mask_min_area_px=1,
                      mask_budget_pixels=100)
        mask, score, _ = score_guided_mask(residual, counts.astype(bool), (gradients, counts), config)
        self.assertTrue(mask.all())
        np.testing.assert_allclose(score, .135)
        other, _, _ = score_guided_mask(residual, counts.astype(bool), (gradients, counts),
            {**config, 'mask_residual_threshold':999, 'mask_gradient_threshold':999,
             'patch_size_px':1, 'max_regions':0})
        np.testing.assert_array_equal(mask, other)

    def test_composite_budget_keeps_highest_scores_and_never_grows(self):
        residual = np.arange(100, dtype=np.float32).reshape(10, 10)/100
        zero = np.zeros_like(residual)
        mask, score, stats = score_guided_mask(residual, np.ones_like(residual, bool),
            (zero, zero), dict(score_smoothing_sigma_px=0, min_score=.12,
                               mask_min_area_px=1, mask_budget_pixels=10))
        self.assertEqual(mask.sum(), 10)
        self.assertTrue(mask[-1].all())
        self.assertGreaterEqual(score[mask].min(), .9-1e-6)
        self.assertEqual(stats['budget_pixels'], 10)

    def test_composite_gradient_mean_respects_projected_soft_mass(self):
        residual = np.zeros((15, 15), np.float32)
        counts = np.full_like(residual, 100)
        counts[7, 7] = 1
        values = np.zeros_like(residual)
        values[7, 7] = 1
        mask, score, _ = score_guided_mask(residual, np.ones_like(residual, bool),
            (values, counts), dict(score_smoothing_sigma_px=1, normalize_gradient=False,
                                   gradient_weight=1., min_score=.12, mask_min_area_px=1))
        self.assertLess(score[7, 7], .01)
        self.assertFalse(mask.any())

    def test_composite_ignores_unreliable_depth_when_allocating_budget(self):
        residual = np.full((10, 10), .5, np.float32)
        confidence = np.ones_like(residual)
        confidence[:, :5] = 0
        zero = np.zeros_like(residual)
        mask, _, _ = score_guided_mask(residual, np.ones_like(residual, bool), (zero, zero),
            dict(score_smoothing_sigma_px=0, mask_min_area_px=1, mask_budget_pixels=10), confidence)
        self.assertEqual(mask.sum(), 10)
        self.assertFalse(mask[:, :5].any())

    def test_composite_preserves_original_full_projection_normalization_domain(self):
        residual = np.full((10, 10), .2, np.float32)
        valid = np.zeros_like(residual, bool)
        valid[:, :5] = True
        counts = np.ones_like(residual)
        values = np.zeros_like(residual)
        values[:, 5:] = 10
        _, _, stats = score_guided_mask(residual, valid, (values, counts),
            dict(score_smoothing_sigma_px=0, mask_min_area_px=1))
        self.assertEqual(stats['gradient_scale'], 10.)

    def test_composite_missing_observations_contribute_zero_to_residual_mean(self):
        residual = np.full((15, 15), .2, np.float32)
        valid = np.ones_like(residual, bool)
        valid[7, 7] = False
        zero = np.zeros_like(residual)
        _, priority, _ = score_guided_mask(residual, valid, (zero, zero),
            dict(score_smoothing_sigma_px=1, mask_min_area_px=1))
        self.assertLess(priority[7, 8], .199)

    def test_composite_empty_evidence_and_zero_budget_are_safe(self):
        residual = np.full((10, 10), .5, np.float32)
        zero = np.zeros_like(residual)
        for valid, config in [(np.zeros_like(residual, bool), {}),
                              (np.ones_like(residual, bool), dict(mask_budget_pixels=0))]:
            mask, priority, _ = score_guided_mask(residual, valid, (zero, zero), config)
            self.assertFalse(mask.any())
            self.assertTrue(np.isfinite(priority).all())

    def test_composite_keeps_irregular_shape_and_excludes_low_score_holes(self):
        residual = np.zeros((64, 64), np.float32)
        residual[10:15, 10:40] = .2
        residual[10:40, 10:15] = .2
        zero = np.zeros_like(residual)
        mask, _, _ = score_guided_mask(residual, np.ones_like(residual, bool), (zero, zero),
            dict(score_smoothing_sigma_px=0, mask_min_area_px=1, mask_budget_pixels=500))
        np.testing.assert_array_equal(mask, residual >= .12)
        self.assertFalse(mask[30, 30])

    def test_target_budget_is_spent_inside_mask(self):
        features = dict(keypoints=torch.tensor([[[2., 2.], [5., 5.], [6., 6.]]]),
                        keypoint_scores=torch.tensor([[.99, .7, .8]]),
                        descriptors=torch.arange(12).reshape(1, 3, 4))
        mask = np.zeros((10, 10), bool)
        mask[4:8, 4:8] = True
        selected = masked_features(features, mask, 1)
        np.testing.assert_array_equal(selected['keypoints'].numpy(), [[[6., 6.]]])
        np.testing.assert_array_equal(selected['descriptors'].numpy(), [[[8, 9, 10, 11]]])
        self.assertEqual(masked_features(features, np.zeros_like(mask), 1)['keypoints'].shape[1], 0)

    def test_mask_keeps_problem_pixels_and_search_window(self):
        score = np.zeros((30, 30), np.float32)
        score[10:20, 10:20] = .5
        score[:5, :5] = 1
        valid = np.ones_like(score, bool)
        valid[15, 15] = False
        zero = np.zeros_like(score)
        mask, _ = problem_mask(score, valid, [(8, 8, 22, 22)], (zero, zero),
                               dict(mask_min_area_px=4, mask_dilation_px=1))
        self.assertTrue(mask[12, 12])
        self.assertFalse(mask[15, 15])
        self.assertFalse(mask[2, 2])
        self.assertFalse(mask[8, 8])

    def test_global_mask_finds_problems_outside_old_patches_and_at_image_border(self):
        score = np.zeros((120, 240), np.float32)
        score[:8, 210:220] = .5  # Old border/window selection would exclude this.
        score[50:60, 92:100] = .5  # Cross the former 96px grid boundary.
        valid = np.ones_like(score, bool)
        zero = np.zeros_like(score)
        config = dict(mask_min_area_px=4, mask_dilation_px=0)
        mask, _, _ = global_problem_mask(score, valid, (zero, zero), config)
        np.testing.assert_array_equal(mask, score > .12)
        other, _, _ = global_problem_mask(score, valid, (zero, zero),
            {**config, 'patch_size_px': 1, 'max_regions': 0, 'border_margin_px': 1000})
        np.testing.assert_array_equal(mask, other)

    def test_global_mask_uses_either_pixel_residual_or_ownership_gradient(self):
        score = np.zeros((40, 40), np.float32)
        score[4:10, 4:10] = .5
        counts = np.zeros_like(score)
        counts[25:30, 25:30] = 1
        mask, _, signals = global_problem_mask(score, np.ones_like(score, bool),
            (counts.copy(), counts), dict(mask_dilation_px=0, mask_min_area_px=4))
        self.assertTrue(mask[6, 6])
        self.assertTrue(mask[27, 27])
        self.assertFalse(mask[17, 17])
        self.assertTrue(signals['residual_trigger'][6, 6])
        self.assertTrue(signals['gradient_trigger'][27, 27])

    def test_irregular_mask_is_not_replaced_by_its_bounding_rectangle(self):
        score = np.zeros((64, 64), np.float32)
        score[10:15, 10:40] = .5
        score[10:40, 10:15] = .5
        zero = np.zeros_like(score)
        mask, _, _ = global_problem_mask(score, np.ones_like(score, bool), (zero, zero),
                                         dict(mask_dilation_px=0, mask_min_area_px=4))
        np.testing.assert_array_equal(mask, score > .12)
        self.assertFalse(mask[30, 30])

    def test_full_frame_matching_does_not_crop_or_match_an_empty_mask(self):
        first = np.zeros((120, 240, 3), np.uint8)
        second = first.copy()
        mask = np.zeros((120, 240), bool)
        with patch('pair_utils.match_images', return_value=np.empty((0, 4))) as matcher:
            full_image_masked_matches(first, second, mask, None, None)
            matcher.assert_not_called()
            mask[5, 230] = True
            full_image_masked_matches(first, second, mask, None, None)
            self.assertIs(matcher.call_args.args[0], first)
            self.assertIs(matcher.call_args.args[1], second)
            self.assertIs(matcher.call_args.kwargs['target_mask'], mask)

    def test_pending_spacing_has_no_square_cell_boundary(self):
        points = np.array([[0, 0, 95, 20], [0, 0, 97, 20], [0, 0, 120, 20]], np.float32)
        ids = spaced_indices(points, np.array([1., .9, .8]), np.zeros((40, 140)),
                             dict(pending_min_distance_px=8, pending_max_active=10))
        np.testing.assert_array_equal(ids, [0, 2])

    def test_pending_survives_rejection_and_is_removed_only_when_retained(self):
        pool = PendingMatches()
        points = np.array([[10, 10, 20, 20], [30, 30, 40, 40]], np.float32)
        pool.add(points)
        pool.add(points)
        self.assertEqual(len(pool.matches), 2)
        pool.remove_retained(points[:1])
        np.testing.assert_array_equal(pool.matches, points[1:])

    def test_spatial_budget_does_not_get_consumed_by_one_dense_cell(self):
        points = np.zeros((20, 4), np.float32)
        points[:, 2:] = 2
        points[-1, 2:] = 40
        ids = balanced_indices(points, np.ones(20), np.zeros((64, 64)),
                               dict(pending_cell_px=24, pending_per_cell=2, pending_max_active=2))
        self.assertIn(19, ids)

    def test_global_improvement_cannot_hide_problem_region_regression(self):
        mask = np.zeros((10, 10), bool)
        mask[:2] = True
        coverage = np.ones_like(mask)
        before = dict(score=np.ones((10, 10)), coverage=coverage)
        score = np.full((10, 10), .5)
        score[mask] = 1.1
        self.assertFalse(region_gate(before, dict(score=score, coverage=coverage), mask, {})['accepted'])

    def test_local_improvement_cannot_hide_loss_of_local_coverage(self):
        mask = np.zeros((10, 10), bool)
        mask[:2] = True
        coverage = np.ones_like(mask)
        before = dict(score=np.ones((10, 10)), coverage=coverage)
        after_coverage = coverage.copy()
        after_coverage[0] = False
        after = dict(score=np.full((10, 10), .5), coverage=after_coverage)
        self.assertFalse(region_gate(before, after, mask,
                                     dict(feedback_min_coverage_ratio=.8))['accepted'])

    def test_gray_band_points_can_repair_a_motion_with_weak_old_ownership(self):
        cfg = yaml.safe_load((Path(__file__).parent / 'configs' /
            'kitti360_full_feedback_four_pairs_ransac8_depth_fit_masked_repair.yaml').read_text())
        rng = np.random.default_rng(11)
        xyz = rng.normal(0, .2, (36, 3)).astype(np.float32)
        xyz[:, 2] += 20
        target = xyz + np.array([1., 0., 0.], np.float32)
        pixels = np.array([(x*8+100, y*8+100) for y in range(6) for x in range(6)], np.float32)
        anchors = dict(x0=xyz, x1=target, matches=np.column_stack((pixels, pixels+10)),
                       confidence=np.full(36, .8, np.float32))
        models = np.stack([np.eye(4), np.eye(4)])
        models[1, 0, 3] = .4
        state = dict(H=models, q=np.tile([.99, .01], (36, 1)), reference=0)
        self.assertTrue(np.all((minimum_errors(anchors, models) > .5) &
                               (minimum_errors(anchors, models) < .8)))
        proposals = fit_proposals(state, anchors, dict(uv=pixels), cfg, 0)
        repairs = [p for p in proposals if p['kind'] == 'repair']
        self.assertTrue(repairs)
        self.assertEqual(repairs[0]['component'], 1)
        self.assertLess(minimum_errors(anchors, repairs[0]['models']).max(), 1e-4)

    def test_repair_preserves_identity_across_later_prune(self):
        cycles = [dict(cycle=0, proposals=[
            dict(kind='repair', component=2, accepted=True),
            dict(kind='prune', component=1, accepted=True)])]
        identities, events = motion_lineage(3, cycles)
        self.assertEqual(identities, [0, 2])
        self.assertEqual(events[0]['repaired_identity'], 2)

    def test_repaired_vehicle_keeps_m1_while_contaminating_background_returns_to_m0(self):
        cfg = yaml.safe_load((Path(__file__).parent / 'configs' /
            'kitti360_full_feedback_four_pairs_ransac8_depth_fit_masked_repair.yaml').read_text())
        xyz = np.array([[0, 0, 20], [-4, 0, 20]], np.float32)
        uv = np.array([[10, 10], [20, 10]], np.float32)
        target = xyz.copy()
        target[0, 0] += 1
        anchors = dict(x0=xyz, x1=target, matches=np.column_stack((uv, uv)),
                       confidence=np.ones(2, np.float32))
        models = np.stack([np.eye(4), np.eye(4)])
        models[1, 0, 3] = 1
        state = dict(q=np.array([[.01, .99], [.01, .99]], np.float32), reference=0)
        gmap = dict(xyz=xyz, uv=uv, color=np.zeros((2, 3), np.float32))
        q = warm_ownership(state, list(models), anchors, anchors, gmap,
                           np.zeros((30, 30, 3), np.uint8), cfg)
        np.testing.assert_array_equal(q.argmax(axis=1), [1, 0])


if __name__ == '__main__':
    unittest.main()

"""Core-grown surface coverage must fill uncertain bodies without crossing stops."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP)) if str(EXP) not in sys.path else None
from ownership.core_region_completion import (build_region_completion, region_completion_loss,
                                             initialize_region_probabilities)
from ownership.object_connectivity_constraints import build_connectivity_state


def scene(edit=None, three_objects=False):
    snapshots = {}
    n = 10
    for first in range(2):
        costs = np.array([[.9, .05]]*3+[[.5, .45]]*5+[[.05, .9]]*2)
        if three_objects:
            costs = np.column_stack((costs, np.ones(n)))
        snap = dict(first=first, second=first+1, gaussian_ids=np.arange(n)+100,
                    object_ids=np.arange(costs.shape[1]), costs=costs, weights=np.ones_like(costs),
                    track_ids=np.r_[np.arange(3), np.full(n-3, -1)],
                    payload=dict(uv=np.column_stack((np.arange(n), np.ones(n))),
                                 xyz=np.column_stack((np.arange(n)*.1, np.zeros(n), np.full(n, 5.))),
                                 color=np.full((n, 3), .3), object_model_valid=np.ones(costs.shape[1], bool)))
        if edit:
            edit(snap)
        snapshots[first, first+1] = snap
    window = SimpleNamespace(_active=snapshots, _active_gids=np.arange(n)+100,
                             object_ids=list(range(costs.shape[1])))
    state = build_connectivity_state(window)
    return window, state


class CoreRegionCompletionTests(unittest.TestCase):
    def build(self, edit=None, three_objects=False, **options):
        window, state = scene(edit, three_objects)
        return build_region_completion(window, state, dict(enabled=True, **options))

    def test_grows_beyond_three_cores_into_uncertain_body_and_stops_at_road(self):
        result = self.build()
        np.testing.assert_array_equal(np.unique(result['rows']), np.arange(8))
        self.assertTrue(np.all(result['columns'] == 1))
        self.assertFalse(result['report']['targets_from_current_posterior'])
        self.assertFalse(result['report']['supplies_motion_cores'])

    def test_depth_discontinuity_stops_growth_and_does_not_jump_across(self):
        def edit(snap):
            snap['payload']['xyz'][5:, 2] += 2
        np.testing.assert_array_equal(np.unique(self.build(edit)['rows']), np.arange(5))

    def test_sharp_image_color_edge_stops_growth(self):
        def edit(snap):
            snap['payload']['color'][5:] = 1
        np.testing.assert_array_equal(np.unique(self.build(edit)['rows']), np.arange(5))

    def test_background_evidence_stops_growth_even_with_continuous_geometry(self):
        def edit(snap):
            snap['costs'][5] = [.01, .45]
        np.testing.assert_array_equal(np.unique(self.build(edit)['rows']), np.arange(5))

    def test_other_motion_evidence_stops_growth(self):
        def edit(snap):
            snap['costs'][5, 2] = .01
        np.testing.assert_array_equal(np.unique(self.build(edit, True)['rows']), np.arange(5))

    def test_no_motion_core_means_no_completion(self):
        window, state = scene()
        state['core_by_object'] = {}
        self.assertEqual(len(build_region_completion(window, state, dict(enabled=True))['rows']), 0)

    def test_current_invalid_motion_does_not_gain_targets_from_old_cores(self):
        def edit(snap):
            snap['payload']['object_model_valid'][1] = snap['first'] == 0
        result = self.build(edit)
        self.assertEqual(set(result['by_pair']), {(0, 1)})

    def test_incoming_rgbd_fit_can_complete_last_visible_frame_without_outgoing_motion(self):
        def edit(snap):
            models = np.repeat(np.eye(4)[None], 2, axis=0)
            models[0, 2, 3] = 1.
            snap['payload'].update(H_object=models, K=np.array([[10., 0, 2], [0, 10., 2], [0, 0, 1.]]))
            snap['payload']['object_model_valid'][1] = snap['first'] == 0
            snap['render_observations'] = dict(source=np.full((8, 16, 3), 76, np.uint8),
                source_depth=np.full((8, 16), 5.), source_confidence=np.ones((8, 16)))
        result = self.build(edit)
        self.assertEqual(set(result['by_pair']), {(0, 1), (1, 2)})
        current = result['report']['frames'][1]['objects']['1']
        self.assertEqual(current['observation_direction'], 'incoming')
        self.assertGreater(current['target_count'], 3)

    def test_incoming_check_requires_actual_rgbd_support(self):
        def edit(snap):
            models = np.repeat(np.eye(4)[None], 2, axis=0)
            models[0, 2, 3] = 1.
            snap['payload'].update(H_object=models, K=np.eye(3))
            snap['payload']['object_model_valid'][1] = snap['first'] == 0
            snap['render_observations'] = dict(source=np.full((8, 16, 3), 76, np.uint8),
                source_depth=np.full((8, 16), np.nan), source_confidence=np.ones((8, 16)))
        result = self.build(edit)
        self.assertEqual(set(result['by_pair']), {(0, 1)})

    def test_weak_or_missing_measurement_blocks_propagation(self):
        for value in (0., np.nan):
            def edit(snap):
                snap['weights'][5, 1] = value
            np.testing.assert_array_equal(np.unique(self.build(edit)['rows']), np.arange(5))

    def test_metric_distance_bounds_growth(self):
        result = self.build(max_distance_m=.21)
        np.testing.assert_array_equal(np.unique(result['rows']), np.arange(5))

    def test_distant_camera_origins_do_not_break_same_frame_geometry(self):
        def edit(snap):
            snap['payload']['xyz'][:, 0] += 1000*snap['first']
        np.testing.assert_array_equal(np.unique(self.build(edit)['rows']), np.arange(8))

    def test_height_exclusion_stops_growth(self):
        window, state = scene()
        state['height_keep'][5] = False
        result = build_region_completion(window, state, dict(enabled=True))
        np.testing.assert_array_equal(np.unique(result['rows']), np.arange(5))

    def test_diffuse_core_does_not_expand_into_a_large_background_region(self):
        window, state = scene()
        for snap in window._active.values():
            snap['payload']['xyz'][2, 0] = 20
        result = build_region_completion(window, state, dict(enabled=True))
        self.assertEqual(len(result['rows']), 0)
        self.assertEqual(result['report']['frames'][0]['objects']['1']['action'], 'skip_spread_core')

    def test_competing_core_regions_leave_overlap_uncertain(self):
        def edit(snap):
            snap['costs'][:3] = [.9, .05, .9]
            snap['costs'][3:7] = [.6, .4, .4]
            snap['costs'][7:] = [.9, .9, .05]
            snap['track_ids'][7:] = [7, 8, 9]
        result = self.build(edit, True)
        np.testing.assert_array_equal(np.unique(result['rows'][result['columns'] == 1]), [0, 1, 2])
        np.testing.assert_array_equal(np.unique(result['rows'][result['columns'] == 2]), [7, 8, 9])

    def test_targets_do_not_change_costs_cores_or_input_posteriors(self):
        window, state = scene()
        before = [s['costs'].copy() for s in window._active.values()]
        cores = state['core_by_object'][1].copy()
        build_region_completion(window, state, dict(enabled=True))
        np.testing.assert_array_equal(state['core_by_object'][1], cores)
        for old, snap in zip(before, window._active.values()):
            np.testing.assert_array_equal(snap['costs'], old)
            self.assertNotIn('region_completion_gaussian_ids', snap['payload'])

    def test_region_loss_fills_body_without_training_outside_region(self):
        result = self.build()
        logits = torch.nn.Parameter(torch.tensor([[0., -2.]]*10))
        before = logits.detach().clone()
        optimizer = torch.optim.Adam([logits], lr=.15)
        rows, cols = torch.tensor(result['rows']), torch.tensor(result['columns'])
        weights = torch.tensor(result['weights'])
        for _ in range(35):
            optimizer.zero_grad()
            loss = region_completion_loss(logits.log_softmax(1), rows, cols, weights)
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.all(logits.softmax(1)[:8, 1] > .9))
        torch.testing.assert_close(logits[8:], before[8:])

    def test_only_lk_config_enables_completion(self):
        from run_sequence import load_config
        lg = load_config(EXP/'configs/kitti360_sequence.yaml')
        lk = load_config(EXP/'configs/kitti360_sequence_flow.yaml')
        self.assertFalse(lg['object_window'].get('region_completion', {}).get('enabled', False))
        self.assertTrue(lk['object_window']['region_completion']['enabled'])

    def test_zero_probability_surface_gets_soft_start_without_changing_other_rows(self):
        region = self.build()
        before = np.tile([1., 0.], (10, 1))
        result, changed = initialize_region_probabilities(before, region, .1)
        self.assertEqual(changed, 8)
        np.testing.assert_allclose(result[:8], [[.9, .1]]*8)
        np.testing.assert_array_equal(result[8:], before[8:])
        np.testing.assert_array_equal(before, [[1., 0.]]*10)

    def test_no_core_old_winner_cannot_lock_verified_body_to_background(self):
        from ownership.object_spatial_constraints import prepare_spatial_constraints
        window, _ = scene(three_objects=True)
        before = np.tile([.01, .05, .94], (10, 1))
        window._render_initial_probabilities = before.copy()
        result = prepare_spatial_constraints(window, dict(
            connectivity=dict(enabled=True, no_core_to_background=True),
            region_completion=dict(enabled=True)))
        self.assertFalse(result['hard_background_mask'][:8].any())
        self.assertTrue(result['hard_background_mask'][8:].all())
        self.assertTrue(result['no_core_disabled_columns'][2])
        np.testing.assert_allclose(result['no_core_initial_probabilities'][:8, 1], .1)
        np.testing.assert_array_equal(window._render_initial_probabilities, before)
        self.assertEqual(result['report']['connectivity']['graph']['objects']['1']['core_count'], 3)

    def test_invalid_or_nonfinite_growth_options_rejected(self):
        for options in ({'max_distance_m': 0}, {'max_distance_m': float('nan')},
                        {'min_weight': -1}, {'min_color_affinity': 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.build(**options)


if __name__ == '__main__':
    unittest.main()

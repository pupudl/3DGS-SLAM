"""CPU checks for evidence-selected cores and revisable connectivity losses."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / 'ownership/object_connectivity_constraints.py'
_spec = importlib.util.spec_from_file_location('object_connectivity_constraints', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
build_connectivity_state = _module.build_connectivity_state
update_connectivity_targets = _module.update_connectivity_targets
no_core_background_policy = _module.no_core_background_policy


def snapshot(ids, uv, costs, tracks=None, xyz=None, colors=None):
    ids = np.asarray(ids, np.int64)
    uv = np.asarray(uv, float)
    xyz = (np.column_stack((uv[:, 0]*.1, uv[:, 1]*.1, np.full(len(ids), 5.)))
           if xyz is None else np.asarray(xyz, float))
    return dict(gaussian_ids=ids, object_ids=np.array([0, 1]),
                costs=np.asarray(costs, float), weights=np.ones((len(ids), 2)),
                track_ids=np.full(len(ids), -1) if tracks is None else np.asarray(tracks),
                payload=dict(uv=uv, xyz=xyz,
                             color=np.zeros((len(ids), 3)) if colors is None else np.asarray(colors)))


def window(snapshots):
    gids = np.unique(np.concatenate([snap['gaussian_ids'] for snap in snapshots]))
    return SimpleNamespace(_active_gids=gids, object_ids=[0, 1],
                           _active={(i, i+1): snap for i, snap in enumerate(snapshots)})


def scene(background_count=12, contiguous=False, untracked_background_core=False):
    count = 3+background_count
    x = np.r_[np.arange(3), np.arange(background_count)+(3 if contiguous else 10)]
    uv = np.column_stack((x, np.zeros(count)))
    costs = np.vstack((np.tile([.9, .05], (3, 1)),
                       np.tile([.99, .01] if untracked_background_core else [.05, .9],
                               (background_count, 1))))
    tracks = np.r_[np.arange(3), np.full(background_count, -1)]
    snap = snapshot(np.arange(count), uv, costs, tracks)
    copies = 3 if untracked_background_core else 2
    return window([snap]*copies)


class ObjectConnectivityTests(unittest.TestCase):
    def test_large_wrong_background_block_is_not_selected_by_probability_or_area(self):
        state = build_connectivity_state(scene(background_count=40))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        p = np.tile([.01, .99], (43, 1))
        result = update_connectivity_targets(state, p)
        np.testing.assert_array_equal(result['penalty'][:3, 1], 0)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))
        self.assertFalse(result['report']['forces_background_assignment'])

    def test_actual_track_motion_support_wins_over_larger_untracked_false_island(self):
        state = build_connectivity_state(scene(40, untracked_background_core=True))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        self.assertEqual(state['diagnostics']['objects']['1']['rejected_core_candidates'], 40)

    def test_background_is_exempt_from_connectivity_and_anchors(self):
        state = build_connectivity_state(scene())
        result = update_connectivity_targets(state, np.tile([.05, .95], (15, 1)))
        np.testing.assert_array_equal(result['penalty'][:, 0], 0)
        self.assertFalse(np.any(result['seed_columns'] == 0))
        self.assertEqual(result['report']['objects']['0']['action'], 'background_exempt')

    def test_shared_id_bridges_partial_views_without_cross_camera_xyz_distance(self):
        first = snapshot([0, 1, 2], [[0, 0], [1, 0], [2, 0]], [[.9, .05]]*3,
                         tracks=[10, 11, 12])
        # Different camera coordinate origins must never be concatenated into
        # the metric graph. A shared visible ID is the temporal bridge.
        last = snapshot([2, 3, 4], [[0, 0], [1, 0], [2, 0]], [[.9, .05]]*3,
                        xyz=[[100., 0, 5], [100.1, 0, 5], [100.2, 0, 5]])
        state = build_connectivity_state(window([first, first, last]))
        result = update_connectivity_targets(state, np.tile([.05, .95], (5, 1)))
        np.testing.assert_array_equal(result['penalty'][:, 1], 0)
        self.assertEqual(result['report']['objects']['1']['protected_count'], 5)
        self.assertTrue(np.any(np.all(state['edges'] == [2, 3], axis=1)))

    def test_no_reliable_core_does_not_delete_or_penalize_any_object(self):
        original = scene()
        state = build_connectivity_state(window([next(iter(original._active.values()))]))
        self.assertNotIn(1, state['core_by_object'])
        result = update_connectivity_targets(state, np.tile([.01, .99], (15, 1)))
        np.testing.assert_array_equal(result['penalty'], 0)
        self.assertEqual(len(result['seed_rows']), 0)
        self.assertEqual(result['report']['objects']['1']['action'], 'skip_no_reliable_core')

    def test_no_core_policy_forces_winning_gaussians_to_exact_background(self):
        original = scene()
        current = window([next(iter(original._active.values()))])
        state = build_connectivity_state(current, dict(no_core_to_background=True))
        initial = np.tile([.01, .99], (15, 1))
        result = no_core_background_policy(state, initial)
        np.testing.assert_array_equal(result['probabilities'], np.tile([1., 0.], (15, 1)))
        self.assertTrue(result['forced_background_mask'].all())
        self.assertEqual(result['report']['disabled_object_ids'], [1])
        np.testing.assert_array_equal(initial, np.tile([.01, .99], (15, 1)))

    def test_no_core_mass_moves_to_o0_without_increasing_supported_object_mass(self):
        current = scene()
        snap = next(iter(current._active.values()))
        snap['object_ids'] = np.array([2, 0, 1])
        snap['costs'] = np.column_stack((np.full(15, .9), snap['costs']))
        snap['weights'] = np.ones((15, 3))
        current.object_ids = [2, 0, 1]
        state = build_connectivity_state(current, dict(no_core_to_background=True))
        initial = np.tile([.1, .1, .8], (15, 1))
        initial[0] = [.7, .1, .2]
        initial[2] = [.1, .8, .1]
        result = no_core_background_policy(state, initial)
        np.testing.assert_array_equal(result['disabled_columns'], [True, False, False])
        np.testing.assert_array_equal(result['probabilities'][0], [0., 1., 0.])
        np.testing.assert_allclose(result['probabilities'][1], [0., .2, .8])
        np.testing.assert_allclose(result['probabilities'][2], [0., .9, .1])
        self.assertEqual(result['report']['forced_background_gaussians'], 1)

    def test_fragmented_core_candidates_also_fall_back_to_background(self):
        snap = snapshot([0, 1, 2], [[0, 0], [10, 0], [20, 0]], [[.9, .05]]*3,
                        tracks=[10, 11, 12])
        state = build_connectivity_state(window([snap, snap]), dict(no_core_to_background=True))
        self.assertEqual(state['diagnostics']['objects']['1']['action'], 'skip_fragmented_core')
        result = no_core_background_policy(state, np.tile([.01, .99], (3, 1)))
        np.testing.assert_array_equal(result['probabilities'], np.tile([1., 0.], (3, 1)))

    def test_no_core_fallback_requires_the_background_column(self):
        state = dict(active_gids=np.array([7]), object_ids=np.array([1, 2]),
                     core_by_object={}, options=dict(no_core_to_background=True))
        with self.assertRaisesRegex(ValueError, 'background object ID'):
            no_core_background_policy(state, [[.1, .9]])

    def test_unobserved_high_probability_island_is_not_declared_disconnected(self):
        current = scene(background_count=3)
        for snap in current._active.values():
            snap['costs'][3:] = np.nan
        state = build_connectivity_state(current)
        result = update_connectivity_targets(state, np.tile([.01, .99], (6, 1)))
        np.testing.assert_array_equal(result['penalty'][3:], 0)

    def test_car_road_weak_bridge_is_rejected_by_observed_motion_compatibility(self):
        state = build_connectivity_state(scene(background_count=8, contiguous=True))
        self.assertTrue(np.any(np.all(state['edges'] == [2, 3], axis=1)))
        result = update_connectivity_targets(state, np.tile([.01, .99], (11, 1)))
        np.testing.assert_array_equal(result['penalty'][:3, 1], 0)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))

    def test_posterior_surface_mode_preserves_cost_core_but_connects_winning_body(self):
        current = scene(background_count=8, contiguous=True)
        strict = build_connectivity_state(current)
        posterior = build_connectivity_state(current, dict(bridge_require_cost_compatibility=False))
        np.testing.assert_array_equal(posterior['core_by_object'][1], strict['core_by_object'][1])
        np.testing.assert_array_equal(posterior['compatible'], strict['compatible'])
        p = np.tile([.01, .99], (11, 1))
        old = update_connectivity_targets(strict, p)
        new = update_connectivity_targets(posterior, p)
        self.assertTrue(np.all(old['penalty'][3:, 1] > 0))
        np.testing.assert_array_equal(new['penalty'][:, 1], np.zeros(11))
        self.assertEqual(new['report']['objects']['1']['protected_count'], 11)
        self.assertFalse(new['report']['bridge_require_cost_compatibility'])

    def test_posterior_surface_mode_does_not_certify_motion_cores_through_raw_cost_gaps(self):
        uv = np.column_stack((np.arange(5), np.zeros(5)))
        costs = np.array([[.9, .05], [.05, .9], [.9, .05], [.05, .9], [.9, .05]])
        snap = snapshot(np.arange(5), uv, costs, tracks=np.arange(5))
        state = build_connectivity_state(
            window([snap, snap]), dict(bridge_require_cost_compatibility=False))
        self.assertNotIn(1, state['core_by_object'])
        self.assertEqual(state['diagnostics']['objects']['1']['action'], 'skip_fragmented_core')

    def test_posterior_surface_mode_still_penalizes_disconnected_winning_island(self):
        state = build_connectivity_state(
            scene(background_count=5, contiguous=False),
            dict(bridge_require_cost_compatibility=False))
        result = update_connectivity_targets(state, np.tile([.01, .99], (8, 1)))
        self.assertEqual(result['report']['objects']['1']['protected_count'], 3)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))

    def test_posterior_surface_mode_cannot_bridge_via_background_or_unobserved_or_height(self):
        for gap in ('background', 'unobserved', 'height'):
            with self.subTest(gap=gap):
                state = build_connectivity_state(
                    scene(background_count=5, contiguous=True),
                    dict(bridge_require_cost_compatibility=False, halo_hops=0))
                p = np.tile([.01, .99], (8, 1))
                if gap == 'background':
                    p[3] = [.51, .49]
                elif gap == 'unobserved':
                    state['observed'][3, 1] = False
                else:
                    state['height_keep'][3] = False
                result = update_connectivity_targets(state, p)
                self.assertEqual(result['report']['objects']['1']['protected_count'], 3)
                self.assertTrue(np.all(result['penalty'][4:, 1] > 0))

    def test_posterior_surface_mode_background_core_is_seed_not_geometric_bridge(self):
        state = build_connectivity_state(
            scene(background_count=5, contiguous=True),
            dict(bridge_require_cost_compatibility=False, halo_hops=0))
        p = np.tile([.01, .99], (8, 1))
        p[2] = [.99, .01]
        result = update_connectivity_targets(state, p)
        np.testing.assert_array_equal(result['seed_rows'], [0, 1, 2])
        self.assertEqual(result['report']['objects']['1']['protected_count'], 2)
        self.assertTrue(np.all(result['penalty'][3:, 1] > 0))

    def test_depth_discontinuity_blocks_edges_but_body_color_change_is_soft(self):
        uv = np.column_stack((np.arange(6), np.zeros(6)))
        xyz = np.column_stack((np.arange(6)*.1, np.zeros(6), [5, 5, 5, 10, 10, 10]))
        color = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]*2)
        snap = snapshot(np.arange(6), uv, [[.9, .05]]*6, tracks=np.arange(6), xyz=xyz, colors=color)
        state = build_connectivity_state(window([snap, snap]))
        self.assertTrue(np.any(np.all(state['edges'] == [0, 1], axis=1)))
        self.assertFalse(np.any(np.all(state['edges'] == [2, 3], axis=1)))
        self.assertGreater(float(state['adjacency'][0, 1]), 0.)

    def test_known_impossible_height_rejects_core_but_missing_plane_does_not(self):
        current = scene()
        height = np.full(15, 10.)
        state = build_connectivity_state(current, height_info=dict(height=height, valid=np.ones(15, bool)))
        self.assertNotIn(1, state['core_by_object'])
        state = build_connectivity_state(current, height_info=dict(height=height, valid=np.zeros(15, bool)))
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        state = build_connectivity_state(current, height_info=dict(
            height=np.ones(15), valid=np.ones(15, bool), allowed=np.zeros(15, bool)))
        self.assertNotIn(1, state['core_by_object'])

    def test_inherited_labels_and_init_are_not_motion_evidence(self):
        current = scene()
        for snap in current._active.values():
            snap['object_labels'] = np.ones(15, int)
            snap['q_init'] = np.tile([.001, .999], (15, 1))
            snap['costs'][:] = [.2, .2]
        state = build_connectivity_state(current)
        self.assertFalse(state['core_by_object'])

    def test_targets_are_differentiable_without_mutating_probabilities_or_snapshot(self):
        import torch
        current = scene()
        before = next(iter(current._active.values()))['costs'].copy()
        state = build_connectivity_state(current)
        logits = torch.zeros((15, 2), requires_grad=True)
        probabilities = torch.softmax(logits, dim=1)
        targets = update_connectivity_targets(state, probabilities.detach().numpy())
        losses = _module.connectivity_losses(probabilities, targets)
        (losses['disconnected']+losses['anchor']).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.all(logits.grad[:3, 1] < 0))
        self.assertTrue(torch.all(logits.grad[3:, 1] > 0))
        np.testing.assert_array_equal(next(iter(current._active.values()))['costs'], before)
        np.testing.assert_allclose(probabilities.detach().numpy(), .5)

    def test_invalid_probability_input_is_rejected(self):
        state = build_connectivity_state(scene())
        with self.assertRaises(ValueError):
            update_connectivity_targets(state, np.ones((15, 2)))


if __name__ == '__main__':
    unittest.main()

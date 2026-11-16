"""Mandatory final connectivity on actual assigned cores, not stale cost cores."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

from ownership.final_connectivity import finalize_connectivity
from ownership.object_window import ObjectWindow


def scene(n=6, objects=(0, 1), core=None, edges=None):
    objects = np.asarray(objects, np.int64)
    k = len(objects)
    if edges is None:
        edges = [(i, i+1) for i in range(n-1)]
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    left = np.r_[edges[:, 0], edges[:, 1]]
    right = np.r_[edges[:, 1], edges[:, 0]]
    graph = sparse.csr_matrix((np.ones(len(left)), (left, right)), shape=(n, n))
    return dict(active_gids=np.arange(n), object_ids=objects, adjacency=graph,
                observed=np.ones((n, k), bool), compatible=np.ones((n, k), bool),
                height_keep=np.ones(n, bool), mean_advantage=np.ones((n, k)),
                core_by_object={1: np.array([0, 1, 2])} if core is None else core,
                options=dict(core_min_points=3))


def project(state, p, **kwargs):
    options = dict(assignment_support=np.ones(np.shape(p), bool),
                   min_probability=.6, min_margin=.1)
    options.update(kwargs)
    return finalize_connectivity(state, p, **options)


class FinalConnectivityTests(unittest.TestCase):
    def test_cost_core_now_background_cannot_protect_a_foreground_island(self):
        state = scene()
        p = np.array([[.99, .01]]*3+[[.01, .99]]*3)
        before = p.copy()
        result = project(state, p)
        np.testing.assert_array_equal(result['disabled_columns'], [False, True])
        np.testing.assert_array_equal(result['labels'], np.zeros(6, np.int64))
        np.testing.assert_array_equal(result['probabilities'][3:], [[1., 0.]]*3)
        np.testing.assert_array_equal(p, before)

    def test_one_owned_core_is_not_enough(self):
        state = scene()
        p = np.array([[.01, .99], [.99, .01], [.99, .01]]+[[.01, .99]]*3)
        result = project(state, p)
        self.assertTrue(result['disabled_columns'][1])
        self.assertEqual(result['report']['objects']['1']['owned_core_count'], 1)
        self.assertTrue(np.all(result['labels'] == 0))

    def test_disconnected_island_is_exact_background(self):
        state = scene(edges=[(0, 1), (1, 2), (3, 4), (4, 5)])
        result = project(state, np.tile([.01, .99], (6, 1)))
        np.testing.assert_array_equal(result['labels'], [1, 1, 1, 0, 0, 0])
        self.assertFalse(result['disabled_columns'][1])
        self.assertTrue(np.all(result['forced_background_mask'][3:]))
        np.testing.assert_array_equal(result['probabilities'][3:], [[1., 0.]]*3)

    def test_owned_cores_split_into_insufficient_branches_disable_the_object(self):
        state = scene(n=4, core={1: np.arange(4)}, edges=[(0, 1), (2, 3)])
        result = project(state, np.tile([.01, .99], (4, 1)))
        self.assertTrue(result['disabled_columns'][1])
        np.testing.assert_array_equal(result['labels'], [0, 0, 0, 0])
        self.assertEqual(result['report']['objects']['1']['action'],
                         'disable_fragmented_final_core')

    def test_small_reliable_core_beats_large_unanchored_island(self):
        edges = [(0, 1), (1, 2)]+[(i, i+1) for i in range(3, 19)]
        state = scene(n=20, edges=edges)
        result = project(state, np.tile([.01, .99], (20, 1)))
        np.testing.assert_array_equal(result['labels'][:3], 1)
        np.testing.assert_array_equal(result['labels'][3:], 0)

    def test_competing_core_branches_use_evidence_not_component_area(self):
        state = scene(n=10, core={1: np.arange(6)},
                      edges=[(0, 1), (1, 2), (3, 4), (4, 5), (5, 6),
                             (6, 7), (7, 8), (8, 9)])
        state['mean_advantage'][:3, 1] = 2.
        state['mean_advantage'][3:, 1] = 1.
        result = project(state, np.tile([.01, .99], (10, 1)))
        np.testing.assert_array_equal(result['labels'], [1, 1, 1]+[0]*7)

    def test_shared_gid_graph_keeps_a_partial_view_without_its_own_core(self):
        # The graph is the union of camera-local edges [0,1,2] and [2,3,4,5].
        # Shared GID 2, not distances between different camera coordinate frames,
        # connects the second partial view to the reliable first-view core.
        state = scene(edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
        result = project(state, np.tile([.01, .99], (6, 1)))
        np.testing.assert_array_equal(result['labels'], np.ones(6, np.int64))

    def test_background_winner_cannot_be_a_bridge(self):
        state = scene(n=7)
        p = np.tile([.01, .99], (7, 1))
        p[3] = [.99, .01]
        result = project(state, p)
        np.testing.assert_array_equal(result['labels'], [1, 1, 1, 0, 0, 0, 0])

    def test_height_rejected_row_cannot_be_a_bridge(self):
        state = scene(n=7)
        state['height_keep'][3] = False
        result = project(state, np.tile([.01, .99], (7, 1)))
        np.testing.assert_array_equal(result['labels'], [1, 1, 1, 0, 0, 0, 0])

    def test_supplied_hard_background_cannot_anchor_or_bridge(self):
        state = scene(n=7)
        hard = np.zeros(7, bool)
        hard[3] = True
        result = project(state, np.tile([.01, .99], (7, 1)), hard_background_mask=hard)
        np.testing.assert_array_equal(result['labels'], [1, 1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(result['probabilities'][hard], [[1., 0.]])

    def test_low_confidence_argmax_can_bridge_without_receiving_a_hard_label(self):
        state = scene(n=7)
        p = np.tile([.01, .99], (7, 1))
        p[3] = [.45, .55]
        result = project(state, p)
        np.testing.assert_array_equal(result['labels'], [1, 1, 1, -1, 1, 1, 1])
        np.testing.assert_allclose(result['probabilities'], p)

    def test_unobserved_or_incompatible_rows_do_not_bridge(self):
        for field in ('observed', 'compatible'):
            with self.subTest(field=field):
                state = scene(n=7)
                state[field][3, 1] = False
                result = project(state, np.tile([.01, .99], (7, 1)))
                np.testing.assert_array_equal(result['labels'], [1, 1, 1, 0, 0, 0, 0])

    def test_posterior_surface_path_can_cross_a_raw_cost_incompatible_body(self):
        state = scene(n=7)
        state['compatible'][3, 1] = False
        p = np.tile([.01, .99], (7, 1))
        strict = project(state, p)
        np.testing.assert_array_equal(strict['labels'], [1, 1, 1, 0, 0, 0, 0])
        state['options']['bridge_require_cost_compatibility'] = False
        result = project(state, p)
        np.testing.assert_array_equal(result['labels'], np.ones(7, np.int64))
        self.assertFalse(result['report']['bridge_require_cost_compatibility'])

    def test_posterior_surface_path_keeps_strict_motion_cores(self):
        state = scene()
        state['options']['bridge_require_cost_compatibility'] = False
        state['compatible'][1, 1] = False
        result = project(state, np.tile([.01, .99], (6, 1)))
        self.assertEqual(result['report']['objects']['1']['owned_core_count'], 2)
        self.assertTrue(result['disabled_columns'][1])
        np.testing.assert_array_equal(result['labels'], np.zeros(6, np.int64))

    def test_posterior_surface_path_still_rejects_background_missing_and_height_gaps(self):
        for gap in ('background', 'unobserved', 'height', 'hard_background'):
            with self.subTest(gap=gap):
                state = scene(n=7)
                state['options']['bridge_require_cost_compatibility'] = False
                state['compatible'][3, 1] = False
                p = np.tile([.01, .99], (7, 1))
                hard = np.zeros(7, bool)
                if gap == 'background':
                    # Foreground probability clears .35, but O0 wins.
                    p[3] = [.51, .49]
                elif gap == 'unobserved':
                    state['observed'][3, 1] = False
                elif gap == 'height':
                    state['height_keep'][3] = False
                else:
                    hard[3] = True
                result = project(state, p, hard_background_mask=hard)
                expected_gap = -1 if gap == 'background' else 0
                np.testing.assert_array_equal(result['labels'], [1, 1, 1, expected_gap, 0, 0, 0])

    def test_posterior_surface_path_other_object_cannot_bridge(self):
        state = scene(n=10, objects=(0, 1, 2),
                      core={1: np.arange(3), 2: np.arange(7, 10)})
        state['options']['bridge_require_cost_compatibility'] = False
        p = np.tile([.01, .98, .01], (10, 1))
        p[3] = [.01, .4, .59]
        p[7:] = [.01, .01, .98]
        result = project(state, p, min_probability=.55)
        np.testing.assert_array_equal(result['labels'][4:7], [0, 0, 0])

    def test_posterior_surface_path_still_requires_bridge_probability(self):
        # Isolate low-probability O1 argmax from disable transfers by giving
        # both competing objects reliable, owned anchors of their own.
        state = scene(n=13, objects=(0, 1, 2, 3),
                      core={1: np.arange(3), 2: np.arange(7, 10), 3: np.arange(10, 13)})
        state['options']['bridge_require_cost_compatibility'] = False
        p = np.tile([.01, .97, .01, .01], (13, 1))
        p[3] = [.32, .34, .17, .17]
        p[7:10] = [.01, .01, .97, .01]
        p[10:] = [.01, .01, .01, .97]
        result = project(state, p)
        np.testing.assert_array_equal(result['labels'][3:7], [0, 0, 0, 0])

    def test_mass_transfer_rechecks_and_disables_a_second_objects_core(self):
        state = scene(n=6, objects=(0, 1, 2), core={1: np.arange(3), 2: np.arange(3, 6)})
        p = np.array([[.10, .61, .29]]*3+[[.99, .005, .005]]*3)
        # A initially passes the .3 margin: .61-.29=.32. Removing unsupported
        # B makes O0 .39, so A's margin drops to .22 and its core becomes invalid.
        result = project(state, p, min_margin=.3)
        self.assertGreaterEqual(result['report']['projection_updates'], 2)
        self.assertTrue(np.all(result['disabled_columns'][1:]))
        np.testing.assert_array_equal(result['labels'], np.zeros(6, np.int64))

    def test_object_column_permutation_does_not_change_projection(self):
        state = scene(n=6, objects=(0, 1, 2), core={1: np.arange(3), 2: np.arange(3, 6)})
        p = np.array([[.10, .61, .29]]*3+[[.99, .005, .005]]*3)
        original = project(state, p, min_margin=.3)
        order = np.array([2, 0, 1])
        permuted = {**state, 'object_ids': state['object_ids'][order],
                    'observed': state['observed'][:, order],
                    'compatible': state['compatible'][:, order],
                    'mean_advantage': state['mean_advantage'][:, order]}
        other = project(permuted, p[:, order], min_margin=.3)
        np.testing.assert_array_equal(other['labels'], original['labels'])
        np.testing.assert_allclose(other['probabilities'][:, np.argsort(order)],
                                   original['probabilities'])
        np.testing.assert_array_equal(other['disabled_columns'][np.argsort(order)],
                                      original['disabled_columns'])

    def test_empty_window_and_background_only_are_valid(self):
        for n in (0, 3):
            with self.subTest(n=n):
                state = scene(n=n, objects=(0,), core={})
                result = project(state, np.ones((n, 1)))
                np.testing.assert_array_equal(result['labels'], np.zeros(n, np.int64))
                self.assertFalse(result['report']['loss_improvement_claimed'])

    def test_invalid_probabilities_or_masks_are_rejected(self):
        state = scene()
        with self.assertRaises(ValueError):
            project(state, np.ones((6, 2)))
        with self.assertRaises(ValueError):
            project(state, np.tile([.01, .99], (6, 1)), hard_background_mask=np.zeros(5))


class FinalConnectivityWindowRouteTests(unittest.TestCase):
    def make_window(self, render=False, *, frames=(0, 1, 2), window_frames=3):
        return ObjectWindow(dict(sequence=dict(frames=frames), object_window=dict(
            window_frames=window_frames, window_step=window_frames-1,
            temperature=.25, prior_weight=.05,
            connectivity=dict(enabled=True, no_core_to_background=True,
                              final_hard_assignment=True),
            render=dict(enabled=render))))

    def update(self, window, first, second):
        uv = np.column_stack(([0, 1, 2, 10, 11, 12], np.zeros(6)))
        payload = dict(uv=uv, xyz=np.column_stack((uv[:, 0]*.1, np.zeros(6), np.full(6, 5.))),
                       color=np.zeros((6, 3)), scale=np.full((6, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], 2, axis=0))
        observations = dict(source=np.zeros((2, 16, 3), np.uint8),
                            target=np.zeros((2, 16, 3), np.uint8),
                            source_depth=np.full((2, 16), 5.), target_depth=np.full((2, 16), 5.),
                            source_confidence=np.ones((2, 16)), target_confidence=np.ones((2, 16)))
        # Both branches are attractive to the CPU solver, but only the first
        # has enough advantage over O0 to establish a reliable motion core.
        costs = np.array([[.9, .05]]*3+[[.6, .4]]*3)
        return window.update(first, second, np.arange(6), np.array([0, 1]),
                             np.tile([.01, .99], (6, 1)), costs,
                             track_ids=np.arange(6), payload=payload, render_observations=observations)

    def test_cpu_completed_window_projects_but_collecting_window_does_not(self):
        window = self.make_window()
        with patch('ownership.final_connectivity.finalize_connectivity',
                   wraps=finalize_connectivity) as final:
            collecting = self.update(window, 0, 1)
            final.assert_not_called()
            self.assertNotIn('final_connectivity', collecting['report'])
            np.testing.assert_array_equal(collecting['object_labels'], np.ones(6, np.int64))
            completed = self.update(window, 1, 2)
            self.assertEqual(final.call_count, 1)
        np.testing.assert_array_equal(completed['object_labels'], [1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(window._active_connectivity_background,
                                      [False, False, False, True, True, True])
        self.assertTrue(completed['report']['final_connectivity']['projection_only'])
        self.assertFalse(completed['report']['final_connectivity']['objective_evaluated'])

    def test_mock_render_returns_state_and_uses_the_same_final_projection(self):
        window = self.make_window(render=True)

        def render(current, cfg, *, priors):
            from ownership.object_connectivity_constraints import build_connectivity_state
            state = build_connectivity_state(current, cfg['connectivity'])
            self.assertIn('height_keep', state)
            return dict(probabilities=np.tile([.01, .99], (6, 1)),
                        render_support=np.ones((6, 2), bool), connectivity_state=state,
                        report=dict(final_losses=dict(total=.123), update_accepted=False))

        with patch('ownership.object_render_optimizer.optimize_render_window', side_effect=render) as solve:
            self.update(window, 0, 1)
            solve.assert_not_called()
            completed = self.update(window, 1, 2)
            self.assertEqual(solve.call_count, 1)
        np.testing.assert_array_equal(completed['object_labels'], [1, 1, 1, 0, 0, 0])
        report = completed['report']
        self.assertEqual(report['render_optimization']['final_losses']['total'], .123)
        self.assertEqual(report['render_optimization']['reported_losses_stage'],
                         'before_final_connectivity_projection')
        self.assertIn('final_connectivity', report)
        self.assertNotIn('final_losses', report['final_connectivity'])

    def test_final_partial_window_also_projects(self):
        window = self.make_window(frames=(0, 1, 2), window_frames=4)
        self.update(window, 0, 1)
        completed = self.update(window, 1, 2)
        self.assertEqual(completed['report']['window_schedule']['reason'], 'final_partial_window')
        np.testing.assert_array_equal(completed['object_labels'], [1, 1, 1, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

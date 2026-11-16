"""A valid moving surface need not have a valid background projection."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from ownership.object_connectivity_constraints import build_connectivity_state, update_connectivity_targets
from ownership.final_connectivity import finalize_connectivity


def state_with_occluded_background(*, own_cost=.05, own_weight=1.):
    n = 7
    uv = np.column_stack((np.arange(n), np.zeros(n)))
    costs = np.tile([.9, .05], (n, 1))
    weights = np.ones((n, 2))
    costs[3, 0], weights[3, 0] = np.nan, 0.
    costs[3, 1], weights[3, 1] = own_cost, own_weight
    # Only the three confirmed points are independent tracked cost cores.
    costs[4:, 0] = .05
    snap = dict(gaussian_ids=np.arange(n), object_ids=np.array([0, 1]),
                costs=costs, weights=weights, track_ids=np.r_[np.arange(3), [-1]*4],
                payload=dict(uv=uv, xyz=np.column_stack((uv[:, 0]*.1, uv[:, 1], np.full(n, 5.))),
                             color=np.zeros((n, 3))))
    window = SimpleNamespace(_active_gids=np.arange(n), object_ids=[0, 1],
                             _active={(0, 1):snap, (1, 2):snap})
    return build_connectivity_state(window, dict(bridge_require_cost_compatibility=False,
                                    bridge_require_background_observation=False))


class SurfaceModelObservationTests(unittest.TestCase):
    def project(self, state, p=None):
        p = np.tile([.01, .99], (7, 1)) if p is None else p
        return finalize_connectivity(state, p, assignment_support=np.ones((7, 2), bool),
                                     min_probability=.6, min_margin=.1)

    def test_own_fit_can_connect_surface_without_becoming_a_motion_core(self):
        state = state_with_occluded_background()
        self.assertFalse(state['observed'][3, 1])
        self.assertTrue(state['model_observed'][3, 1])
        np.testing.assert_array_equal(state['core_by_object'][1], [0, 1, 2])
        targets = update_connectivity_targets(state, np.tile([.01, .99], (7, 1)))
        self.assertEqual(targets['report']['objects']['1']['protected_count'], 7)
        np.testing.assert_array_equal(self.project(state)['labels'], [1]*7)

    def test_legacy_paired_observation_still_rejects_missing_background_bridge(self):
        state = state_with_occluded_background()
        state['options']['bridge_require_background_observation'] = True
        np.testing.assert_array_equal(self.project(state)['labels'], [1, 1, 1, 0, 0, 0, 0])

    def test_missing_bad_or_unreliable_object_fit_cannot_connect(self):
        for cost, weight in ((np.nan, 1.), (2., 1.), (.05, 0.), (.05, .01)):
            with self.subTest(cost=cost, weight=weight):
                state = state_with_occluded_background(own_cost=cost, own_weight=weight)
                self.assertFalse(state['model_observed'][3, 1])
                np.testing.assert_array_equal(self.project(state)['labels'], [1, 1, 1, 0, 0, 0, 0])

    def test_surface_fit_never_replaces_a_missing_comparative_core(self):
        state = state_with_occluded_background()
        state['observed'][1, 1] = False
        self.assertTrue(self.project(state)['disabled_columns'][1])

    def test_background_winner_still_cannot_connect(self):
        state = state_with_occluded_background()
        p = np.tile([.01, .99], (7, 1)); p[3] = [.99, .01]
        np.testing.assert_array_equal(self.project(state, p)['labels'], [1, 1, 1, 0, 0, 0, 0])


if __name__ == '__main__':
    unittest.main()

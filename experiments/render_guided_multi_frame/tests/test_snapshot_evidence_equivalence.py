"""Sorted evidence must preserve the prior pairwise statistical objective."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ownership.object_window import ObjectWindow


def pairwise_reference(window, snap, global_count, candidate_ids=None):
    """Frozen original quadratic implementation, retained only as an oracle."""
    costs, weights = snap['costs'], snap['weights']
    n, count = costs.shape
    statistics = np.zeros((3, n, count))
    energy, support, good_fit = statistics
    valid = np.isfinite(costs) & (weights > 0)
    if candidate_ids is not None:
        valid &= np.isin(snap['object_ids'], candidate_ids)[None]
    valid_count = valid.sum(axis=1)
    any_fit = np.any(valid & (costs < window.max_fit_cost), axis=1)
    comparable = (valid_count >= 2) & any_fit
    clipped = np.minimum(costs, window.max_cost)
    reliability = np.minimum(weights, window.max_weight)
    for j in range(count):
        for k in range(j+1, count):
            visible = np.flatnonzero(comparable & valid[:, j] & valid[:, k])
            if not len(visible):
                continue
            weight = np.minimum(reliability[visible, j], reliability[visible, k])/valid_count[visible]
            difference = weight*(clipped[visible, j]-clipped[visible, k])
            energy[visible, j] += difference
            energy[visible, k] -= difference
            support[visible, j] += weight
            support[visible, k] += weight
        visible = np.flatnonzero(comparable & valid[:, j] & (costs[:, j] < window.max_fit_cost))
        good_fit[visible, j] += reliability[visible, j]
    if global_count == 1:
        for j in range(count):
            visible = np.flatnonzero(valid[:, j] & any_fit)
            support[visible, j] += reliability[visible, j]
            good_fit[visible, j] += reliability[visible, j]
    return statistics


class SnapshotEvidenceEquivalenceTests(unittest.TestCase):
    def compare(self, costs, weights, candidate_ids=None, global_count=None):
        costs = np.asarray(costs, np.float64)
        weights = np.broadcast_to(np.asarray(weights, np.float64), costs.shape).copy()
        window = ObjectWindow(dict(max_cost=3., max_fit_cost=1., max_weight=1.))
        snap = dict(costs=costs, weights=weights, object_ids=np.arange(costs.shape[1])*3+2)
        costs_before, weights_before = costs.copy(), weights.copy()
        count = costs.shape[1] if global_count is None else global_count
        expected = pairwise_reference(window, snap, count, candidate_ids)
        actual = window._snapshot_evidence(snap, count, candidate_ids)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
        np.testing.assert_array_equal(costs, costs_before)
        np.testing.assert_array_equal(weights, weights_before)
        return expected, actual

    def test_random_missing_data_retired_columns_and_nondefault_ids(self):
        rng = np.random.default_rng(291)
        for columns in (1, 2, 3, 7, 17, 60):
            costs = rng.exponential(1.2, size=(137, columns))
            weights = rng.uniform(0, 2, size=costs.shape)
            costs[rng.random(costs.shape) < .12] = np.nan
            costs[rng.random(costs.shape) < .12] = np.inf
            weights[rng.random(costs.shape) < .2] = 0
            for keep in (None, np.arange(columns)[::2]*3+2, [], [2]):
                self.compare(costs, weights, keep)

    def test_reliability_ties_and_clipping(self):
        rng = np.random.default_rng(42)
        self.compare(rng.random((97, 50))*10,
                     rng.choice([0., .01, .5, 1., 2., 1e308], size=(97, 50)))

    def test_all_missing_no_fit_or_zero_weight_is_zero(self):
        for costs, weights in (([[np.nan, np.inf, 0]], 0.),
                               ([[10., 10., 10.]], 1.), ([[0., 0., 0.]], 0.)):
            expected, _ = self.compare(costs, weights)
            self.assertFalse(np.any(expected))

    def test_exact_fit_threshold_and_zero_cost(self):
        self.compare([[0., 1., 3.], [1., 1., 1.], [.99999999999, 1., np.inf]], 1.)

    def test_extreme_costs_and_weights_are_clipped_before_accumulation(self):
        self.compare([[0., 1e308, np.inf, np.nan], [1e-308, 1e308, 2., 0.]],
                     [[1e308, 1e-308, 1., 1.], [1., 1., 1e-308, 0.]])

    def test_global_single_candidate_keeps_independent_special_branch(self):
        self.compare([[0., .2, 4.], [np.inf, 0., .5]], 1., global_count=1)
        self.compare([[0.], [.5], [1.], [np.inf], [np.nan]], 1., global_count=1)

    def test_tiny_positive_support_is_not_lost_by_subtracting_self(self):
        expected, actual = self.compare([[0., .5], [.2, .7]],
                                        [[1., 1e-308], [1e-308, 1.]])
        np.testing.assert_array_equal(actual[1] > 0, expected[1] > 0)
        self.assertTrue((actual[1] > 0).all())

    def test_column_permutation_preserves_result(self):
        rng = np.random.default_rng(12)
        costs, weights = rng.random((20, 20))*3, rng.random((20, 20))
        _, original = self.compare(costs, weights)
        permutation = rng.permutation(20)
        _, permuted = self.compare(costs[:, permutation], weights[:, permutation])
        np.testing.assert_allclose(permuted[:, :, np.argsort(permutation)], original,
                                   atol=2e-14, rtol=2e-13)

    def test_empty_rows_and_no_candidate_retained(self):
        self.compare(np.empty((0, 7)), np.empty((0, 7)))
        self.compare(np.ones((19, 7)), 1., candidate_ids=[])

    def test_fixed_fifty_thousand_row_chunk_boundary(self):
        rng = np.random.default_rng(19)
        costs, weights = rng.random((50_007, 3)), rng.random((50_007, 3))
        costs[[49_999, 50_000, 50_001], 1] = np.inf
        self.compare(costs, weights)


if __name__ == '__main__':
    unittest.main()

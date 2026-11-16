"""Fixed-lag regressions for bounded Gaussian state and departing evidence."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


_path = Path(__file__).resolve().parents[1] / "ownership/object_window.py"
_spec = importlib.util.spec_from_file_location("streaming_object_window", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectWindow = _module.ObjectWindow


def config(window_frames=3, render=False):
    return dict(sequence=dict(streaming=True),
                object_window=dict(window_frames=window_frames, temperature=.2,
                                   prior_weight=.03, max_fit_cost=.8,
                                   render=dict(enabled=render)))


def update(window, first, gids=(42,), oids=(0, 1), costs=None, q=None, **kwargs):
    n, k = len(gids), len(oids)
    costs = np.full((n, k), np.nan) if costs is None else np.asarray(costs, float)
    q = np.full((n, k), 1. / k) if q is None else np.asarray(q, float)
    return window.update(first, first + 1, np.asarray(gids, np.int64),
                         np.asarray(oids, np.int64), q, costs, **kwargs)


class StreamingWindowTests(unittest.TestCase):
    def test_long_roll_keeps_only_active_gaussians_frames_and_object_columns(self):
        window = ObjectWindow(config(window_frames=4))
        for frame in range(30):
            oid = 1 + frame // 5
            result = update(window, frame, gids=(frame, frame + 1), oids=(0, oid),
                            costs=[[.05, .9], [.9, .05]])
            active = {int(g) for snap in window._active.values() for g in snap["gaussian_ids"]}
            for table in (window._initial, window._first_object_ids,
                          window._state, window._labels):
                self.assertEqual(set(table), active)
            self.assertTrue(set(window._boundary) <= active)
            self.assertLessEqual(len(active), 4)
            self.assertLessEqual(len(window._frames), 4)
            self.assertLessEqual(len(window._active), 3)
            self.assertLessEqual(len(window.object_ids), 3)
            self.assertEqual(result["object_probabilities"].shape, (2, len(window.object_ids)))
        report = window.to_dict()
        self.assertEqual(report["total_gaussians"], 31)
        self.assertEqual(report["retained_gaussians"], 4)
        self.assertEqual(report["forgotten_gaussians"], 27)
        self.assertEqual(report["finalized_pairs"], 27)
        self.assertFalse(report["pixel_observations_marginalized"])
        self.assertEqual(report["boundary_policy"],
                         "exact_centre_track_sufficient_statistics;pixel_posterior_warm_start")
        with self.assertRaises(KeyError):
            window.get_state(np.asarray([0], np.int64))

    def test_departing_factors_match_full_history_exactly_without_overlap_votes(self):
        streaming = ObjectWindow(config())
        full = ObjectWindow(dict(config()["object_window"], window_frames=100))
        observations = ([.1, .9], [.9, .1], [np.nan, np.nan], [.1, .9],
                        [.2, .8], [2., 3.], [.6, .1], [.3, .7], [.1, .9])
        for frame, costs in enumerate(observations):
            kwargs = dict(costs=[costs], q=[[.99, .01] if frame == 0 else [.01, .99]],
                          weights=[[.6, .8]])
            expected = update(full, frame, **kwargs)
            actual = update(streaming, frame, **kwargs)
            np.testing.assert_allclose(actual["object_probabilities"],
                                       expected["object_probabilities"], atol=1e-14)
            np.testing.assert_array_equal(actual["object_labels"], expected["object_labels"])
            # Replacing a still-active factor must not fold its previous
            # contents into the boundary or use the current posterior as data.
            if frame == 4:
                kwargs["costs"] = [[.8, .2]]
                expected = update(full, frame, **kwargs)
                actual = update(streaming, frame, **kwargs)
                np.testing.assert_allclose(actual["object_probabilities"],
                                           expected["object_probabilities"], atol=1e-14)
        np.testing.assert_allclose(streaming._prior(42), [.99, .01])
        before = actual["object_probabilities"].copy()
        actual = update(streaming, 9)
        np.testing.assert_allclose(actual["object_probabilities"], before, atol=1e-14)

    def test_departing_good_fit_and_support_survive_missing_active_observations(self):
        window = ObjectWindow(config(window_frames=2))
        initial = update(window, 0, costs=[[.01, .95]])
        for frame in range(1, 7):
            result = update(window, frame, q=[[0., 1.]])
            np.testing.assert_allclose(result["object_probabilities"], initial["object_probabilities"])
            self.assertEqual(result["object_labels"].tolist(), [0])
        single = ObjectWindow(config(window_frames=2))
        update(single, 0, oids=(0,), costs=[[.01]])
        result = update(single, 1, oids=(0,))
        self.assertEqual(result["object_labels"].tolist(), [0])

    def test_columns_shrink_and_reappear_with_original_first_prior_identity(self):
        window = ObjectWindow(config(window_frames=2))
        update(window, 0, gids=(5,), oids=(9, 3), q=[[.9, .1]], costs=[[.05, .9]])
        result = update(window, 1, gids=(5,), oids=(3, 12), q=[[.001, .999]])
        self.assertEqual(result["object_ids"].tolist(), [3, 12])
        np.testing.assert_allclose(window._prior(5), np.asarray([.1, .001]) / .101)
        result = update(window, 2, gids=(5,), oids=(12, 9, 3), q=[[1., 0., 0.]])
        self.assertEqual(result["object_ids"].tolist(), [3, 12, 9])
        np.testing.assert_allclose(window._prior(5), np.asarray([.1, .001, .9]) / 1.001)
        # Departed O9's factor was pruned when that candidate retired. Its
        # immutable first initializer remains correctly associated with O9.
        np.testing.assert_allclose(window._boundary[5][0], [.425, 0., 0.])
        self.assertEqual(window._initial[5].tolist(), [.9, .1])

    def test_exports_freeze_at_eviction_and_finalized_updates_are_rejected(self):
        window = ObjectWindow(config())
        with tempfile.TemporaryDirectory() as temporary:
            directories = [Path(temporary) / str(i) for i in range(8)]
            update(window, 0, costs=[[.1, .9]], output_dir=directories[0])
            update(window, 1, costs=[[.9, .1]], output_dir=directories[1])
            file0 = directories[0] / "object_state.npz"
            frozen = file0.read_bytes()
            for frame in range(2, 8):
                update(window, frame, costs=[[.1, .9]], output_dir=directories[frame])
                self.assertEqual(file0.read_bytes(), frozen)
            with np.load(file0) as archived:
                np.testing.assert_array_equal(archived["window_frames"], [0, 1, 2])
            self.assertEqual(window.to_dict()["finalized_pair_exports"], 6)
            before = window.to_dict()
            with self.assertRaisesRegex(ValueError, "finalized pair"):
                update(window, 0)
            self.assertEqual(window.to_dict(), before)
            self.assertEqual(file0.read_bytes(), frozen)

    def test_monotonic_id_high_water_rejects_reuse_without_lifetime_id_set(self):
        window = ObjectWindow(config(window_frames=2))
        update(window, 0, gids=(10,))
        update(window, 1, gids=(11,))
        with self.assertRaisesRegex(ValueError, "Gaussian IDs cannot be reused"):
            update(window, 2, gids=(10,))
        # The active pair is replaceable and a genuinely new larger ID works.
        update(window, 1, gids=(11,))
        update(window, 2, gids=(12,))
        self.assertEqual(window.to_dict()["total_gaussians"], 3)
        self.assertEqual(window.to_dict()["gaussian_id_high_water_mark"], 12)

    def test_retained_lower_gaussian_ids_and_fresh_ids_share_reordered_candidates(self):
        window = ObjectWindow(config(window_frames=2))
        update(window, 0, gids=(10,), oids=(9, 3), q=[[.9, .1]])
        original_columns = window._first_object_ids[10]
        update(window, 1, gids=(11, 10), oids=(3, 9), q=[[.7, .3], [.99, .01]])
        np.testing.assert_allclose(window._prior(10), [.9, .1])
        np.testing.assert_allclose(window._prior(11), [.3, .7])
        update(window, 2, gids=(10, 12, 13), oids=(3, 9),
               q=[[.99, .01], [.4, .6], [.2, .8]])
        self.assertIs(window._first_object_ids[10], original_columns)
        self.assertIs(window._first_object_ids[12], window._first_object_ids[13])
        self.assertFalse(window._first_object_ids[12].flags.writeable)
        np.testing.assert_allclose(window._prior(10), [.9, .1])
        self.assertEqual(window.to_dict()["gaussian_id_high_water_mark"], 13)
        with self.assertRaises(KeyError):
            window.get_state(np.asarray([11], np.int64))

    def test_render_warm_start_remaps_and_never_replaces_first_prior(self):
        window = ObjectWindow(config(window_frames=2, render=True))
        observations = dict(source=np.zeros((2, 2, 3), np.uint8),
                            target=np.zeros((2, 2, 3), np.uint8),
                            source_depth=np.ones((2, 2)), target_depth=np.ones((2, 2)),
                            source_confidence=np.ones((2, 2)), target_confidence=np.ones((2, 2)))
        captured = []

        def solve(current, cfg, *, priors):
            captured.append((current._render_initial_probabilities.copy(), current._prior(5).copy()))
            probabilities = np.array([[.2, .8]] if len(captured) == 1 else [[.4, .6]])
            return dict(probabilities=probabilities, render_support=np.ones_like(probabilities, bool),
                        report={"render_passes": 1})

        solver = types.SimpleNamespace(optimize_render_window=solve)
        payload = dict(xyz=np.array([[0., 0., 2.]]), color=np.ones((1, 3)),
                       scale=np.full((1, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], 2, axis=0))
        with patch.dict("sys.modules", {"ownership.object_render_optimizer": solver}):
            update(window, 0, gids=(5,), oids=(9, 3), q=[[.9, .1]], payload=payload,
                   render_observations=observations)
            update(window, 1, gids=(5,), oids=(3, 12), q=[[.001, .999]], payload=payload,
                   render_observations=observations)
            np.testing.assert_allclose(captured[1][0], np.asarray([[.8, .001]]) / .801)
            np.testing.assert_allclose(captured[1][1], np.asarray([.1, .001]) / .101)
            self.assertEqual(window._render_energy.shape, (1, 2))
            empty_payload = dict(payload, xyz=np.empty((0, 3)), color=np.empty((0, 3)),
                                 scale=np.empty((0, 3)), H_object=np.eye(4)[None])
            update(window, 2, gids=(), oids=(0,), payload=empty_payload,
                   render_observations=observations)
            self.assertEqual(window._render_initial_probabilities.shape, (0, 1))
            self.assertEqual(window._render_energy.shape, (0, 1))
            self.assertEqual(window._state, {})
            self.assertEqual(window._boundary, {})
            self.assertEqual(window._render_report, {})


if __name__ == "__main__":
    unittest.main()

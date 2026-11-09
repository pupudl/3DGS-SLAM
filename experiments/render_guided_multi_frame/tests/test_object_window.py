"""CPU regression tests for shared, revisable Gaussian object attributes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


_path = Path(__file__).resolve().parents[1] / "object_window.py"
_spec = importlib.util.spec_from_file_location("sequence_object_window", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectWindow = _module.ObjectWindow


class ObjectWindowTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(window_frames=6, temperature=.2, prior_weight=.03,
                        max_fit_cost=.8)

    def update(self, window, first, costs, q=None, **kwargs):
        costs = np.asarray(costs, float)
        if costs.ndim == 1:
            costs = costs[None]
        if q is None:
            q = np.tile([.01, .99], (len(costs), 1))
        return window.update(first, first + 1, np.arange(len(costs), dtype=np.int64),
                             np.asarray([0, 1]), q, costs, **kwargs)

    def test_inherited_wrong_label_is_initialization_not_two_observations(self):
        window = ObjectWindow(self.cfg)
        # B is carried along as O1 by a track, but no independent observation
        # differentiates the models in these first two pairs.
        self.update(window, 0, [np.nan, np.nan])
        second = self.update(window, 1, [np.nan, np.nan])
        self.assertEqual(second["object_labels"][0], -1)
        result = self.update(window, 2, [.05, .95])
        self.assertEqual(result["object_labels"][0], 0)
        self.assertGreater(result["object_probabilities"][0, 0], .9)

    def test_four_reliable_car_observations_survive_one_bad_in_any_order(self):
        observations = [[.95, .05]] * 4 + [[.05, .95]]
        outputs = []
        for values in (observations, observations[::-1]):
            window = ObjectWindow(self.cfg)
            for first, costs in enumerate(values):
                result = self.update(window, first, costs)
            outputs.append(result["object_probabilities"])
            self.assertEqual(result["object_labels"][0], 1)
        np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-14)

    def test_window_rewrites_older_results_and_freezes_dropped_snapshots(self):
        window = ObjectWindow(dict(self.cfg, window_frames=3))
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / str(i) for i in range(3)]
            self.update(window, 0, [np.nan, np.nan], output_dir=outputs[0])
            before = np.load(outputs[0] / "object_state.npz")["object_probabilities"]
            self.update(window, 1, [.05, .95], output_dir=outputs[1])
            file0 = outputs[0] / "object_state.npz"
            with np.load(file0) as snapshot:
                self.assertEqual(snapshot["object_labels"][0], 0)
                self.assertFalse(np.allclose(before, snapshot["object_probabilities"]))
                np.testing.assert_array_equal(snapshot["window_frames"], [0, 1, 2])
            frozen = file0.read_bytes()
            result = self.update(window, 2, [.95, .05], output_dir=outputs[2])
            self.assertEqual(frozen, file0.read_bytes())
            with np.load(outputs[1] / "object_state.npz") as snapshot:
                np.testing.assert_allclose(snapshot["object_probabilities"], result["object_probabilities"])
                np.testing.assert_array_equal(snapshot["window_frames"], [1, 2, 3])
                json.loads(str(snapshot["report_json"]))

    def test_repeated_inherited_initializers_do_not_accumulate_or_reset(self):
        window = ObjectWindow(self.cfg)
        first = self.update(window, 0, [.1, .9])
        # Later observations are uninformative and their track initializer
        # disagrees. They must neither add label votes nor replace the prior.
        for t in range(1, 4):
            result = self.update(window, t, [np.nan, np.nan], q=[[1., 0.]])
        np.testing.assert_allclose(first["object_probabilities"], result["object_probabilities"])

    def test_replacing_pair_does_not_double_count_observation(self):
        window = ObjectWindow(self.cfg)
        first = self.update(window, 0, [.1, .9])
        result = self.update(window, 0, [.1, .9])
        np.testing.assert_allclose(first["object_probabilities"], result["object_probabilities"])

    def test_missing_candidate_has_no_zero_cost_reward(self):
        window = ObjectWindow(self.cfg)
        result = self.update(window, 0, [.1, np.nan])
        self.assertEqual(result["object_labels"][0], -1)
        before = result["object_probabilities"].copy()
        result = self.update(window, 1, [.1, np.inf])
        np.testing.assert_allclose(before, result["object_probabilities"])
        # O2 is completely missing; it cannot win by having accumulated zero
        # error while the two actually observed candidates have positive error.
        result = window.update(2, 3, np.array([0]), np.array([0, 1, 2]),
                               [[.01, .98, .01]], [[.05, .7, np.nan]])
        self.assertEqual(result["object_labels"][0], 0)
        self.assertLess(result["object_probabilities"][0, 2], .2)

    def test_all_bad_costs_are_unknown_and_do_not_erase_reliable_history(self):
        window = ObjectWindow(self.cfg)
        result = self.update(window, 0, [3., 2.])
        self.assertEqual(result["object_labels"][0], -1)
        good = self.update(window, 1, [.9, .1])
        result = self.update(window, 2, [1., 100.])
        np.testing.assert_allclose(good["object_probabilities"], result["object_probabilities"])
        self.assertEqual(result["object_labels"][0], 1)

    def test_inputs_and_local_motion_probabilities_remain_unchanged(self):
        window = ObjectWindow(self.cfg)
        q = np.array([[.01, .99]])
        local = np.array([[.1, .2, .7]])
        costs = np.array([[.05, .95]])
        weights = np.array([1.])
        payload = {"xyz": np.array([[1., 2., 3.]])}
        with tempfile.TemporaryDirectory() as temporary:
            self.update(window, 0, costs, q=q, weights=weights,
                        q_motion_initial=local, payload=payload, output_dir=temporary)
            np.testing.assert_array_equal(q, [[.01, .99]])
            np.testing.assert_array_equal(local, [[.1, .2, .7]])
            # Mutating caller-owned arrays after update cannot change snapshot.
            q[:] = 0
            local[:] = 0
            costs[:] = 0
            weights[:] = 0
            payload["xyz"][:] = 0
            self.update(window, 1, [np.nan, np.nan])
            with np.load(Path(temporary) / "object_state.npz") as snapshot:
                np.testing.assert_array_equal(snapshot["q_motion_initial"], [[.1, .2, .7]])
                np.testing.assert_array_equal(snapshot["q_object_initial"], [[.01, .99]])
                np.testing.assert_array_equal(snapshot["observation_costs"], [[.05, .95]])
                np.testing.assert_array_equal(snapshot["xyz"], [[1., 2., 3.]])

    def test_new_object_columns_are_global_and_old_pairs_are_padded(self):
        window = ObjectWindow(self.cfg)
        with tempfile.TemporaryDirectory() as temporary:
            window.update(0, 1, np.array([42]), np.array([9, 3]),
                          [[.9, .1]], [[.05, .9]], output_dir=temporary)
            result = window.update(1, 2, np.array([42, 77]), np.array([12, 3, 9]),
                                   [[.9, .09, .01], [.99, .005, .005]],
                                   [[.9, .9, .05], [.01, .9, .9]])
            np.testing.assert_array_equal(result["object_ids"], [9, 3, 12])
            np.testing.assert_array_equal(result["object_labels"], [9, 12])
            with np.load(Path(temporary) / "object_state.npz") as snapshot:
                self.assertEqual(snapshot["object_probabilities"].shape, (1, 3))
                np.testing.assert_array_equal(snapshot["q_motion_initial"], [[.9, .1]])
                np.testing.assert_array_equal(snapshot["evidence_object_ids"], [9, 3])

    def test_candidate_weights_compare_only_jointly_available_evidence(self):
        window = ObjectWindow(self.cfg)
        result = self.update(window, 0, [.01, .9], weights=[[1., 0.]])
        self.assertEqual(result["object_labels"][0], -1)
        result = self.update(window, 1, [.01, .9], weights=np.array([.9]))
        self.assertEqual(result["object_labels"][0], 0)

    def test_older_pose_columns_and_track_exports_follow_shared_state(self):
        window = ObjectWindow(self.cfg)
        with tempfile.TemporaryDirectory() as temporary:
            window.update(0, 1, np.array([42]), np.array([9, 3]),
                          [[.01, .99]], [[np.nan, np.nan]],
                          output_dir=temporary, track_ids=[99],
                          payload={"H_object": np.stack([np.eye(4), 2 * np.eye(4)]),
                                   "object_track_labels": np.array([3])})
            window.update(1, 2, np.array([42]), np.array([3, 9, 12]),
                          [[.99, .01, 0.]], [[.95, .05, np.nan]])
            with np.load(Path(temporary) / "object_state.npz") as snapshot:
                np.testing.assert_array_equal(snapshot["object_ids"], [9, 3, 12])
                np.testing.assert_array_equal(snapshot["H_object"][:2],
                                              np.stack([np.eye(4), 2 * np.eye(4)]))
                self.assertTrue(np.isnan(snapshot["H_object"][2]).all())
                np.testing.assert_array_equal(snapshot["object_model_valid"], [True, True, False])
                np.testing.assert_array_equal(snapshot["object_track_ids"], [99])
                # The stale caller payload is replaced with joint-window state.
                np.testing.assert_array_equal(snapshot["object_track_labels"], [9])

    def test_invalid_shapes_and_negative_values_are_rejected(self):
        for kwargs in (dict(weights=[-1.]), dict(weights=[[1., np.nan]])):
            with self.assertRaises(ValueError):
                self.update(ObjectWindow(self.cfg), 0, [.1, .9], **kwargs)
        with self.assertRaises(ValueError):
            self.update(ObjectWindow(self.cfg), 0, [-.1, .9])
        with self.assertRaises(ValueError):
            self.update(ObjectWindow(self.cfg), 0, [.1, .9], q=[[1.]])

    def test_single_global_object_requires_actual_absolute_fit(self):
        window = ObjectWindow(self.cfg)
        result = window.update(0, 1, np.array([42, 77]), np.array([9]),
                               [[1.], [1.]], [[.05], [np.nan]])
        np.testing.assert_array_equal(result["object_labels"], [9, -1])

    def test_empty_observation_batch_and_invalid_payload_are_safe(self):
        window = ObjectWindow(self.cfg)
        result = window.update(0, 1, np.empty(0, np.int64), np.array([0, 1]),
                               np.empty((0, 2)), np.empty((0, 2)))
        self.assertEqual(result["object_probabilities"].shape, (0, 2))
        with self.assertRaises(ValueError):
            self.update(window, 1, [.1, .9],
                        payload={"H_object": np.full((2, 4, 4), 'invalid')})
        self.assertEqual(window.to_dict()["updates"], 1)
        self.assertEqual(window.to_dict()["total_gaussians"], 0)

    def test_render_solver_shares_history_keeps_first_prior_and_freezes_motion_initialization(self):
        window = ObjectWindow(dict(self.cfg, render={"enabled": True}))
        payload = dict(xyz=np.array([[0., 0., 2.]]), color=np.ones((1, 3)),
                       scale=np.full((1, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], 2, axis=0))
        observations = dict(source=np.full((4, 4, 3), 100, np.uint8),
                            target=np.full((4, 4, 3), 120, np.uint8),
                            source_depth=np.full((4, 4), 2.), target_depth=np.full((4, 4), 2.),
                            source_confidence=np.ones((4, 4)), target_confidence=np.ones((4, 4)))
        captured = []

        def solve(current, cfg):
            captured.append((current._render_initial_probabilities.copy(), current._prior(0)))
            return dict(probabilities=np.array([[.1, .9]] if len(captured) == 1 else [[.9, .1]]),
                        render_support=np.ones((1, 2), bool), report={"render_passes": 2})

        solver = types.SimpleNamespace(optimize_render_window=solve)
        motion = np.array([[.2, .3, .5]])
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
                "sys.modules", {"object_render_optimizer": solver}):
            first_dir = Path(temporary) / "first"
            result = self.update(window, 0, [np.nan, np.nan], q=[[.975, .025]],
                                 payload=payload, render_observations=observations,
                                 q_motion_initial=motion, output_dir=first_dir)
            self.assertEqual(result["object_labels"].tolist(), [1])
            observations["source"][:] = 0
            self.assertTrue(np.all(window._active[(0, 1)]["render_observations"]["source"] == 100))
            result = self.update(window, 1, [np.nan, np.nan], q=[[.01, .99]],
                                 payload=payload, render_observations=observations)
            np.testing.assert_allclose(captured[0][0], [[.975, .025]])
            np.testing.assert_allclose(captured[1][0], [[.1, .9]])
            np.testing.assert_allclose(captured[1][1], [.975, .025])
            with np.load(first_dir / "object_state.npz") as saved:
                np.testing.assert_array_equal(saved["object_labels"], [0])
                np.testing.assert_allclose(saved["object_probabilities"], result["object_probabilities"])
                np.testing.assert_array_equal(saved["q_motion_initial"], motion)
                self.assertNotIn("source", saved.files)
                self.assertEqual(json.loads(str(saved["report_json"]))["render_optimization"]["render_passes"], 2)

    def test_render_mode_requires_raw_images_before_registering_a_snapshot(self):
        window = ObjectWindow(dict(self.cfg, render={"enabled": True}))
        payload = dict(xyz=np.array([[0., 0., 2.]]), color=np.ones((1, 3)),
                       scale=np.full((1, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], 2, axis=0))
        with self.assertRaisesRegex(ValueError, "RGB-D"):
            self.update(window, 0, [.1, .9], payload=payload)
        self.assertEqual(window.object_ids, [])
        self.assertEqual(window._active, {})


if __name__ == "__main__":
    unittest.main()

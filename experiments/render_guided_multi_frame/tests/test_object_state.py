"""CPU checks for continuing identity and current-observation warm fitting."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / "object_state.py"
_spec = importlib.util.spec_from_file_location("sequence_object_state", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectRegistry = _module.ObjectRegistry
from hybrid_motion import _core_labels, motion_probabilities, motion_costs


def pose(x=0., y=0.):
    H = np.eye(4)
    H[:2, 3] = [x, y]
    return H


def anchors_for(translations, group_size=8):
    rng = np.random.default_rng(23)
    K = np.asarray([[300., 0., 320.], [0., 300., 180.], [0., 0., 1.]])
    x0 = rng.uniform([-3., -1.5, 8.], [3., 1.5, 12.], (len(translations)*group_size, 3))
    labels = np.repeat(np.arange(len(translations)), group_size)
    models = [pose(t) for t in translations]
    x1 = np.vstack([x0[labels == k] @ H[:3, :3].T + H[:3, 3]
                    for k, H in enumerate(models)])

    def project(x):
        return x[:, :2] / x[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]

    anchors = dict(x0=x0, x1=x1,
                   matches=np.column_stack((project(x0), project(x1))),
                   confidence=np.ones(len(x0)))
    return anchors, K, models, labels


class ObjectIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(tracking=dict(prior_weight=.25))
        self.registry = ObjectRegistry(self.cfg)
        self.motion = dict(hybrid=dict(min_support=6, max_components=8,
                                       depth_weight=.05, core_margin=.1,
                                       refit_max_nfev=20))

    def finish(self, first, second, labels, tracks=None, reference=0):
        labels = np.asarray(labels, np.int32)
        count = max(int(labels.max())+1, 1)
        tracks = np.arange(len(labels)) if tracks is None else np.asarray(tracks)
        q = np.eye(count)[np.maximum(labels, 0)]
        return self.registry.finish_pair(first, second, [pose(k) for k in range(count)],
                                          q, tracks, reference, f"pairs/{first}_{second}",
                                          labels=labels)

    def test_local_indices_can_swap_and_reference_does_not_force_background(self):
        first = self.finish(1, 2, [0]*8+[1]*8)
        second = self.finish(2, 3, [1]*8+[0]*8, reference=0)
        self.assertEqual(first, {"0": 0, "1": 1})
        self.assertEqual(second, {"0": 1, "1": 0})
        state = self.registry.to_dict()
        self.assertEqual(state["objects"]["1"]["frames"], [1, 2, 3])
        self.assertEqual(state["objects"]["0"]["appearances"][1]["local_model"], 1)

    def test_first_reference_creates_object_zero_even_if_not_local_zero(self):
        mapping = self.finish(1, 2, [0]*8+[1]*8, reference=1)
        self.assertEqual(mapping, {"0": 1, "1": 0})

    def test_global_assignment_preserves_more_continuing_track_votes_than_greedy(self):
        self.finish(1, 2, [0]*8+[1]*8)
        # Local 0 has 5 votes for old 0 and 4 for old 1; local 1 has 3 for old
        # 0. Global assignment keeps 7 votes rather than greedily keeping 5.
        tracks = np.r_[np.arange(5), np.arange(8, 12), np.arange(5, 8)]
        mapping = self.finish(2, 3, [0]*9+[1]*3, tracks=tracks)
        self.assertEqual(mapping, {"0": 1, "1": 0})

    def test_split_assigns_new_child_id_and_records_parent(self):
        self.finish(1, 2, [0]*8+[1]*8)
        mapping = self.finish(2, 3, [0]*8+[1]*4+[2]*4)
        self.assertEqual(mapping["0"], 0)
        self.assertEqual(set(mapping.values()), {0, 1, 2})
        state = self.registry.to_dict()
        self.assertEqual(state["objects"]["2"]["parent_id"], 1)
        event = next(row for row in state["events"] if row["type"] == "split")
        self.assertEqual(event["parent_id"], 1)
        self.assertEqual(set(event["object_ids"]), {1, 2})

    def test_merge_retains_historical_object_and_records_association(self):
        self.finish(1, 2, [0]*8+[1]*8+[2]*8)
        mapping = self.finish(2, 3, [0]*8+[1]*16)
        state = self.registry.to_dict()
        survivor = mapping["1"]
        absorbed = ({1, 2} - {survivor}).pop()
        self.assertEqual(state["objects"][str(absorbed)]["merged_into"], survivor)
        self.assertEqual(state["objects"][str(absorbed)]["frames"], [1, 2])
        merge = next(row for row in state["events"] if row["type"] == "merge")
        self.assertEqual(set(merge["source_object_ids"]), {1, 2})
        self.assertEqual(len(state["objects"]), 3)

    def test_current_core_can_correct_a_tracks_old_object(self):
        self.finish(1, 2, [0]*8+[1]*8)
        self.finish(2, 3, [0]*8+[1]*7+[0])
        state = self.registry.to_dict()
        self.assertEqual(state["track_objects"]["15"], 0)
        self.assertIn(15, state["objects"]["1"]["track_ids"])
        self.assertIn(15, state["objects"]["0"]["track_ids"])
        event = next(row for row in state["events"] if row["type"] == "track_reassignment")
        self.assertEqual((event["source_object_id"], event["object_id"], event["tracks"]),
                         (1, 0, 1))

    def test_disappearing_tracks_do_not_delete_object_history(self):
        self.finish(1, 2, [0]*8+[1]*8)
        self.finish(2, 3, [0]*8, tracks=np.arange(8))
        state = self.registry.to_dict()
        self.assertEqual(state["objects"]["1"]["last_frame"], 2)
        self.assertEqual(state["objects"]["1"]["frames"], [1, 2])
        self.assertEqual(state["track_objects"]["8"], 1)

    def test_gray_anchors_do_not_create_or_overwrite_hard_track_evidence(self):
        self.finish(1, 2, [0]*8+[1]*8)
        ids = np.r_[np.arange(16), 99]
        labels = np.r_[np.zeros(8, int), np.ones(7, int), -1, -1]
        q = np.eye(2)[np.r_[np.zeros(8, int), np.ones(7, int), 0, 1]]
        self.registry.finish_pair(2, 3, [pose(), pose(1)], q, ids, 0, "pair", labels=labels)
        state = self.registry.to_dict()
        self.assertEqual(state["track_objects"]["15"], 1)
        self.assertNotIn("99", state["track_objects"])
        self.assertNotIn(99, state["objects"]["1"]["track_ids"])

    def test_six_continuing_gray_tracks_reconnect_object_identity(self):
        self.finish(1, 2, [0]*8+[1]*8)
        labels = np.r_[np.zeros(8, int), np.full(6, -1), np.ones(6, int)]
        tracks = np.r_[np.arange(14), np.arange(100, 106)]
        q = np.vstack((np.tile([1., 0.], (8, 1)),
                       np.tile([.1, .9], (6, 1)), np.tile([0., 1.], (6, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1)], q, tracks, 0,
                                              "pair", labels=labels)
        self.assertEqual(mapping, {"0": 0, "1": 1})
        event = next(row for row in self.registry.events
                     if row["type"] == "identity_from_soft_tracks")
        self.assertEqual(event["support_track_ids"], list(range(8, 14)))
        self.assertAlmostEqual(event["vote"], 4.8)
        np.testing.assert_array_equal(labels[8:14], -np.ones(6, int))

    def test_soft_votes_cannot_replace_an_existing_hard_identity(self):
        self.finish(1, 2, [0]*8+[1]*8+[2]*8)
        tracks = np.r_[np.arange(8), 8, np.arange(16, 24)]
        labels = np.r_[np.zeros(8, int), 1, np.full(8, -1)]
        q = np.vstack((np.tile([1., 0.], (8, 1)), [0., 1.],
                       np.tile([.01, .99], (8, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1)], q, tracks, 0,
                                              "pair", labels=labels)
        self.assertEqual(mapping, {"0": 0, "1": 1})
        self.assertFalse(any(row["type"] == "identity_from_soft_tracks"
                             for row in self.registry.events))
        self.assertEqual(self.registry.track_objects[16], 2)

    def test_uniform_gray_probabilities_do_not_force_an_identity(self):
        self.finish(1, 2, [0]*8+[1]*8)
        labels = np.r_[np.zeros(8, int), np.full(6, -1), np.ones(6, int)]
        tracks = np.r_[np.arange(14), np.arange(100, 106)]
        q = np.vstack((np.tile([1., 0.], (8, 1)),
                       np.tile([.5, .5], (6, 1)), np.tile([0., 1.], (6, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1)], q, tracks, 0,
                                              "pair", labels=labels)
        self.assertEqual(mapping, {"0": 0, "1": 2})
        self.assertFalse(any(row["type"] == "identity_from_soft_tracks"
                             for row in self.registry.events))

    def test_soft_identity_does_not_reassign_its_gray_supporters_hard_membership(self):
        self.finish(1, 2, [0]*8+[1]*8+[2]*8)
        # Six old object-1 tracks reconnect local 1. A grey object-2 track
        # also favours local 1, but that relative preference must not overwrite
        # its own previously recorded hard object or posterior.
        tracks = np.r_[np.arange(14), 16, np.arange(100, 106)]
        labels = np.r_[np.zeros(8, int), np.full(7, -1), np.ones(6, int)]
        q = np.vstack((np.tile([1., 0.], (8, 1)), np.tile([.1, .9], (6, 1)),
                       [.01, .99], np.tile([0., 1.], (6, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1)], q, tracks, 0,
                                              "pair", labels=labels)
        self.assertEqual(mapping["1"], 1)
        self.assertEqual(self.registry.track_objects[16], 2)
        self.assertEqual(self.registry.track_probabilities[16], {2: 1.})
        self.assertNotIn(16, self.registry.objects[1]["track_ids"])
        event = next(row for row in self.registry.events
                     if row["type"] == "identity_from_soft_tracks")
        self.assertNotIn(16, event["support_track_ids"])

    def test_prepare_first_pair_preserves_discovered_geometry(self):
        anchors, K, models, labels = anchors_for([.1, 1.])
        result, current, q, report = self.registry.prepare_pair(
            1, 2, anchors, K, self.motion, models, labels, np.arange(16))
        np.testing.assert_array_equal(result, models)
        np.testing.assert_array_equal(current, labels)
        expected, _ = motion_probabilities(motion_costs(anchors, models, K, self.motion["hybrid"]),
                                            self.motion["hybrid"])
        np.testing.assert_array_equal(q, expected)
        self.assertEqual(report["initial_local_to_object"], [-1, -1])
        self.assertEqual(report["inherited_tracks"], 0)

    def test_warm_fit_uses_current_pair_anchors_instead_of_previous_pose(self):
        self.finish(1, 2, [0]*8+[1]*8)
        anchors, K, current_models, _ = anchors_for([.15, 1.2])
        # Only background was discovered. Tracks must recover a new current
        # vehicle pose, whose translation differs from its saved 1->2 pose.
        _, discovered_labels = _core_labels(anchors, current_models[:1], K, self.motion["hybrid"])
        models, labels, q, report = self.registry.prepare_pair(
            2, 3, anchors, K, self.motion, current_models[:1], discovered_labels, np.arange(16))
        self.assertEqual(len(models), 2)
        np.testing.assert_allclose(models[1], current_models[1], atol=1e-7)
        np.testing.assert_array_equal(labels, [0]*8+[1]*8)
        self.assertEqual(report["initial_local_to_object"], [0, 1])
        self.assertEqual(report["warm_fits"][1]["action"], "append_missing")
        np.testing.assert_allclose(q.sum(axis=1), 1.)
        self.assertEqual(len(self.registry.pairs), 1)

    def test_current_fit_improves_existing_candidate_without_appending_alias(self):
        self.finish(1, 2, [0]*8+[1]*8)
        anchors, K, actual, _ = anchors_for([.15, 1.2])
        discovered = [actual[0].copy(), pose(1.17)]
        _, labels = _core_labels(anchors, discovered, K, self.motion["hybrid"])
        models, current, q, report = self.registry.prepare_pair(
            2, 3, anchors, K, self.motion, discovered, labels, np.arange(16))
        self.assertEqual(len(models), 2)
        np.testing.assert_allclose(models[1], actual[1], atol=1e-7)
        np.testing.assert_array_equal(current, [0]*8+[1]*8)
        self.assertEqual(report["warm_fits"][1]["action"], "refit_existing")

    def test_small_soft_prior_can_be_overruled_by_current_evidence(self):
        # Track 15 was labelled vehicle, but its present observation follows
        # background. The prior should give a small pull, not fix its label.
        self.finish(1, 2, [0]*8+[1]*8)
        anchors, K, models, labels = anchors_for([.15, 1.2])
        anchors["x1"][15] = anchors["x0"][15] + models[0][:3, 3]
        x = anchors["x1"][15]
        anchors["matches"][15, 2:] = x[:2]/x[2]*[K[0, 0], K[1, 1]] + K[:2, 2]
        _, labels = _core_labels(anchors, models, K, self.motion["hybrid"])
        current_models, current, q, report = self.registry.prepare_pair(
            2, 3, anchors, K, self.motion, models, labels, np.arange(16))
        self.assertEqual(current[15], 0)
        self.assertEqual(q[15].argmax(), 0)
        self.assertGreater(q[15, 1], .24)
        self.assertLess(q[15, 1], .26)
        self.assertEqual(report["prior_mixed_tracks"], 16)

    def test_prepare_retains_newly_discovered_independent_object(self):
        self.finish(1, 2, [0]*8+[1]*8)
        anchors, K, discovered, labels = anchors_for([.15, 1.2, -1.])
        tracks = np.r_[np.arange(16), np.arange(100, 108)]
        models, current, q, report = self.registry.prepare_pair(
            2, 3, anchors, K, self.motion, discovered, labels, tracks)
        self.assertEqual(len(models), 3)
        np.testing.assert_allclose(models[2], discovered[2])
        np.testing.assert_array_equal(current, labels)
        self.assertEqual(report["initial_local_to_object"], [0, 1, -1])
        self.assertEqual(q.shape, (24, 3))

    def test_to_dict_is_independent_and_serializable(self):
        import json
        self.finish(1, 2, [0]*8+[1]*8)
        state = self.registry.to_dict()
        json.dumps(state)
        state["objects"]["0"]["frames"].append(999)
        self.assertNotIn(999, self.registry.to_dict()["objects"]["0"]["frames"])


class WindowIdentityTests(unittest.TestCase):
    def setUp(self):
        self.registry = ObjectRegistry(dict(object_window=dict(enabled=True),
                                            tracking=dict(prior_weight=.25)))

    def finish(self, first, second, labels, tracks=None, reference=0):
        labels = np.asarray(labels, np.int32)
        count = max(int(labels.max())+1, 1)
        tracks = np.arange(len(labels)) if tracks is None else np.asarray(tracks)
        q = np.eye(count)[np.maximum(labels, 0)]
        return self.registry.finish_pair(first, second, [pose(k) for k in range(count)],
                                         q, tracks, reference, "pair", labels=labels)

    def test_several_local_candidates_can_continue_one_object_without_split(self):
        self.finish(1, 2, [0]*8+[1]*8)
        mapping = self.finish(2, 3, [0]*8+[1]*4+[2]*4)
        self.assertEqual(mapping, {"0": 0, "1": 1, "2": 1})
        self.assertEqual(len(self.registry.objects), 2)
        appearances = self.registry.objects[1]["appearances"]
        self.assertEqual([row["local_model"] for row in appearances[1:]], [1, 2])
        self.assertFalse(any(event["type"] == "split" for event in self.registry.events))

    def test_pair_labels_do_not_overwrite_inherited_track_state(self):
        self.finish(1, 2, [0]*8+[1]*8)
        old = dict(self.registry.track_probabilities[15])
        mapping = self.finish(2, 3, [0]*8+[1]*7+[0])
        self.assertEqual(mapping["0"], 0)
        self.assertEqual(self.registry.track_objects[15], 1)
        self.assertEqual(self.registry.track_probabilities[15], old)
        self.assertFalse(any(event["type"] == "track_reassignment"
                             for event in self.registry.events))
        # A genuinely new track still gets an initial object state.
        self.finish(3, 4, [0]*8+[1]*8, tracks=np.r_[np.arange(8), np.arange(100, 108)])
        self.assertIn(100, self.registry.track_objects)

    def test_soft_association_can_reuse_an_object_with_a_hard_local_alias(self):
        self.finish(1, 2, [0]*8+[1]*8)
        tracks = np.r_[np.arange(16), np.arange(100, 106)]
        labels = np.r_[np.zeros(8, int), np.ones(2, int), np.full(6, -1),
                       np.full(6, 2)]
        q = np.vstack((np.tile([1., 0., 0.], (8, 1)),
                       np.tile([0., 1., 0.], (2, 1)),
                       np.tile([.05, .05, .9], (12, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1), pose(2)],
                                            q, tracks, 0, "pair", labels=labels)
        self.assertEqual(mapping, {"0": 0, "1": 1, "2": 1})
        event = next(row for row in self.registry.events
                     if row["type"] == "identity_from_soft_tracks")
        self.assertEqual(event["support_track_ids"], list(range(10, 16)))

    def test_tied_soft_observations_do_not_force_an_identity(self):
        self.finish(1, 2, [0]*8+[1]*8)
        labels = np.r_[np.zeros(8, int), np.ones(2, int), np.full(6, -1),
                       np.full(6, 2)]
        q = np.vstack((np.tile([1., 0., 0.], (8, 1)),
                       np.tile([0., 1., 0.], (2, 1)),
                       np.full((6, 3), 1/3), np.tile([0., 0., 1.], (6, 1))))
        mapping = self.registry.finish_pair(2, 3, [pose(), pose(1), pose(2)],
                                            q, np.r_[np.arange(16), np.arange(100, 106)],
                                            0, "pair", labels=labels)
        self.assertNotEqual(mapping["2"], 1)
        self.assertFalse(any(row["type"] == "identity_from_soft_tracks"
                             for row in self.registry.events))

    def test_missing_candidate_does_not_merge_or_delete_old_objects(self):
        self.finish(1, 2, [0]*8+[1]*8+[2]*8)
        self.finish(2, 3, [0]*8+[1]*16)
        self.assertTrue({0, 1, 2}.issubset(self.registry.objects))
        for oid in (0, 1, 2):
            self.assertIsNone(self.registry.objects[oid]["merged_into"])
        self.assertFalse(any(row["type"] == "merge" for row in self.registry.events))
        self.assertEqual(self.registry.track_objects[23], 2)

    def test_prepare_keeps_conflicting_local_aliases_for_window_resolution(self):
        self.finish(1, 2, [0]*8+[1]*16)
        anchors, K, models, labels = anchors_for([.15, 1.2, .5])
        motion = dict(hybrid=dict(min_support=6, max_components=8, depth_weight=.05,
                                  core_margin=.1, refit_max_nfev=20))
        result, current, q, report = self.registry.prepare_pair(
            2, 3, anchors, K, motion, models, labels, np.arange(24))
        self.assertEqual(report["initial_local_to_object"], [0, 1, 1])
        self.assertEqual(report["warm_fits"][1]["action"], "ambiguous_local_aliases")
        np.testing.assert_allclose(result[1:], models[1:])
        np.testing.assert_array_equal(current, labels)
        np.testing.assert_array_equal(q.argmax(axis=1), labels)
        np.testing.assert_allclose(q.sum(axis=1), 1.)

    def test_only_joint_window_commit_revises_confident_track_membership(self):
        self.finish(1, 2, [0]*8+[1]*8)
        self.finish(2, 3, [0]*8+[1]*7+[0])
        self.registry.commit_window_tracks([15, 14], [0, 1], [[.97, .03], [.6, .4]], [0, -1])
        self.assertEqual(self.registry.track_objects[15], 0)
        self.assertEqual(self.registry.track_objects[14], 1)
        self.assertAlmostEqual(self.registry.track_probabilities[14][0], .6)
        self.assertIn(15, self.registry.objects[1]["track_ids"])
        self.assertIn(15, self.registry.objects[0]["track_ids"])
        event = next(row for row in self.registry.events if row["type"] == "track_reassignment")
        self.assertEqual(event["source"], "object_window")
        self.assertEqual((event["source_object_id"], event["object_id"], event["tracks"]), (1, 0, 1))

    def test_conflicting_duplicate_window_rows_cannot_use_latest_label(self):
        self.finish(1, 2, [0]*8+[1]*8)
        self.registry.commit_window_tracks([15, 15], [0, 1], [[.1, .9], [.9, .1]], [1, 0])
        self.assertEqual(self.registry.track_objects[15], 1)
        self.assertEqual(self.registry.track_probabilities[15], {0: .5, 1: .5})


if __name__ == "__main__":
    unittest.main()

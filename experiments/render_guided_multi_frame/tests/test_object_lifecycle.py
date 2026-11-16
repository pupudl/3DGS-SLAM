"""CPU checks for object births, bounded absence, and global identity lifetime."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import tempfile
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / "tracking/object_state.py"
_spec = importlib.util.spec_from_file_location("lifecycle_object_state", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ObjectRegistry = _module.ObjectRegistry


class ObjectLifecycleTests(unittest.TestCase):
    def registry(self, **options):
        return ObjectRegistry(dict(sequence=dict(streaming=True),
                                   object_window=dict(enabled=True),
                                   object_lifecycle=options))

    @staticmethod
    def finish(registry, first, second, labels, tracks=None, reference=0):
        labels = np.asarray(labels, np.int32)
        tracks = np.arange(len(labels)) if tracks is None else np.asarray(tracks)
        count = max(int(labels.max())+1, 1)
        q = np.eye(count)[np.maximum(labels, 0)]
        return registry.finish_pair(first, second, [np.eye(4) for _ in range(count)],
                                    q, tracks, reference, f"pair_{first}_{second}", labels=labels)

    def initialize(self, registry):
        self.finish(registry, 0, 1, [0]*8)
        registry.observe_frame(0, [0])
        registry.observe_frame(1, [0])

    def vehicle(self, registry, first=10, tracks=None):
        tracks = np.r_[np.arange(8), np.arange(100, 108)] if tracks is None else tracks
        return self.finish(registry, first, first+1, [0]*8+[1]*8, tracks)

    def test_vehicle_10_to_16_closes_at_last_seen_and_new_vehicle_gets_new_id(self):
        registry = self.registry(missing_grace_frames=2)
        self.initialize(registry)
        for frame in range(2, 10):
            registry.observe_frame(frame, [0])
        mapping = self.vehicle(registry)
        self.assertEqual(mapping, {"0": 0, "1": 1})
        self.assertEqual(registry.objects[1]["status"], "provisional")
        self.assertIsNone(registry.objects[1]["first_frame"])
        self.assertIsNone(registry.objects[1]["last_frame"])
        self.assertFalse(any(event["type"] == "birth" and event["object_id"] == 1
                             for event in registry.events))
        for frame in range(10, 17):
            registry.observe_frame(frame, [0, 1])
        registry.observe_frame(17, [0])
        self.assertEqual(registry.objects[1]["status"], "missing")
        registry.observe_frame(18, [0])
        car = registry.objects[1]
        self.assertEqual((car["first_observed_frame"], car["last_observed_frame"]), (10, 16))
        self.assertEqual((car["status"], car["end_frame"], car["retired_at_frame"]),
                         ("retired", 16, 18))
        self.assertEqual(registry.candidate_object_ids(), [0])
        self.assertTrue(set(range(100, 108)).isdisjoint(registry.track_objects))
        # Even reused stale track numbers cannot reactivate the retired ID.
        mapping = self.vehicle(registry, 20)
        self.assertEqual(mapping, {"0": 0, "1": 2})
        registry.observe_frame(20, [0, 2])
        self.assertEqual(registry.objects[2]["first_observed_frame"], 20)
        self.assertEqual(registry.objects[1]["last_observed_frame"], 16)

    def test_brief_occlusion_reactivates_by_continuing_tracks_without_new_birth(self):
        registry = self.registry(missing_grace_frames=3)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0])
        registry.observe_frame(12, [0])
        self.assertEqual(registry.objects[1]["status"], "missing")
        self.assertEqual(self.vehicle(registry, 13), {"0": 0, "1": 1})
        registry.observe_frame(13, [0, 1])
        self.assertEqual(registry.objects[1]["status"], "active")
        self.assertEqual(registry.objects[1]["missing_frames"], 0)
        births = [event for event in registry.events
                  if event["type"] == "birth" and event["object_id"] == 1]
        self.assertEqual(len(births), 1)
        self.assertTrue(any(event["type"] == "reactivation" for event in registry.events))

    def test_unrelated_tracks_do_not_reactivate_a_missing_vehicle(self):
        registry = self.registry(missing_grace_frames=3)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0])
        mapping = self.vehicle(registry, 12, np.r_[np.arange(8), np.arange(200, 208)])
        self.assertEqual(mapping, {"0": 0, "1": 2})
        registry.observe_frame(12, [0, 2])
        self.assertEqual(registry.objects[1]["status"], "missing")
        self.assertEqual(registry.objects[2]["first_observed_frame"], 12)

    def test_frame_stride_does_not_consume_missing_grace(self):
        registry = self.registry(missing_grace_frames=3)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(1000, [0])
        self.assertEqual(registry.objects[1]["status"], "missing")
        self.assertEqual(registry.objects[1]["missing_frames"], 1)
        registry.observe_frame(2000, [0])
        self.assertEqual(registry.objects[1]["missing_frames"], 2)
        registry.observe_frame(3000, [0])
        self.assertEqual(registry.objects[1]["retired_at_frame"], 3000)
        self.assertEqual(registry.objects[1]["end_frame"], 10)

    def test_same_source_target_frame_adds_presence_without_aging_or_double_count(self):
        registry = self.registry(missing_grace_frames=3)
        self.initialize(registry)
        registry.observe_frame(10, [0])
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(10, [0, 1])
        self.assertEqual(registry.objects[1]["first_observed_frame"], 10)
        self.assertEqual(registry.objects[1]["observation_count"], 1)
        registry.observe_frame(11, [0])
        registry.observe_frame(11, [0])
        self.assertEqual(registry.objects[1]["missing_frames"], 1)
        registry.observe_frame(11, [0, 1])
        self.assertEqual(registry.objects[1]["missing_frames"], 0)
        self.assertEqual(registry.objects[1]["observation_count"], 2)
        self.assertEqual(registry.to_dict()["lifecycle"]["processed_frames"], 4)

    def test_same_frame_positive_backfill_corrects_tentative_retirement_only(self):
        registry = self.registry(missing_grace_frames=1)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0])
        self.assertEqual(registry.objects[1]["status"], "retired")
        registry.observe_frame(11, [0, 1])
        self.assertEqual(registry.objects[1]["status"], "active")
        self.assertIsNone(registry.objects[1]["end_frame"])
        registry.observe_frame(12, [0])
        registry.observe_frame(13, [0, 1])
        self.assertEqual(registry.objects[1]["status"], "retired")
        self.assertEqual(registry.objects[1]["last_observed_frame"], 11)

    def test_next_source_keeps_just_retired_identity_through_prepare_finish_and_commit(self):
        from test_object_state import anchors_for
        registry = self.registry(missing_grace_frames=2)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0, 1])
        registry.observe_frame(12, [0])
        registry.observe_frame(13, [0])
        self.assertEqual(registry.candidate_object_ids(), [0])
        self.assertNotIn(100, registry.track_objects)
        tracks = np.r_[np.arange(8), np.arange(100, 108)]
        registry.retain_tracks(tracks)
        anchors, intrinsics, models, labels = anchors_for([.15, 1.2])
        motion = dict(hybrid=dict(min_support=6, max_components=8, depth_weight=.05,
                                  core_margin=.1, refit_max_nfev=20))
        models, labels, q, report = registry.prepare_pair(
            13, 14, anchors, intrinsics, motion, models, labels, tracks)
        self.assertEqual(report["initial_local_to_object"], [0, 1])
        self.assertEqual(report["inherited_tracks"], 16)
        self.assertEqual(report["prior_mixed_tracks"], 16)
        mapping = registry.finish_pair(13, 14, models, q, tracks, 0, "pair", labels=labels)
        self.assertEqual(mapping, {"0": 0, "1": 1})
        self.assertEqual(len(registry.objects), 2)
        # The backend's window commit precedes current-source lifecycle
        # correction and therefore masks this retired column at this instant.
        registry.commit_window_tracks([100], [0, 1], [[.01, .99]], [1])
        registry.observe_frame(13, [0, 1])
        registry.observe_frame(14, [0, 1])
        self.assertEqual(registry.track_objects[100], 1)
        self.assertEqual(registry.track_probabilities[100], {1: 1.})
        self.assertEqual(registry.objects[1]["last_observed_frame"], 14)
        self.assertEqual(registry.objects[1]["status"], "active")

    def test_unrecovered_correction_tracks_expire_on_advance_or_cache_eviction(self):
        advanced = self.registry(missing_grace_frames=1)
        self.initialize(advanced)
        self.vehicle(advanced)
        advanced.observe_frame(10, [0, 1])
        advanced.observe_frame(11, [0])
        self.assertTrue(advanced._retirement_track_objects)
        advanced.observe_frame(12, [0])
        self.assertFalse(advanced._retirement_track_objects)
        self.assertFalse(advanced._retirement_track_probabilities)
        self.assertEqual(self.vehicle(advanced, 12)["1"], 2)

        registry = self.registry(missing_grace_frames=1)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0])
        self.assertTrue(registry._retirement_track_objects)
        registry.retain_tracks(np.arange(8))
        self.assertFalse(registry._retirement_track_objects)
        self.assertFalse(registry._retirement_track_probabilities)
        registry.observe_frame(12, [0])
        mapping = self.vehicle(registry, 12)
        self.assertEqual(mapping["1"], 2)

    def test_persistent_background_survives_track_turnover(self):
        registry = self.registry()
        self.initialize(registry)
        registry.retain_tracks([])
        mapping = self.finish(registry, 2, 3, [0]*8, np.arange(100, 108))
        self.assertEqual(mapping, {"0": 0})
        registry.observe_frame(2, [])
        registry.observe_frame(3, [])
        self.assertEqual(registry.objects[0]["status"], "active")
        self.assertEqual(registry.candidate_object_ids(), [0])
        self.assertTrue(any(event["type"] == "background_reference_fallback"
                             for event in registry.events))

    def test_changed_reference_does_not_overwrite_tracked_vehicle_as_background(self):
        registry = self.registry()
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        mapping = self.finish(registry, 11, 12, [1]*8+[0]*8,
                              np.r_[np.arange(8), np.arange(100, 108)], reference=0)
        self.assertEqual(mapping, {"0": 1, "1": 0})
        # With only vehicle tracks left, its reference ID is still O1.
        mapping = self.finish(registry, 12, 13, [0]*8, np.arange(100, 108))
        self.assertEqual(mapping, {"0": 1})

    def test_proposed_motion_without_current_posterior_support_is_not_a_birth(self):
        registry = self.registry()
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0])
        registry.observe_frame(11, [0])
        self.assertEqual(registry.objects[1]["status"], "retired")
        self.assertFalse(registry.objects[1]["confirmed"])
        self.assertIsNone(registry.objects[1]["first_observed_frame"])
        self.assertFalse(any(event["type"] == "birth" and event["object_id"] == 1
                             for event in registry.events))

    def test_presence_report_contains_current_ids_and_only_new_retirements(self):
        registry = self.registry(missing_grace_frames=1)
        self.initialize(registry)
        self.vehicle(registry)
        report = registry.observe_frame(10, [0, 1])
        self.assertEqual(report["observed_object_ids"], [0, 1])
        self.assertEqual(report["active_object_ids"], [0, 1])
        report = registry.observe_frame(11, [0])
        self.assertEqual(report["newly_retired_object_ids"], [1])
        self.assertEqual(report["retired_object_count"], 1)
        report = registry.observe_frame(12, [0, 1])
        self.assertEqual(report["observed_object_ids"], [0])
        self.assertEqual(report["newly_retired_object_ids"], [])
        self.assertEqual(report["retired_object_count"], 1)

    def test_event_journal_preserves_all_events_despite_tiny_memory_limit(self):
        registry = self.registry(event_limit=1)
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory)/"nested"/"events.jsonl"
            registry.set_event_log(event_path)
            self.initialize(registry)
            self.vehicle(registry)
            registry.observe_frame(10, [0, 1])
            registry.observe_frame(11, [0])
            registry.observe_frame(12, [0])
            rows = [json.loads(line) for line in event_path.read_text().splitlines()]
            self.assertEqual(len(rows), registry.to_dict()["lifecycle"]["event_count"])
            self.assertEqual([row["event_index"] for row in rows], list(range(1, len(rows)+1)))
            self.assertEqual(len(registry.events), 1)
            self.assertTrue(any(row["type"] == "proposal" for row in rows))
            self.assertTrue(any(row["type"] == "birth" for row in rows))
            self.assertTrue(any(row["type"] == "retirement" for row in rows))
            with self.assertRaises(ValueError):
                self.registry().set_event_log(event_path)
            with self.assertRaises(ValueError):
                registry.set_event_log(Path(directory)/"late.jsonl")

    def test_backend_pair_presence_stops_at_source_exit_without_window_history_renewal(self):
        # Exercise the real backend's lifecycle bridge with small, already
        # optimized observations. Rendering and local motion fitting are not
        # needed to verify which time instant each observation belongs to.
        backend_path = _path.parents[1]/"ownership/gaussian_object_backend.py"
        spec = importlib.util.spec_from_file_location("lifecycle_backend", backend_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        registry = self.registry(missing_grace_frames=2)
        backend = module.SequenceObjectBackend(registry.cfg)
        self.initialize(registry)
        self.vehicle(registry)
        for frame in range(10, 16):
            report = backend._observe_current_pair(
                registry, frame, frame+1, dict(object_labels=np.array([0, 1])),
                np.array([0, 1]), np.array([True, True]),
                np.array([[0., np.nan], [np.nan, 0.]]), np.eye(2))
        self.assertEqual(report["target"]["observed_object_ids"], [0, 1])
        # The car is in source16 but exits before target17. Its source
        # posterior stays valid; there is no measured target fit for O1.
        report = backend._observe_current_pair(
            registry, 16, 17, dict(object_labels=np.array([0, 1])),
            np.array([0, 1]), np.array([True, True]),
            np.array([[0., np.nan], [np.nan, np.nan]]), np.array([[1., 0.], [0., 0.]]))
        self.assertEqual(report["source"]["observed_object_ids"], [0, 1])
        self.assertEqual(report["target"]["observed_object_ids"], [0])
        self.assertEqual(registry.objects[1]["missing_frames"], 1)
        backend._observe_current_pair(
            registry, 17, 18, dict(object_labels=np.array([0])),
            np.array([0]), np.array([True]), np.zeros((1, 1)), np.ones((1, 1)))
        self.assertEqual(registry.objects[1]["status"], "retired")
        self.assertEqual(registry.objects[1]["first_observed_frame"], 10)
        self.assertEqual(registry.objects[1]["last_observed_frame"], 16)
        self.assertEqual(registry.objects[1]["retired_at_frame"], 18)

    def test_pruning_retains_reachable_priors_and_bounded_object_history(self):
        registry = self.registry(history_limit=3, event_limit=5)
        self.initialize(registry)
        for frame in range(2, 15):
            self.finish(registry, frame, frame+1, [0]*8, np.arange(frame*100, frame*100+8))
            registry.observe_frame(frame, [0])
            registry.retain_tracks(np.arange(frame*100, frame*100+8))
        state = registry.to_dict()
        self.assertLessEqual(len(state["pairs"]), 3)
        self.assertLessEqual(len(state["events"]), 5)
        self.assertLessEqual(len(state["objects"]["0"]["appearances"]), 3)
        self.assertLessEqual(len(state["objects"]["0"]["frames"]), 3)
        self.assertEqual(len(state["objects"]["0"]["track_ids"]), 8)
        self.assertEqual(len(state["track_objects"]), 8)
        self.assertEqual(len(state["track_probabilities"]), 8)
        self.assertEqual(state["lifecycle"]["pair_count"], 14)
        self.assertEqual(state["objects"]["0"]["first_observed_frame"], 0)
        json.dumps(state)

    def test_archived_window_labels_cannot_restore_retired_track_priors(self):
        registry = self.registry(missing_grace_frames=1)
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0])
        registry.commit_window_tracks([100], [0, 1], [[.1, .9]], [1])
        self.assertNotIn(100, registry.track_objects)
        self.assertEqual(registry.track_probabilities[100], {0: 1.})

    def test_finalize_closes_interval_without_adding_an_unobserved_frame(self):
        registry = self.registry()
        self.initialize(registry)
        self.vehicle(registry)
        registry.observe_frame(10, [0, 1])
        registry.observe_frame(11, [0, 1])
        registry.finalize_sequence()
        registry.finalize_sequence()
        self.assertEqual(registry.objects[1]["retirement_reason"], "sequence_end")
        self.assertEqual(registry.objects[1]["end_frame"], 11)
        self.assertEqual(registry.objects[1]["frames"], [10, 11])
        self.assertEqual(registry.objects[0]["status"], "active")
        self.assertTrue(registry.to_dict()["lifecycle"]["finalized"])
        with self.assertRaises(ValueError):
            registry.observe_frame(12, [0, 1])

    def test_explicit_disable_preserves_legacy_history_and_candidates(self):
        registry = self.registry(enabled=False)
        self.initialize(registry)
        self.vehicle(registry)
        registry.retain_tracks([])
        registry.finalize_sequence()
        self.assertEqual(registry.candidate_object_ids(), [0, 1])
        self.assertIn(100, registry.track_objects)
        self.assertNotIn("status", registry.objects[1])


if __name__ == "__main__":
    unittest.main()

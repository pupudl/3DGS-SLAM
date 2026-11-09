"""CPU integration checks between pair outputs and the object-window backend."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("object_window_sequence_runner", HERE / "run_sequence.py")
_module = importlib.util.module_from_spec(_spec)
with patch.object(sys, "path", [str(HERE), *sys.path]):
    _spec.loader.exec_module(_module)
ObjectRegistry = _module.ObjectRegistry
SequencePairContext = _module.SequencePairContext
configure_object_window = _module.configure_object_window


class FixedTracks:
    def __init__(self, count):
        self.ids = np.arange(count, dtype=np.int64)
        self.calls = []

    def register_matches(self, first, second, matches):
        self.calls.append((first, second, matches.copy()))
        return self.ids.copy()


class RecordingBackend:
    def __init__(self):
        self.calls = []

    def process(self, first, second, gmap, anchors, ids, mapping, state, K,
                *, output, registry, **observations):
        # Pair identity must already exist, while current noisy local labels
        # must not yet have overwritten inherited track identity.
        self.before = registry.to_dict()
        self.calls.append(dict(first=first, second=second, gmap=gmap,
                               anchors=anchors, ids=ids, mapping=mapping,
                               state=state, K=K, output=output, **observations))
        registry.commit_window_tracks([7], [0, 1], [[.99, .01]], [0])
        return dict(assigned_object=registry.track_objects[7], joint_window=True)


class SequenceObjectWindowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "pair"
        self.output.mkdir()
        self.K = np.array([[80., 0., 16.], [0., 80., 12.], [0., 0., 1.]])
        self.cfg = dict(object_window=dict(enabled=True), motion=dict(hybrid={}))
        rng = np.random.default_rng(4)
        xyz = rng.uniform([-.5, -.5, 7.], [.5, .5, 9.], (8, 3))
        H = np.tile(np.eye(4), (2, 1, 1))
        H[1, 0, 3] = 1.
        labels = np.array([0]*4+[1]*3+[0])
        xyz1 = xyz + H[labels, :3, 3]

        def project(points):
            return points[:, :2]/points[:, 2:] * self.K.diagonal()[:2] + self.K[:2, 2]

        self.anchors = dict(x0=xyz, x1=xyz1,
                            matches=np.column_stack((project(xyz), project(xyz1))),
                            confidence=np.ones(8))
        self.gmap = dict(xyz=xyz.copy(), uv=project(xyz), color=np.zeros((8, 3)),
                         scale=np.full((8, 3), .01))
        self.state = dict(H=H, q=np.array([[.9, .1]]*4+[[.1, .9]]*4),
                          q_anchor=np.eye(2)[labels], reference=0)
        self.observations = dict(source=np.zeros((24, 32, 3), np.uint8),
                                 target=np.zeros((24, 32, 3), np.uint8),
                                 source_depth=np.ones((24, 32)), target_depth=np.ones((24, 32)),
                                 source_confidence=np.ones((24, 32)),
                                 target_confidence=np.ones((24, 32)))

    def registry(self, enabled=True):
        registry = ObjectRegistry(dict(object_window=dict(enabled=enabled)))
        registry.finish_pair(0, 1, self.state["H"], np.eye(2)[[0]*4+[1]*4],
                             np.arange(8), 0, "previous", labels=[0]*4+[1]*4)
        return registry

    def test_configuration_freezes_q_and_structural_updates_without_mutating_input(self):
        cfg = dict(object_window=dict(enabled=True), joint=dict(tau_final=.3),
                   discovery=dict(enable_repair=True, enable_birth=True),
                   ownership=dict(refresh_after_feedback=True))
        original = deepcopy(cfg)
        configured = configure_object_window(cfg)
        self.assertEqual(cfg, original)
        self.assertTrue(configured["joint"]["freeze_motion_ownership"])
        for operation in ("birth", "split", "merge", "prune"):
            self.assertFalse(configured["discovery"][f"enable_{operation}"])
        self.assertTrue(configured["discovery"]["enable_repair"])
        self.assertEqual(configured["joint"]["tau_final"], .3)
        configured["ownership"]["refresh_after_feedback"] = False
        self.assertTrue(cfg["ownership"]["refresh_after_feedback"])

    def test_configuration_without_window_keeps_legacy_flags(self):
        cfg = dict(joint=dict(freeze_motion_ownership=False),
                   discovery=dict(enable_birth=True))
        self.assertEqual(configure_object_window(cfg), cfg)
        self.assertEqual(configure_object_window(dict(object_window=dict(enabled=True)))["joint"],
                         dict(freeze_motion_ownership=True))

    def test_finish_passes_raw_observations_after_mapping_and_commits_before_export(self):
        registry, backend = self.registry(), RecordingBackend()
        context = SequencePairContext(1, 2, FixedTracks(8), registry, self.root, backend)
        initial_q, anchor_q = self.state["q"].copy(), self.state["q_anchor"].copy()
        report = context.finish(self.anchors, self.gmap, self.state, self.K,
                                self.output, self.cfg, observations=self.observations)
        self.assertEqual(len(backend.calls), 1)
        call = backend.calls[0]
        self.assertEqual(call["mapping"], {"0": 0, "1": 1})
        self.assertEqual(backend.before["pairs"][-1]["first"], 1)
        self.assertEqual(backend.before["track_objects"]["7"], 1)
        self.assertEqual(registry.track_objects[7], 0)
        for key, value in self.observations.items():
            self.assertIs(call[key], value)
        for key, value in (("gmap", self.gmap), ("anchors", self.anchors),
                           ("state", self.state), ("K", self.K)):
            self.assertIs(call[key], value)
        np.testing.assert_array_equal(self.state["q"], initial_q)
        np.testing.assert_array_equal(self.state["q_anchor"], anchor_q)
        with np.load(self.output / "gaussian_state.npz") as saved:
            np.testing.assert_array_equal(saved["q"], initial_q)
        with np.load(self.output / "pair_tracks.npz") as saved:
            np.testing.assert_array_equal(saved["q_anchor"], anchor_q)
            self.assertEqual(saved["labels"][-1], 0)
        self.assertEqual(report["object_window"], dict(assigned_object=0, joint_window=True))
        self.assertEqual(report, json.loads((self.output / "temporal_state.json").read_text()))
        self.assertIs(context.final_report, report)

    def test_missing_observations_fail_before_registry_tracks_or_files_change(self):
        registry, tracks = self.registry(), FixedTracks(8)
        original = registry.to_dict()
        backend = RecordingBackend()
        context = SequencePairContext(1, 2, tracks, registry, self.root, backend)
        with self.assertRaisesRegex(ValueError, "raw pair observations"):
            context.finish(self.anchors, self.gmap, self.state, self.K,
                           self.output, self.cfg)
        self.assertEqual(registry.to_dict(), original)
        self.assertEqual(tracks.calls, [])
        self.assertEqual(backend.calls, [])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_incomplete_observations_fail_before_registry_or_output_changes(self):
        registry, tracks = self.registry(), FixedTracks(8)
        original = registry.to_dict()
        backend = RecordingBackend()
        context = SequencePairContext(1, 2, tracks, registry, self.root, backend)
        incomplete = dict(self.observations)
        del incomplete["target_depth"]
        with self.assertRaisesRegex(ValueError, "raw pair observations"):
            context.finish(self.anchors, self.gmap, self.state, self.K,
                           self.output, self.cfg, observations=incomplete)
        self.assertEqual(registry.to_dict(), original)
        self.assertEqual(tracks.calls, [])
        self.assertEqual(backend.calls, [])
        self.assertEqual(list(self.output.iterdir()), [])

    def test_finish_without_backend_keeps_legacy_pair_exports_and_track_updates(self):
        registry = self.registry(enabled=False)
        context = SequencePairContext(1, 2, FixedTracks(8), registry, self.root)
        report = context.finish(self.anchors, self.gmap, self.state, self.K,
                                self.output, dict(motion=dict(hybrid={})))
        self.assertEqual(registry.track_objects[7], 0)
        self.assertNotIn("object_window", report)
        self.assertEqual(report["inherited_3d_tracks"], 8)
        self.assertTrue((self.output / "pair_tracks.npz").is_file())
        self.assertTrue((self.output / "gaussian_state.npz").is_file())


if __name__ == "__main__":
    unittest.main()

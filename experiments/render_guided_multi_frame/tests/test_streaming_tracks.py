"""CPU regressions for incremental, bounded-memory track persistence."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import sqlite3
import tempfile
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / "tracking/feature_tracks.py"
_spec = importlib.util.spec_from_file_location("streaming_feature_tracks", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FeatureTrackStore = _module.FeatureTrackStore


class StreamingTrackTests(unittest.TestCase):
    def store(self, cache_frames=2):
        return FeatureTrackStore({"sequence": {"streaming": True},
                                  "tracking": {"cache_frames": cache_frames,
                                               "join_radius_px": .5}}, device="cpu")

    def test_long_chain_releases_history_and_exports_all_observations(self):
        store = self.store()
        persistent = None
        newly_created = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first_chunk = None
            for frame in range(1, 61):
                # The second feature is deliberately far from the preceding
                # pair's target, creating one short track per pair.
                ids = store.register_matches(frame, frame + 1, [
                    [frame, 0, frame + 1, 0],
                    [100 + frame * 10, 5, 101 + frame * 10, 5]])
                if persistent is None:
                    persistent = int(ids[0])
                self.assertEqual(int(ids[0]), persistent)
                newly_created.append(int(ids[1]))
                summary = store.save(output)
                self.assertLessEqual(len(store.frames), 2)
                self.assertTrue(all(len(history) <= 2 for history in store.tracks.values()))
                self.assertLessEqual(len(store.tracks), 3)
                self.assertEqual(summary["tracks"], frame + 1)
                self.assertEqual(summary["observations"], (frame + 1) + frame * 2)
                if frame == 1:
                    first_chunk = (output / "observation_chunks/0000000001.npz").read_bytes()
                self.assertEqual(first_chunk,
                    (output / "observation_chunks/0000000001.npz").read_bytes())
            self.assertEqual(newly_created, list(range(1, 61)))
            self.assertEqual(store.current_track_ids, {persistent, newly_created[-1]})
            summary = store.save(output, final=True)
            self.assertEqual(summary["frame_count"], 61)
            self.assertEqual(summary["tracks_with_three_or_more_frames"], 1)
            self.assertEqual(len(list((output / "observation_chunks").glob("*.npz"))), 61)
            self.assertFalse((output / "recent_observations.npz").exists())
            with np.load(output / "observations.npz") as observations:
                self.assertEqual(len(observations["track_ids"]), 181)
                np.testing.assert_array_equal(observations["frames"][:61], np.arange(1, 62))
                np.testing.assert_array_equal(observations["lengths"], [61] + [2] * 60)
            tracks = json.loads((output / "tracks.json").read_text())
            self.assertEqual(tracks[0]["frames"], list(range(1, 62)))
            self.assertEqual(tracks[-1]["track_id"], 60)
            manifest = json.loads((output / "tracks_manifest.json").read_text())
            self.assertTrue(manifest["finalized"])
            self.assertEqual(manifest["sealed_through"], 61)
            self.assertFalse(manifest["resume_supported"])

    def test_latest_target_can_gain_no_depth_tracks_before_being_sealed(self):
        store = self.store(cache_frames=1)
        with tempfile.TemporaryDirectory() as directory:
            store.register_matches(10, 11, [[1, 1, 2, 1]])
            first_summary = store.save(directory)
            self.assertEqual(first_summary["observations"], 2)
            self.assertEqual(set(store.frames), {11})
            ids = store.register_matches(11, 12, [[2, 1, 3, 1], [20, 20, 21, 20]])
            summary = store.save(directory)
            self.assertEqual(summary["observations"], 5)
            self.assertEqual(summary["tracks"], 2)
            # Duplicate saves must not increment cumulative track lengths.
            self.assertEqual(store.save(directory)["observations"], 5)
            with np.load(Path(directory) / "observation_chunks/0000000011.npz") as chunk:
                np.testing.assert_array_equal(chunk["track_ids"], ids)
                np.testing.assert_allclose(chunk["uv"], [[2, 1], [20, 20]])
            store.save(directory, final=True)
            with np.load(Path(directory) / "observations.npz") as archive:
                np.testing.assert_array_equal(archive["lengths"], [3, 2])
                np.testing.assert_array_equal(archive["frames"], [10, 11, 12, 11, 12])

    def test_conflicts_remain_rejected_after_pruning(self):
        store = self.store(cache_frames=1)
        with tempfile.TemporaryDirectory() as directory:
            ids = store.register_matches(0, 1, [[10, 10, 11, 10], [30, 30, 31, 30]])
            store.save(directory)
            store.register_matches(1, 2, [[11, 10, 12, 10], [31, 30, 32, 30]])
            conflict = store.register_matches(1, 2, [[11, 10, 32, 30], [11, 10, 50, 50]])
            np.testing.assert_array_equal(conflict, [-1, -1])
            np.testing.assert_array_equal(store.track_ids_for_frame(2, [[12, 10], [32, 30]]), ids)
            summary = store.save(directory, final=True)
            self.assertEqual(summary["tracks"], 2)
            self.assertEqual(summary["conflicting_matches"], 2)
            self.assertEqual(summary["observations"], 6)

    def test_cannot_recreate_past_frame_or_reuse_an_existing_archive(self):
        store = self.store(cache_frames=1)
        with tempfile.TemporaryDirectory() as directory:
            store.register_matches(0, 1, [[1, 1, 2, 1]])
            store.save(directory)
            with self.assertRaises(ValueError):
                store.register_matches(0, 1, [[1, 1, 2, 1]])
            with self.assertRaises(ValueError):
                store.observations_for_frame(0)
            with self.assertRaises(FileExistsError):
                self.store().save(directory)
            with tempfile.TemporaryDirectory() as other:
                with self.assertRaises(ValueError):
                    store.save(other)

    def test_features_for_adjacent_pairs_are_shared_with_one_frame_cache(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")

        class Extractor:
            def __init__(self):
                self.config = {"max_num_keypoints": 1, "detection_threshold": .01}
                self.calls = 0

            def __call__(self, data):
                self.calls += 1
                return {"keypoints": torch.tensor([[[5., 5.]]]),
                        "keypoint_scores": torch.tensor([[.9]]),
                        "descriptors": torch.ones((1, 1, 2))}, None

        class Matcher:
            def __call__(self, data):
                return {"matches0": torch.tensor([[0]])}

        store = self.store(cache_frames=1)
        extractor, matcher = Extractor(), Matcher()
        image = np.zeros((16, 16, 3), np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            for first in range(8):
                store.match_pair(first, first + 1, image, image, extractor, matcher)
                store.save(directory)
                self.assertEqual(set(store.frames), {first + 1})
            self.assertEqual(extractor.calls, 9)
            self.assertEqual(store.current_track_ids, {0})
            store.save(directory, final=True)
            with sqlite3.connect(Path(directory) / "track_index.sqlite3") as database:
                self.assertEqual(database.execute("SELECT length FROM tracks").fetchall(), [(9,)])

    def test_final_save_of_empty_store_and_repeat_finalize(self):
        store = self.store()
        with tempfile.TemporaryDirectory() as directory:
            summary = store.save(directory, final=True)
            self.assertEqual(summary["observations"], 0)
            self.assertEqual(store.save(directory, final=True), summary)
            with np.load(Path(directory) / "observations.npz") as archive:
                self.assertEqual(archive["uv"].shape, (0, 2))
            self.assertEqual(json.loads((Path(directory) / "tracks.json").read_text()), [])


if __name__ == "__main__":
    unittest.main()

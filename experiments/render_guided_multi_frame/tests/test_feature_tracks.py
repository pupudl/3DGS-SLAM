"""CPU checks for cross-frame identity and shared supplemental features."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / "feature_tracks.py"
_spec = importlib.util.spec_from_file_location("sequence_feature_tracks", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
FeatureTrackStore = _module.FeatureTrackStore


class TrackIdentityTests(unittest.TestCase):
    def setUp(self):
        self.store = FeatureTrackStore({"tracking": {"join_radius_px": 0.5}}, device="cpu")

    def test_chain_and_new_tracks(self):
        ids = self.store.register_matches(1, 2, [[10, 10, 12, 10], [30, 30, 32, 30]])
        next_ids = self.store.register_matches(2, 3,
                                               [[12, 10, 14, 10], [60, 60, 62, 60]])
        self.assertEqual(ids[0], next_ids[0])
        self.assertNotIn(next_ids[1], ids)
        self.assertEqual(sorted(self.store.tracks[int(ids[0])]), [1, 2, 3])
        self.assertEqual(sorted(self.store.tracks[int(ids[1])]), [1, 2])

    def test_repeated_match_does_not_create_track(self):
        initial = self.store.register_matches(1, 2, [[10, 10, 12, 10]])
        repeated = self.store.register_matches(1, 2, [[10.1, 10, 12.1, 10]])
        np.testing.assert_array_equal(initial, repeated)
        self.assertEqual(len(self.store.tracks), 1)

    def test_existing_tracks_are_not_merged(self):
        ids = self.store.register_matches(1, 2, [[10, 10, 12, 10], [30, 30, 32, 30]])
        conflict = self.store.register_matches(1, 2, [[10, 10, 32, 30]])
        self.assertEqual(conflict[0], -1)
        np.testing.assert_array_equal(
            self.store.track_ids_for_frame(2, [[12, 10], [32, 30]]), ids)
        self.assertEqual(len(self.store.tracks), 2)

    def test_track_cannot_have_two_points_in_one_frame(self):
        ids = self.store.register_matches(1, 2, [[10, 10, 12, 10]])
        conflict = self.store.register_matches(1, 2, [[10, 10, 20, 20]])
        self.assertEqual(conflict[0], -1)
        self.assertEqual(len(self.store.observations_for_frame(2)["uv"]), 1)
        self.assertEqual(self.store.track_ids_for_frame(2, [[20, 20]])[0], -1)
        self.assertEqual(self.store.track_ids_for_frame(1, [[10, 10]])[0], ids[0])

    def test_observations_exist_without_any_depth(self):
        self.store.register_matches(1, 2, [[10, 10, 12, 10]])
        self.store.register_matches(2, 3, [[12, 10, 14, 10]])
        with tempfile.TemporaryDirectory() as directory:
            summary = self.store.save(directory)
            obs = np.load(Path(directory) / "observations.npz")
            np.testing.assert_array_equal(obs["frames"], [1, 2, 3])
            np.testing.assert_array_equal(obs["track_ids"], [0, 0, 0])
            np.testing.assert_allclose(obs["uv"], [[10, 10], [12, 10], [14, 10]])
            self.assertEqual(summary["tracks_with_three_or_more_frames"], 1)


class SharedFeatureTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        class Extractor:
            def __init__(self):
                self.config = {"max_num_keypoints": 1, "detection_threshold": .01}
                self.calls = 0

            def __call__(self, data):
                self.calls += 1
                features = {
                    "keypoints": torch.tensor([[[5., 5.], [15., 15.], [25., 25.]]]),
                    "keypoint_scores": torch.tensor([[.9, .8, .7]]),
                    "descriptors": torch.eye(3)[None],
                }
                return features, None

        class Matcher:
            def __call__(self, data):
                # Descriptor labels allow only the same feature to match.
                d0 = data["descriptors0"][0]
                d1 = data["descriptors1"][0]
                scores = d0 @ d1.T
                values, indices = scores.max(dim=1)
                indices[values < .5] = -1
                return {"matches0": indices[None]}

        self.extractor = Extractor()
        self.matcher = Matcher()
        self.store = FeatureTrackStore({
            "features": {"initial_keypoints": 1, "supplemental_keypoints": 3,
                         "supplemental_threshold": 1e-5}}, device="cpu")
        self.image = np.zeros((32, 32, 3), np.uint8)

    def test_shared_frame_is_extracted_only_once(self):
        self.store.match_pair(1, 2, self.image, self.image, self.extractor, self.matcher)
        self.store.match_pair(2, 3, self.image, self.image, self.extractor, self.matcher)
        self.assertEqual(self.extractor.calls, 3)
        self.assertEqual(self.extractor.config["max_num_keypoints"], 1)
        self.assertEqual(self.extractor.config["detection_threshold"], .01)

    def test_mask_supplement_tracks_continue_beyond_initial_top_k(self):
        self.store.match_pair(1, 2, self.image, self.image, self.extractor, self.matcher)
        mask = np.zeros((32, 32), bool)
        mask[15, 15] = True
        added = self.store.match_mask(1, 2, self.image, self.image,
                                     self.extractor, self.matcher, mask)
        np.testing.assert_allclose(added, [[15, 15, 15, 15]])
        supplement_id = self.store.track_ids_for_frame(2, [[15, 15]])[0]
        # The target's same low-ranked point is outside initial top-K. It has
        # to be available to match a tracked source feature in the next frame.
        subsequent = self.store.match_pair(2, 3, self.image, self.image,
                                           self.extractor, self.matcher)
        self.assertTrue(np.any(np.all(subsequent[:, :2] == [15, 15], axis=1)))
        self.assertEqual(self.store.track_ids_for_frame(3, [[15, 15]])[0], supplement_id)


if __name__ == "__main__":
    unittest.main()

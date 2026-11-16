"""Regressions for forward LK tracks with descriptors at their true endpoints."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch


_PATH = Path(__file__).resolve().parents[1] / "tracking/feature_tracks.py"
if str(_PATH.parents[1]) not in sys.path:
    sys.path.insert(0, str(_PATH.parents[1]))
_SPEC = importlib.util.spec_from_file_location("flow_feature_tracks", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FeatureTrackStore = _MODULE.FeatureTrackStore


class FakeExtractor:
    """Return a detector catalogue independently of the sampled flow endpoints."""

    def __init__(self, catalogues, descriptor=None):
        self.catalogues = catalogues
        self.descriptor = descriptor
        self.calls = 0
        self.sampled = []
        self.config = {"max_num_keypoints": 1024,
                       "detection_threshold": .01,
                       "dense_descriptor_mode": "coarse"}

    def _describe(self, frame, points):
        if self.descriptor is not None:
            return self.descriptor(frame, points)
        result = torch.zeros((*points.shape[:-1], 3), dtype=points.dtype,
                             device=points.device)
        result[..., 0] = 1.
        return result

    def __call__(self, data):
        self.calls += 1
        uv = self.catalogues[min(self.calls - 1, len(self.catalogues) - 1)]
        points = torch.as_tensor(uv, dtype=torch.float32).reshape(1, -1, 2)
        features = {"keypoints": points,
                    "keypoint_scores": torch.linspace(.99, .5, points.shape[1])[None],
                    "descriptors": self._describe(self.calls, points)}
        # A marker identifying the frame is sufficient for the fake sampler;
        # the production extractor returns its normalized coarse descriptor map.
        coarse = torch.full((1, 3, 12, 16), float(self.calls))
        return features, coarse

    def sample_descriptors_at(self, points, coarse):
        frame = int(coarse[0, 0, 0, 0].item())
        self.sampled.append((frame, points.detach().cpu().numpy().copy()))
        return self._describe(frame, points)


class ForbiddenMatcher:
    def __call__(self, data):
        raise AssertionError("The LK path must not invoke the descriptor matcher")


def translated_flow(delta=(2., 0.), status=None, error=None):
    """A standard OpenCV LK result with predictable subpixel endpoints."""
    def calculate(previous, current, points, initial=None, *args, **kwargs):
        p = np.asarray(points, np.float32).reshape(-1, 1, 2)
        n = len(p)
        return (p + np.asarray(delta, np.float32),
                np.ones((n, 1), np.uint8) if status is None else np.asarray(status, np.uint8).reshape(n, 1),
                np.zeros((n, 1), np.float32) if error is None else np.asarray(error, np.float32).reshape(n, 1))
    return calculate


class FlowTrackTests(unittest.TestCase):
    def store(self, initial=2, supplemental=4, streaming=False, flow=None):
        return FeatureTrackStore({
            "features": {"initial_keypoints": initial,
                         "supplemental_keypoints": supplemental},
            "tracking": {"matcher": "lk_descriptor", "join_radius_px": 1.5,
                         "cache_frames": 1, "catalog_keypoints": 32,
                         "flow": flow or {}},
            "sequence": {"streaming": streaming}}, device="cpu")

    def setUp(self):
        self.image = np.zeros((96, 128, 3), np.uint8)
        self.matcher = ForbiddenMatcher()

    def match(self, store, first, second, extractor, **kwargs):
        return store.match_pair(first, second, self.image, self.image,
                                extractor, self.matcher, **kwargs)

    def test_only_forward_lk_is_called_and_descriptors_use_its_endpoint(self):
        extractor = FakeExtractor([[[10, 10]], [[70, 70]]])
        store = self.store(initial=1)
        first = np.full_like(self.image, 30)
        second = np.full_like(self.image, 90)
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow((2.25, .75))) as lk:
            matches = store.match_pair(1, 2, first, second, extractor, self.matcher)
        self.assertEqual(lk.call_count, 1)
        args = lk.call_args.args
        np.testing.assert_array_equal(args[0], cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))
        np.testing.assert_array_equal(args[1], cv2.cvtColor(second, cv2.COLOR_BGR2GRAY))
        np.testing.assert_allclose(matches, [[10, 10, 12.25, 10.75]])
        target_samples = [uv.reshape(-1, 2) for frame, uv in extractor.sampled if frame == 2]
        self.assertTrue(any(np.any(np.linalg.norm(uv - [12.25, 10.75], axis=1) < 1e-5)
                            for uv in target_samples))

    def test_true_non_detector_endpoint_continues_the_same_track(self):
        extractor = FakeExtractor([[[10, 10]], [[60, 60]], [[80, 60]]])
        store = self.store(initial=1)
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow((2.25, .75))):
            self.match(store, 1, 2, extractor)
            track = int(store.track_ids_for_frame(2, [[12.25, 10.75]], radius=.01)[0])
            self.assertGreaterEqual(track, 0)
            matches = self.match(store, 2, 3, extractor)
        expected = np.array([12.25, 10.75, 14.5, 11.5], np.float32)
        self.assertTrue(np.any(np.linalg.norm(matches - expected, axis=1) < 1e-5))
        self.assertEqual(store.track_ids_for_frame(3, [[14.5, 11.5]], radius=.01)[0], track)
        self.assertEqual(sorted(store.tracks[track]), [1, 2, 3])
        self.assertEqual(extractor.calls, 3)

    def test_feedback_is_repeatable_without_duplicate_tracks_and_continues_low_rank_point(self):
        extractor = FakeExtractor([[[10, 10], [30, 30]]] * 3)
        store = self.store(initial=1)
        mask = np.zeros(self.image.shape[:2], bool)
        mask[29:32, 31:34] = True
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow()):
            self.match(store, 1, 2, extractor)
            store.match_mask(1, 2, self.image, self.image, extractor, self.matcher, mask)
            extra_track = int(store.track_ids_for_frame(2, [[32, 30]], radius=.01)[0])
            self.assertGreaterEqual(extra_track, 0)
            track_count = len(store.tracks)
            store.match_mask(1, 2, self.image, self.image, extractor, self.matcher, mask)
            self.assertEqual(len(store.tracks), track_count)
            self.assertEqual(store.track_ids_for_frame(2, [[32, 30]], radius=.01)[0], extra_track)
            self.match(store, 2, 3, extractor)
        self.assertEqual(store.track_ids_for_frame(3, [[34, 30]], radius=.01)[0], extra_track)

    def test_descriptor_rejection_does_not_register_the_rejected_observation(self):
        def descriptor(frame, points):
            result = torch.zeros((*points.shape[:-1], 3), dtype=points.dtype)
            rejected = (points[..., 0] > 20) & (frame == 2)
            result[..., 0] = (~rejected).float()
            result[..., 1] = rejected.float()
            return result

        extractor = FakeExtractor([[[10, 10], [30, 30]]] * 2, descriptor)
        store = self.store()
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow()):
            matches = self.match(store, 1, 2, extractor)
        np.testing.assert_allclose(matches, [[10, 10, 12, 10]])
        self.assertEqual(len(store.tracks), 1)
        self.assertEqual(store.track_ids_for_frame(2, [[32, 30]], radius=.01)[0], -1)

    def test_nearby_existing_tracks_keep_distinct_identities(self):
        store = self.store()
        original = store.register_matches(0, 1, [[18, 20, 20, 20], [22, 20, 24, 20]])
        extractor = FakeExtractor([[[20, 20], [24, 20]], [[70, 70], [80, 70]]])

        def converge(previous, current, points, initial=None, *args, **kwargs):
            uv = np.asarray(points, np.float32).reshape(-1, 2)
            out = np.column_stack((np.where(uv[:, 0] < 22, 35., 36.1), uv[:, 1]))
            return out.astype(np.float32).reshape(-1, 1, 2), np.ones((len(uv), 1), np.uint8), np.zeros((len(uv), 1), np.float32)

        with patch("cv2.calcOpticalFlowPyrLK", side_effect=converge):
            matches = self.match(store, 1, 2, extractor)
        self.assertEqual(len(matches), 2)
        np.testing.assert_array_equal(store.track_ids_for_frame(2, [[35, 20], [36.1, 20]], radius=.01), original)
        self.assertEqual(len(store.tracks), 2)

    def test_source_target_and_feedback_masks_apply_to_actual_flow_points(self):
        catalogues = [[[10, 10], [30, 30], [50, 50]]] * 2
        source = np.ones(self.image.shape[:2], bool)
        target = np.ones_like(source)
        source[10, 10] = False
        target[30, 32] = False
        feedback = np.zeros_like(source)
        feedback[50, 52] = True
        for mode in ("pair", "feedback"):
            with self.subTest(mode=mode):
                store, extractor = self.store(initial=3), FakeExtractor(catalogues)
                with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow()):
                    if mode == "pair":
                        matches = self.match(store, 1, 2, extractor,
                                             source_mask=source, target_mask=target)
                    else:
                        matches = store.match_mask(1, 2, self.image, self.image,
                            extractor, self.matcher, feedback,
                            source_mask=source, target_mask=target)
                np.testing.assert_allclose(matches, [[50, 50, 52, 50]])

    def test_failed_out_of_bounds_and_nonfinite_flow_are_not_registered(self):
        extractor = FakeExtractor([[[10, 10], [30, 30], [50, 50], [70, 70]]] * 2)
        store = self.store(initial=4)

        def invalid(previous, current, points, initial=None, *args, **kwargs):
            return (np.array([[[12, 10]], [[32, 30]], [[130, 50]], [[np.nan, 70]]], np.float32),
                    np.array([[1], [0], [1], [1]], np.uint8), np.zeros((4, 1), np.float32))

        with patch("cv2.calcOpticalFlowPyrLK", side_effect=invalid):
            matches = self.match(store, 1, 2, extractor)
        np.testing.assert_allclose(matches, [[10, 10, 12, 10]])
        self.assertEqual(len(store.tracks), 1)

    def test_configured_lk_error_limit_is_applied_before_track_registration(self):
        extractor = FakeExtractor([[[10, 10], [30, 30]]] * 2)
        store = self.store(flow={"max_error": 5.})
        with patch("cv2.calcOpticalFlowPyrLK",
                   side_effect=translated_flow(error=[1., 6.])):
            matches = self.match(store, 1, 2, extractor)
        np.testing.assert_allclose(matches, [[10, 10, 12, 10]])
        self.assertEqual(len(store.tracks), 1)

    def test_track_cap_counts_feedback_and_prioritizes_existing_tracks_next_pair(self):
        extractor = FakeExtractor([[[10, 10], [30, 30], [50, 50]]] * 3)
        store = self.store(initial=1, flow={"max_tracks": 2})
        with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow()):
            self.match(store, 1, 2, extractor)
            store.match_mask(1, 2, self.image, self.image, extractor, self.matcher,
                             np.ones(self.image.shape[:2], bool))
            ids = store.track_ids_for_frame(2, [[12, 10], [32, 30]], radius=.01)
            self.assertTrue(np.all(ids >= 0))
            self.assertEqual(len(store.tracks), 2)
            self.assertEqual(len(store.observations_for_frame(2)["uv"]), 2)
            self.assertEqual(store.track_ids_for_frame(2, [[52, 50]], radius=.01)[0], -1)
            # Repeated calls can return the same observations at the cap.
            self.match(store, 1, 2, extractor)
            self.assertEqual(len(store.tracks), 2)
            self.match(store, 2, 3, extractor)
        np.testing.assert_array_equal(store.track_ids_for_frame(3, [[14, 10], [34, 30]], radius=.01), ids)
        self.assertEqual(len(store.observations_for_frame(3)["uv"]), 2)
        self.assertEqual(len(store.tracks), 2)

    def test_invalid_track_caps_are_rejected(self):
        for value in (0, -1, 1.5, True, "2", None, float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.store(flow={"max_tracks": value})

    def test_invalid_flow_parameters_fail_even_with_no_seeds(self):
        from tracking.flow_correspondence import forward_flow
        invalid = [
            ("window_size", 1), ("window_size", 4), ("window_size", 3.5),
            ("max_level", -1), ("max_level", 1.5),
            ("max_iterations", 0), ("max_iterations", 2.5),
            ("epsilon", 0), ("epsilon", -.1),
            ("min_eigenvalue", -1), ("max_error", -1),
            ("descriptor_min_similarity", -1.1),
            ("descriptor_min_similarity", 1.1),
        ]
        for key in ("window_size", "max_level", "max_iterations", "epsilon",
                    "min_eigenvalue", "max_error", "descriptor_min_similarity"):
            invalid.extend((key, value) for value in (float("nan"), float("inf"), True, None, "bad"))
        extractor = FakeExtractor([[]])
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                with patch("cv2.calcOpticalFlowPyrLK") as lk:
                    with self.assertRaises(ValueError):
                        forward_flow(self.image, self.image, np.empty((0, 2), np.float32),
                                     extractor, None, None, {name: value})
                    lk.assert_not_called()

    def test_empty_features_and_missing_flow_result_are_safe(self):
        for catalogue, result in [([], None), ([[10, 10]], (None, None, None))]:
            with self.subTest(catalogue=catalogue):
                store, extractor = self.store(), FakeExtractor([catalogue, catalogue])
                with patch("cv2.calcOpticalFlowPyrLK", return_value=result):
                    matches = self.match(store, 1, 2, extractor)
                self.assertEqual(matches.shape, (0, 4))
                self.assertEqual(len(store.tracks), 0)

    def test_streaming_one_frame_cache_keeps_non_detector_track_and_full_archive(self):
        extractor = FakeExtractor([[[10, 10]], [[60, 60]], [[80, 60]]])
        store = self.store(initial=1, streaming=True)
        with tempfile.TemporaryDirectory() as directory:
            with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow((2.25, .75))):
                self.match(store, 1, 2, extractor)
                track = int(store.track_ids_for_frame(2, [[12.25, 10.75]], radius=.01)[0])
                store.save(directory)
                self.assertEqual(set(store.frames), {2})
                self.match(store, 2, 3, extractor)
                self.assertEqual(store.track_ids_for_frame(3, [[14.5, 11.5]], radius=.01)[0], track)
                store.save(directory, final=True)
            with np.load(Path(directory) / "observations.npz") as archive:
                own = archive["track_ids"] == track
                np.testing.assert_array_equal(archive["frames"][own], [1, 2, 3])
                np.testing.assert_allclose(archive["uv"][own], [[10, 10], [12.25, 10.75], [14.5, 11.5]])
            self.assertEqual(extractor.calls, 3)

    def test_streaming_sealed_source_is_rejected_even_if_still_cached(self):
        extractor = FakeExtractor([[[10, 10]], [[60, 60]]])
        store = self.store(initial=1, streaming=True)
        store.cache_frames = 2
        with tempfile.TemporaryDirectory() as directory:
            with patch("cv2.calcOpticalFlowPyrLK", side_effect=translated_flow()):
                self.match(store, 1, 2, extractor)
                store.save(directory)
                self.assertIn(1, store.frames)
                with self.assertRaises(ValueError):
                    self.match(store, 1, 2, extractor)

    def test_real_lk_tracks_subpixel_translation_of_random_texture(self):
        rng = np.random.default_rng(20261115)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (96, 128), np.uint8), (3, 3), .6)
        first = np.repeat(gray[..., None], 3, axis=2)
        delta = np.array([2.25, -1.5], np.float32)
        second = cv2.warpAffine(first, np.array([[1, 0, delta[0]], [0, 1, delta[1]]], np.float32),
                                (128, 96), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        points = np.array([[24, 24], [48, 24], [72, 48], [96, 64]], np.float32)
        extractor = FakeExtractor([points, points + delta])
        store = self.store(initial=4)
        matches = store.match_pair(1, 2, first, second, extractor, self.matcher)
        self.assertEqual(len(matches), len(points))
        np.testing.assert_allclose(matches[:, 2:] - matches[:, :2], np.tile(delta, (len(matches), 1)), atol=.15)


if __name__ == "__main__":
    unittest.main()

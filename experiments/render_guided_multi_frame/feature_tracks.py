"""Shared frame features and persistent 2D tracks for sequential pair updates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


class FeatureTrackStore:
    """Keep one feature catalogue per frame and join matches through that catalogue.

    Feature extraction is shared by initial and mask-guided matching.  Tracks
    are purely 2D: stereo/depth filtering in the pair estimator does not erase
    an observation.  A conflicting correspondence is left out rather than
    merging two established track identities.
    """

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.features_cfg = cfg.get("features", cfg)
        self.tracking_cfg = cfg.get("tracking", cfg.get("sequence", {}))
        self.device = device
        self.radius = float(self.tracking_cfg.get("join_radius_px", 1.5))
        self.frames = {}
        self.tracks = {}
        self._next_track = 0
        self.conflicting_matches = 0

    def _frame(self, frame):
        frame = int(frame)
        if frame not in self.frames:
            self.frames[frame] = {
                "uv": np.empty((0, 2), np.float32),
                "track_ids": np.empty(0, np.int64),
                "features": None,
                "catalog_ids": np.empty(0, np.int64),
            }
        return self.frames[frame]

    def _keypoint_ids(self, frame, uv, append=False, radius=None):
        record = self._frame(frame)
        uv = np.asarray(uv, np.float32).reshape(-1, 2)
        ids = np.full(len(uv), -1, np.int64)
        threshold = self.radius if radius is None else radius
        if len(record["uv"]) and len(uv):
            distance, nearest = cKDTree(record["uv"]).query(uv)
            close = distance <= threshold
            ids[close] = nearest[close]
        if append:
            additions = []
            original_count = len(record["uv"])
            for i in np.flatnonzero(ids < 0):
                if additions:
                    distances = np.linalg.norm(np.asarray(additions) - uv[i], axis=1)
                    best = int(np.argmin(distances))
                    if distances[best] <= threshold:
                        ids[i] = original_count + best
                        continue
                ids[i] = original_count + len(additions)
                additions.append(uv[i])
            if additions:
                record["uv"] = np.vstack((record["uv"], np.asarray(additions, np.float32)))
                record["track_ids"] = np.r_[record["track_ids"],
                                             np.full(len(additions), -1, np.int64)]
        return ids

    def register_matches(self, first, second, matches):
        """Record all matched 2D observations; return persistent IDs, -1 on conflict."""
        first, second = int(first), int(second)
        matches = np.asarray(matches, np.float32).reshape(-1, 4)
        ids0 = self._keypoint_ids(first, matches[:, :2], append=True)
        ids1 = self._keypoint_ids(second, matches[:, 2:], append=True)
        frame0, frame1 = self._frame(first), self._frame(second)
        result = np.full(len(matches), -1, np.int64)
        for i, (id0, id1) in enumerate(zip(ids0, ids1)):
            track0 = int(frame0["track_ids"][id0])
            track1 = int(frame1["track_ids"][id1])
            if track0 >= 0 and track1 >= 0 and track0 != track1:
                self.conflicting_matches += 1
                continue
            track = max(track0, track1)
            if track < 0:
                track = self._next_track
                self._next_track += 1
                self.tracks[track] = {}
            observations = self.tracks[track]
            if ((first in observations and observations[first] != int(id0)) or
                    (second in observations and observations[second] != int(id1))):
                self.conflicting_matches += 1
                continue
            observations[first] = int(id0)
            observations[second] = int(id1)
            frame0["track_ids"][id0] = track
            frame1["track_ids"][id1] = track
            result[i] = track
        return result

    def track_ids_for_frame(self, frame, uv, radius=None):
        ids = self._keypoint_ids(frame, uv, radius=radius)
        result = np.full(len(ids), -1, np.int64)
        known = ids >= 0
        result[known] = self._frame(frame)["track_ids"][ids[known]]
        return result

    def observations_for_frame(self, frame):
        record = self._frame(frame)
        ids = np.flatnonzero(record["track_ids"] >= 0)
        return {"track_ids": record["track_ids"][ids].copy(),
                "keypoint_ids": ids, "uv": record["uv"][ids].copy()}

    @staticmethod
    def _image_tensor(image, device):
        import torch
        # Input images use the same BGR convention as the two-frame runner.
        rgb = np.ascontiguousarray(image[..., ::-1])
        return torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)[None] / 255.

    def _extract(self, frame, image, extractor, device):
        import torch
        record = self._frame(frame)
        if record["features"] is not None:
            return record
        options = extractor.config
        old_limit = options["max_num_keypoints"]
        old_threshold = options.get("detection_threshold", 5e-5)
        # A single low-threshold catalogue supplies both initial and feedback
        # features. Target-mask matching spends its budget inside the mask.
        options["max_num_keypoints"] = int(self.tracking_cfg.get("catalog_keypoints", -1))
        options["detection_threshold"] = min(
            old_threshold, float(self.features_cfg.get("supplemental_threshold", old_threshold)))
        try:
            with torch.inference_mode():
                features, _ = extractor({"image": self._image_tensor(image, device)})
                order = torch.argsort(features["keypoint_scores"][0], descending=True)
                features = {key: value[:, order].detach().cpu() for key, value in features.items()}
        finally:
            options["max_num_keypoints"] = old_limit
            options["detection_threshold"] = old_threshold
        record["catalog_ids"] = self._keypoint_ids(
            frame, features["keypoints"][0].numpy(), append=True)
        record["features"] = features
        return record

    @staticmethod
    def _top_ids(record, limit, tracked=False, mask=None):
        points = record["features"]["keypoints"][0].numpy()
        ids = np.arange(len(points), dtype=np.int64)
        if mask is not None:
            xy = np.rint(points).astype(np.int64)
            h, w = mask.shape
            valid = ((xy[:, 0] >= 0) & (xy[:, 0] < w) &
                     (xy[:, 1] >= 0) & (xy[:, 1] < h))
            allowed = np.zeros(len(points), bool)
            allowed[valid] = np.asarray(mask, bool)[xy[valid, 1], xy[valid, 0]]
            ids = ids[allowed]
        selected = ids if limit < 0 else ids[:limit]
        if tracked:
            existing = ids[record["track_ids"][record["catalog_ids"][ids]] >= 0]
            selected = np.union1d(selected, existing)
        return selected

    def _match(self, first, second, image0, image1, extractor, matcher,
               limit, mask=None, device=None):
        import torch
        device = device or self.device
        frame0 = self._extract(first, image0, extractor, device)
        frame1 = self._extract(second, image1, extractor, device)
        ids0 = self._top_ids(frame0, limit, tracked=True)
        # Supplementary source tracks can have low detector scores. Expose
        # the target catalogue when extending tracks, so target top-K does
        # not silently terminate them before descriptor matching.
        extending = np.any(frame0["track_ids"] >= 0)
        target_limit = -1 if mask is None and extending else limit
        ids1 = self._top_ids(frame1, target_limit, tracked=True, mask=mask)
        if not len(ids0) or not len(ids1):
            return np.empty((0, 4), np.float32)
        with torch.inference_mode():
            features0 = {key + "0": value[:, ids0].to(device)
                         for key, value in frame0["features"].items()}
            features1 = {key + "1": value[:, ids1].to(device)
                         for key, value in frame1["features"].items()}
            data = {**features0, **features1,
                    "image0": self._image_tensor(image0, device),
                    "image1": self._image_tensor(image1, device)}
            indices = matcher(data)["matches0"][0].detach().cpu().numpy()
        good = indices >= 0
        if not np.any(good):
            return np.empty((0, 4), np.float32)
        uv0 = frame0["features"]["keypoints"][0, ids0[good]].numpy()
        uv1 = frame1["features"]["keypoints"][0, ids1[indices[good]]].numpy()
        matches = np.column_stack((uv0, uv1)).astype(np.float32)
        tracks = self.register_matches(first, second, matches)
        return matches[tracks >= 0]

    def match_pair(self, first, second, image0, image1, extractor, matcher, device=None):
        return self._match(first, second, image0, image1, extractor, matcher,
                           int(self.features_cfg.get("initial_keypoints", 1024)), device=device)

    def match_mask(self, first, second, image0, image1, extractor, matcher, mask, device=None):
        return self._match(first, second, image0, image1, extractor, matcher,
                           int(self.features_cfg.get("supplemental_keypoints", 2048)),
                           mask=np.asarray(mask, bool), device=device)

    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        observations = [(track, frame, keypoint, *self.frames[frame]["uv"][keypoint])
                        for track, history in sorted(self.tracks.items())
                        for frame, keypoint in sorted(history.items())]
        table = np.asarray(observations, float).reshape(-1, 5)
        unique = np.asarray(sorted(self.tracks), np.int64)
        np.savez_compressed(
            output_dir / "observations.npz", track_ids=table[:, 0].astype(np.int64),
            frames=table[:, 1].astype(np.int64), keypoint_ids=table[:, 2].astype(np.int64),
            uv=table[:, 3:].astype(np.float32), unique_track_ids=unique,
            first_frames=np.asarray([min(self.tracks[t]) for t in unique], np.int64),
            last_frames=np.asarray([max(self.tracks[t]) for t in unique], np.int64),
            lengths=np.asarray([len(self.tracks[t]) for t in unique], np.int64))
        features_dir = output_dir / "features"
        features_dir.mkdir(exist_ok=True)
        for frame, record in sorted(self.frames.items()):
            fields = {"keypoints": record["uv"], "track_ids": record["track_ids"],
                      "catalog_keypoint_ids": record["catalog_ids"]}
            if record["features"] is not None:
                fields.update({key: value[0].numpy()
                               for key, value in record["features"].items()
                               if key != "keypoints"})
            np.savez_compressed(features_dir / f"{frame:010d}.npz", **fields)
        summary = {"frames": sorted(self.frames), "tracks": len(unique),
                   "observations": len(table), "conflicting_matches": self.conflicting_matches,
                   "tracks_with_three_or_more_frames":
                       sum(len(history) >= 3 for history in self.tracks.values())}
        (output_dir / "tracks_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "tracks.json").write_text(json.dumps([
            {"track_id": int(t), "frames": sorted(self.tracks[t]),
             "first_frame": min(self.tracks[t]), "last_frame": max(self.tracks[t]),
             "length": len(self.tracks[t])} for t in unique
        ], indent=2), encoding="utf-8")
        return summary

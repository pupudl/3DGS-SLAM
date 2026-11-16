"""Shared frame features and persistent 2D tracks for sequential pair updates."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import tempfile

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
        self.matcher_method = self.tracking_cfg.get("matcher", "lightglue")
        if self.matcher_method not in ("lightglue", "lk_descriptor"):
            raise ValueError("tracking.matcher must be lightglue or lk_descriptor")
        self.flow_cfg = self.tracking_cfg.get("flow", {})
        self.max_flow_tracks = self.flow_cfg.get("max_tracks", 4096)
        if (isinstance(self.max_flow_tracks, bool) or not isinstance(self.max_flow_tracks, int)
                or self.max_flow_tracks < 1):
            raise ValueError("tracking.flow.max_tracks must be a positive integer")
        self._flow_pair = None
        self._flow_attempted = set()
        self._flow_proposals = {}
        self._flow_reports = []
        self.radius = float(self.tracking_cfg.get("join_radius_px", 1.5))
        self.frames = {}
        self.tracks = {}
        self._next_track = 0
        self.conflicting_matches = 0
        self.streaming = bool(cfg.get("sequence", {}).get("streaming", False))
        self.cache_frames = max(1, int(self.tracking_cfg.get("cache_frames", 2)))
        self._archive_dir = None
        self._sealed_through = None
        self._finalized = False

    @property
    def current_track_ids(self):
        """IDs observed in the latest frame, suitable for next-pair association."""
        if not self.frames:
            return set()
        ids = self.frames[max(self.frames)]["track_ids"]
        return set(map(int, ids[ids >= 0]))

    def _frame(self, frame):
        frame = int(frame)
        if frame not in self.frames:
            if (self.streaming and self._sealed_through is not None and
                    frame <= self._sealed_through):
                raise ValueError(f"Frame {frame} has been archived and evicted")
            self.frames[frame] = {
                "uv": np.empty((0, 2), np.float32),
                "track_ids": np.empty(0, np.int64),
                "features": None,
                "catalog_ids": np.empty(0, np.int64),
                "descriptor_map": None,
            }
        return self.frames[frame]

    def _keypoint_ids(self, frame, uv, append=False, radius=None):
        record = self._frame(frame)
        uv = np.asarray(uv, np.float32).reshape(-1, 2)
        ids = np.full(len(uv), -1, np.int64)
        threshold = self.radius if radius is None else radius
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("keypoint join radius must be finite and nonnegative")
        if not np.isfinite(uv).all():
            raise ValueError("keypoint coordinates must be finite float32 values")
        if len(record["uv"]) and len(uv):
            distance, nearest = cKDTree(record["uv"]).query(uv)
            close = distance <= threshold
            ids[close] = nearest[close]
        if append:
            additions = []
            original_count = len(record["uv"])
            # Search only nearby *accepted* additions, preserving the original
            # sequential nearest-neighbour rule (including first-index ties).
            # A doubled cell width also covers float32 subtraction rounding
            # and norm underflow at a zero/tiny radius. Final distances still
            # use the same float32 norm as the original implementation.
            smallest = float(np.nextafter(np.float32(0), np.float32(1)))
            width = 2. * max(float(threshold), math.sqrt(smallest))
            buckets = {}
            for i in np.flatnonzero(ids < 0):
                cell = tuple(math.floor(float(v) / width) for v in uv[i])
                candidates = sorted(index
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                    for index in buckets.get((cell[0]+dx, cell[1]+dy), ()))
                if candidates:
                    points = np.asarray([additions[j] for j in candidates], np.float32)
                    distances = np.linalg.norm(points - uv[i], axis=1)
                    best = int(np.argmin(distances))
                    if distances[best] <= threshold:
                        ids[i] = original_count + candidates[best]
                        continue
                ids[i] = original_count + len(additions)
                buckets.setdefault(cell, []).append(len(additions))
                additions.append(uv[i])
            if additions:
                record["uv"] = np.vstack((record["uv"], np.asarray(additions, np.float32)))
                record["track_ids"] = np.r_[record["track_ids"],
                                             np.full(len(additions), -1, np.int64)]
        return ids

    def register_matches(self, first, second, matches):
        """Record all matched 2D observations; return persistent IDs, -1 on conflict."""
        first, second = int(first), int(second)
        if self.streaming and (self._finalized or
                (self._sealed_through is not None and
                 min(first, second) <= self._sealed_through)):
            raise ValueError("Streaming matches must extend the latest unsealed frame")
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
                features, descriptor_map = extractor({"image": self._image_tensor(image, device)})
                order = torch.argsort(features["keypoint_scores"][0], descending=True)
                features = {key: value[:, order].detach().cpu() for key, value in features.items()}
        finally:
            options["max_num_keypoints"] = old_limit
            options["detection_threshold"] = old_threshold
        record["catalog_ids"] = self._keypoint_ids(
            frame, features["keypoints"][0].numpy(), append=True)
        record["features"] = features
        if self.matcher_method == "lk_descriptor":
            if descriptor_map is None or not hasattr(extractor, "sample_descriptors_at"):
                raise ValueError("LK descriptor tracking requires a coarse descriptor map and endpoint sampler")
            if (descriptor_map.ndim != 4 or
                    descriptor_map.shape[1] != features["descriptors"].shape[-1]):
                raise ValueError("Use SuperPoint(dense_descriptor_mode='coarse') for LK tracking")
            record["descriptor_map"] = descriptor_map.detach().cpu()
        return record

    @staticmethod
    def _allowed(uv, mask):
        if mask is None:
            return np.ones(len(uv), bool)
        xy = np.rint(uv).astype(np.int64)
        h, w = mask.shape
        valid = ((xy[:, 0] >= 0) & (xy[:, 0] < w) &
                 (xy[:, 1] >= 0) & (xy[:, 1] < h))
        keep = np.zeros(len(uv), bool)
        keep[valid] = np.asarray(mask, bool)[xy[valid, 1], xy[valid, 0]]
        return keep

    def _flow_source_ids(self, record, limit, source_mask):
        # Existing endpoints need not coincide with any current detector point.
        tracked = np.flatnonzero(record["track_ids"] >= 0)
        detected = record["catalog_ids"][self._top_ids(record, limit, mask=source_mask)]
        detected = np.unique(detected[record["track_ids"][detected] < 0])
        if len(tracked) and len(detected):
            distances, _ = cKDTree(record["uv"][tracked]).query(record["uv"][detected])
            detected = detected[distances > self.radius]
        ids = np.r_[tracked, detected]
        return ids[self._allowed(record["uv"][ids], source_mask)]

    def _register_flow_matches(self, first, second, source_ids, matches):
        """Store exact flow endpoints; preserve close, distinct established IDs."""
        first, second = int(first), int(second)
        frame0, frame1 = self._frame(first), self._frame(second)
        result = np.full(len(matches), -1, np.int64)
        original_count = len(frame1["uv"])
        occupied = np.flatnonzero(frame1["track_ids"] >= 0)
        occupied_count = len(occupied)
        buckets = {}

        def cell(uv):
            return (math.floor(float(uv[0]) * 4),
                    math.floor(float(uv[1]) * 4))

        for uv in frame1["uv"][occupied]:
            buckets.setdefault(cell(uv), []).append(uv)
        # Exact detector-coordinate reuse retains the original first-index
        # rule. Newly tracked subpixel endpoints are appended in one batch.
        exact = {}
        for index, uv in enumerate(frame1["uv"]):
            exact.setdefault(tuple(uv), index)
        pending_uv = np.empty((len(matches), 2), frame1["uv"].dtype)
        pending_tracks = np.empty(len(matches), np.int64)
        pending_count = 0
        for i, (id0, match) in enumerate(zip(source_ids, matches)):
            id0 = int(id0)
            track = int(frame0["track_ids"][id0])
            if track >= 0 and second in self.tracks[track]:
                id1 = self.tracks[track][second]
                existing_uv = (frame1["uv"][id1] if id1 < original_count else
                               pending_uv[id1-original_count])
                if np.allclose(existing_uv, match[2:], atol=1e-5, rtol=0):
                    result[i] = track
                continue
            if (track >= 0 and first in self.tracks[track] and
                    self.tracks[track][first] != id0):
                continue
            if occupied_count >= self.max_flow_tracks:
                continue
            uv = match[2:]
            key = cell(uv)
            nearby = [point for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                      for point in buckets.get((key[0]+dx, key[1]+dy), ())]
            if nearby:
                # Preserve the strict 0.25-pixel rejection with the same
                # float32 norm, while checking only adjacent occupied cells.
                distance = np.linalg.norm(np.asarray(nearby)-uv, axis=1)
                if np.any(distance < .25):
                    continue
            id1 = exact.get(tuple(uv))
            if id1 is None:
                id1 = original_count + pending_count
                pending_uv[pending_count] = uv
                exact[tuple(uv)] = id1
                pending_count += 1
            if track < 0:
                track = self._next_track
                self._next_track += 1
                self.tracks[track] = {}
            self.tracks[track].update({first: id0, second: id1})
            frame0["track_ids"][id0] = track
            if id1 < original_count:
                frame1["track_ids"][id1] = track
            else:
                pending_tracks[id1-original_count] = track
            # Keep references to persistent coordinate storage, not to the
            # caller's match array, while registering subsequent endpoints.
            accepted_uv = (frame1["uv"][id1] if id1 < original_count else
                           pending_uv[id1-original_count])
            buckets.setdefault(key, []).append(accepted_uv)
            occupied_count += 1
            result[i] = track
        if pending_count:
            frame1["uv"] = np.vstack((frame1["uv"], pending_uv[:pending_count]))
            frame1["track_ids"] = np.r_[frame1["track_ids"], pending_tracks[:pending_count]]
        return result

    def _match_flow(self, first, second, image0, image1, extractor, limit,
                    mask=None, device=None, source_mask=None, target_mask=None):
        from tracking.flow_correspondence import forward_flow
        first, second = int(first), int(second)
        if self.streaming and (self._finalized or
                (self._sealed_through is not None and
                 min(first, second) <= self._sealed_through)):
            raise ValueError("Streaming matches must extend the latest unsealed frame")
        device = device or self.device
        frame0 = self._extract(first, image0, extractor, device)
        frame1 = self._extract(second, image1, extractor, device)
        pair = (first, second)
        if self._flow_pair != pair:
            self._flow_pair = pair
            self._flow_attempted = set()
            self._flow_proposals = {}
        # Mask feedback can cheaply propagate the remaining catalogue, then
        # select its actual target endpoints inside the requested region.
        ids = self._flow_source_ids(frame0, -1 if mask is not None else limit, source_mask)
        new_ids = np.asarray([i for i in ids if int(i) not in self._flow_attempted], np.int64)
        report = dict(seedcount=0, accepted=0, elapsed_seconds=0.)
        if len(new_ids):
            uv0 = frame0["uv"][new_ids]
            uv1, keep, report = forward_flow(image0, image1, uv0, extractor,
                frame0["descriptor_map"], frame1["descriptor_map"], self.flow_cfg)
            self._flow_attempted.update(map(int, new_ids))
            for index in np.flatnonzero(keep):
                self._flow_proposals[int(new_ids[index])] = np.r_[uv0[index], uv1[index]].astype(np.float32)
        selected = [int(i) for i in ids if int(i) in self._flow_proposals]
        matches = np.asarray([self._flow_proposals[i] for i in selected], np.float32).reshape(-1, 4)
        allowed = target_mask if mask is None else np.asarray(mask, bool)
        if mask is not None and target_mask is not None:
            allowed = allowed & target_mask
        keep = self._allowed(matches[:, 2:], allowed)
        source_ids = np.asarray(selected, np.int64)[keep]
        matches = matches[keep]
        tracks = self._register_flow_matches(first, second, source_ids, matches)
        self._flow_reports.append(dict(first=first, second=second,
            stage="feedback" if mask is not None else "initial", **report,
            cached_proposals=len(selected)-int(np.sum(np.isin(selected, new_ids))),
            returned_matches=int(np.sum(tracks >= 0)), rejected_registration=int(np.sum(tracks < 0))))
        return matches[tracks >= 0]

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
               limit, mask=None, device=None, source_mask=None, target_mask=None):
        import torch
        device = device or self.device
        frame0 = self._extract(first, image0, extractor, device)
        frame1 = self._extract(second, image1, extractor, device)
        ids0 = self._top_ids(frame0, limit, tracked=True, mask=source_mask)
        # Supplementary source tracks can have low detector scores. Expose
        # the target catalogue when extending tracks, so target top-K does
        # not silently terminate them before descriptor matching.
        extending = np.any(frame0["track_ids"] >= 0)
        target_limit = -1 if mask is None and extending else limit
        allowed = target_mask if mask is None else mask
        if mask is not None and target_mask is not None:
            allowed = mask & target_mask
        ids1 = self._top_ids(frame1, target_limit, tracked=True, mask=allowed)
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

    def match_pair(self, first, second, image0, image1, extractor, matcher, device=None,
                   source_mask=None, target_mask=None):
        if self.matcher_method == "lk_descriptor":
            return self._match_flow(first, second, image0, image1, extractor,
                int(self.features_cfg.get("initial_keypoints", 1024)), device=device,
                source_mask=source_mask, target_mask=target_mask)
        return self._match(first, second, image0, image1, extractor, matcher,
                           int(self.features_cfg.get("initial_keypoints", 1024)), device=device,
                           source_mask=source_mask, target_mask=target_mask)

    def match_mask(self, first, second, image0, image1, extractor, matcher, mask, device=None,
                   source_mask=None, target_mask=None):
        if self.matcher_method == "lk_descriptor":
            return self._match_flow(first, second, image0, image1, extractor,
                int(self.features_cfg.get("supplemental_keypoints", 2048)),
                mask=np.asarray(mask, bool), device=device, source_mask=source_mask,
                target_mask=target_mask)
        return self._match(first, second, image0, image1, extractor, matcher,
                           int(self.features_cfg.get("supplemental_keypoints", 2048)),
                           mask=np.asarray(mask, bool), device=device,
                           source_mask=source_mask, target_mask=target_mask)

    def save(self, output_dir, final=False):
        """Persist tracks; streaming saves also release frames beyond the cache.

        Call once after every completed pair and with ``final=True`` after the
        last pair. Streaming archives are incremental and do not support resume:
        a new store must use a fresh output directory. The final save exports
        the legacy observations.npz/tracks.json using disk-backed arrays.
        """
        if self.streaming:
            result = self._save_streaming(output_dir, final=final)
            self._save_flow_reports(output_dir)
            return result
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
        self._save_flow_reports(output_dir)
        return summary

    def _save_flow_reports(self, output_dir):
        if self._flow_reports:
            with (Path(output_dir) / "flow_tracking.jsonl").open("a") as journal:
                for row in self._flow_reports:
                    journal.write(json.dumps(row) + "\n")
            self._flow_reports.clear()

    def _open_archive(self, output_dir):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        database = output_dir / "track_index.sqlite3"
        if self._archive_dir is not None and output_dir != self._archive_dir:
            raise ValueError("A streaming store cannot change its archive directory")
        if self._archive_dir is None:
            if any(output_dir.iterdir()):
                raise FileExistsError(
                    f"Track output already exists at {output_dir}; streaming resume is unsupported")
            with sqlite3.connect(database) as connection:
                connection.executescript("""
                    CREATE TABLE stats (name TEXT PRIMARY KEY, value INTEGER NOT NULL);
                    INSERT INTO stats VALUES ('tracks', 0), ('observations', 0),
                        ('tracks_with_three_or_more_frames', 0), ('frame_count', 0);
                    CREATE TABLE frames (
                        frame INTEGER PRIMARY KEY, sealed INTEGER NOT NULL DEFAULT 0);
                    CREATE TABLE tracks (
                        track_id INTEGER PRIMARY KEY, first_frame INTEGER NOT NULL,
                        last_frame INTEGER NOT NULL, length INTEGER NOT NULL);
                    CREATE TABLE observations (
                        track_id INTEGER NOT NULL, frame INTEGER NOT NULL,
                        keypoint_id INTEGER NOT NULL, u REAL NOT NULL, v REAL NOT NULL,
                        PRIMARY KEY (track_id, frame)) WITHOUT ROWID;
                    CREATE TRIGGER count_observation AFTER INSERT ON observations BEGIN
                        UPDATE stats SET value = value + 1 WHERE name = 'observations';
                        UPDATE stats SET value = value + 1 WHERE name = 'tracks'
                            AND NOT EXISTS (SELECT 1 FROM tracks WHERE track_id = NEW.track_id);
                        UPDATE stats SET value = value + 1
                            WHERE name = 'tracks_with_three_or_more_frames'
                            AND EXISTS (SELECT 1 FROM tracks
                                WHERE track_id = NEW.track_id AND length = 2);
                        INSERT INTO tracks VALUES (NEW.track_id, NEW.frame, NEW.frame, 1)
                            ON CONFLICT (track_id) DO UPDATE SET
                                first_frame = MIN(first_frame, NEW.frame),
                                last_frame = MAX(last_frame, NEW.frame), length = length + 1;
                    END;
                """)
            self._archive_dir = output_dir
        # Keep SQLite's page cache bounded too; all cumulative metadata lives on disk.
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA cache_size = -2048")
        connection.execute("PRAGMA temp_store = FILE")
        return connection

    @staticmethod
    def _write_features(path, record):
        fields = {"keypoints": record["uv"], "track_ids": record["track_ids"],
                  "catalog_keypoint_ids": record["catalog_ids"]}
        if record["features"] is not None:
            fields.update({key: value[0].numpy()
                           for key, value in record["features"].items()
                           if key != "keypoints"})
        np.savez_compressed(path, **fields)

    @staticmethod
    def _frame_observations(frame, record):
        ids = np.flatnonzero(record["track_ids"] >= 0)
        return {"track_ids": record["track_ids"][ids],
                "frames": np.full(len(ids), frame, np.int64),
                "keypoint_ids": ids, "uv": record["uv"][ids]}

    def _save_streaming(self, output_dir, final=False):
        connection = self._open_archive(output_dir)
        output_dir = self._archive_dir
        features_dir = output_dir / "features"
        chunks_dir = output_dir / "observation_chunks"
        features_dir.mkdir(exist_ok=True)
        chunks_dir.mkdir(exist_ok=True)
        latest = max(self.frames) if self.frames else None
        try:
            with connection:
                for frame, record in sorted(self.frames.items()):
                    if self._sealed_through is not None and frame <= self._sealed_through:
                        continue
                    sealed = final or frame != latest
                    observations = self._frame_observations(frame, record)
                    # A target frame can acquire additional 2D observations when
                    # it becomes the next source. The unique key prevents recounts.
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO frames (frame) VALUES (?)", (frame,))
                    if inserted.rowcount:
                        connection.execute(
                            "UPDATE stats SET value = value + 1 WHERE name = 'frame_count'")
                    connection.executemany(
                        "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?)",
                        ((int(track), frame, int(keypoint), float(uv[0]), float(uv[1]))
                         for track, keypoint, uv in zip(observations["track_ids"],
                             observations["keypoint_ids"], observations["uv"])))
                    self._write_features(features_dir / f"{frame:010d}.npz", record)
                    if sealed:
                        np.savez_compressed(chunks_dir / f"{frame:010d}.npz", **observations)
                        connection.execute("UPDATE frames SET sealed = 1 WHERE frame = ?", (frame,))
                    else:
                        np.savez_compressed(output_dir / "recent_observations.npz", **observations)
                sealed_row = connection.execute(
                    "SELECT MAX(frame) FROM frames WHERE sealed = 1").fetchone()
                self._sealed_through = sealed_row[0]
            self._evict_old_frames()
            summary = dict(connection.execute("SELECT name, value FROM stats"))
            first, last = connection.execute("SELECT MIN(frame), MAX(frame) FROM frames").fetchone()
            summary.update(first_frame=first, last_frame=last,
                           cached_frames=sorted(self.frames),
                           conflicting_matches=self.conflicting_matches,
                           streaming=True, finalized=bool(final or self._finalized))
            if final and not self._finalized:
                self._export_archive(connection, output_dir, summary)
                (output_dir / "recent_observations.npz").unlink(missing_ok=True)
                self._finalized = True
            (output_dir / "tracks_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            manifest = {
                "format": "feature_tracks_streaming_v1", "database": "track_index.sqlite3",
                "observation_chunks": "observation_chunks/{frame:010d}.npz",
                "features": "features/{frame:010d}.npz",
                "sealed_through": self._sealed_through,
                "recent_observations": None if self._finalized else "recent_observations.npz",
                "finalized": self._finalized, "resume_supported": False,
            }
            (output_dir / "tracks_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
            return summary
        finally:
            connection.close()

    def _evict_old_frames(self):
        retained = set(sorted(self.frames)[-self.cache_frames:])
        for frame in list(self.frames):
            if frame not in retained:
                del self.frames[frame]
        for track, history in list(self.tracks.items()):
            for frame in list(history):
                if frame not in retained:
                    del history[frame]
            if not history:
                del self.tracks[track]

    @staticmethod
    def _export_archive(connection, output_dir, summary):
        """Create legacy files without loading the sequence or a whole track."""
        with tempfile.TemporaryDirectory(prefix=".tracks_export_", dir=output_dir) as temporary:
            arrays = {}
            n_observations, n_tracks = summary["observations"], summary["tracks"]
            for key in ("track_ids", "frames", "keypoint_ids", "uv",
                        "unique_track_ids", "first_frames", "last_frames", "lengths"):
                count = n_tracks if key in (
                    "unique_track_ids", "first_frames", "last_frames", "lengths") else n_observations
                shape = (count, 2) if key == "uv" else (count,)
                arrays[key] = np.lib.format.open_memmap(
                    Path(temporary) / f"{key}.npy", mode="w+", shape=shape,
                    dtype=np.float32 if key == "uv" else np.int64)
            rows = connection.execute(
                "SELECT track_id, frame, keypoint_id, u, v FROM observations ORDER BY track_id, frame")
            start = 0
            while batch := rows.fetchmany(16384):
                stop = start + len(batch)
                for column, key in enumerate(("track_ids", "frames", "keypoint_ids")):
                    arrays[key][start:stop] = np.fromiter(
                        (row[column] for row in batch), dtype=np.int64, count=len(batch))
                arrays["uv"][start:stop] = np.asarray([row[3:] for row in batch], np.float32)
                start = stop
            rows = connection.execute(
                "SELECT track_id, first_frame, last_frame, length FROM tracks ORDER BY track_id")
            start = 0
            while batch := rows.fetchmany(16384):
                stop = start + len(batch)
                for column, key in enumerate(("unique_track_ids", "first_frames", "last_frames", "lengths")):
                    arrays[key][start:stop] = np.fromiter(
                        (row[column] for row in batch), dtype=np.int64, count=len(batch))
                start = stop
            np.savez_compressed(output_dir / "observations.npz", **arrays)
            # Release mappings before removing the temporary backing files.
            arrays.clear()
        with (output_dir / "tracks.json").open("w", encoding="utf-8") as stream:
            stream.write("[\n")
            separator = ""
            for track, first, last, length in connection.execute(
                    "SELECT track_id, first_frame, last_frame, length FROM tracks ORDER BY track_id"):
                stream.write(separator + '{"track_id": ' + str(track) + ', "frames": [')
                frame_separator = ""
                for (frame,) in connection.execute(
                        "SELECT frame FROM observations WHERE track_id = ? ORDER BY frame", (track,)):
                    stream.write(frame_separator + str(frame))
                    frame_separator = ", "
                stream.write(f'], "first_frame": {first}, "last_frame": {last}, "length": {length}}}')
                separator = ",\n"
            stream.write("\n]\n")

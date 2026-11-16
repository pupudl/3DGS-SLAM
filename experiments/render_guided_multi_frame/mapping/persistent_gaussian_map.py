"""CPU storage for a persistent, object-owned Gaussian map.

Map IDs are independent of the front end's observation Gaussian IDs. Each map
Gaussian owns one geometry/appearance vector and one Adam history. Background
coordinates are world coordinates; dynamic coordinates belong to that object's
fixed canonical frame. Coordinate conversion is the caller's responsibility.

Only map rows and object metadata persist here: image pixels, tracks and frame
observations remain in the bounded optimization window. Integrating a repeated
surface associates it with an existing row and never overwrites learned values
with a fresh RGB-D measurement. Association uses a separate immutable measured
RGB anchor, so learned appearance cannot make an existing surface look new.
"""
from __future__ import annotations

import json
import os
import copy
from pathlib import Path
import tempfile
import uuid

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


PARAMETER_WIDTHS = {
    "xyz": 3,
    "log_scales": 3,
    "rotations": 4,
    "colors": 3,
    "opacity_logits": 1,
}
_STATUSES = {"active", "missing", "archived", "retired"}
_FORMAT_VERSION = 2


def _array(values, shape, name, dtype=np.float32):
    result = np.array(values, dtype=dtype, copy=True)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _ids(values, name="map_ids"):
    raw = np.asarray(values)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    result = raw.astype(np.int64, copy=True)
    if np.any(result < 0) or len(np.unique(result)) != len(result):
        raise ValueError(f"{name} must contain unique nonnegative integers")
    return result


def _object_id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("object_id must be a nonnegative integer")
    return int(value)


def _normalized_quaternions(values, n):
    rotations = _array(values, (n, 4), "rotations")
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("rotations must contain nonzero wxyz quaternions")
    return rotations / norms


class PersistentGaussianMap:
    """Persistent CPU map blocks, with explicit ownership edits and stable IDs.

    ``cfg`` accepts the ``gaussian_map`` section or a complete experiment config.
    ``association_distance`` and ``color_distance`` are Euclidean thresholds in
    canonical metres and RGB [0,1], respectively. ``voxel_size`` accelerates
    same-batch deduplication; it does not quantize stored Gaussian positions.
    """

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg
        section = cfg.get("gaussian_map", cfg)
        self.cfg = dict(section)
        self.association_distance = float(section.get("association_distance", .06))
        self.color_distance = float(section.get("color_distance", .25))
        self.voxel_size = float(section.get("voxel_size", self.association_distance))
        self.initial_scale = float(section.get("initial_scale", .02))
        self.initial_opacity = float(section.get("initial_opacity", .5))
        for name in ("association_distance", "voxel_size", "initial_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.color_distance) or self.color_distance < 0:
            raise ValueError("color_distance must be finite and nonnegative")
        if not np.isfinite(self.initial_opacity) or not 0 < self.initial_opacity < 1:
            raise ValueError("initial_opacity must be strictly between zero and one")
        self._blocks = {}
        self._metadata = {}
        self._trees = {}
        self._dirty = set()
        # These compact integer arrays route globally contiguous IDs to blocks;
        # there is deliberately no historical observation/pixel lookup table.
        self._owners = np.empty(0, np.int64)
        self._rows = np.empty(0, np.int64)
        self._next_id = 0
        self._live_count = 0
        self._ownership_events = []
        self._topology_events = []
        self._revision = 0
        self._last_save_directory = None
        self._saved_files = {}

    def __len__(self):
        """Number of live rows; ``next_id`` includes permanently pruned IDs."""
        return self._live_count

    @property
    def object_ids(self):
        return sorted(self._blocks)

    @property
    def next_id(self):
        return self._next_id

    @property
    def ownership_events(self):
        return copy.deepcopy(self._ownership_events)

    @property
    def topology_events(self):
        return copy.deepcopy(self._topology_events)

    @property
    def revision(self):
        return self._revision

    @property
    def next_revision(self):
        return self._revision + 1

    def is_live(self, map_ids):
        """Return a boolean mask, accepting repeated, unknown and pruned IDs."""
        values = np.asarray(map_ids)
        if values.ndim != 1 or (values.size and not np.issubdtype(values.dtype, np.integer)):
            raise ValueError("map_ids must be a one-dimensional integer array")
        valid = (values >= 0) & (values < self._next_id)
        result = np.zeros(values.shape, bool)
        result[valid] = self._owners[values[valid].astype(np.int64)] >= 0
        return result

    @staticmethod
    def _empty_block():
        block = {name: np.empty((0, width), np.float32)
                 for name, width in PARAMETER_WIDTHS.items()}
        block.update(map_ids=np.empty(0, np.int64),
                     initial_xyz=np.empty((0, 3), np.float32),
                     initial_log_scales=np.empty((0, 3), np.float32),
                     association_colors=np.empty((0, 3), np.float32),
                     observation_count=np.empty(0, np.int64),
                     last_seen_frame=np.empty(0, np.int64))
        block["optimizer_state"] = {
            name: {"exp_avg": np.empty((0, width), np.float32),
                   "exp_avg_sq": np.empty((0, width), np.float32),
                   "row_steps": np.empty((0, 1), np.int64)}
            for name, width in PARAMETER_WIDTHS.items()}
        return block

    def _ensure_object(self, object_id):
        if object_id not in self._blocks:
            self._blocks[object_id] = self._empty_block()
            self._metadata[object_id] = dict(status="active", first_frame=None,
                                             last_frame=None)
            self._dirty.add(object_id)
        return self._blocks[object_id]

    def _resolve(self, map_ids):
        ids = _ids(map_ids)
        if np.any(ids >= self._next_id):
            raise KeyError("map_ids contains an unknown map ID")
        if np.any(self._owners[ids] < 0):
            raise KeyError("map_ids contains a pruned map ID")
        return ids, self._owners[ids], self._rows[ids]

    def integrate(self, object_id, xyz, colors, scales=None, rotations=None,
                  opacity=None, frame_id=None, existing_map_ids=None, max_new=None):
        """Associate canonical RGB-D samples, inserting previously unseen rows.

        Returns ``map_ids`` aligned with the input samples, and ``inserted`` /
        ``matched`` counts. Several samples can return the same map ID. The
        optional ``existing_map_ids`` is an equally sized integer vector with
        -1 for unknown entries; hints are accepted only after owner, geometry
        and color checks. Incorrect hints fall back to ordinary association.
        ``max_new`` bounds inserted rows without restricting association. Rows
        that cannot be associated after the budget is exhausted return ID -1
        and contribute to ``skipped``; they do not create persistent state.
        """
        oid = _object_id(object_id)
        raw_xyz = np.asarray(xyz)
        if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3:
            raise ValueError("xyz must have shape [N,3]")
        n = len(raw_xyz)
        if max_new is not None and (isinstance(max_new, (bool, np.bool_)) or
                                    not isinstance(max_new, (int, np.integer)) or max_new < 0):
            raise ValueError("max_new must be a nonnegative integer or None")
        points = _array(raw_xyz, (n, 3), "xyz")
        rgb = _array(colors, (n, 3), "colors")
        if np.any((rgb < 0) | (rgb > 1)):
            raise ValueError("colors must be RGB in [0,1]")
        if frame_id is not None:
            if (isinstance(frame_id, (bool, np.bool_)) or
                    not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
                raise ValueError("frame_id must be a nonnegative integer or None")
            frame_id = int(frame_id)
        if scales is None:
            sizes = np.full((n, 3), self.initial_scale, np.float32)
        else:
            sizes = np.asarray(scales, dtype=np.float32)
            if sizes.ndim == 0:
                sizes = np.full((n, 3), float(sizes), np.float32)
            elif sizes.shape == (n,):
                sizes = np.repeat(sizes[:, None], 3, axis=1)
            sizes = _array(sizes, (n, 3), "scales")
        if np.any(sizes <= 0):
            raise ValueError("scales must be positive")
        if rotations is None:
            quaternions = np.zeros((n, 4), np.float32)
            quaternions[:, 0] = 1
        else:
            quaternions = _normalized_quaternions(rotations, n)
        if opacity is None:
            alpha = np.full((n, 1), self.initial_opacity, np.float32)
        else:
            alpha = np.asarray(opacity, dtype=np.float32)
            if alpha.ndim == 0:
                alpha = np.full((n, 1), float(alpha), np.float32)
            elif alpha.shape == (n,):
                alpha = alpha[:, None]
            alpha = _array(alpha, (n, 1), "opacity")
        if np.any((alpha <= 0) | (alpha >= 1)):
            raise ValueError("opacity must be strictly between zero and one")
        if existing_map_ids is None:
            hints = np.full(n, -1, np.int64)
        else:
            raw_hints = np.asarray(existing_map_ids)
            if (raw_hints.shape != (n,) or
                    not np.issubdtype(raw_hints.dtype, np.integer) or
                    np.any(raw_hints < -1)):
                raise ValueError("existing_map_ids must be integer [N], with -1 for unknown")
            hints = raw_hints.astype(np.int64, copy=True)
        if n == 0:
            return dict(map_ids=np.empty(0, np.int64), inserted=0, matched=0, skipped=0)

        block = self._ensure_object(oid)
        assigned = np.full(n, -1, np.int64)
        valid_hint = (hints >= 0) & (hints < self._next_id)
        candidate_samples = np.flatnonzero(valid_hint)
        if len(candidate_samples):
            candidate_ids = hints[candidate_samples]
            same_owner = self._owners[candidate_ids] == oid
            candidate_samples = candidate_samples[same_owner]
            candidate_rows = self._rows[hints[candidate_samples]]
            distance = np.linalg.norm(points[candidate_samples] - block["xyz"][candidate_rows], axis=1)
            color = np.linalg.norm(rgb[candidate_samples] - block["association_colors"][candidate_rows], axis=1)
            accepted = (distance <= self.association_distance) & (color <= self.color_distance)
            assigned[candidate_samples[accepted]] = hints[candidate_samples[accepted]]

        remaining = np.flatnonzero(assigned < 0)
        if len(block["map_ids"]) and len(remaining):
            tree = self._trees.get(oid)
            if tree is None:
                tree = cKDTree(block["xyz"])
                self._trees[oid] = tree
            # Bounded chunks avoid materializing every radius neighbourhood of
            # an entire image at once for dense surfaces.
            for start in range(0, len(remaining), 2048):
                samples = remaining[start:start + 2048]
                neighborhoods = tree.query_ball_point(points[samples], self.association_distance)
                for sample, candidates in zip(samples, neighborhoods):
                    if not candidates:
                        continue
                    candidates = np.asarray(candidates, np.int64)
                    color = np.linalg.norm(block["association_colors"][candidates] - rgb[sample], axis=1)
                    candidates = candidates[color <= self.color_distance]
                    if len(candidates):
                        distances = np.sum((block["xyz"][candidates] - points[sample]) ** 2, axis=1)
                        order = np.lexsort((block["map_ids"][candidates], distances))
                        assigned[sample] = block["map_ids"][candidates[order[0]]]

        new_samples = []
        voxels = {}
        radius = int(np.ceil(self.association_distance / self.voxel_size))
        offsets = [(x, y, z) for x in range(-radius, radius + 1)
                   for y in range(-radius, radius + 1)
                   for z in range(-radius, radius + 1)]
        radius_squared = self.association_distance ** 2
        color_squared = self.color_distance ** 2
        for sample in np.flatnonzero(assigned < 0):
            key = tuple(np.floor(points[sample] / self.voxel_size).astype(np.int64))
            best = None
            best_distance = np.inf
            for offset in offsets:
                for old_sample in voxels.get(tuple(a + b for a, b in zip(key, offset)), ()):
                    distance = float(np.sum((points[old_sample] - points[sample]) ** 2))
                    if (distance <= radius_squared and distance < best_distance and
                            np.sum((rgb[old_sample] - rgb[sample]) ** 2) <= color_squared):
                        best, best_distance = old_sample, distance
            if best is None:
                if max_new is not None and len(new_samples) >= max_new:
                    continue
                assigned[sample] = self._next_id + len(new_samples)
                new_samples.append(int(sample))
                voxels.setdefault(key, []).append(int(sample))
            else:
                assigned[sample] = assigned[best]

        inserted = len(new_samples)
        if inserted:
            indices = np.asarray(new_samples, np.int64)
            old_count = len(block["map_ids"])
            new_ids = np.arange(self._next_id, self._next_id + inserted, dtype=np.int64)
            new_values = dict(xyz=points[indices], log_scales=np.log(sizes[indices]),
                              rotations=quaternions[indices], colors=rgb[indices],
                              opacity_logits=np.log(alpha[indices] / (1 - alpha[indices])))
            for name, values in new_values.items():
                block[name] = np.concatenate((block[name], values), axis=0)
            for name, values in (("initial_xyz", new_values["xyz"]),
                                 ("initial_log_scales", new_values["log_scales"]),
                                 ("association_colors", new_values["colors"])):
                block[name] = np.concatenate((block[name], values.copy()), axis=0)
            block["map_ids"] = np.concatenate((block["map_ids"], new_ids))
            block["observation_count"] = np.concatenate((block["observation_count"], np.zeros(inserted, np.int64)))
            block["last_seen_frame"] = np.concatenate((block["last_seen_frame"], np.full(inserted, -1, np.int64)))
            for state in block["optimizer_state"].values():
                for key, values in state.items():
                    state[key] = np.concatenate((values, np.zeros((inserted, values.shape[1]), values.dtype)))
            self._owners = np.concatenate((self._owners, np.full(inserted, oid, np.int64)))
            self._rows = np.concatenate((self._rows, np.arange(old_count, old_count + inserted, dtype=np.int64)))
            self._next_id += inserted
            self._live_count += inserted
            self._trees.pop(oid, None)
        # Count supporting samples, without retaining their pixel identities.
        observed_ids, counts = np.unique(assigned[assigned >= 0], return_counts=True)
        rows = self._rows[observed_ids]
        block["observation_count"][rows] += counts
        if frame_id is not None:
            block["last_seen_frame"][rows] = np.maximum(block["last_seen_frame"][rows], frame_id)
            metadata = self._metadata[oid]
            metadata["first_frame"] = frame_id if metadata["first_frame"] is None else min(metadata["first_frame"], frame_id)
            metadata["last_frame"] = frame_id if metadata["last_frame"] is None else max(metadata["last_frame"], frame_id)
        self._dirty.add(oid)
        skipped = int(np.count_nonzero(assigned < 0))
        return dict(map_ids=assigned, inserted=inserted, matched=n - inserted - skipped, skipped=skipped)

    def select(self, map_ids, include_optimizer=True):
        """Copy unique map rows in requested order, including their Adam state."""
        ids, owners, rows = self._resolve(map_ids)
        n = len(ids)
        result = {name: np.empty((n, width), np.float32) for name, width in PARAMETER_WIDTHS.items()}
        result.update(map_ids=ids, object_ids=owners.copy(),
                      initial_xyz=np.empty((n, 3), np.float32),
                      initial_log_scales=np.empty((n, 3), np.float32),
                      association_colors=np.empty((n, 3), np.float32),
                      observation_count=np.empty(n, np.int64),
                      last_seen_frame=np.empty(n, np.int64))
        moments = ({name: {"exp_avg": np.empty((n, width), np.float32),
                           "exp_avg_sq": np.empty((n, width), np.float32),
                           "row_steps": np.empty((n, 1), np.int64)}
                    for name, width in PARAMETER_WIDTHS.items()} if include_optimizer else {})
        for oid in np.unique(owners):
            positions = np.flatnonzero(owners == oid)
            selected_rows = rows[positions]
            block = self._blocks[int(oid)]
            for name in (*PARAMETER_WIDTHS, "initial_xyz", "initial_log_scales",
                         "association_colors", "observation_count", "last_seen_frame"):
                result[name][positions] = block[name][selected_rows]
            for name, state in moments.items():
                for key in state:
                    state[key][positions] = block["optimizer_state"][name][key][selected_rows]
        if include_optimizer:
            result["optimizer_state"] = dict(map_ids=ids.copy(), params=moments)
        return result

    def record_observations(self, map_ids, frame_id):
        """Record trusted shared-ID observations without reassociating geometry.

        Unlike ``integrate`` hints, these IDs must already be established by
        the caller's observation identity layer. Repeated IDs count repeated
        supporting samples. Validation finishes before any state changes;
        learned parameters, association anchors and Adam remain untouched.
        This changes support metadata only and does not advance map revision.
        """
        raw = np.asarray(map_ids)
        if (raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)) or
                np.any(raw < 0)):
            raise ValueError("map_ids must be a one-dimensional nonnegative integer array")
        if (isinstance(frame_id, (bool, np.bool_)) or
                not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
            raise ValueError("frame_id must be a nonnegative integer")
        frame_id = int(frame_id)
        ids, counts = np.unique(raw.astype(np.int64, copy=True), return_counts=True)
        ids, owners, rows = self._resolve(ids)
        for oid in np.unique(owners):
            positions = np.flatnonzero(owners == oid)
            selected_rows = rows[positions]
            block = self._blocks[int(oid)]
            block["observation_count"][selected_rows] += counts[positions]
            block["last_seen_frame"][selected_rows] = np.maximum(
                block["last_seen_frame"][selected_rows], frame_id)
            metadata = self._metadata[int(oid)]
            metadata["first_frame"] = frame_id if metadata["first_frame"] is None else min(metadata["first_frame"], frame_id)
            metadata["last_frame"] = frame_id if metadata["last_frame"] is None else max(metadata["last_frame"], frame_id)
            self._dirty.add(int(oid))
        return dict(observations=len(raw), gaussians=len(ids))

    def commit(self, map_ids, parameters, optimizer_state=None):
        """Commit optimized rows atomically, leaving initial priors untouched.

        Ownership/IDs cannot be changed. Partial parameter dictionaries are
        accepted. RGB is clamped to [0,1], wxyz quaternions are normalized, and
        optimizer moments are reordered by their own stable ``map_ids``.
        """
        ids, owners, rows = self._resolve(map_ids)
        n = len(ids)
        unknown = set(parameters) - set(PARAMETER_WIDTHS)
        if unknown:
            raise ValueError(f"unknown or immutable parameters: {sorted(unknown)}")
        values = {}
        for name, raw in parameters.items():
            values[name] = (_normalized_quaternions(raw, n) if name == "rotations" else
                            _array(raw, (n, PARAMETER_WIDTHS[name]), name))
            if name == "colors":
                values[name] = np.clip(values[name], 0, 1)
        moment_values = {}
        if optimizer_state is not None:
            state_ids = _ids(optimizer_state.get("map_ids", ids), "optimizer_state.map_ids")
            if len(state_ids) != n or set(state_ids.tolist()) != set(ids.tolist()):
                raise ValueError("optimizer_state.map_ids must match committed map_ids")
            order = np.argsort(state_ids)
            reorder = order[np.searchsorted(state_ids[order], ids)]
            states = optimizer_state.get("params", {})
            if set(states) - set(PARAMETER_WIDTHS):
                raise ValueError("optimizer_state has an unknown parameter")
            for name, state in states.items():
                width = PARAMETER_WIDTHS[name]
                avg = _array(state["exp_avg"], (n, width), f"{name}.exp_avg")[reorder]
                square = _array(state["exp_avg_sq"], (n, width), f"{name}.exp_avg_sq")[reorder]
                raw_steps = np.asarray(state["row_steps"])
                if (raw_steps.shape != (n, 1) or not np.issubdtype(raw_steps.dtype, np.integer) or
                        np.any(raw_steps < 0) or np.any(square < 0)):
                    raise ValueError("Adam squared moments and integer row_steps must be nonnegative")
                moment_values[name] = dict(exp_avg=avg, exp_avg_sq=square,
                                          row_steps=raw_steps.astype(np.int64, copy=True)[reorder])
        for oid in np.unique(owners):
            positions = np.flatnonzero(owners == oid)
            selected_rows = rows[positions]
            block = self._blocks[int(oid)]
            for name, data in values.items():
                block[name][selected_rows] = data[positions]
            for name, state in moment_values.items():
                for key, data in state.items():
                    block["optimizer_state"][name][key][selected_rows] = data[positions]
            self._dirty.add(int(oid))
            if "xyz" in values:
                self._trees.pop(int(oid), None)

    def import_rows(self, batch, object_id):
        """Copy selected rows into this map under freshly allocated local IDs.

        This is an exact boundary seed import, rather than surface association:
        nearby rows remain independent, measured appearance/initial priors and
        Adam histories persist, and the caller's arrays are never modified.
        The batch must contain rows from one canonical owner. Coordinate frames
        must already agree between the source and destination maps.
        """
        oid = _object_id(object_id)
        old_ids = _ids(batch['map_ids'])
        n = len(old_ids)
        owners = np.asarray(batch['object_ids'])
        if (owners.shape != (n,) or not np.issubdtype(owners.dtype, np.integer) or
                np.any(owners != oid)):
            raise ValueError('Imported rows must belong to the requested canonical owner')
        selected = {name: _array(batch[name], (n, width), name)
                    for name, width in PARAMETER_WIDTHS.items()}
        norms = np.linalg.norm(selected['rotations'], axis=1)
        if np.any(norms < 1e-12) or not np.allclose(norms, 1., atol=1e-5):
            raise ValueError('Imported rotations must already be normalized wxyz quaternions')
        selected['initial_xyz'] = _array(batch['initial_xyz'], (n, 3), 'initial_xyz')
        selected['initial_log_scales'] = _array(batch['initial_log_scales'], (n, 3), 'initial_log_scales')
        selected['association_colors'] = _array(batch['association_colors'], (n, 3), 'association_colors')
        for key in ('colors', 'association_colors'):
            if np.any((selected[key] < 0) | (selected[key] > 1)):
                raise ValueError(f'{key} must be RGB in [0,1]')
        for key in ('observation_count', 'last_seen_frame'):
            raw = np.asarray(batch[key])
            lower = -1 if key == 'last_seen_frame' else 0
            if (raw.shape != (n,) or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < lower)):
                raise ValueError(f'Invalid imported {key}')
            selected[key] = raw.astype(np.int64, copy=True)
        state = batch['optimizer_state']
        state_ids = _ids(state['map_ids'], 'optimizer_state.map_ids')
        if not np.array_equal(state_ids, old_ids) or set(state['params']) != set(PARAMETER_WIDTHS):
            raise ValueError('Imported optimizer rows and parameters must match map rows')
        moments = {}
        for name, width in PARAMETER_WIDTHS.items():
            source = state['params'][name]
            avg = _array(source['exp_avg'], (n, width), f'{name}.exp_avg')
            square = _array(source['exp_avg_sq'], (n, width), f'{name}.exp_avg_sq')
            steps = np.asarray(source['row_steps'])
            if (np.any(square < 0) or steps.shape != (n, 1) or
                    not np.issubdtype(steps.dtype, np.integer) or np.any(steps < 0)):
                raise ValueError('Imported Adam squared moments and integer steps must be nonnegative')
            moments[name] = dict(exp_avg=avg, exp_avg_sq=square, row_steps=steps.astype(np.int64, copy=True))
        new_ids = np.arange(self._next_id, self._next_id+n, dtype=np.int64)
        selected['map_ids'] = new_ids
        selected['optimizer_state'] = dict(map_ids=new_ids.copy(), params=moments)
        if n:
            self._owners = np.r_[self._owners, np.full(n, -1, np.int64)]
            self._rows = np.r_[self._rows, np.full(n, -1, np.int64)]
            self._append_rows(oid, selected)
            self._next_id += n
            self._live_count += n
        return new_ids.copy()

    @staticmethod
    def _event_arguments(frame_id, reason):
        if frame_id is not None and (isinstance(frame_id, (bool, np.bool_)) or
                                     not isinstance(frame_id, (int, np.integer)) or frame_id < 0):
            raise ValueError("frame_id must be a nonnegative integer or None")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a nonempty string")
        return None if frame_id is None else int(frame_id)

    def _event(self, kind, frame_id, reason, **values):
        self._revision += 1
        return dict(kind=kind, revision=self._revision, frame_id=frame_id,
                    reason=reason, **values)

    def _remove_rows(self, ids):
        """Remove resolved rows and refresh routes; caller owns count/journals."""
        owners = self._owners[ids].copy()
        rows = self._rows[ids].copy()
        for oid in np.unique(owners):
            block = self._blocks[int(oid)]
            keep = np.ones(len(block["map_ids"]), bool)
            keep[rows[owners == oid]] = False
            for name in block:
                if name == "optimizer_state":
                    for state in block[name].values():
                        for key in state:
                            state[key] = state[key][keep].copy()
                else:
                    block[name] = block[name][keep].copy()
            self._rows[block["map_ids"]] = np.arange(len(block["map_ids"]), dtype=np.int64)
            self._trees.pop(int(oid), None)
            self._dirty.add(int(oid))
        self._owners[ids] = -1
        self._rows[ids] = -1

    def _append_rows(self, oid, batch):
        """Append a selected-style batch with already allocated stable IDs."""
        block = self._ensure_object(oid)
        ids = batch["map_ids"]
        start = len(block["map_ids"])
        for name in block:
            if name == "optimizer_state":
                for parameter, state in block[name].items():
                    for key in state:
                        state[key] = np.concatenate((state[key], batch[name]["params"][parameter][key]), axis=0)
            else:
                block[name] = np.concatenate((block[name], batch[name]), axis=0)
        self._owners[ids] = oid
        self._rows[ids] = np.arange(start, start + len(ids), dtype=np.int64)
        self._trees.pop(oid, None)
        self._dirty.add(oid)

    @staticmethod
    def _subset(batch, positions):
        selected = {name: value[positions].copy() for name, value in batch.items()
                    if name != "optimizer_state"}
        selected["optimizer_state"] = dict(map_ids=selected["map_ids"].copy(), params={
            name: {key: value[positions].copy() for key, value in state.items()}
            for name, state in batch["optimizer_state"]["params"].items()})
        return selected

    def reassign(self, map_ids, new_object_id, transform, frame_id=None,
                 reason="ownership_correction"):
        """Move live rows between canonical frames without changing map IDs.

        ``transform`` is rigid old-canonical -> new-canonical, [4,4] or [N,4,4].
        Together with compatible object poses it preserves the observed world
        centre and covariance. Learned/initial centres and local quaternions
        transform; RGB, opacity and scale axes stay unchanged. XYZ/quaternion
        Adam histories reset because their coordinate systems changed. Other
        moments persist. Existing rows already owned by the target are no-ops.
        An append-only revisioned event records every effective edit.
        """
        ids, owners, _ = self._resolve(map_ids)
        oid = _object_id(new_object_id)
        frame_id = self._event_arguments(frame_id, reason)
        n = len(ids)
        raw = np.asarray(transform)
        if raw.shape == (4, 4):
            transforms = np.broadcast_to(_array(raw, (4, 4), "transform", dtype=np.float64), (n, 4, 4)).copy()
        else:
            transforms = _array(raw, (n, 4, 4), "transform", dtype=np.float64)
        if not np.all(np.isfinite(transforms)):
            raise ValueError("transform must be finite")
        rotation = transforms[:, :3, :3]
        if (not np.allclose(transforms[:, 3, :], [0., 0., 0., 1.], atol=1e-6) or
                not np.allclose(rotation @ np.swapaxes(rotation, 1, 2), np.eye(3), atol=1e-5) or
                not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)):
            raise ValueError("transform must contain proper rigid transforms")
        changed = owners != oid
        if not np.any(changed):
            return dict(kind="reassign", revision=self._revision, frame_id=frame_id,
                        reason=reason, map_ids=[], old_object_ids=[], new_object_id=oid,
                        reset_optimizer_parameters=["xyz", "rotations"])
        ids, owners, transforms = ids[changed], owners[changed], transforms[changed]
        batch = self.select(ids)
        rotation, translation = transforms[:, :3, :3], transforms[:, :3, 3]
        for name in ("xyz", "initial_xyz"):
            batch[name] = _array(np.einsum("nij,nj->ni", rotation, batch[name]) + translation,
                                 (len(ids), 3), name)
        old_quaternion = batch["rotations"]
        old_matrices = Rotation.from_quat(old_quaternion[:, [1, 2, 3, 0]]).as_matrix()
        new_quaternion = Rotation.from_matrix(rotation @ old_matrices).as_quat()[:, [3, 0, 1, 2]]
        batch["rotations"] = new_quaternion.astype(np.float32)
        for parameter in ("xyz", "rotations"):
            for values in batch["optimizer_state"]["params"][parameter].values():
                values.fill(0)
        self._remove_rows(ids)
        self._append_rows(oid, batch)
        metadata = self._metadata[oid]
        seen = batch["last_seen_frame"]
        first_observation = frame_id if frame_id is not None else (int(seen[seen >= 0].min()) if np.any(seen >= 0) else None)
        first = [value for value in [first_observation, metadata["first_frame"]] if value is not None]
        if first:
            metadata["first_frame"] = min(first)
        last = [int(value) for value in seen if value >= 0]
        if frame_id is not None:
            last.append(frame_id)
        if metadata["last_frame"] is not None:
            last.append(metadata["last_frame"])
        if last:
            metadata["last_frame"] = max(last)
        event = self._event("reassign", frame_id, reason, map_ids=ids.tolist(),
                            old_object_ids=owners.tolist(), new_object_id=oid,
                            transforms=transforms.tolist(),
                            reset_optimizer_parameters=["xyz", "rotations"])
        self._ownership_events.append(event)
        return copy.deepcopy(event)

    def append_from_parent(self, parent_ids, parameters=None, frame_id=None, reason="clone"):
        """Allocate children of live parents, permitting repeated parent IDs.

        Children inherit ownership, measured RGB association anchors and support
        metadata. Optional Gaussian parameters override parent values. Child
        geometry priors start at the resulting geometry and all Adam state is
        zero. Append before pruning split parents; IDs are never reused.
        Returns new stable IDs in parent input order.
        """
        raw_ids = np.asarray(parent_ids)
        if (raw_ids.ndim != 1 or (raw_ids.size and not np.issubdtype(raw_ids.dtype, np.integer)) or
                np.any(raw_ids < 0)):
            raise ValueError("parent_ids must be a one-dimensional nonnegative integer array")
        parents = raw_ids.astype(np.int64, copy=True)
        unique, inverse = np.unique(parents, return_inverse=True)
        self._resolve(unique)
        frame_id = self._event_arguments(frame_id, reason)
        n = len(parents)
        parameters = {} if parameters is None else parameters
        unknown = set(parameters) - set(PARAMETER_WIDTHS)
        if unknown:
            raise ValueError(f"unknown or immutable parameters: {sorted(unknown)}")
        values = {}
        for name, raw in parameters.items():
            values[name] = (_normalized_quaternions(raw, n) if name == "rotations" else
                            _array(raw, (n, PARAMETER_WIDTHS[name]), name))
            if name == "colors":
                values[name] = np.clip(values[name], 0., 1.)
        if n == 0:
            return np.empty(0, np.int64)
        batch = self._subset(self.select(unique), inverse)
        batch.update(values)
        batch["initial_xyz"] = batch["xyz"].copy()
        batch["initial_log_scales"] = batch["log_scales"].copy()
        for state in batch["optimizer_state"]["params"].values():
            for value in state.values():
                value.fill(0)
        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        batch["map_ids"] = new_ids
        batch["optimizer_state"]["map_ids"] = new_ids.copy()
        owners = batch["object_ids"]
        self._owners = np.concatenate((self._owners, np.full(n, -1, np.int64)))
        self._rows = np.concatenate((self._rows, np.full(n, -1, np.int64)))
        self._next_id += n
        self._live_count += n
        for oid in np.unique(owners):
            self._append_rows(int(oid), self._subset(batch, np.flatnonzero(owners == oid)))
        self._topology_events.append(self._event("append", frame_id, reason,
            parent_ids=parents.tolist(), map_ids=new_ids.tolist(), object_ids=owners.tolist()))
        return new_ids.copy()

    def remove(self, map_ids, frame_id=None, reason="prune"):
        """Physically remove live rows, retaining ID tombstones and a journal."""
        ids, owners, _ = self._resolve(map_ids)
        frame_id = self._event_arguments(frame_id, reason)
        if not len(ids):
            return dict(kind="remove", revision=self._revision, frame_id=frame_id,
                        reason=reason, map_ids=[], old_object_ids=[])
        self._remove_rows(ids)
        self._live_count -= len(ids)
        event = self._event("remove", frame_id, reason, map_ids=ids.tolist(),
                            old_object_ids=owners.tolist())
        self._topology_events.append(event)
        return copy.deepcopy(event)

    def export_object(self, object_id, include_optimizer=False):
        oid = _object_id(object_id)
        if oid not in self._blocks:
            raise KeyError(f"unknown object {oid}")
        result = self.select(self._blocks[oid]["map_ids"], include_optimizer=include_optimizer)
        result["metadata"] = dict(self._metadata[oid])
        return result

    def ids_for_object(self, object_id):
        return self._blocks[_object_id(object_id)]["map_ids"].copy()

    def set_status(self, object_id, status, last_seen_frame=None):
        oid = _object_id(object_id)
        if status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}")
        if last_seen_frame is not None:
            if (isinstance(last_seen_frame, (bool, np.bool_)) or
                    not isinstance(last_seen_frame, (int, np.integer)) or last_seen_frame < 0):
                raise ValueError("last_seen_frame must be a nonnegative integer or None")
        self._ensure_object(oid)
        if last_seen_frame is not None:
            previous = self._metadata[oid]["last_frame"]
            self._metadata[oid]["last_frame"] = int(last_seen_frame) if previous is None else max(previous, int(last_seen_frame))
        self._metadata[oid]["status"] = status

    def get_status(self, object_id):
        return self._metadata[_object_id(object_id)]["status"]

    def to_dict(self):
        """Small JSON-compatible metadata; it does not duplicate map arrays."""
        return dict(format_version=_FORMAT_VERSION, gaussian_count=len(self), next_id=self._next_id,
                    live_count=len(self), allocated_count=self._next_id,
                    pruned_count=self._next_id - len(self), revision=self.revision,
                    next_revision=self.next_revision,
                    ownership_event_count=len(self._ownership_events),
                    topology_event_count=len(self._topology_events),
                    object_count=len(self._blocks), objects={
                        str(oid): dict(self._metadata[oid], gaussian_count=len(self._blocks[oid]["map_ids"]))
                        for oid in self.object_ids})

    summary = to_dict

    @staticmethod
    def _atomic_npz(path, values):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **values)
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def save(self, directory):
        """Write changed object blocks and atomically replace the manifest.

        Archived blocks remain on CPU/disk and are available for historical
        rendering. GPU activity is managed by the optimization-window adapter.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        same_directory = self._last_save_directory == directory.resolve()
        objects = {}
        new_files = {}
        for oid in self.object_ids:
            filename = self._saved_files.get(oid)
            block = self._blocks[oid]
            if (not same_directory or oid in self._dirty or filename is None or
                    not (directory / filename).exists()):
                # New generations keep the old manifest readable even if a
                # process stops halfway through writing several object blocks.
                filename = f"object_{oid:06d}_{uuid.uuid4().hex}.npz"
                arrays = {name: value for name, value in block.items() if name != "optimizer_state"}
                for name, state in block["optimizer_state"].items():
                    for key, values in state.items():
                        arrays[f"adam_{name}_{key}"] = values
                self._atomic_npz(directory / filename, arrays)
            new_files[oid] = filename
            objects[str(oid)] = dict(self._metadata[oid], filename=filename,
                                     gaussian_count=len(block["map_ids"]))
        settings = {name: getattr(self, name) for name in
                    ("association_distance", "color_distance", "voxel_size", "initial_scale", "initial_opacity")}
        manifest = dict(format_version=_FORMAT_VERSION, next_id=self._next_id,
                        live_count=len(self), allocated_count=self._next_id,
                        pruned_ids=np.flatnonzero(self._owners < 0).tolist(),
                        revision=self.revision, next_revision=self.next_revision,
                        ownership_events=self._ownership_events,
                        topology_events=self._topology_events,
                        settings=settings, objects=objects)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=".manifest.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, directory / "manifest.json")
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        self._last_save_directory = directory.resolve()
        if same_directory:
            for oid, filename in self._saved_files.items():
                if filename != new_files.get(oid):
                    (directory / filename).unlink(missing_ok=True)
        self._saved_files = new_files
        self._dirty.clear()
        return directory / "manifest.json"

    @staticmethod
    def validate_journal(manifest, owners):
        """Validate event provenance using only a compact final owner array.

        Returns copied ownership/topology journals and the final revision. This
        is shared with the read-only auditor, which never loads the full map.
        """
        next_id = len(owners)
        ownership_events = copy.deepcopy(manifest.get("ownership_events", []))
        topology_events = copy.deepcopy(manifest.get("topology_events", []))
        events = ownership_events + topology_events
        revision = manifest.get("revision", 0)
        if (not isinstance(revision, int) or isinstance(revision, bool) or revision < 0 or
                sorted(event.get("revision", -1) for event in events) != list(range(1, revision + 1)) or
                manifest.get("next_revision", revision + 1) != revision + 1):
            raise ValueError("invalid map event revisions")
        for event in events:
            ids = _ids(event.get("map_ids", []), "event.map_ids")
            if np.any(ids >= next_id):
                raise ValueError("event contains an unknown map ID")
            PersistentGaussianMap._event_arguments(event.get("frame_id"), event.get("reason"))
            kind = event.get("kind")
            if kind == "reassign":
                if event not in ownership_events:
                    raise ValueError("reassignment event appears in topology journal")
                _object_id(event.get("new_object_id"))
                old = event.get("old_object_ids", [])
                if len(old) != len(ids) or np.asarray(event.get("transforms")).shape != (len(ids), 4, 4):
                    raise ValueError("invalid reassignment event")
                for oid in old:
                    _object_id(oid)
            elif kind in ("append", "remove"):
                if event not in topology_events:
                    raise ValueError("topology event appears in ownership journal")
                key = "object_ids" if kind == "append" else "old_object_ids"
                if len(event.get(key, [])) != len(ids):
                    raise ValueError("invalid topology ownership metadata")
                for oid in event[key]:
                    _object_id(oid)
                if kind == "append":
                    parents = np.asarray(event.get("parent_ids", []))
                    if (parents.shape != (len(ids),) or not np.issubdtype(parents.dtype, np.integer) or
                            np.any(parents < 0) or np.any(parents >= next_id)):
                        raise ValueError("invalid topology parent IDs")
            else:
                raise ValueError("unknown map event kind")
        # Reverse the journal from the saved routing table. This verifies that
        # every reassignment, append and prune is consistent with later edits,
        # while original RGB-D insertions need no per-row event history.
        historical_owners = np.asarray(owners, np.int64).copy()
        for event in sorted(events, key=lambda value: value["revision"], reverse=True):
            ids = np.asarray(event["map_ids"], np.int64)
            kind = event["kind"]
            if kind == "reassign":
                if np.any(historical_owners[ids] != event["new_object_id"]):
                    raise ValueError("ownership journal disagrees with saved ownership")
                transforms = np.asarray(event["transforms"], np.float64)
                rotation = transforms[:, :3, :3]
                if (not np.all(np.isfinite(transforms)) or
                        not np.allclose(transforms[:, 3], [0., 0., 0., 1.], atol=1e-6) or
                        not np.allclose(rotation @ np.swapaxes(rotation, 1, 2), np.eye(3), atol=1e-5) or
                        not np.allclose(np.linalg.det(rotation), 1., atol=1e-5)):
                    raise ValueError("ownership journal contains a nonrigid transform")
                historical_owners[ids] = event["old_object_ids"]
            elif kind == "remove":
                if np.any(historical_owners[ids] != -1):
                    raise ValueError("prune journal disagrees with saved topology")
                historical_owners[ids] = event["old_object_ids"]
            else:
                parents = np.asarray(event["parent_ids"], np.int64)
                if (np.any(historical_owners[ids] != event["object_ids"]) or
                        np.any(historical_owners[parents] != event["object_ids"]) or np.any(parents >= ids)):
                    raise ValueError("append journal disagrees with parent ownership")
                historical_owners[ids] = -2  # child not allocated yet at this revision
        if np.any(historical_owners == -1):
            raise ValueError("pruned map IDs have no matching removal event")
        return ownership_events, topology_events, revision

    @classmethod
    def load(cls, directory):
        """Restore all CPU blocks, stable IDs, metadata and row-wise Adam state."""
        directory = Path(directory)
        with (directory / "manifest.json").open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        version = manifest.get("format_version")
        if version not in (1, _FORMAT_VERSION):
            raise ValueError("unsupported persistent Gaussian map format")
        next_id = manifest.get("next_id")
        if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 0:
            raise ValueError("invalid next_id in map manifest")
        result = cls(manifest.get("settings", {}))
        result._owners = np.full(next_id, -1, np.int64)
        result._rows = np.full(next_id, -1, np.int64)
        result._next_id = next_id
        for key, metadata in manifest.get("objects", {}).items():
            oid = _object_id(int(key))
            filename = metadata["filename"]
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("invalid object block filename")
            if metadata["status"] not in _STATUSES:
                raise ValueError("invalid object status in manifest")
            block = result._ensure_object(oid)
            with np.load(directory / filename, allow_pickle=False) as saved:
                ids = _ids(saved["map_ids"])
                if (len(ids) != metadata["gaussian_count"] or np.any(ids >= next_id) or
                        np.any(result._owners[ids] != -1)):
                    raise ValueError("object block IDs disagree with manifest")
                n = len(ids)
                block["map_ids"] = ids
                for name, width in PARAMETER_WIDTHS.items():
                    block[name] = _array(saved[name], (n, width), name)
                norms = np.linalg.norm(block["rotations"], axis=1)
                if (np.any(norms < 1e-12) or np.any((block["colors"] < 0) | (block["colors"] > 1))):
                    raise ValueError("invalid saved Gaussian rotations or colors")
                for name in ("initial_xyz", "initial_log_scales"):
                    block[name] = _array(saved[name], (n, 3), name)
                # Early checkpoints did not separate learned appearance from
                # measurements. They can still be read; new checkpoints always
                # preserve the immutable anchor explicitly.
                block["association_colors"] = _array(
                    saved["association_colors"] if "association_colors" in saved else block["colors"],
                    (n, 3), "association_colors")
                if np.any((block["association_colors"] < 0) | (block["association_colors"] > 1)):
                    raise ValueError("invalid saved association colors")
                for name in ("observation_count", "last_seen_frame"):
                    values = saved[name]
                    if values.shape != (n,) or not np.issubdtype(values.dtype, np.integer):
                        raise ValueError(f"invalid saved {name}")
                    block[name] = values.astype(np.int64, copy=True)
                for name, width in PARAMETER_WIDTHS.items():
                    for state_key, state_width in (("exp_avg", width), ("exp_avg_sq", width), ("row_steps", 1)):
                        values = saved[f"adam_{name}_{state_key}"]
                        if state_key == "row_steps":
                            if (values.shape != (n, 1) or not np.issubdtype(values.dtype, np.integer) or
                                    np.any(values < 0)):
                                raise ValueError("invalid saved Adam row_steps")
                            values = values.astype(np.int64, copy=True)
                        else:
                            values = _array(values, (n, state_width), f"adam_{name}_{state_key}")
                            if state_key == "exp_avg_sq" and np.any(values < 0):
                                raise ValueError("invalid saved Adam squared moments")
                        block["optimizer_state"][name][state_key] = values
                result._owners[ids] = oid
                result._rows[ids] = np.arange(n, dtype=np.int64)
            result._metadata[oid] = {name: metadata[name] for name in ("status", "first_frame", "last_frame")}
            result._saved_files[oid] = filename
        missing = np.flatnonzero(result._owners < 0)
        pruned_ids = _ids(manifest.get("pruned_ids", []), "pruned_ids")
        if (np.any(pruned_ids >= next_id) or not np.array_equal(np.sort(pruned_ids), missing)):
            raise ValueError("map manifest is missing allocated map IDs or has invalid pruned IDs")
        result._live_count = next_id - len(pruned_ids)
        if manifest.get("live_count", result._live_count) != result._live_count:
            raise ValueError("live_count disagrees with map blocks")
        if manifest.get("allocated_count", next_id) != next_id:
            raise ValueError("allocated_count disagrees with next_id")
        result._ownership_events, result._topology_events, result._revision = cls.validate_journal(manifest, result._owners)
        result._last_save_directory = directory.resolve()
        result._dirty.clear()
        return result

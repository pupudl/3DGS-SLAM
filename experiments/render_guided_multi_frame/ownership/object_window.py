"""Fixed-lag optimization of persistent Gaussian object attributes.

Local motion probabilities are immutable initialization, never observations.
Every shared Gaussian has one object-logit vector.  Each active pair supplies
fresh costs from its actual images/geometry, and contributes equally regardless
of arrival order (apart from explicit reliability weights).

In the explicit CPU baseline, for jointly visible candidates j,k, c_j-c_k adds
opposite signed energies to j and k.  A missing candidate contributes neither
an error nor a zero-error reward.  Let a_i be the sum of these energies and pi_i
the Gaussian's *first* initialization, padded for newly discovered objects.
The optimized objective is

    sum_i <p_i, a_i> + prior_weight KL(p_i || pi_i) - temperature H(p_i).

It has the exact minimizer
    p_i = softmax((prior_weight log(pi_i) - a_i) /
                  (prior_weight + temperature)).

With ``render.enabled``, the shared logits instead receive gradients from actual
RGB/depth Gaussian rasterizations across the active window. The centre costs
remain an optional auxiliary term. Previous posteriors only warm-start the
optimizer; the fixed first initialization supplies its weak prior. Geometry and
object motions remain fixed, and the original motion probabilities stay frozen.
"""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np


class ObjectWindow:
    """Maintain persistent attributes and rewrite active pair snapshots.

    ``cfg`` accepts a complete config containing ``object_window`` or the
    section itself. ``window_frames`` counts image IDs, not pair updates.
    ``window_step`` advances that many sampled frames after a full window.
    Rendering runs on complete windows and on the sequence's final partial
    window; intervening updates use measured centre costs for provisional state.
    The default step=1 preserves the legacy per-pair rolling solver.
    ``object_ids`` are the global IDs of the input matrix columns; several
    local motions may have already been aggregated into the same object.
    With ``sequence.streaming``, finalized pair exports preserve history and
    per-Gaussian state exists only inside the active lag. New Gaussian IDs must
    be allocated monotonically so a constant-size high-water mark detects reuse.
    """

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg
        self.cfg = dict(cfg.get("object_window", cfg))
        self.streaming = bool(cfg.get("sequence", {}).get("streaming", False))
        self.render_enabled = bool(self.cfg.get("render", {}).get("enabled", False))
        self.gaussian_opacity = float(cfg.get("joint", {}).get("gaussian_opacity", .5))
        self._render_report = {}
        self.window_size = int(self.cfg.get("window_frames", 5))
        self.window_step = int(self.cfg.get('window_step', 1))
        sequence_frames = cfg.get('sequence', {}).get('frames')
        self._sequence_end = (int(sequence_frames[-1]) if sequence_frames is not None and
                              len(sequence_frames) else None)
        self.temperature = float(self.cfg.get("temperature", .25))
        self.prior_weight = float(self.cfg.get("prior_weight", .05))
        self.prior_floor = float(self.cfg.get("prior_floor", .001))
        self.max_cost = float(self.cfg.get("max_cost", 1.))
        self.max_fit_cost = float(self.cfg.get("max_fit_cost", 1.))
        self.max_weight = float(self.cfg.get("max_weight", 1.))
        self.min_probability = float(self.cfg.get("min_probability", .6))
        self.min_margin = float(self.cfg.get("min_margin", .1))
        self.archive_compression = int(self.cfg.get('archive_compression', 1))
        if not 0 <= self.archive_compression <= 9:
            raise ValueError('archive_compression must be in [0,9] (0 stores without compression)')
        if self.window_size < 2:
            raise ValueError("window_frames must count at least two images")
        if not 1 <= self.window_step < self.window_size:
            raise ValueError('window_step must be positive and smaller than window_frames')
        positive = (self.temperature, self.prior_floor, self.max_cost,
                    self.max_fit_cost, self.max_weight)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("temperature, floors, costs, and max_weight must be finite and positive")
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if not (0 <= self.min_probability <= 1 and 0 <= self.min_margin <= 1):
            raise ValueError("probability and margin thresholds must be in [0,1]")
        self.object_ids = []
        # Raw snapshots keep their original candidate columns for diagnostics.
        # Exclude discarded IDs while those snapshots remain in the lag.
        self._discarded_object_ids = set()
        self._initial = {}
        self._state = {}
        self._labels = {}
        self._active_gids = np.empty(0, np.int64)
        self._active_probabilities = np.empty((0, 0), float)
        self._active_labels = np.empty(0, np.int64)
        self._active_hard_background = np.empty(0, bool)
        self._active_height_background = np.empty(0, bool)
        self._active_connectivity_background = np.empty(0, bool)
        self._active_no_core_disabled = np.empty(0, bool)
        self._spatial_report = {}
        self._final_connectivity_report = {}
        self._frames = set()
        self._active = {}
        self._updates = 0
        self._last_report = {}
        # The streaming mode keeps only the fixed lag. Original candidate IDs
        # travel with each immutable first prior, so pruning/reordering current
        # probability columns cannot silently change its meaning.
        self._first_object_ids = {}
        self._boundary = {}
        self._max_gaussian_id = -1
        self._latest_frame = None
        self._finalized_frame = None
        self._finalized_pairs = 0
        self._forgotten_gaussians = 0
        self._created_gaussians = 0
        self._finalized_exports = 0
        self._boundary_pairs = 0

    @staticmethod
    def _ids(values, name):
        raw = np.asarray(values)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array")
        result = raw.astype(np.int64, copy=True)
        if np.any(result < 0) or len(np.unique(result)) != len(result):
            raise ValueError(f"{name} must contain unique nonnegative IDs")
        return result

    def update(self, first, second, gaussian_ids, object_ids, q_init, costs,
               weights=None, output_dir=None, track_ids=None, payload=None,
               q_motion_initial=None, render_observations=None, force_optimize=False):
        """Add/replace a pair's raw evidence and jointly reoptimize its window.

        ``q_init`` and ``costs`` have shape [N,K]; weights may be [N] or [N,K].
        NaN/+inf costs, or zero weights, denote unavailable evidence. Negative
        costs/weights are rejected. A pair with no valid candidate comparison
        cannot reinforce inherited labels. Repeating the same pair replaces
        its evidence rather than counting it twice. All arrays are copied.

        ``q_motion_initial`` optionally saves the original local-motion matrix
        (which need not have K columns). Otherwise it aliases a copy of q_init.
        ``payload`` contains extra non-object arrays for visualization exports.
        ``render_observations`` contains fixed source/target RGB-D/confidence
        arrays for differentiable rendering. These are not exported into NPZs.
        ``force_optimize`` flushes an incomplete window when no sequence end
        was configured (for example, a caller consuming a live stream).
        """
        first, second = int(first), int(second)
        if first == second:
            raise ValueError("a pair must contain two distinct image IDs")
        gids = self._ids(gaussian_ids, "gaussian_ids")
        oids = self._ids(object_ids, "object_ids")
        if not len(oids):
            raise ValueError("at least one object column is required")
        n, k = len(gids), len(oids)
        q = np.array(q_init, dtype=np.float64, copy=True)
        c = np.array(costs, dtype=np.float64, copy=True)
        if q.shape != (n, k) or c.shape != (n, k):
            raise ValueError("q_init and costs must have shape [N,K]")
        if np.any(~np.isfinite(q)) or np.any(q < 0):
            raise ValueError("q_init must be finite and nonnegative")
        if np.any(c[np.isfinite(c)] < 0) or np.any(np.isneginf(c)):
            raise ValueError("costs must be nonnegative, NaN, or +inf")
        if weights is None:
            w = np.ones((n, k), dtype=np.float64)
        else:
            w = np.array(weights, dtype=np.float64, copy=True)
            if w.shape == (n,):
                w = np.repeat(w[:, None], k, axis=1)
            if w.shape != (n, k) or np.any(~np.isfinite(w)) or np.any(w < 0):
                raise ValueError("weights must be finite nonnegative [N] or [N,K]")
        motion_q = np.array(q if q_motion_initial is None else q_motion_initial, copy=True)
        if (motion_q.ndim != 2 or len(motion_q) != n or
                np.any(~np.isfinite(motion_q)) or np.any(motion_q < 0)):
            raise ValueError("q_motion_initial must be a finite nonnegative [N,M] array")
        tracks = None if track_ids is None else np.array(track_ids, dtype=np.int64, copy=True)
        if tracks is not None and tracks.shape != (n,):
            raise ValueError("track_ids must have shape [N]")
        extra = {}
        reserved = {"first", "second", "gaussian_ids", "object_ids", "object_labels",
                    "object_probabilities", "object_logits", "q_motion_initial",
                    "q_object_initial", "evidence_object_ids", "observation_costs",
                    "observation_weights", "window_frames", "window_pairs",
                    "report_json", "track_ids", "height_forced_background",
                    "connectivity_forced_background", "connectivity_disabled_object_ids"}
        for name, value in (payload or {}).items():
            array = np.array(value, copy=True)
            if name in reserved or array.dtype.hasobject:
                raise ValueError(f"payload key {name!r} is reserved or has object dtype")
            extra[str(name)] = array
        if "H_object" in extra and extra["H_object"].shape != (k, 4, 4):
            raise ValueError("payload H_object must have shape [K,4,4]")
        if "H_object" in extra:
            extra["H_object"] = np.asarray(extra["H_object"], dtype=float)
            if np.isinf(extra["H_object"]).any():
                raise ValueError("payload H_object may contain missing NaNs but not infinite poses")
        if "object_model_valid" in extra and extra["object_model_valid"].shape != (k,):
            raise ValueError("payload object_model_valid must have shape [K]")
        observations = None
        if self.render_enabled:
            required_geometry = {"xyz", "color", "scale", "K", "H_object"}
            if required_geometry - extra.keys():
                raise ValueError("Render optimization requires saved Gaussian geometry, K and object motions")
            required = {"source", "target", "source_depth", "target_depth",
                        "source_confidence", "target_confidence"}
            if render_observations is None or required - render_observations.keys():
                raise ValueError("Render optimization requires source/target RGB-D and confidence observations")
            observations = {}
            for endpoint in ("source", "target"):
                image = np.asarray(render_observations[endpoint])
                depth = np.asarray(render_observations[endpoint + "_depth"])
                confidence = np.asarray(render_observations[endpoint + "_confidence"])
                if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or
                        depth.shape != image.shape[:2] or confidence.shape != depth.shape):
                    raise ValueError("Render observations need uint8 BGR images and matching depth/confidence maps")
                if (not np.isfinite(confidence).all() or np.any(confidence < 0) or
                        np.any(confidence > 1)):
                    raise ValueError("Render confidence must be finite in [0,1]")
                observations[endpoint] = image.copy()
                observations[endpoint + "_depth"] = depth.copy()
                observations[endpoint + "_confidence"] = confidence.copy()
        elif (self.cfg.get('height', {}).get('enabled', False) and
              self.cfg.get('height', {}).get('hard_upper_bound', False) and
              render_observations is not None and 'source' in render_observations):
            # The CPU baseline also needs the measured source plane for the
            # same hard rule; retaining target RGB-D is unnecessary here.
            image = np.asarray(render_observations['source'])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError('Height observations require a uint8 source BGR image')
            observations = dict(source=image.copy())
            if 'source_confidence' in render_observations:
                confidence = np.asarray(render_observations['source_confidence'])
                if (confidence.shape != image.shape[:2] or not np.isfinite(confidence).all() or
                        np.any(confidence < 0) or np.any(confidence > 1)):
                    raise ValueError('Height source confidence must match the image and be in [0,1]')
                observations['source_confidence'] = confidence.copy()

        # No object or Gaussian state changes before all input validation passes.
        if self.streaming:
            frames = self._validate_stream_update(first, second, gids)
            oids.setflags(write=False)
        else:
            frames = self._next_window_frames(first, second)
        self.object_ids.extend(int(o) for o in oids
                               if int(o) not in self.object_ids and int(o) not in self._discarded_object_ids)
        columns = {o: j for j, o in enumerate(self.object_ids)}
        initial_columns = np.asarray([columns.get(int(o), -1) for o in oids], np.int64)
        for gid, row in zip(gids, q):
            gid = int(gid)
            if gid not in self._initial:
                mass = float(row.sum())
                normalized = row / mass if mass > 0 else np.full(k, 1. / k)
                initial = np.full(len(self.object_ids), self.prior_floor)
                retained = initial_columns >= 0
                initial[initial_columns[retained]] = np.maximum(normalized[retained], self.prior_floor)
                if self.streaming:
                    self._initial[gid] = np.maximum(normalized, self.prior_floor)
                    # Every Gaussian born in this update shares this immutable
                    # column descriptor; avoid millions of tiny copied arrays.
                    self._first_object_ids[gid] = oids
                else:
                    self._initial[gid] = initial
                self._created_gaussians += 1
        snapshot = dict(first=first, second=second, gaussian_ids=gids,
                        object_ids=oids, q_init=q, costs=c, weights=w,
                        q_motion_initial=motion_q, track_ids=tracks, payload=extra,
                        render_observations=observations,
                        output_dir=None if output_dir is None else Path(output_dir))
        self._active[(first, second)] = snapshot
        self._frames.update((first, second))
        frame_set = set(frames)
        departing = {pair: snap for pair, snap in self._active.items()
                     if pair[0] not in frame_set or pair[1] not in frame_set}
        self._active = {pair: snap for pair, snap in self._active.items()
                        if pair not in departing}
        self._discarded_object_ids.intersection_update(
            int(oid) for snap in self._active.values() for oid in snap['object_ids'])
        if self.streaming:
            self._advance_stream(departing, frames)
            self._max_gaussian_id = max(self._max_gaussian_id, int(gids.max()) if n else -1)
            self._latest_frame = max(first, second, self._latest_frame or -1)
        elif self.window_step > 1:
            self._frames = set(frames)
        self._updates += 1
        final = bool(force_optimize or second == self._sequence_end)
        render_now = self.window_step == 1 or len(frames) == self.window_size or final
        reason = ('per_pair_rolling' if self.window_step == 1 else
                  'full_window' if len(frames) == self.window_size else
                  'final_partial_window' if final else 'collecting_window')
        if self.render_enabled and not render_now:
            self._render_report = dict(reason='waiting_for_window_completion',
                                       optimized_parameter=None, motion_probabilities_optimized=False,
                                       iterations_completed=0, nonzero_gradient_steps=0,
                                       render_passes=0,
                                       provisional_solver='retained_posteriors_and_new_row_centre_costs')
        self._optimize(render_now=render_now)
        empty_pruning = self._prune_empty_objects() if (
            self.streaming and self.cfg.get('prune_empty_objects', False) and render_now) else None
        self._last_report = dict(
            updates=self._updates, window_frames=frames,
            window_pairs=[list(pair) for pair in sorted(self._active)],
            active_gaussians=len(self._active_gids),
            total_gaussians=self._created_gaussians, object_ids=list(self.object_ids),
            temperature=self.temperature, prior_weight=self.prior_weight,
            objective="bounded_pairwise_cost + prior_weight*KL - temperature*entropy",
            motion_probabilities_optimized=False)
        self._last_report['window_schedule'] = dict(
            window_size=self.window_size, window_step=self.window_step,
            overlap_frames=self.window_size-self.window_step,
            complete=len(frames) == self.window_size,
            render_optimization_performed=bool(self.render_enabled and render_now and len(self._active_gids)),
            reason=reason)
        if empty_pruning is not None:
            self._last_report['empty_object_pruning'] = empty_pruning
        if self._spatial_report:
            self._last_report['spatial_constraints'] = self._spatial_report
        if self._final_connectivity_report:
            self._last_report['final_connectivity'] = self._final_connectivity_report
        if self.streaming:
            self._last_report.update(
                streaming=True, retained_gaussians=len(self._initial),
                retained_first_priors=len(self._initial),
                retained_posteriors=len(self._state), retained_frames=len(self._frames),
                retained_pairs=len(self._active), retained_objects=len(self.object_ids),
                boundary_gaussians=len(self._boundary),
                boundary_pairs_marginalized=self._boundary_pairs,
                finalized_pairs=self._finalized_pairs,
                finalized_pair_exports=self._finalized_exports,
                forgotten_gaussians=self._forgotten_gaussians,
                gaussian_id_high_water_mark=self._max_gaussian_id,
                finalized_through_frame=self._finalized_frame,
                boundary_policy="exact_centre_track_sufficient_statistics;pixel_posterior_warm_start",
                pixel_observations_marginalized=False,
                retired_candidate_columns="discarded;first_initialization_preserved",
                finalized_outputs="last_active_window_export_is_frozen")
        if self.render_enabled:
            self._last_report.update(
                optimizer="shared_logits_differentiable_gaussian_raster",
                objective="window_RGB_depth_coverage_raster + optional_centre_cost + first_prior_KL",
                render_optimization=self._render_report)
            if any(self.cfg.get(section, {}).get('enabled', False) for section in ('height', 'connectivity')):
                self._last_report['objective'] += ' + height_band + core_guided_connectivity'
        self._write_active()
        return self.get_state(gids)

    def _prior(self, gid):
        initial = self._initial[gid]
        row = np.full(len(self.object_ids), self.prior_floor)
        if self.streaming:
            original = dict(zip(self._first_object_ids[gid], initial))
            row = np.asarray([original.get(oid, self.prior_floor) for oid in self.object_ids])
        else:
            row[:len(initial)] = initial
        return row / row.sum()

    def _priors(self, gids):
        """Batch the fixed first priors by their original column descriptor.

        Descriptors are shared by Gaussians born together. Map columns once
        per descriptor and normalize whole arrays, rather than constructing
        a dictionary and a normalized tiny array for every Gaussian.
        """
        result = np.full((len(gids), len(self.object_ids)), self.prior_floor)
        groups = {}
        for row, gid in enumerate(gids):
            gid = int(gid)
            descriptor = self._first_object_ids[gid] if self.streaming else None
            key = id(descriptor) if self.streaming else len(self._initial[gid])
            if key not in groups:
                groups[key] = (descriptor, [], [])
            groups[key][1].append(row)
            groups[key][2].append(self._initial[gid])
        for descriptor, rows, values in groups.values():
            values = np.stack(values)
            if self.streaming:
                original = {int(oid): j for j, oid in enumerate(descriptor)}
                target = [j for j, oid in enumerate(self.object_ids) if oid in original]
                source = [original[self.object_ids[j]] for j in target]
            else:
                source = target = list(range(values.shape[1]))
            result[np.ix_(rows, target)] = values[:, source]
        result /= result.sum(axis=1, keepdims=True)
        return result

    def _prune_empty_objects(self):
        """Discard foreground IDs with no accepted Gaussian in the solved lag.

        Waiting for a complete/final window lets provisional objects collect
        evidence first. Unknown labels stay unknown when unused soft mass is
        transferred to background. Immutable observations and first priors are
        retained in their original columns, and finalized exports stay frozen.
        """
        background = int(self.cfg.get('height', {}).get('background_object_id', 0))
        supported, counts = np.unique(self._active_labels[self._active_labels >= 0], return_counts=True)
        removed = [oid for oid in self.object_ids if oid != background and oid not in supported]
        report = dict(removed_object_ids=removed, scope='solved_active_window',
                      accepted_gaussians={str(int(oid)): int(n) for oid, n in zip(supported, counts)})
        if not removed:
            return report
        if background not in self.object_ids:
            raise ValueError('Empty-object pruning requires a background object column')
        old_ids = list(self.object_ids)
        keep = np.asarray([oid not in removed for oid in old_ids])
        background_column = old_ids.index(background)

        def probabilities(values):
            values = values.copy()
            values[..., background_column] += values[..., ~keep].sum(axis=-1)
            return values[..., keep]

        self._active_probabilities = probabilities(self._active_probabilities)
        for gid, row in zip(self._active_gids, self._active_probabilities):
            self._state[int(gid)] = row.copy()
        self._boundary = {gid: stats[:, keep] for gid, stats in self._boundary.items()}
        self._active_no_core_disabled = self._active_no_core_disabled[keep]
        if hasattr(self, '_render_initial_probabilities'):
            self._render_initial_probabilities = probabilities(self._render_initial_probabilities)
            self._render_energy = self._render_energy[:, keep]
        self.object_ids = [oid for oid in old_ids if oid not in removed]
        self._discarded_object_ids.update(removed)
        return report

    def _next_window_frames(self, first, second):
        current = sorted(self._frames)
        if self.window_step > 1 and len(current) == self.window_size and max(first, second) > current[-1]:
            current = current[self.window_step:]
        if self.window_step > 1 and current and min(first, second) < current[0]:
            raise ValueError('Pair endpoints precede the active frame window')
        frames = sorted(set(current) | {first, second})
        if self.window_step == 1:
            frames = frames[-self.window_size:]
        if len(frames) > self.window_size or first not in frames or second not in frames:
            raise ValueError('Pair endpoints must fit inside the active frame window')
        return frames

    def _validate_stream_update(self, first, second, gids):
        pair = (first, second)
        if pair not in self._active:
            if (self._latest_frame is not None and max(pair) < self._latest_frame or
                    self._finalized_frame is not None and min(pair) <= self._finalized_frame):
                raise ValueError("streaming cannot revisit a finalized pair; use saved pair exports")
        frames = self._next_window_frames(first, second)
        if first not in frames or second not in frames:
            raise ValueError("streaming pair endpoints must fit inside the active frame window")
        if any(int(gid) not in self._initial and int(gid) <= self._max_gaussian_id
               for gid in gids):
            raise ValueError("streaming requires monotonically allocated new Gaussian IDs; "
                             "finalized Gaussian IDs cannot be reused")
        return frames

    @staticmethod
    def _remap(values, old_ids, new_ids, fill):
        """Remap the last axis by global ID, including newly appended columns."""
        lookup = {int(oid): j for j, oid in enumerate(old_ids[:values.shape[-1]])}
        result = np.full(values.shape[:-1] + (len(new_ids),), fill, dtype=values.dtype)
        for new, oid in enumerate(new_ids):
            old = lookup.get(int(oid))
            if old is not None:
                result[..., new] = values[..., old]
        return result

    def _advance_stream(self, departing, frames):
        """Eliminate departing factors once, then forget finalized parameters.

        Centre-cost factors are linear in each independent categorical row.
        Their three additive arrays are sufficient statistics, so no posterior
        is fed back as a new observation. This is exact for retained candidate
        columns. Retired columns are deliberately discarded. Raster factors
        couple many Gaussians and are not exactly marginalized by this method;
        their final posterior supplies only the ordinary solver warm start.
        """
        old_ids = list(self.object_ids)
        candidates = {int(oid) for snap in self._active.values() for oid in snap["object_ids"]
                      if int(oid) not in self._discarded_object_ids}
        background = int(self.cfg.get("height", {}).get("background_object_id", 0))
        if background in old_ids:
            candidates.add(background)
        self.object_ids = [oid for oid in old_ids if oid in candidates]
        active_gids = (np.unique(np.concatenate([snap["gaussian_ids"] for snap in self._active.values()]))
                       if self._active else np.empty(0, np.int64))
        keep = set(map(int, active_gids))
        forgotten = set(self._initial) - keep
        self._forgotten_gaussians += len(forgotten)
        for table in (self._initial, self._first_object_ids, self._state, self._labels, self._boundary):
            for gid in forgotten:
                table.pop(gid, None)
        same_columns = old_ids == self.object_ids
        for gid, row in self._state.items():
            if same_columns and len(row) == len(self.object_ids):
                continue
            remapped = self._remap(row, old_ids, self.object_ids, self.prior_floor)
            self._state[gid] = remapped / remapped.sum()
        for gid, stats in list(self._boundary.items()):
            if same_columns and stats.shape[-1] == len(self.object_ids):
                continue
            remapped = self._remap(stats, old_ids, self.object_ids, 0.)
            if np.any(remapped):
                self._boundary[gid] = remapped
            else:
                del self._boundary[gid]
        columns = {int(oid): j for j, oid in enumerate(self.object_ids)}
        for snap in departing.values():
            statistics = self._snapshot_evidence(snap, len(old_ids), old_ids)
            source_columns = [j for j, oid in enumerate(snap["object_ids"]) if int(oid) in columns]
            target_columns = [columns[int(snap["object_ids"][j])] for j in source_columns]
            for row, gid in enumerate(snap["gaussian_ids"]):
                gid = int(gid)
                if gid not in keep:
                    continue
                incoming = statistics[:, row, source_columns]
                if not np.any(incoming):
                    continue
                stats = self._boundary.setdefault(gid, np.zeros((3, len(self.object_ids))))
                stats[:, target_columns] += incoming
            self._boundary_pairs += 1
        self._finalized_pairs += len(departing)
        self._finalized_exports += sum(snap["output_dir"] is not None and
                                      (snap["output_dir"] / "object_state.npz").is_file()
                                      for snap in departing.values())
        dropped_frames = self._frames - set(frames)
        if dropped_frames:
            self._finalized_frame = max(max(dropped_frames), self._finalized_frame or -1)
        self._frames = set(frames)

    def _snapshot_evidence(self, snap, global_count, candidate_ids=None):
        """Return pairwise statistics in the snapshot's immutable columns.

        Sorting each comparable row by reliability makes all pair minima an
        exclusive prefix plus a constant-weight suffix. This is the same
        pairwise objective with O(K log K) work instead of O(K**2). Fixed-size
        chunks bound temporary arrays; absent/retired candidates remain zero.
        """
        costs, weights = snap["costs"], snap["weights"]
        n, count = costs.shape
        statistics = np.zeros((3, n, count))
        columns = (np.arange(count) if candidate_ids is None else
                   np.flatnonzero(np.isin(snap['object_ids'], candidate_ids)))
        if not len(columns):
            return statistics
        higher = np.arange(len(columns)-1, -1, -1)
        for start in range(0, n, 50_000):
            end = min(n, start+50_000)
            c0, w0 = costs[start:end], weights[start:end]
            if len(columns) != count:
                c0, w0 = c0[:, columns], w0[:, columns]
            valid = np.isfinite(c0) & (w0 > 0)
            valid_count = valid.sum(axis=1)
            any_fit = np.any(valid & (c0 < self.max_fit_cost), axis=1)
            rows = np.flatnonzero((valid_count >= 2) & any_fit)
            if len(rows):
                reliability = np.where(valid[rows], np.minimum(w0[rows], self.max_weight), 0.)
                clipped = np.where(valid[rows], np.minimum(c0[rows], self.max_cost), 0.)
                order = np.argsort(reliability, axis=1)
                reliability = np.take_along_axis(reliability, order, axis=1)
                clipped = np.take_along_axis(clipped, order, axis=1)
                prefix_r = np.zeros_like(reliability)
                prefix_rc = np.zeros_like(clipped)
                suffix_c = np.zeros_like(clipped)
                if len(columns) > 1:
                    prefix_r[:, 1:] = np.cumsum(reliability[:, :-1], axis=1)
                    prefix_rc[:, 1:] = np.cumsum(reliability[:, :-1]*clipped[:, :-1], axis=1)
                    suffix_c[:, :-1] = np.cumsum(clipped[:, :0:-1], axis=1)[:, ::-1]
                weighted_count = prefix_r+reliability*higher[None, :]
                denominator = valid_count[rows, None]
                energy = (clipped*weighted_count-prefix_rc-reliability*suffix_c)/denominator
                support = weighted_count/denominator
                statistics[0, (start+rows)[:, None], columns[order]] = energy
                statistics[1, (start+rows)[:, None], columns[order]] = support
                good = valid[rows] & (c0[rows] < self.max_fit_cost)
                statistics[2][np.ix_(start+rows, columns)] = np.where(
                    good, np.minimum(w0[rows], self.max_weight), 0.)
            if global_count == 1:
                extra = np.where(valid & any_fit[:, None], np.minimum(w0, self.max_weight), 0.)
                positions = np.ix_(np.arange(start, end), columns)
                statistics[1][positions] += extra
                statistics[2][positions] += extra
        return statistics

    def _optimize(self, render_now=True):
        count = len(self.object_ids)
        columns = {o: j for j, o in enumerate(self.object_ids)}
        active_gids = (np.unique(np.concatenate([snap["gaussian_ids"] for snap in self._active.values()]))
                       if self._active else np.empty(0, np.int64))
        energy = np.zeros((len(active_gids), count))
        support = np.zeros_like(energy)
        good_fit = np.zeros_like(energy)
        if self.streaming:
            for row, gid in enumerate(active_gids):
                stats = self._boundary.get(int(gid))
                if stats is not None:
                    energy[row], support[row], good_fit[row] = stats
        for pair in sorted(self._active):
            snap = self._active[pair]
            retained = np.asarray([j for j, oid in enumerate(snap['object_ids']) if int(oid) in columns], np.int64)
            col = np.asarray([columns[int(snap['object_ids'][j])] for j in retained], np.int64)
            rows = np.searchsorted(active_gids, snap["gaussian_ids"])
            local_energy, local_support, local_good_fit = self._snapshot_evidence(snap, count, self.object_ids)
            positions = np.ix_(rows, col)
            energy[positions] += local_energy[:, retained]
            support[positions] += local_support[:, retained]
            good_fit[positions] += local_good_fit[:, retained]
        priors = self._priors(active_gids)
        render_support = np.zeros_like(support, dtype=bool)
        hard_background = np.zeros(len(active_gids), bool)
        height_background = np.zeros(len(active_gids), bool)
        connectivity_background = np.zeros(len(active_gids), bool)
        disabled_columns = np.zeros(count, bool)
        self._spatial_report = {}
        self._final_connectivity_report = {}
        connectivity_state = None
        height_options = self.cfg.get('height', {})
        connectivity_options = self.cfg.get('connectivity', {})
        hard_height = height_options.get('enabled', False) and height_options.get('hard_upper_bound', False)
        no_core_fallback = (connectivity_options.get('enabled', False) and
                            connectivity_options.get('no_core_to_background', False) and render_now)
        final_connectivity = (connectivity_options.get('enabled', False) and
                              connectivity_options.get('final_hard_assignment', False) and render_now)
        if self.streaming and self.render_enabled:
            # Empty updates must also release the previous GPU solver's CPU
            # input buffers, rather than retaining a finalized parameter set.
            self._render_initial_probabilities = priors.copy()
            self._render_energy = energy
            if not len(active_gids):
                self._render_report = {}
        if self.render_enabled and render_now and len(active_gids):
            # The first prior stays fixed in either mode. A collecting state
            # predates newly born objects and must not lock old rows to O0
            # merely because a short logit optimization cannot escape it.
            initialization = self.cfg.get('render', {}).get('initialization', 'previous')
            if initialization == 'measurement':
                logits = ((self.prior_weight * np.log(priors) - energy) /
                          (self.temperature + self.prior_weight))
                logits -= logits.max(axis=1, keepdims=True)
                initial = np.exp(logits)
                initial /= initial.sum(axis=1, keepdims=True)
            elif initialization == 'previous':
                initial = priors.copy()
                for row, gid in enumerate(active_gids):
                    previous = self._state.get(int(gid))
                    if previous is not None:
                        initial[row] = self.prior_floor
                        initial[row, :len(previous)] = previous
                        initial[row] /= initial[row].sum()
            else:
                raise ValueError('render.initialization must be previous or measurement')
            self._active_gids = active_gids
            self._render_initial_probabilities = initial
            self._render_energy = energy
            from ownership.object_render_optimizer import optimize_render_window
            result = optimize_render_window(self, self.cfg, priors=priors)
            probabilities = np.asarray(result["probabilities"], np.float64)
            render_support = np.asarray(result["render_support"], bool)
            if (probabilities.shape != energy.shape or render_support.shape != energy.shape or
                    not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                    not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("Render optimizer returned invalid shared object probabilities/support")
            self._render_report = result["report"]
            self._render_report['initialization'] = initialization
            hard_background = np.asarray(result.get('hard_background_mask', hard_background), bool)
            height_background = np.asarray(result.get('height_background_mask', hard_background), bool)
            connectivity_background = np.asarray(result.get('connectivity_background_mask', connectivity_background), bool)
            disabled_columns = np.asarray(result.get('no_core_disabled_columns', disabled_columns), bool)
            connectivity_state = result.get('connectivity_state')
        else:
            denominator = self.temperature + self.prior_weight
            logits = (self.prior_weight * np.log(priors) - energy) / denominator
            if len(active_gids):
                logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            if self.render_enabled and not render_now:
                # Window overlap is a warm state, not fresh measurements. Keep
                # its solved posterior intact while collecting the next window.
                # Only newly introduced rows use the cheap centre initializer.
                for row, gid in enumerate(active_gids):
                    previous = self._state.get(int(gid))
                    if previous is None:
                        continue
                    probabilities[row] = self.prior_floor
                    probabilities[row, :len(previous)] = previous
                    probabilities[row] /= probabilities[row].sum()
                    label = self._labels.get(int(gid), -1)
                    if label in columns:
                        render_support[row, columns[label]] = True
            if (hard_height or no_core_fallback or final_connectivity) and len(active_gids):
                from ownership.object_spatial_constraints import prepare_spatial_constraints
                self._active_gids = active_gids
                self._render_initial_probabilities = probabilities
                # Before a batch is complete, only the mandatory height rule
                # runs. Core absence is decided from the completed window.
                spatial_cfg = {**self.cfg, 'connectivity': (connectivity_options if
                                                           (no_core_fallback or final_connectivity) else
                                                           {'enabled': False})}
                spatial = prepare_spatial_constraints(self, spatial_cfg)
                hard_background = spatial['hard_background_mask']
                height_background = spatial['height_background_mask']
                connectivity_background = spatial['connectivity_background_mask']
                disabled_columns = spatial['no_core_disabled_columns']
                connectivity_state = spatial['connectivity_state']
                if spatial['no_core_initial_probabilities'] is not None:
                    probabilities = spatial['no_core_initial_probabilities'].copy()
                self._spatial_report = spatial['report']
        if any(mask.shape != (len(active_gids),) for mask in
               (hard_background, height_background, connectivity_background)) or disabled_columns.shape != (count,):
            raise ValueError('Hard background masks must align with active Gaussian/object IDs')
        background_id = int((height_options if hard_height else connectivity_options).get('background_object_id', 0))
        if hard_height and (no_core_fallback or final_connectivity) and background_id != int(connectivity_options.get('background_object_id', 0)):
            raise ValueError('Height and connectivity constraints must use the same background object ID')
        if (hard_height or no_core_fallback or final_connectivity) and len(active_gids):
            if background_id not in self.object_ids:
                raise ValueError('Hard constraints require the configured background object ID')
            bg = self.object_ids.index(background_id)
            mass = probabilities[:, disabled_columns].sum(1)
            probabilities[:, disabled_columns] = 0.
            probabilities[:, bg] += mass
            probabilities[hard_background] = 0.
            probabilities[hard_background, bg] = 1.
        labels = np.full(len(active_gids), -1, np.int64)
        if len(active_gids):
            winner = probabilities.argmax(axis=1)
            rows = np.arange(len(active_gids))
            best = probabilities[rows, winner]
            second = (np.partition(probabilities, -2, axis=1)[:, -2]
                      if count > 1 else np.zeros(len(active_gids)))
            observed = (((support[rows, winner] > 0) & (good_fit[rows, winner] > 0)) |
                        render_support[rows, winner])
            confident = (observed &
                         (best >= self.min_probability) & (best - second >= self.min_margin))
            labels[confident] = np.asarray(self.object_ids)[winner[confident]]
        # Hard spatial exclusion is sufficient evidence for mandatory background;
        # no photometric support is required to avoid an unknown (-1) label.
        labels[hard_background] = background_id
        if final_connectivity and len(active_gids):
            if connectivity_state is None:
                raise ValueError('Final connectivity requires the completed window connectivity state')
            from ownership.final_connectivity import finalize_connectivity
            final = finalize_connectivity(
                connectivity_state, probabilities,
                assignment_support=((support > 0) & (good_fit > 0)) | render_support,
                min_probability=self.min_probability, min_margin=self.min_margin,
                hard_background_mask=hard_background)
            probabilities, labels = final['probabilities'], final['labels']
            connectivity_background |= final['forced_background_mask'] & ~hard_background
            hard_background |= final['forced_background_mask']
            disabled_columns |= final['disabled_columns']
            self._final_connectivity_report = final['report']
            if self.render_enabled:
                # The render objective was evaluated before this mandatory
                # assignment projection; its loss is not a score of the clamp.
                self._render_report['reported_losses_stage'] = 'before_final_connectivity_projection'
                self._render_report['saved_probabilities_include_final_connectivity_projection'] = True
        self._active_gids = active_gids
        self._active_probabilities = probabilities
        self._active_labels = labels
        self._active_hard_background = hard_background
        self._active_height_background = height_background
        self._active_connectivity_background = connectivity_background
        self._active_no_core_disabled = disabled_columns
        # The legacy mode retains finalized posteriors. Streaming removes
        # finalized rows in _advance_stream; exports are their durable record.
        for gid, probability, label in zip(active_gids, probabilities, labels):
            self._state[int(gid)] = probability.copy()
            self._labels[int(gid)] = int(label)

    def get_state(self, gaussian_ids=None):
        """Return copied columns/probabilities; streaming finalized IDs are absent."""
        gids = (np.asarray(sorted(self._initial), dtype=np.int64) if gaussian_ids is None
                else self._ids(gaussian_ids, "gaussian_ids"))
        if any(int(g) not in self._initial for g in gids):
            raise KeyError("unknown Gaussian ID")
        probs = np.empty((len(gids), len(self.object_ids)), dtype=np.float64)
        labels = np.full(len(gids), -1, dtype=np.int64)
        positions = np.searchsorted(self._active_gids, gids)
        present = positions < len(self._active_gids)
        present[present] &= self._active_gids[positions[present]] == gids[present]
        probs[present] = self._active_probabilities[positions[present]]
        labels[present] = self._active_labels[positions[present]]
        for row in np.flatnonzero(~present):
            gid = gids[row]
            gid = int(gid)
            state = self._state.get(gid)
            p = self._prior(gid)
            if state is not None:
                p = np.full(len(self.object_ids), self.prior_floor)
                p[:len(state)] = state
            p = p / p.sum()
            probs[row] = p
            winner = int(np.argmax(p))
            oid = self.object_ids[winner]
            second = float(np.partition(p, -2)[-2]) if len(p) > 1 else 0.
            # Inactive rows keep their last observed label, never convert a
            # confident initializer into an observed hard assignment.
            observed = self._labels.get(gid, -1) == oid
            if (observed and p[winner] >= self.min_probability and
                    p[winner] - second >= self.min_margin):
                labels[row] = oid
        return dict(gaussian_ids=gids.copy(), object_ids=np.asarray(self.object_ids, np.int64),
                    object_probabilities=probs, object_labels=labels,
                    object_logits=np.log(np.maximum(probs, np.finfo(float).tiny)),
                    report=self.to_dict())

    def _write_active(self):
        for pair in sorted(self._active):
            snap = self._active[pair]
            if snap["output_dir"] is None:
                continue
            state = self.get_state(snap["gaussian_ids"])
            arrays = {k: v for k, v in state.items() if k != "report"}
            arrays.update(snap["payload"])
            rows = np.searchsorted(self._active_gids, snap['gaussian_ids'])
            arrays['height_forced_background'] = self._active_height_background[rows]
            arrays['connectivity_forced_background'] = self._active_connectivity_background[rows]
            arrays['connectivity_disabled_object_ids'] = np.asarray(self.object_ids, np.int64)[self._active_no_core_disabled]
            # Motion payload columns refer to this pair's original candidates.
            # Match them explicitly to the expanding global probability columns.
            if "H_object" in arrays:
                poses = np.full((len(self.object_ids), 4, 4), np.nan)
                valid = np.zeros(len(self.object_ids), bool)
                original = snap["payload"]["H_object"]
                original_valid = snap["payload"].get(
                    "object_model_valid", np.isfinite(original).all(axis=(1, 2)))
                # Keep motions in the same immutable columns as q_init and
                # measured costs, including candidates removed from the final
                # posterior. Otherwise exact solver replay loses their Hs.
                arrays['evidence_H_object'] = original
                arrays['evidence_object_model_valid'] = original_valid
                columns = {o: j for j, o in enumerate(self.object_ids)}
                for j, oid in enumerate(snap["object_ids"]):
                    if int(oid) not in columns:
                        continue
                    column = columns[int(oid)]
                    poses[column] = original[j]
                    valid[column] = original_valid[j]
                arrays["H_object"] = poses
                arrays["object_model_valid"] = valid
            arrays.update(first=np.asarray(snap["first"]), second=np.asarray(snap["second"]),
                          q_motion_initial=snap["q_motion_initial"],
                          q_object_initial=snap["q_init"], evidence_object_ids=snap["object_ids"],
                          observation_costs=snap["costs"], observation_weights=snap["weights"],
                          window_frames=np.asarray(self._last_report["window_frames"], np.int64),
                          window_pairs=np.asarray(self._last_report["window_pairs"], np.int64),
                          report_json=np.asarray(json.dumps(self.to_dict(), sort_keys=True)))
            if snap["track_ids"] is not None:
                arrays["track_ids"] = snap["track_ids"]
                observed = np.flatnonzero(snap["track_ids"] >= 0)
                tracks, inverse = np.unique(snap["track_ids"][observed], return_inverse=True)
                track_labels = np.full(len(tracks), -1, dtype=np.int64)
                track_probabilities = np.zeros((len(tracks), len(self.object_ids)), float)
                np.add.at(track_probabilities, inverse, state["object_probabilities"][observed])
                if len(tracks):
                    track_probabilities /= np.bincount(inverse)[:, None]
                    best = track_probabilities.argmax(axis=1)
                    oids = np.asarray(self.object_ids)[best]
                    maximum = track_probabilities[np.arange(len(tracks)), best]
                    second = (np.partition(track_probabilities, -2, axis=1)[:, -2]
                              if len(self.object_ids) > 1 else np.zeros(len(tracks)))
                    supported = np.zeros(len(tracks), bool)
                    np.logical_or.at(supported, inverse,
                                     state["object_labels"][observed] == oids[inverse])
                    confident = (supported & (maximum >= self.min_probability) &
                                 (maximum-second >= self.min_margin))
                    track_labels[confident] = oids[confident]
                arrays["object_track_ids"] = tracks
                arrays["object_track_labels"] = track_labels
                arrays["object_track_probabilities"] = track_probabilities
            snap["output_dir"].mkdir(parents=True, exist_ok=True)
            destination = snap["output_dir"] / "object_state.npz"
            temporary = destination.with_suffix(".tmp.npz")
            # Standard NPZ, with tunable compression and atomic replacement.
            # Level 1 avoids blocking each active-window rewrite on zlib's
            # default level 6; all fields remain readable by numpy.load.
            compression = zipfile.ZIP_DEFLATED if self.archive_compression else zipfile.ZIP_STORED
            try:
                with zipfile.ZipFile(temporary, 'w', compression=compression,
                                     compresslevel=self.archive_compression or None,
                                     allowZip64=True) as archive:
                    for name, array in arrays.items():
                        with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                            np.lib.format.write_array(stream, np.asanyarray(array), allow_pickle=False)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)

    def to_dict(self):
        """Small JSON-serializable window summary (arrays live in NPZ files)."""
        return json.loads(json.dumps(self._last_report))

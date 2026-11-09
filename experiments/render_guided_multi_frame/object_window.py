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

import numpy as np


class ObjectWindow:
    """Maintain persistent attributes and rewrite active pair snapshots.

    ``cfg`` accepts a complete config containing ``object_window`` or the
    section itself. ``window_frames`` counts image IDs, not pair updates.
    ``object_ids`` are the global IDs of the input matrix columns; several
    local motions may have already been aggregated into the same object.
    """

    def __init__(self, cfg=None):
        cfg = {} if cfg is None else cfg
        self.cfg = dict(cfg.get("object_window", cfg))
        self.render_enabled = bool(self.cfg.get("render", {}).get("enabled", False))
        self.gaussian_opacity = float(cfg.get("joint", {}).get("gaussian_opacity", .5))
        self._render_report = {}
        self.window_size = int(self.cfg.get("window_frames", 5))
        self.temperature = float(self.cfg.get("temperature", .25))
        self.prior_weight = float(self.cfg.get("prior_weight", .05))
        self.prior_floor = float(self.cfg.get("prior_floor", .001))
        self.max_cost = float(self.cfg.get("max_cost", 1.))
        self.max_fit_cost = float(self.cfg.get("max_fit_cost", 1.))
        self.max_weight = float(self.cfg.get("max_weight", 1.))
        self.min_probability = float(self.cfg.get("min_probability", .6))
        self.min_margin = float(self.cfg.get("min_margin", .1))
        if self.window_size < 2:
            raise ValueError("window_frames must count at least two images")
        positive = (self.temperature, self.prior_floor, self.max_cost,
                    self.max_fit_cost, self.max_weight)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("temperature, floors, costs, and max_weight must be finite and positive")
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        if not (0 <= self.min_probability <= 1 and 0 <= self.min_margin <= 1):
            raise ValueError("probability and margin thresholds must be in [0,1]")
        self.object_ids = []
        self._initial = {}
        self._state = {}
        self._labels = {}
        self._active_gids = np.empty(0, np.int64)
        self._active_probabilities = np.empty((0, 0), float)
        self._active_labels = np.empty(0, np.int64)
        self._frames = set()
        self._active = {}
        self._updates = 0
        self._last_report = {}

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
               q_motion_initial=None, render_observations=None):
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
                    "report_json", "track_ids"}
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

        # No object or Gaussian state changes before all input validation passes.
        self.object_ids.extend(int(o) for o in oids if int(o) not in self.object_ids)
        columns = {o: j for j, o in enumerate(self.object_ids)}
        initial_columns = np.asarray([columns[int(o)] for o in oids])
        for gid, row in zip(gids, q):
            gid = int(gid)
            if gid not in self._initial:
                mass = float(row.sum())
                normalized = row / mass if mass > 0 else np.full(k, 1. / k)
                initial = np.full(len(self.object_ids), self.prior_floor)
                initial[initial_columns] = np.maximum(normalized, self.prior_floor)
                self._initial[gid] = initial
        snapshot = dict(first=first, second=second, gaussian_ids=gids,
                        object_ids=oids, q_init=q, costs=c, weights=w,
                        q_motion_initial=motion_q, track_ids=tracks, payload=extra,
                        render_observations=observations,
                        output_dir=None if output_dir is None else Path(output_dir))
        self._active[(first, second)] = snapshot
        self._frames.update((first, second))
        frames = sorted(self._frames)[-self.window_size:]
        frame_set = set(frames)
        self._active = {pair: snap for pair, snap in self._active.items()
                        if pair[0] in frame_set and pair[1] in frame_set}
        self._updates += 1
        self._optimize()
        self._last_report = dict(
            updates=self._updates, window_frames=frames,
            window_pairs=[list(pair) for pair in sorted(self._active)],
            active_gaussians=len({int(g) for snap in self._active.values()
                                  for g in snap["gaussian_ids"]}),
            total_gaussians=len(self._initial), object_ids=list(self.object_ids),
            temperature=self.temperature, prior_weight=self.prior_weight,
            objective="bounded_pairwise_cost + prior_weight*KL - temperature*entropy",
            motion_probabilities_optimized=False)
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
        row[:len(initial)] = initial
        return row / row.sum()

    def _optimize(self):
        count = len(self.object_ids)
        columns = {o: j for j, o in enumerate(self.object_ids)}
        active_gids = (np.unique(np.concatenate([snap["gaussian_ids"] for snap in self._active.values()]))
                       if self._active else np.empty(0, np.int64))
        energy = np.zeros((len(active_gids), count))
        support = np.zeros_like(energy)
        good_fit = np.zeros_like(energy)
        for pair in sorted(self._active):
            snap = self._active[pair]
            col = np.asarray([columns[int(o)] for o in snap["object_ids"]])
            rows = np.searchsorted(active_gids, snap["gaussian_ids"])
            costs, weights = snap["costs"], snap["weights"]
            valid = np.isfinite(costs) & (weights > 0)
            valid_count = valid.sum(axis=1)
            any_fit = np.any(valid & (costs < self.max_fit_cost), axis=1)
            # All-bad/occluded comparisons are outliers, not association
            # evidence. They cannot override reliable earlier observations.
            comparable = (valid_count >= 2) & any_fit
            clipped = np.minimum(costs, self.max_cost)
            reliability = np.minimum(weights, self.max_weight)
            # Gaussian IDs are unique within a snapshot: these vectorized
            # indexed updates cannot lose repeated-index accumulation. Only
            # the small candidate-pair loop is in Python, never the point loop.
            for j in range(len(col)):
                for k in range(j + 1, len(col)):
                    visible = np.flatnonzero(comparable & valid[:, j] & valid[:, k])
                    if not len(visible):
                        continue
                    weight = (np.minimum(reliability[visible, j], reliability[visible, k]) /
                              valid_count[visible])
                    difference = weight * (clipped[visible, j] - clipped[visible, k])
                    indices = rows[visible]
                    energy[indices, col[j]] += difference
                    energy[indices, col[k]] -= difference
                    support[indices, col[j]] += weight
                    support[indices, col[k]] += weight
                visible = np.flatnonzero(comparable & valid[:, j] & (costs[:, j] < self.max_fit_cost))
                good_fit[rows[visible], col[j]] += reliability[visible, j]
            # With exactly one global object there is no categorical decision
            # to optimize. Its actual absolute fit can nevertheless establish
            # membership; an absent/bad fit still yields an unknown hard label.
            if count == 1:
                visible = np.flatnonzero(valid[:, 0] & any_fit)
                support[rows[visible], 0] += reliability[visible, 0]
                good_fit[rows[visible], 0] += reliability[visible, 0]
        priors = (np.stack([self._prior(int(gid)) for gid in active_gids])
                  if len(active_gids) else np.empty_like(energy))
        render_support = np.zeros_like(support, dtype=bool)
        if self.render_enabled and len(active_gids):
            # Warm starts do not change the objective's fixed first prior.
            initial = priors.copy()
            for row, gid in enumerate(active_gids):
                previous = self._state.get(int(gid))
                if previous is not None:
                    initial[row] = self.prior_floor
                    initial[row, :len(previous)] = previous
                    initial[row] /= initial[row].sum()
            self._active_gids = active_gids
            self._render_initial_probabilities = initial
            self._render_energy = energy
            from object_render_optimizer import optimize_render_window
            result = optimize_render_window(self, self.cfg)
            probabilities = np.asarray(result["probabilities"], np.float64)
            render_support = np.asarray(result["render_support"], bool)
            if (probabilities.shape != energy.shape or render_support.shape != energy.shape or
                    not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
                    not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
                raise ValueError("Render optimizer returned invalid shared object probabilities/support")
            self._render_report = result["report"]
        else:
            denominator = self.temperature + self.prior_weight
            logits = (self.prior_weight * np.log(priors) - energy) / denominator
            if len(active_gids):
                logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
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
        self._active_gids = active_gids
        self._active_probabilities = probabilities
        self._active_labels = labels
        # Compact finalized posteriors remain available if a surface leaves
        # the window; the much larger observations/payloads are discarded.
        for gid, probability, label in zip(active_gids, probabilities, labels):
            self._state[int(gid)] = probability.copy()
            self._labels[int(gid)] = int(label)

    def get_state(self, gaussian_ids=None):
        """Return copied global columns, probabilities, and uncertain hard IDs."""
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
            # Motion payload columns refer to this pair's original candidates.
            # Match them explicitly to the expanding global probability columns.
            if "H_object" in arrays:
                poses = np.full((len(self.object_ids), 4, 4), np.nan)
                valid = np.zeros(len(self.object_ids), bool)
                original = snap["payload"]["H_object"]
                original_valid = snap["payload"].get(
                    "object_model_valid", np.isfinite(original).all(axis=(1, 2)))
                columns = {o: j for j, o in enumerate(self.object_ids)}
                for j, oid in enumerate(snap["object_ids"]):
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
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)

    def to_dict(self):
        """Small JSON-serializable window summary (arrays live in NPZ files)."""
        return json.loads(json.dumps(self._last_report))

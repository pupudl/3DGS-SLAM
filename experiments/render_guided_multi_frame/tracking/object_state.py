"""Persistent object identities and revisable track priors for pair updates.

The pair optimizer owns geometry.  This registry owns identity and history:
local motion indices may change every pair, while object IDs keep their past.
Historical tracks fit a fresh transform from the *current* pair's observations;
the previous pair's transform is never copied to the new time interval.
Streaming mode archives object intervals and retains only bounded recent
history plus track priors reachable from the current feature cache.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

from frontend.hybrid_motion import (_core_labels, _refit, _rigid, motion_costs,
                           motion_probabilities)


class ObjectRegistry:
    """Associate current motion components through continuing feature tracks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracking_cfg = cfg.get("tracking", {})
        self.identity_cfg = self.tracking_cfg.get("motion_identity", {})
        self.prior_weight = float(self.tracking_cfg.get("prior_weight", .25))
        self.window_enabled = bool(cfg.get("object_window", {}).get("enabled", False))
        self.lifecycle_cfg = cfg.get("object_lifecycle", {})
        self.lifecycle_enabled = bool(self.lifecycle_cfg.get(
            "enabled", cfg.get("sequence", {}).get("streaming", False)))
        self.missing_grace_frames = int(self.lifecycle_cfg.get("missing_grace_frames", 2))
        self.history_limit = int(self.lifecycle_cfg.get("history_limit", 32))
        self.event_limit = int(self.lifecycle_cfg.get("event_limit", 256))
        if self.lifecycle_enabled and (self.missing_grace_frames < 1 or
                                       self.history_limit < 1 or self.event_limit < 1):
            raise ValueError("Object lifecycle grace and history limits must be positive")
        self.objects = {}
        self.events = []
        self.pairs = []
        self.track_objects = {}
        self.track_probabilities = {}
        self._next_object = 0
        self._observation_step = -1
        self._last_observation_frame = None
        self._pair_count = 0
        self._event_count = 0
        self._observed_frame_count = 0
        self._current_observed_ids = set()
        self._finalized = False
        self._candidate_ids = set()
        self._retired_count = 0
        self._event_log_path = None
        self._pair_source_frame = None
        self._identity_core_pair = None
        self._identity_cores = {}
        # Independent geometry proposals survive a provisional soft-window
        # posterior for one adjacent pair. They never overwrite final labels.
        self._geometric_identity_pair = None
        self._geometric_identity_tracks = {}
        # A target visibility test can be corrected when the same image
        # becomes the next source. Keep only those just-retired hard track
        # identities until that source has been processed, without adding
        # retired columns to the normal candidate pool.
        self._retirement_track_objects = {}
        self._retirement_track_probabilities = {}

    def set_event_log(self, path):
        """Append every subsequent event to a durable JSONL journal.

        Install before the first pair so bounded in-memory history cannot
        discard events before they reach disk. Existing nonempty journals
        are rejected to avoid silently mixing separate sequence runs.
        """
        if self._event_count:
            raise ValueError("Install the object event log before recording any events")
        event_path = Path(path)
        if event_path.exists() and event_path.stat().st_size:
            raise ValueError("Object event log already contains events")
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_log_path = event_path

    def _event(self, row):
        if self._event_log_path is not None:
            with self._event_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event_index=self._event_count+1, **row),
                                        sort_keys=True)+"\n")
        self.events.append(row)
        self._event_count += 1
        if self.lifecycle_enabled:
            del self.events[:-self.event_limit]

    def candidate_object_ids(self):
        """Return identities eligible for the bounded, current optimization.

        Retired object metadata remains available in ``objects``. Its column
        must not be added to every future Gaussian probability matrix.
        """
        return sorted(self._candidate_ids if self.lifecycle_enabled else self.objects)

    def _eligible_track_objects(self):
        candidates = set(self.candidate_object_ids())
        tracks = (dict(self._retirement_track_objects) if self._correction_tracks_available() else {})
        tracks.update({track: oid for track, oid in self.track_objects.items() if oid in candidates})
        return tracks

    def _correction_tracks_available(self):
        return (self.lifecycle_enabled and self._last_observation_frame is not None and
                self._pair_source_frame == self._last_observation_frame)

    def _begin_pair(self, first):
        self._pair_source_frame = int(first)
        if (self._geometric_identity_pair is not None and
                int(first) != self._geometric_identity_pair[1]):
            self._geometric_identity_pair = None
            self._geometric_identity_tracks.clear()
        if self.lifecycle_enabled and first != self._last_observation_frame:
            self._retirement_track_objects.clear()
            self._retirement_track_probabilities.clear()

    def _prune_geometric_identity(self, retained_tracks=None):
        eligible = set(self.candidate_object_ids())
        keep = None if retained_tracks is None else set(map(int, retained_tracks))
        self._geometric_identity_tracks = {
            track: oid for track, oid in self._geometric_identity_tracks.items()
            if oid in eligible and (keep is None or track in keep)}

    def _identity_track_objects(self, models, labels, track_ids, anchors, K, motion_cfg):
        """Merge only adjacent, freshly revalidated geometric identity votes."""
        tracks = self._eligible_track_objects()
        report = dict(enabled=bool(self.identity_cfg.get('geometric_identity_ledger', False)),
                      previous_pair=(list(self._geometric_identity_pair)
                                     if self._geometric_identity_pair is not None else None),
                      revalidated_track_ids=[], conflicting_track_ids=[],
                      accepted_tracks=[], rejected_tracks=[])
        self._prune_geometric_identity()
        if (not report['enabled'] or not self.identity_cfg.get('enabled', True) or
                self._geometric_identity_pair is None or
                self._geometric_identity_pair[1] != self._pair_source_frame or
                anchors is None or K is None or not len(models)):
            return tracks, report
        hcfg = (motion_cfg or self.cfg.get('motion', {}))
        hcfg = hcfg.get('hybrid', hcfg)
        _, current = _core_labels(anchors, models, K, hcfg)
        confidence = np.asarray(anchors.get('confidence', np.ones(len(current)))).reshape(-1)
        supported, supported_rows = {}, {}
        observed_tracks = set(map(int, np.asarray(track_ids)[np.asarray(track_ids) >= 0]))
        for row, (track, local) in enumerate(zip(track_ids, labels)):
            track, local = int(track), int(local)
            if track < 0 or local < 0:
                continue
            valid = (current[row] == local and np.isfinite(confidence[row]) and
                     confidence[row] > 0)
            supported.setdefault(track, set()).add(local if valid else -1)
            if valid:
                supported_rows.setdefault(track, []).append(row)
        for track, oid in self._geometric_identity_tracks.items():
            locals_ = supported.get(track, set())
            if len(locals_) != 1 or -1 in locals_:
                reason = ('not_observed_in_current_pair' if track not in observed_tracks else
                          'conflicting_current_core' if len(locals_) > 1 else
                          'current_core_unreliable')
                report['rejected_tracks'].append(dict(
                    track_id=int(track), object_id=int(oid), reason=reason))
                continue
            old = tracks.get(track)
            # Current hard foreground conflicts cannot be resolved by a
            # previous proposal. O0 remains revisable by the motion gate.
            if old is not None and old != oid and old != 0:
                report['conflicting_track_ids'].append(track)
                report['rejected_tracks'].append(dict(
                    track_id=int(track), object_id=int(oid),
                    current_foreground_object_id=int(old), reason='current_foreground_conflict'))
                continue
            local = next(iter(locals_))
            discovery = self._identity_cores.get(local)
            in_discovery = (any(tuple(map(float, anchors['matches'][row])) in discovery
                                for row in supported_rows[track]) if discovery is not None else None)
            tracks[track] = oid
            report['revalidated_track_ids'].append(track)
            report['accepted_tracks'].append(dict(
                track_id=int(track), object_id=int(oid), local_model=int(local),
                in_initial_discovery_core=in_discovery,
                reason=('accepted_final_core_outside_discovery' if in_discovery is False else
                        'accepted_current_core')))
        return tracks, report

    def _remember_geometric_identity(self, first, second, models, labels,
                                     track_ids, mapping, anchors, K, motion_cfg):
        self._geometric_identity_pair = None
        self._geometric_identity_tracks = {}
        if (not self.identity_cfg.get('geometric_identity_ledger', False) or
                not self.identity_cfg.get('enabled', True) or anchors is None or
                K is None or not len(models)):
            return
        hcfg = (motion_cfg or self.cfg.get('motion', {}))
        hcfg = hcfg.get('hybrid', hcfg)
        minimum = int(hcfg.get('min_support', 6))
        _, current = _core_labels(anchors, models, K, hcfg)
        confidence = np.asarray(anchors.get('confidence', np.ones(len(current)))).reshape(-1)
        support = {}
        for row, (track, local) in enumerate(zip(track_ids, labels)):
            track, local = int(track), int(local)
            if (track >= 0 and local >= 0 and current[row] == local and
                    np.isfinite(confidence[row]) and confidence[row] > 0):
                support.setdefault(local, set()).add(track)
        identities = {}
        for local, tracks in support.items():
            oid = int(mapping[local])
            if oid == 0 or len(tracks) < minimum:
                continue
            for track in tracks:
                identities.setdefault(track, set()).add(oid)
        self._geometric_identity_pair = (int(first), int(second))
        self._geometric_identity_tracks = {
            track: next(iter(oids)) for track, oids in identities.items() if len(oids) == 1}
        self._prune_geometric_identity()

    def _track_prior(self, track):
        prior = self.track_probabilities.get(track)
        if prior is None and self._correction_tracks_available():
            prior = self._retirement_track_probabilities.get(track)
        return prior

    def _prune_track_probabilities(self, retained_tracks=None):
        self._prune_geometric_identity(retained_tracks)
        eligible = set(self.candidate_object_ids())
        keep = (set(self.track_objects) | set(self.track_probabilities) if retained_tracks is None
                else set(map(int, retained_tracks)))
        self.track_objects = {track: oid for track, oid in self.track_objects.items()
                              if track in keep and oid in eligible}
        probabilities = {}
        for track, prior in self.track_probabilities.items():
            if track not in keep:
                continue
            prior = {oid: float(p) for oid, p in prior.items() if oid in eligible and p > 0}
            mass = sum(prior.values())
            if mass > 0:
                probabilities[track] = {oid: p/mass for oid, p in prior.items()}
        self.track_probabilities = probabilities

    def retain_tracks(self, track_ids):
        """Keep only priors reachable from the feature cache / active window.

        Per-object track lists describe retained support in streaming mode;
        historical object identity and its observed interval remain archived.
        """
        retained = set(map(int, track_ids)) - {-1}
        self._prune_geometric_identity(retained)
        if not self.lifecycle_enabled:
            return
        self._prune_track_probabilities(retained)
        self._retirement_track_objects = {
            track: oid for track, oid in self._retirement_track_objects.items() if track in retained}
        self._retirement_track_probabilities = {
            track: prior for track, prior in self._retirement_track_probabilities.items()
            if track in retained}
        for record in self.objects.values():
            record["track_ids"] = [track for track in record["track_ids"] if track in retained]

    def _confirm_observation(self, frame, oid):
        record = self.objects[oid]
        if record["status"] == "retired":
            # Source evidence for the previous target can improve on that
            # pair's target visibility test. Permit correction within this
            # same processed frame, never historical reidentification.
            if (record["retired_at_step"] != self._observation_step or
                    record["retirement_reason"] != "missing"):
                return
            self._event(dict(type="retirement_corrected", object_id=oid,
                             frame=int(frame), step=self._observation_step))
            self._candidate_ids.add(oid)
            self._retired_count -= 1
            # A previous window commit excludes retired columns. Restore the
            # reliable identity only now that current source observations
            # have actually corrected this retirement.
            for track, old_oid in list(self._retirement_track_objects.items()):
                if old_oid == oid:
                    if self.track_objects.get(track, oid) == oid:
                        self.track_objects[track] = oid
                        self.track_probabilities[track] = dict(
                            self._retirement_track_probabilities.get(track, {oid: 1.}))
                    del self._retirement_track_objects[track]
                    self._retirement_track_probabilities.pop(track, None)
            record.update(retired_at_frame=None, retired_at_step=None,
                          end_frame=None, retirement_reason=None)
        if not record["confirmed"]:
            record.update(confirmed=True, first_frame=int(frame),
                          first_observed_frame=int(frame), first_observed_step=self._observation_step)
            self._event(dict(type="birth", object_id=oid, frame=int(frame),
                             step=self._observation_step,
                             proposed_first_frame=record["proposed_first_frame"]))
        elif record["status"] == "missing":
            self._event(dict(type="reactivation", object_id=oid, frame=int(frame),
                             step=self._observation_step, missing_frames=record["missing_frames"]))
        record.update(status="active", last_frame=int(frame), last_observed_frame=int(frame),
                      last_observed_step=self._observation_step, missing_frames=0)
        record["observation_count"] += 1
        record["frames"].append(int(frame))
        del record["frames"][:-self.history_limit]

    def _retire(self, oid, frame, reason):
        record = self.objects[oid]
        if reason == "missing":
            for track, old_oid in self.track_objects.items():
                if old_oid == oid:
                    self._retirement_track_objects[track] = oid
                    self._retirement_track_probabilities[track] = dict(
                        self.track_probabilities.get(track, {oid: 1.}))
        record.update(status="retired", retired_at_frame=int(frame),
                      retired_at_step=self._observation_step,
                      end_frame=record["last_observed_frame"], retirement_reason=reason)
        self._candidate_ids.discard(oid)
        self._geometric_identity_tracks = {
            track: old for track, old in self._geometric_identity_tracks.items() if old != oid}
        self._retired_count += 1
        self._event(dict(type="retirement" if record["confirmed"] else "proposal_discarded",
                         object_id=oid, frame=int(frame), step=self._observation_step,
                         last_observed_frame=record["last_observed_frame"],
                         end_frame=record["end_frame"], reason=reason))

    def observe_frame(self, frame, object_ids, step=None):
        """Commit current-frame presence after the final ownership posterior.

        Call once for each processed frame with confident *current* observations,
        not all object labels in the overlapping optimization window. Dataset
        frame-number gaps do not age objects: each new call advances one frame.
        Repeating the newest frame adds positive evidence without ageing any
        object again, so a pair's source and the preceding target can overlap.
        """
        if not self.lifecycle_enabled:
            return
        if self._finalized:
            raise ValueError("Cannot observe frames after finalizing the object registry")
        frame = int(frame)
        observed = set(map(int, object_ids)) - {-1}
        if observed - self.objects.keys():
            raise ValueError("Observed object IDs must already exist in the registry")
        if self._last_observation_frame is not None and frame < self._last_observation_frame:
            raise ValueError("Object observations must follow processed-frame order")
        repeated = frame == self._last_observation_frame
        if repeated:
            if step is not None and int(step) != self._observation_step:
                raise ValueError("A repeated frame must use its existing observation step")
        else:
            next_step = self._observation_step+1 if step is None else int(step)
            if next_step <= self._observation_step:
                raise ValueError("Object observation steps must increase for new frames")
            self._observation_step = next_step
            self._observed_frame_count += 1
            self._last_observation_frame = frame
            self._current_observed_ids = set()
            self._retirement_track_objects.clear()
            self._retirement_track_probabilities.clear()
        for oid in sorted(observed - self._current_observed_ids):
            self._confirm_observation(frame, oid)
        self._current_observed_ids.update(observed)
        newly_retired = []
        if not repeated:
            for oid in self.candidate_object_ids():
                record = self.objects[oid]
                if oid == 0 or oid in observed or record["status"] == "retired":
                    continue
                record["missing_frames"] += 1
                if record["status"] == "active":
                    record["status"] = "missing"
                    self._event(dict(type="missing", object_id=oid, frame=frame,
                                     step=self._observation_step,
                                     last_observed_frame=record["last_observed_frame"]))
                if record["missing_frames"] >= self.missing_grace_frames:
                    self._retire(oid, frame, "missing")
                    newly_retired.append(oid)
        self._prune_track_probabilities()
        candidates = self.candidate_object_ids()
        return dict(
            frame=frame, step=self._observation_step,
            observed_object_ids=sorted(
                oid for oid in self._current_observed_ids
                if self.objects[oid]["confirmed"] and
                self.objects[oid]["last_observed_frame"] == frame and
                self.objects[oid]["status"] == "active"),
            active_object_ids=[oid for oid in candidates if self.objects[oid]["status"] == "active"],
            missing_object_ids=[oid for oid in candidates if self.objects[oid]["status"] == "missing"],
            provisional_object_ids=[oid for oid in candidates
                                    if self.objects[oid]["status"] == "provisional"],
            newly_retired_object_ids=newly_retired, retired_object_count=self._retired_count)

    def discard_empty_objects(self, frame, object_ids):
        """Retire IDs rejected by a completed ownership window immediately.

        Retired metadata remains an audit of past proposals/observations; it
        supplies no candidate, track prior, or same-frame revival opportunity.
        """
        if not self.lifecycle_enabled:
            return
        discarded = set(map(int, object_ids)) - {0}
        for oid in sorted(discarded & set(self.candidate_object_ids())):
            self._retire(oid, frame, 'no_gaussians')
        self._current_observed_ids.difference_update(discarded)
        self._retirement_track_objects = {
            track: oid for track, oid in self._retirement_track_objects.items() if oid not in discarded}
        self._retirement_track_probabilities = {
            track: {oid: p for oid, p in prior.items() if oid not in discarded}
            for track, prior in self._retirement_track_probabilities.items()}
        self._prune_track_probabilities()

    def finalize_sequence(self):
        """Close live dynamic intervals at their last observation at sequence end."""
        if not self.lifecycle_enabled or self._finalized:
            return
        if self._last_observation_frame is not None:
            for oid in self.candidate_object_ids():
                record = self.objects[oid]
                if oid != 0 and record["status"] != "retired":
                    self._retire(oid, self._last_observation_frame, "sequence_end")
            self._prune_track_probabilities()
        self._finalized = True
        self._retirement_track_objects.clear()
        self._retirement_track_probabilities.clear()
        self._geometric_identity_pair = None
        self._geometric_identity_tracks.clear()

    def _votes(self, labels, track_ids, count, track_objects=None):
        track_objects = (self._eligible_track_objects() if track_objects is None else track_objects)
        previous = sorted(set(track_objects.values()))
        index = {oid: i for i, oid in enumerate(previous)}
        votes = np.zeros((count, len(previous)), np.int64)
        # One track casts at most one vote, even if a saved pair contains a
        # repeated observation from initial and supplemental matching.
        seen = set()
        for label, track in zip(labels, track_ids):
            track, label = int(track), int(label)
            if label < 0 or track < 0 or track in seen:
                continue
            seen.add(track)
            oid = track_objects.get(track)
            if oid is not None:
                votes[label, index[oid]] += 1
        return previous, votes

    @staticmethod
    def _association(previous, votes, allow_many=False):
        mapping = {}
        if votes.size:
            if allow_many:
                # Local motion discovery may split one persistent object into
                # several hypotheses. Do not manufacture a new object merely
                # because another local candidate has already used its ID.
                for row, evidence in enumerate(votes):
                    column = int(np.argmax(evidence))
                    if evidence[column] > 0 and np.count_nonzero(evidence == evidence[column]) == 1:
                        mapping[int(row)] = int(previous[column])
                return mapping
            rows, columns = linear_sum_assignment(-votes)
            for row, column in zip(rows, columns):
                if votes[row, column] > 0:
                    mapping[int(row)] = int(previous[column])
        return mapping

    def _motion_identity_gate(self, models, labels, q_anchor, track_ids, anchors=None,
                              K=None, motion_cfg=None, background_local=None):
        """Reject historical identity votes contradicted by current motion.

        A track's old identity is a proposal, rather than proof that its current
        motion is still background. Compare each candidate on its own reliable
        support against the strongest *current-pair* representative of the old
        object. Nothing uses a transform from a previous time interval. Several
        compatible local fragments may continue the same object.

        Calls without metric observations retain the historical API behavior.
        The conservative defaults require a well fitted candidate and a large,
        consistent improvement before allowing a split from an existing object.
        """
        enabled = bool(self.identity_cfg.get(
            "enabled", self.cfg.get("sequence", {}).get("streaming", False)))
        report = dict(enabled=enabled, available=anchors is not None and K is not None,
                      rejected=[], representatives={})
        forbidden = set()
        if not enabled or anchors is None or K is None or not len(models):
            return forbidden, report
        hcfg = (motion_cfg or self.cfg.get("motion", {}))
        hcfg = hcfg.get("hybrid", hcfg)
        costs = motion_costs(anchors, models, K, hcfg)
        track_objects, geometric_report = self._identity_track_objects(
            models, labels, track_ids, anchors, K, motion_cfg)
        report['geometric_identity_ledger'] = geometric_report
        # Geometry comparisons below keep discovery's undiluted evidence.
        # Identity alone can additionally use a previously certified same-ID
        # track that became an exclusive metric core after the current refine.
        ledger_identity_support = {}
        for item in geometric_report['accepted_tracks']:
            key = (item['local_model'], item['object_id'])
            ledger_identity_support.setdefault(key, set()).add(item['track_id'])
        report['identity_support'] = {}
        previous, votes = self._votes(labels, track_ids, len(models), track_objects)
        # Soft-only tracks are also identity clues, but never become reliable
        # geometric support merely because their conditional q is large.
        for oid in sorted(set(track_objects.values())):
            if oid not in previous:
                previous.append(oid)
                votes = np.column_stack((votes, np.zeros(len(models), np.int64)))
        rank = votes.astype(float)
        for i, (label, track) in enumerate(zip(labels, track_ids)):
            oid = track_objects.get(int(track))
            if label < 0 and oid in previous:
                rank[:, previous.index(oid)] += np.asarray(q_anchor[i])
        own_winner = np.asarray(q_anchor).argmax(axis=1)
        minimum = int(self.identity_cfg.get("min_support", hcfg.get("min_support", 6)))
        max_cost = float(self.identity_cfg.get("max_candidate_cost", hcfg.get("outlier_cost", 1.)))
        min_old = float(self.identity_cfg.get("min_incompatible_cost", 4.))
        min_gain = float(self.identity_cfg.get("min_cost_gain", 3.))
        min_ratio = float(self.identity_cfg.get("min_cost_ratio", 3.))
        fraction = float(self.identity_cfg.get("min_incompatible_fraction", .7))
        minimum_identity = int(self.identity_cfg.get("min_identity_support", 3))
        def unique_support(rows):
            seen, result = set(), []
            for row in rows:
                track = int(track_ids[row])
                key = (("track", track) if track >= 0 else
                       ("match", *map(float, anchors['matches'][row])))
                if key not in seen:
                    result.append(row)
                    seen.add(key)
            return np.asarray(result, np.int64)
        # Reliability belongs to the current motion, not to the historical
        # identity being tested. Do not recompute these costs for every old ID.
        reliable_support = []
        core_rows = {}
        if self._identity_cores:
            for row, match in enumerate(anchors['matches']):
                key = tuple(map(float, match))
                for local, matches in self._identity_cores.items():
                    if key in matches:
                        core_rows.setdefault(local, []).append(row)
        report['core_evidence'] = {}
        for local, model in enumerate(models):
            _, individual = _core_labels(anchors, [model], K, hcfg)
            reliable = (individual == 0) & (costs[:, local] <= max_cost)
            if local in self._identity_cores:
                # Terminal discovery intentionally recovers only unexplained
                # gray points. Reclassifying its other inliers can add easy
                # background points and dilute a small moving core. Keep that
                # pair's evidence set, rechecking the *current* pose/depth fit.
                rows = np.asarray(core_rows.get(local, []), np.int64)
                support = unique_support(rows[reliable[rows]])
                report['core_evidence'][str(local)] = dict(
                    source='discovery_core_revalidated',
                    discovered_matches=len(self._identity_cores[local]),
                    reliable_unique_support=len(support))
            else:
                support = unique_support(np.flatnonzero((np.asarray(labels) == local) & reliable))
                if len(support) < minimum:
                    support = unique_support(np.flatnonzero(((np.asarray(labels) == local) |
                        ((np.asarray(labels) < 0) & (own_winner == local))) & reliable))
            reliable_support.append(support)
        for column, oid in enumerate(previous):
            if rank[:, column].max(initial=0.) <= 0:
                continue
            # Prefer maximum continuing support, then current core support.
            representative = max(range(len(models)), key=lambda k: (
                float(rank[k, column]), int(np.count_nonzero(labels == k)), -k))
            if oid == 0 and background_local is None:
                # Fresh background features can outnumber continuing ones.
                # Rank broad current support before track history for O0,
                # while protecting candidates identified as known vehicles.
                other = [j for j, old in enumerate(previous) if old != 0]
                compatible = [k for k in range(len(models)) if not (
                    other and np.max(rank[k, other]) > rank[k, column])]
                if compatible:
                    representative = max(compatible, key=lambda k: (
                        int(np.count_nonzero(labels == k)), float(rank[k, column]), -k))
            if oid == 0 and background_local is not None:
                local = int(background_local)
                # A caller's optimizer reference can be a vehicle. Do not
                # override positive, non-background track evidence with it.
                other = [j for j, old in enumerate(previous) if old != 0]
                if (0 <= local < len(models) and
                        not (other and np.max(rank[local, other]) > rank[local, column])):
                    representative = local
            report["representatives"][str(oid)] = int(representative)
            for local in range(len(models)):
                # Independent hard core, or geometrically reliable winners
                # whose saved label is grey, can validate the new motion.
                # Prefer exclusive current core to the wider conditional-q
                # support. Many ambiguous background points can have a slight
                # preference for a moving candidate without belonging to it.
                support = reliable_support[local]
                original_identity = {int(track_ids[i]) for i in support
                    if int(track_ids[i]) >= 0 and track_objects.get(int(track_ids[i])) == oid}
                additional_identity = ledger_identity_support.get((local, oid), set()) - original_identity
                identity_support = len(original_identity | additional_identity)
                report['identity_support'].setdefault(str(oid), {})[str(local)] = dict(
                    discovery_core_tracks=len(original_identity),
                    additional_final_core_track_ids=sorted(additional_identity),
                    reliable_identity_tracks=identity_support)
                if oid != 0 and rank[local, column] > 0 and identity_support < minimum_identity:
                    forbidden.add((int(local), int(oid)))
                    report["rejected"].append(dict(
                        object_id=int(oid), local_model=int(local),
                        representative_local=int(representative), support=int(len(support)),
                        identity_support=int(identity_support),
                        reason="insufficient_compatible_identity_support"))
                    continue
                if local == representative:
                    continue
                if len(support) < minimum:
                    continue
                own, old = costs[support, local], costs[support, representative]
                incompatible = ((old >= min_old) & (old-own >= min_gain) &
                                (old >= min_ratio*np.maximum(own, 1e-6)))
                reason = 'motion_incompatible'
                if (oid == 0 and self.identity_cfg.get('split_background_on_independent_core', False)
                        and float(np.mean(incompatible)) < fraction):
                    # A new moving object's tracks can legitimately all have
                    # belonged to O0. An independently fitted, metric core
                    # that O0 cannot explain must not be absorbed solely by
                    # historical votes. Use discovery's existing gain/core
                    # tests rather than the much larger emergency split cost.
                    outlier = float(hcfg.get('outlier_cost', 1.))
                    margin = float(hcfg.get('core_margin', .1))
                    independent = ((old >= outlier) & (own < outlier) &
                                   (old-own >= margin) &
                                   (old >= min_ratio*np.maximum(own, 1e-6)))
                    quality = np.sqrt(np.clip(np.asarray(anchors.get(
                        'confidence', np.ones(len(labels)))).reshape(-1)[support], .05, 1.))
                    gain = float(np.sum(np.maximum(0., np.minimum(old, outlier)-own)*quality))
                    if (float(np.mean(independent)) >= fraction and
                            gain >= float(hcfg.get('model_penalty', minimum*.20))):
                        incompatible = independent
                        reason = 'independent_current_core_not_explained_by_background'
                if float(np.mean(incompatible)) < fraction:
                    continue
                forbidden.add((int(local), int(oid)))
                report["rejected"].append(dict(
                    object_id=int(oid), local_model=int(local),
                    representative_local=int(representative), support=int(len(support)),
                    reason=reason,
                    candidate_cost_median=float(np.median(own)),
                    inherited_cost_median=float(np.median(old)),
                    incompatible_fraction=float(np.mean(incompatible))))
        return forbidden, report

    @staticmethod
    def _filtered_votes(previous, votes, forbidden):
        votes = votes.copy()
        for local, oid in forbidden:
            if oid in previous:
                votes[local, previous.index(oid)] = 0
        return votes

    def _soft_association(self, mapping, labels, q_anchor, track_ids, forbidden=()):
        """Fill unmatched identities with continuing grey-track preferences.

        This is only an identity clue. A large conditional probability can
        still accompany a poor absolute geometric fit: grey observations stay
        grey and do not become hard track/object evidence through this step.
        Hard associations are fixed before this remaining-submatrix match.
        """
        mapping = dict(mapping)
        remaining = [k for k in range(q_anchor.shape[1]) if k not in mapping]
        track_objects = self._eligible_track_objects()
        previous = sorted(set(track_objects.values()) -
                          (set() if self.window_enabled else set(mapping.values())))
        if not remaining or not previous:
            return mapping, []
        local_index = {k: i for i, k in enumerate(remaining)}
        object_index = {oid: i for i, oid in enumerate(previous)}
        votes = np.zeros((len(remaining), len(previous)), float)
        supporters = {}
        seen = set()
        for label, q, track in zip(labels, q_anchor, track_ids):
            track = int(track)
            oid = track_objects.get(track)
            if label >= 0 or oid not in object_index or track in seen:
                continue
            seen.add(track)
            winner = int(np.argmax(q))
            second = float(np.partition(q, -2)[-2]) if len(q) > 1 else 0.
            weight = float(q[winner]) - second
            if winner in local_index and weight > 0:
                row, column = local_index[winner], object_index[oid]
                votes[row, column] += weight
                supporters.setdefault((row, column), []).append(track)
        votes = self._filtered_votes(previous, votes,
                                     {(remaining.index(k), oid) for k, oid in forbidden
                                      if k in remaining})
        associated = self._association(previous, votes, self.window_enabled)
        object_index = {oid: i for i, oid in enumerate(previous)}
        records = []
        for row, oid in associated.items():
            column = object_index[oid]
            if votes[row, column] > 0:
                local = remaining[row]
                mapping[local] = oid
                records.append(dict(object_id=int(oid), local_model=int(local),
                                    vote=float(votes[row, column]),
                                    support_track_ids=sorted(supporters[(row, column)])))
        return mapping, records

    def prepare_pair(self, first, second, anchors, K, motion_cfg,
                     discovered_models, discovered_labels, track_ids):
        """Fit present-day warm poses and mix a small, revisable track prior.

        Existing hybrid core/cost rules decide whether historical observations
        support a usable pose.  Independent newly discovered models stay in
        the pool.  This step does not commit any object/track assignments.
        """
        hcfg = motion_cfg.get("hybrid", motion_cfg)
        self._begin_pair(first)
        minimum = int(hcfg.get("min_support", motion_cfg.get("min_support", 6)))
        maximum = int(hcfg.get("max_components", motion_cfg.get("max_components", 8)))
        outlier = float(hcfg.get("outlier_cost", 1.))
        models = [np.asarray(H, float).copy() for H in discovered_models]
        track_ids = np.asarray(track_ids, np.int64)
        labels = np.asarray(discovered_labels, np.int32).copy()
        self._identity_core_pair = (int(first), int(second))
        self._identity_cores = ({local: {tuple(map(float, match))
            for match in anchors['matches'][labels == local]}
            for local in range(len(models))} if self.window_enabled else {})
        discovered_costs = motion_costs(anchors, models, K, hcfg)
        discovered_q, _ = motion_probabilities(discovered_costs, hcfg)
        forbidden, identity_report = self._motion_identity_gate(
            models, labels, discovered_q, track_ids, anchors, K, motion_cfg)
        identity_tracks, _ = self._identity_track_objects(
            models, labels, track_ids, anchors, K, motion_cfg)
        previous, votes = self._votes(labels, track_ids, len(models), identity_tracks)
        associated = self._association(previous, self._filtered_votes(
            previous, votes, forbidden), self.window_enabled)
        object_locals = {}
        for local, oid in associated.items():
            object_locals.setdefault(oid, []).append(local)
        object_local = {oid: locals_[0] for oid, locals_ in object_locals.items()
                        if len(locals_) == 1}
        track_objects = self._eligible_track_objects()
        history = np.asarray([track_objects.get(int(t), -1) for t in track_ids])
        confidence = np.asarray(anchors.get("confidence", np.ones(len(track_ids)))).reshape(-1)
        fit_reports = []
        warm_local = {}

        for oid in sorted(set(history) - {-1}):
            ids = np.flatnonzero(history == oid)
            row = dict(object_id=int(oid), inherited_tracks=int(len(ids)))
            # Old background support may now contain a newly moving object.
            # Do not fit its observations into the background warm transform.
            ids = ids[[((int(labels[i]), int(oid)) not in forbidden) for i in ids]]
            row["motion_compatible_tracks"] = int(len(ids))
            if self.window_enabled and len(object_locals.get(int(oid), [])) > 1:
                # Conflicting local candidates are evidence for the window to
                # resolve, not permission to fit one mixed historical rigid
                # motion and overwrite either current candidate.
                row["action"] = "ambiguous_local_aliases"
                fit_reports.append(row)
                continue
            if len(ids) < minimum:
                row["action"] = "insufficient_current_tracks"
                fit_reports.append(row)
                continue
            # Both rigid initialization and refinement consume 2->3 anchors
            # when processing 2->3, regardless of the saved 1->2 transform.
            H = _rigid(anchors["x0"][ids], anchors["x1"][ids], confidence[ids])
            H = _refit(anchors, H, ids, K, hcfg)
            _, fit_labels = _core_labels(anchors, [H], K, hcfg)
            core = ids[fit_labels[ids] == 0]
            row["fit_core_tracks"] = int(len(core))
            if len(core) < minimum:
                row["action"] = "no_current_core_support"
                fit_reports.append(row)
                continue
            H = _refit(anchors, H, core, K, hcfg)
            local = object_local.get(int(oid))
            replaced = False
            if local is not None:
                # A historical prior should not worsen the evidence that
                # discovered this component in the current pair.
                evidence = np.union1d(core, np.flatnonzero(labels == local))
                old_cost = motion_costs(anchors, [models[local]], K, hcfg)[evidence, 0]
                new_cost = motion_costs(anchors, [H], K, hcfg)[evidence, 0]
                if np.minimum(new_cost, outlier).sum() <= np.minimum(old_cost, outlier).sum():
                    models[local] = H
                    warm_local[local] = int(oid)
                    row.update(action="refit_existing", local_model=int(local))
                    replaced = True
            if not replaced and len(models) < maximum:
                _, proposal_labels = _core_labels(anchors, models + [H], K, hcfg)
                added_support = np.count_nonzero(proposal_labels == len(models))
                if added_support >= minimum:
                    warm_local[len(models)] = int(oid)
                    row.update(action="append_missing", local_model=len(models),
                               competitive_core_tracks=int(added_support))
                    models.append(H)
                else:
                    row["action"] = "already_explained"
            elif not replaced:
                row["action"] = "component_budget"
            fit_reports.append(row)

        models_changed = len(models) != len(discovered_models) or any(
            not np.array_equal(current, discovered)
            for current, discovered in zip(models, discovered_models))
        if models_changed:
            costs, labels = _core_labels(anchors, models, K, hcfg)
        else:
            # No accepted warm update: preserve discovery's terminal labels,
            # including its deliberate restriction to unexplained gray rows.
            costs = discovered_costs
        q_anchor, _ = motion_probabilities(costs, hcfg)
        forbidden, final_identity_report = self._motion_identity_gate(
            models, labels, q_anchor, track_ids, anchors, K, motion_cfg)
        identity_tracks, _ = self._identity_track_objects(
            models, labels, track_ids, anchors, K, motion_cfg)
        previous, votes = self._votes(labels, track_ids, len(models), identity_tracks)
        mapping = self._association(previous, self._filtered_votes(
            previous, votes, forbidden), self.window_enabled)
        # A fitted missing object can initially explain only new tracks; keep
        # its historical ID available to the soft prior without duplicating IDs.
        used = set(mapping.values())
        for local, oid in warm_local.items():
            if (local not in mapping and (local, oid) not in forbidden and
                    (self.window_enabled or oid not in used)):
                mapping[local] = oid
                used.add(oid)
        mapping, soft_identity = self._soft_association(mapping, labels, q_anchor, track_ids,
                                                       forbidden)
        object_locals = {}
        for local, oid in mapping.items():
            object_locals.setdefault(oid, []).append(local)
        mixed = 0
        for i, track in enumerate(track_ids):
            track = int(track)
            prior = self._track_prior(track)
            if prior is None and track in track_objects:
                prior = {track_objects[track]: 1.}
            if not prior:
                continue
            prior_local = np.zeros(len(models), float)
            for oid, probability in prior.items():
                if (int(np.argmax(q_anchor[i])), int(oid)) in forbidden:
                    continue
                if oid in object_locals:
                    local_ids = object_locals[oid]
                    weights = q_anchor[i, local_ids].astype(float)
                    total = weights.sum()
                    weights = weights/total if total > 0 else np.full(len(local_ids), 1/len(local_ids))
                    prior_local[local_ids] += probability*weights
            mass = prior_local.sum()
            if mass > 0:
                prior_local /= mass
                q_anchor[i] = ((1-self.prior_weight)*q_anchor[i] +
                               self.prior_weight*prior_local)
                mixed += 1
        report = dict(
            first=int(first), second=int(second), prior_weight=self.prior_weight,
            inherited_tracks=int(np.count_nonzero(history >= 0)),
            prior_mixed_tracks=mixed, warm_fits=fit_reports,
            motion_identity=final_identity_report,
            motion_identity_before_warm=identity_report,
            core_labels_recomputed=models_changed,
            soft_identity=soft_identity,
            initial_local_to_object=[mapping.get(k, -1) for k in range(len(models))],
            supports=[int(np.count_nonzero(labels == k)) for k in range(len(models))])
        return models, labels, q_anchor, report

    def _create_object(self, first, parent=None):
        oid = self._next_object
        self._next_object += 1
        self.objects[oid] = dict(object_id=oid, first_frame=int(first), last_frame=int(first),
                                 frames=[], track_ids=[], appearances=[],
                                 parent_id=parent, merged_into=None)
        if self.lifecycle_enabled:
            self._candidate_ids.add(oid)
            self.objects[oid].update(
                first_frame=None, last_frame=None,
                status="active" if oid == 0 else "provisional", confirmed=False,
                proposed_first_frame=int(first), first_observed_frame=None,
                last_observed_frame=None, first_observed_step=None, last_observed_step=None,
                missing_frames=0, retired_at_frame=None, retired_at_step=None,
                end_frame=None, retirement_reason=None, observation_count=0,
                appearance_count=0)
        return oid

    def finish_pair(self, first, second, models, q_anchor, track_ids,
                    reference, result_path, *, labels=None, background_local=None,
                    anchors=None, K=None, motion_cfg=None):
        """Record pair identity proposals, preserving track states in window mode.

        Grey anchors do not provide new hard identity evidence. In streaming
        mode these are provisional identities; ``observe_frame`` confirms
        their current presence from the final ownership posterior. Object
        metadata is archived and ``retain_tracks`` can release old priors.
        """
        first, second = int(first), int(second)
        self._begin_pair(first)
        if self._identity_core_pair != (first, second):
            self._identity_core_pair, self._identity_cores = None, {}
        q_anchor = np.asarray(q_anchor, float)
        track_ids = np.asarray(track_ids, np.int64)
        labels = (q_anchor.argmax(axis=1).astype(np.int32) if labels is None else
                  np.asarray(labels, np.int32))
        count = len(models)
        forbidden, identity_report = self._motion_identity_gate(
            models, labels, q_anchor, track_ids, anchors, K, motion_cfg,
            background_local=(int(reference) if background_local is None else background_local))
        identity_tracks, _ = self._identity_track_objects(
            models, labels, track_ids, anchors, K, motion_cfg)
        previous, votes = self._votes(labels, track_ids, count, identity_tracks)
        mapping = self._association(previous, self._filtered_votes(
            previous, votes, forbidden), self.window_enabled)
        mapping, soft_identity = self._soft_association(mapping, labels, q_anchor, track_ids,
                                                       forbidden)
        for row in identity_report["rejected"]:
            # Only journal actual historical identity conflicts; potential
            # incompatibility with unrelated objects is diagnostic context.
            column = previous.index(row["object_id"]) if row["object_id"] in previous else None
            if column is not None and votes[row["local_model"], column] > 0:
                self._event(dict(type="identity_motion_rejected", frames=[first, second], **row))
        for row in soft_identity:
            self._event(dict(type="identity_from_soft_tracks", frames=[first, second], **row))
        first_pair = not self.objects
        if first_pair:
            # Only the first reference creates background object 0. Later
            # references are local optimizer choices, not persistent identity.
            mapping[int(reference)] = self._create_object(first)
        elif self.lifecycle_enabled:
            # The initializer's reference is its broad, dispersed-support
            # background hypothesis. Use that evidence only when tracks have
            # not already identified it as a vehicle (and O0 has no other
            # local candidate). A caller may supply stronger geometric
            # background evidence explicitly through background_local.
            background = int(reference if background_local is None else background_local)
            if background < 0 or background >= count:
                raise ValueError("Background local model is outside the candidate models")
            if (background not in mapping and 0 not in mapping.values() and
                    (background, 0) not in forbidden):
                hard_nonbackground = any(
                    previous[column] != 0 and votes[background, column] > 0 and
                    (background, previous[column]) not in forbidden
                    for column in range(len(previous)))
                soft_nonbackground = any(
                    self.track_objects.get(int(track), 0) != 0 and
                    (background, self.track_objects.get(int(track), 0)) not in forbidden and
                    int(np.argmax(q)) == background and
                    float(q[background]) > (float(np.partition(q, -2)[-2]) if len(q) > 1 else 0.)
                    for label, q, track in zip(labels, q_anchor, track_ids) if label < 0)
                if not hard_nonbackground and not soft_nonbackground:
                    mapping[background] = 0
                    self._event(dict(type="background_reference_fallback", frames=[first, second],
                                     object_id=0, local_model=background,
                                     evidence=("explicit_background_local" if background_local is not None
                                               else "dispersed_support_reference")))
        newly_created = []
        for local in range(count):
            if local in mapping:
                continue
            parent = None
            if votes.shape[1] and votes[local].max() > 0 and (
                    not self.window_enabled or
                    (local, int(previous[int(np.argmax(votes[local]))])) in forbidden):
                parent = int(previous[int(np.argmax(votes[local]))])
            oid = self._create_object(first, parent)
            mapping[local] = oid
            newly_created.append((local, oid, parent))
            self._event(dict(type="proposal" if self.lifecycle_enabled else "birth",
                             frames=[first, second], object_id=oid,
                             local_model=local, parent_id=parent))
        if first_pair:
            self._event(dict(type="proposal" if self.lifecycle_enabled else "birth",
                             frames=[first, second], object_id=0,
                             local_model=int(reference), parent_id=None, initial_reference=True))
        for parent in sorted({p for _, _, p in newly_created if p is not None}):
            children = [oid for _, oid, p in newly_created if p == parent]
            if parent in mapping.values():
                children.insert(0, parent)
            self._event(dict(type="split", frames=[first, second],
                                    parent_id=parent, object_ids=children))
        retained = set(mapping.values())
        for local, oid in ([] if self.window_enabled else mapping.items()):
            sources = [previous[column] for column in np.flatnonzero(votes[local] > 0)]
            absorbed = [old for old in sources if old != oid and old not in retained]
            if absorbed:
                for old in absorbed:
                    self.objects[old]["merged_into"] = oid
                self._event(dict(type="merge", frames=[first, second],
                                        object_id=oid, source_object_ids=sorted(set([oid]+absorbed)),
                                        local_model=int(local)))

        string_mapping = {str(k): int(mapping[k]) for k in range(count)}
        pair = dict(first=first, second=second, result_path=str(result_path),
                    reference=int(reference), local_to_object=string_mapping,
                    motion_identity=identity_report)
        self.pairs.append(pair)
        self._pair_count += 1
        if self.lifecycle_enabled:
            del self.pairs[:-self.history_limit]
        for local, oid in mapping.items():
            record = self.objects[oid]
            if not self.lifecycle_enabled:
                record["frames"] = sorted(set(record["frames"]) | {first, second})
                record["last_frame"] = max(record["last_frame"], second)
            # A historical object may reappear after a provisional merge.
            record["merged_into"] = None
            record["appearances"].append(dict(first=first, second=second,
                                              local_model=int(local), result_path=str(result_path),
                                              H=np.asarray(models[local]).tolist()))
            if self.lifecycle_enabled:
                record["appearance_count"] += 1
                del record["appearances"][:-self.history_limit]
            current_tracks = set(map(int, track_ids[(labels == local) & (track_ids >= 0)]))
            record["track_ids"] = sorted(set(record["track_ids"]) | current_tracks)
        reassigned = {}
        for i, (track, local) in enumerate(zip(track_ids, labels)):
            track, local = int(track), int(local)
            if track < 0 or local < 0:
                continue
            oid = mapping[local]
            old = self.track_objects.get(track)
            if self.window_enabled and old is not None:
                # Current local labels propose identity only. A shared window
                # posterior, rather than the newest frame, revises old tracks.
                continue
            if old is not None and old != oid:
                reassigned[(old, oid)] = reassigned.get((old, oid), 0) + 1
            self.track_objects[track] = oid
            posterior = {}
            for column, probability in enumerate(q_anchor[i]):
                if probability > 0:
                    global_id = mapping[column]
                    posterior[global_id] = posterior.get(global_id, 0.) + float(probability)
            self.track_probabilities[track] = posterior
        for (old, new), n in sorted(reassigned.items()):
            self._event(dict(type="track_reassignment", frames=[first, second],
                                    source_object_id=old, object_id=new, tracks=n))
        self._remember_geometric_identity(first, second, models, labels, track_ids,
                                          mapping, anchors, K, motion_cfg)
        return string_mapping

    def commit_window_tracks(self, track_ids, object_ids, probabilities, labels):
        """Commit a joint-window posterior, with global IDs and -1 for uncertainty.

        Soft states are revisable even when no hard label is justified. Track
        support histories are append-only until streaming cache pruning.
        Repeated track rows are averaged,
        so callers cannot accidentally make the latest observation decisive.
        """
        tracks = np.asarray(track_ids, np.int64).reshape(-1)
        object_ids = np.asarray(object_ids, np.int64).reshape(-1)
        probabilities = np.asarray(probabilities, float)
        labels = np.asarray(labels, np.int64).reshape(-1)
        if probabilities.shape != (len(tracks), len(object_ids)) or len(labels) != len(tracks):
            raise ValueError("Window track probabilities and labels have incompatible shapes")
        if len(set(map(int, object_ids))) != len(object_ids) or any(int(o) not in self.objects for o in object_ids):
            raise ValueError("Window object IDs must be unique, existing registry objects")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
            raise ValueError("Window probabilities must be finite and nonnegative")
        allowed_labels = set(map(int, object_ids)) | {-1}
        if any(int(label) not in allowed_labels for label in labels):
            raise ValueError("Window labels must be global object IDs or -1")
        reassigned = {}
        eligible = set(self.candidate_object_ids())
        eligible_columns = np.asarray([int(oid) in eligible for oid in object_ids])
        for track in np.unique(tracks[tracks >= 0]):
            selected = tracks == track
            probability = probabilities[selected].mean(axis=0)
            if self.lifecycle_enabled:
                probability[~eligible_columns] = 0.
            mass = probability.sum()
            if mass <= 0:
                continue
            probability /= mass
            self.track_probabilities[int(track)] = {
                int(oid): float(p) for oid, p in zip(object_ids, probability) if p > 0}
            confident = set(map(int, labels[selected])) - {-1}
            if self.lifecycle_enabled:
                confident &= eligible
            if len(confident) != 1:
                continue
            oid = confident.pop()
            winner = int(np.argmax(probability))
            if (int(object_ids[winner]) != oid or
                    np.count_nonzero(probability == probability[winner]) != 1):
                continue
            old = self.track_objects.get(int(track))
            self.track_objects[int(track)] = oid
            record = self.objects[oid]
            record["track_ids"] = sorted(set(record["track_ids"]) | {int(track)})
            if old is not None and old != oid:
                reassigned[(old, oid)] = reassigned.get((old, oid), 0) + 1
        frames = ([self.pairs[-1]["first"], self.pairs[-1]["second"]]
                  if self.pairs else [])
        for (old, new), count in sorted(reassigned.items()):
            self._event(dict(type="track_reassignment", source="object_window",
                                    frames=frames, source_object_id=old,
                                    object_id=new, tracks=count))

    def to_dict(self):
        state = dict(
            objects={str(k): v for k, v in self.objects.items()},
            events=self.events, pairs=self.pairs,
            track_objects={str(k): int(v) for k, v in self.track_objects.items()},
            track_probabilities={str(k): {str(oid): float(p) for oid, p in v.items()}
                                 for k, v in self.track_probabilities.items()})
        if self.identity_cfg.get('geometric_identity_ledger', False):
            state['geometric_identity_ledger'] = dict(
                pair=(list(self._geometric_identity_pair)
                      if self._geometric_identity_pair is not None else None),
                tracks={str(k): int(v) for k, v in self._geometric_identity_tracks.items()})
        if self.lifecycle_enabled:
            state["lifecycle"] = dict(
                enabled=True, missing_grace_frames=self.missing_grace_frames,
                next_object_id=self._next_object, processed_frames=self._observed_frame_count,
                observation_step=self._observation_step,
                last_observation_frame=self._last_observation_frame,
                candidate_object_ids=self.candidate_object_ids(),
                pair_count=self._pair_count, event_count=self._event_count,
                history_limit=self.history_limit, event_limit=self.event_limit,
                event_log_path=(str(self._event_log_path) if self._event_log_path is not None else None),
                finalized=self._finalized)
        return deepcopy(state)

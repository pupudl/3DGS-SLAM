"""Persistent object identities and revisable track priors for pair updates.

The pair optimizer owns geometry.  This registry owns identity and history:
local motion indices may change every pair, while object IDs keep their past.
Historical tracks fit a fresh transform from the *current* pair's observations;
the previous pair's transform is never copied to the new time interval.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

PAIR_DIR = Path(__file__).resolve().parent / "render_guided_two_frame"
if str(PAIR_DIR) not in sys.path:
    sys.path.insert(0, str(PAIR_DIR))

from hybrid_motion import (_core_labels, _refit, _rigid, motion_costs,
                           motion_probabilities)


class ObjectRegistry:
    """Associate current motion components through continuing feature tracks."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracking_cfg = cfg.get("tracking", {})
        self.prior_weight = float(self.tracking_cfg.get("prior_weight", .25))
        self.window_enabled = bool(cfg.get("object_window", {}).get("enabled", False))
        self.objects = {}
        self.events = []
        self.pairs = []
        self.track_objects = {}
        self.track_probabilities = {}
        self._next_object = 0

    def _votes(self, labels, track_ids, count):
        previous = sorted(set(self.track_objects.values()))
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
            oid = self.track_objects.get(track)
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

    def _soft_association(self, mapping, labels, q_anchor, track_ids):
        """Fill unmatched identities with continuing grey-track preferences.

        This is only an identity clue. A large conditional probability can
        still accompany a poor absolute geometric fit: grey observations stay
        grey and do not become hard track/object evidence through this step.
        Hard associations are fixed before this remaining-submatrix match.
        """
        mapping = dict(mapping)
        remaining = [k for k in range(q_anchor.shape[1]) if k not in mapping]
        previous = sorted(set(self.track_objects.values()) -
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
            oid = self.track_objects.get(track)
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
        minimum = int(hcfg.get("min_support", motion_cfg.get("min_support", 6)))
        maximum = int(hcfg.get("max_components", motion_cfg.get("max_components", 8)))
        outlier = float(hcfg.get("outlier_cost", 1.))
        models = [np.asarray(H, float).copy() for H in discovered_models]
        track_ids = np.asarray(track_ids, np.int64)
        labels = np.asarray(discovered_labels, np.int32).copy()
        previous, votes = self._votes(labels, track_ids, len(models))
        associated = self._association(previous, votes, self.window_enabled)
        object_locals = {}
        for local, oid in associated.items():
            object_locals.setdefault(oid, []).append(local)
        object_local = {oid: locals_[0] for oid, locals_ in object_locals.items()
                        if len(locals_) == 1}
        history = np.asarray([self.track_objects.get(int(t), -1) for t in track_ids])
        confidence = np.asarray(anchors.get("confidence", np.ones(len(track_ids)))).reshape(-1)
        fit_reports = []
        warm_local = {}

        for oid in sorted(set(history) - {-1}):
            ids = np.flatnonzero(history == oid)
            row = dict(object_id=int(oid), inherited_tracks=int(len(ids)))
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

        costs, labels = _core_labels(anchors, models, K, hcfg)
        q_anchor, _ = motion_probabilities(costs, hcfg)
        previous, votes = self._votes(labels, track_ids, len(models))
        mapping = self._association(previous, votes, self.window_enabled)
        # A fitted missing object can initially explain only new tracks; keep
        # its historical ID available to the soft prior without duplicating IDs.
        used = set(mapping.values())
        for local, oid in warm_local.items():
            if local not in mapping and (self.window_enabled or oid not in used):
                mapping[local] = oid
                used.add(oid)
        mapping, soft_identity = self._soft_association(mapping, labels, q_anchor, track_ids)
        object_locals = {}
        for local, oid in mapping.items():
            object_locals.setdefault(oid, []).append(local)
        mixed = 0
        for i, track in enumerate(track_ids):
            track = int(track)
            prior = self.track_probabilities.get(track)
            if prior is None and track in self.track_objects:
                prior = {self.track_objects[track]: 1.}
            if not prior:
                continue
            prior_local = np.zeros(len(models), float)
            for oid, probability in prior.items():
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
        return oid

    def finish_pair(self, first, second, models, q_anchor, track_ids,
                    reference, result_path, *, labels=None):
        """Record pair identity proposals, preserving track states in window mode.

        Grey anchors do not provide new hard identity evidence.  Historical
        objects and disappeared tracks remain stored; a confident current
        assignment can change a track's formerly assigned object.
        """
        first, second = int(first), int(second)
        q_anchor = np.asarray(q_anchor, float)
        track_ids = np.asarray(track_ids, np.int64)
        labels = (q_anchor.argmax(axis=1).astype(np.int32) if labels is None else
                  np.asarray(labels, np.int32))
        count = len(models)
        previous, votes = self._votes(labels, track_ids, count)
        mapping = self._association(previous, votes, self.window_enabled)
        mapping, soft_identity = self._soft_association(mapping, labels, q_anchor, track_ids)
        for row in soft_identity:
            self.events.append(dict(type="identity_from_soft_tracks", frames=[first, second], **row))
        first_pair = not self.pairs
        if first_pair:
            # Only the first reference creates background object 0. Later
            # references are local optimizer choices, not persistent identity.
            mapping[int(reference)] = self._create_object(first)
        newly_created = []
        for local in range(count):
            if local in mapping:
                continue
            parent = None
            if not self.window_enabled and votes.shape[1] and votes[local].max() > 0:
                parent = int(previous[int(np.argmax(votes[local]))])
            oid = self._create_object(first, parent)
            mapping[local] = oid
            newly_created.append((local, oid, parent))
            self.events.append(dict(type="birth", frames=[first, second], object_id=oid,
                                    local_model=local, parent_id=parent))
        if first_pair:
            self.events.append(dict(type="birth", frames=[first, second], object_id=0,
                                    local_model=int(reference), parent_id=None,
                                    initial_reference=True))
        for parent in sorted({p for _, _, p in newly_created if p is not None}):
            children = [oid for _, oid, p in newly_created if p == parent]
            if parent in mapping.values():
                children.insert(0, parent)
            self.events.append(dict(type="split", frames=[first, second],
                                    parent_id=parent, object_ids=children))
        retained = set(mapping.values())
        for local, oid in ([] if self.window_enabled else mapping.items()):
            sources = [previous[column] for column in np.flatnonzero(votes[local] > 0)]
            absorbed = [old for old in sources if old != oid and old not in retained]
            if absorbed:
                for old in absorbed:
                    self.objects[old]["merged_into"] = oid
                self.events.append(dict(type="merge", frames=[first, second],
                                        object_id=oid, source_object_ids=sorted(set([oid]+absorbed)),
                                        local_model=int(local)))

        string_mapping = {str(k): int(mapping[k]) for k in range(count)}
        pair = dict(first=first, second=second, result_path=str(result_path),
                    reference=int(reference), local_to_object=string_mapping)
        self.pairs.append(pair)
        for local, oid in mapping.items():
            record = self.objects[oid]
            record["frames"] = sorted(set(record["frames"]) | {first, second})
            record["last_frame"] = max(record["last_frame"], second)
            # A historical object may reappear after a provisional merge.
            record["merged_into"] = None
            record["appearances"].append(dict(first=first, second=second,
                                              local_model=int(local), result_path=str(result_path),
                                              H=np.asarray(models[local]).tolist()))
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
            self.events.append(dict(type="track_reassignment", frames=[first, second],
                                    source_object_id=old, object_id=new, tracks=n))
        return string_mapping

    def commit_window_tracks(self, track_ids, object_ids, probabilities, labels):
        """Commit a joint-window posterior, with global IDs and -1 for uncertainty.

        Soft states are revisable even when no hard label is justified. Track
        support histories are append-only. Repeated track rows are averaged,
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
        for track in np.unique(tracks[tracks >= 0]):
            selected = tracks == track
            probability = probabilities[selected].mean(axis=0)
            mass = probability.sum()
            if mass <= 0:
                continue
            probability /= mass
            self.track_probabilities[int(track)] = {
                int(oid): float(p) for oid, p in zip(object_ids, probability) if p > 0}
            confident = set(map(int, labels[selected])) - {-1}
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
            self.events.append(dict(type="track_reassignment", source="object_window",
                                    frames=frames, source_object_id=old,
                                    object_id=new, tracks=count))

    def to_dict(self):
        return deepcopy(dict(
            objects={str(k): v for k, v in self.objects.items()},
            events=self.events, pairs=self.pairs,
            track_objects={str(k): int(v) for k, v in self.track_objects.items()},
            track_probabilities={str(k): {str(oid): float(p) for oid, p in v.items()}
                                 for k, v in self.track_probabilities.items()}))

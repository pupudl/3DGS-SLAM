"""Bounded observation links for correcting persistent map ownership.

Only measured, confidently labelled observations can propose a transfer. The
map itself never provides ownership or lifecycle evidence. Transfers use two
known owner poses at the same instant, preserving that reference geometry.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class MapOwnershipLinks:
    def __init__(self, cfg):
        self.cfg = dict(cfg.get('ownership_correction', {}))
        self.enabled = bool(self.cfg.get('enabled', False))
        self.links = {}
        self.last_frames = {}
        self.pending = {}
        self.min_probability = float(self.cfg.get('min_probability', .85))
        self.min_margin = float(self.cfg.get('min_margin', .2))
        self.dominant_fraction = float(self.cfg.get('dominant_fraction', .9))
        self.confirmations = int(self.cfg.get('confirmation_updates', 2))
        self.max_links = int(self.cfg.get('max_observation_links',
                                         int(cfg.get('max_samples_per_frame', 30000)) *
                                         int(cfg.get('window_frames', 5))))
        self.max_descendants = int(self.cfg.get('max_descendants_per_observation', 64))
        if (not .5 <= self.min_probability <= 1 or not 0 <= self.min_margin <= 1 or
                not .5 < self.dominant_fraction <= 1 or self.confirmations < 1 or
                self.max_links < 1 or self.max_descendants < 1):
            raise ValueError('Invalid map ownership correction configuration')

    def remember(self, observation_ids, map_ids, frame):
        if not self.enabled or observation_ids is None:
            return
        for oid, mid in zip(observation_ids, map_ids):
            if oid >= 0 and mid >= 0:
                previous = self.links.get(int(oid), ())
                self.links[int(oid)] = tuple(dict.fromkeys((int(mid),)+previous))[:self.max_descendants]
                self.last_frames[int(oid)] = int(frame)
        if len(self.links) > self.max_links:
            oldest = sorted(self.links, key=lambda oid: (self.last_frames[oid], oid))
            for oid in oldest[:len(self.links)-self.max_links]:
                self.links.pop(oid)
                self.last_frames.pop(oid)

    def retain(self, store, first_frame, window=None, extra_map_ids=()):
        keep = (window._initial if window is not None else None)
        for oid, ids in list(self.links.items()):
            live = tuple(mid for mid, yes in zip(ids, store.is_live(np.asarray(ids, np.int64))) if yes)
            if not live or (oid not in keep if keep is not None else self.last_frames[oid] < first_frame):
                self.links.pop(oid)
                self.last_frames.pop(oid)
            else:
                self.links[oid] = live
        reachable = {mid for mids in self.links.values() for mid in mids}
        reachable.update(map(int, extra_map_ids))
        self.pending = {mid: value for mid, value in self.pending.items()
                        if mid in reachable and store.is_live(np.asarray([mid], np.int64))[0]}

    def hints(self, observation_ids, tracks, track_landmarks, store):
        result = np.full(len(tracks), -1, np.int64)
        for row, track in enumerate(tracks):
            ids = self.links.get(int(observation_ids[row]), ()) if observation_ids is not None else ()
            if not ids and track >= 0 and int(track) in track_landmarks:
                ids = (track_landmarks[int(track)][1],)
            live = [mid for mid, yes in zip(ids, store.is_live(np.asarray(ids, np.int64))) if yes]
            if live:
                result[row] = live[0]
        return result

    def topology(self, parent_ids, child_ids, removed_ids):
        """Keep all descendants reachable from the same measured observation."""
        children = defaultdict(list)
        for parent, child in zip(parent_ids, child_ids):
            children[int(parent)].append(int(child))
        removed = set(map(int, removed_ids))
        for oid, mids in list(self.links.items()):
            updated = []
            for mid in mids:
                if mid not in removed:
                    updated.append(mid)
                updated.extend(children.get(mid, ()))
            self.links[oid] = tuple(dict.fromkeys(updated))[:self.max_descendants]
            if not updated:
                self.links.pop(oid)
                self.last_frames.pop(oid)

    def correct(self, store, frame, update, observation_ids, posterior, hints,
                current_poses, camera, views, registry, window=None):
        report = dict(enabled=self.enabled, transferred=0, pending=0, conflicts=0,
                      skipped_without_pose=0, events=[], linked_observations=len(self.links))
        if not self.enabled:
            return report
        evidence = {}

        def add(ids, state, mapping):
            probabilities = np.asarray(state['object_probabilities'])
            labels = np.asarray(state['object_labels'])
            if len(probabilities) == 0:
                return
            best = probabilities.max(1)
            second = np.partition(probabilities, -2, axis=1)[:, -2] if probabilities.shape[1] > 1 else 0.
            reliable = (labels >= 0) & (best >= self.min_probability) & (best-second >= self.min_margin)
            if 'object_ids' in state:
                objects = np.asarray(state['object_ids'], np.int64)
                if objects.shape != (probabilities.shape[1],):
                    raise ValueError('Ownership probability columns must match object IDs')
                reliable &= labels == objects[probabilities.argmax(1)]
            for row in np.flatnonzero(reliable):
                mids = mapping(row, int(ids[row]))
                if mids:
                    evidence[int(ids[row])] = (int(labels[row]), float(best[row]), mids)

        if window is not None and self.links:
            ids = np.asarray([oid for oid in self.links if oid in window._initial], np.int64)
            if len(ids):
                add(ids, window.get_state(ids), lambda row, oid: self.links[oid])
        ids = np.asarray(observation_ids if observation_ids is not None else
                         -1-np.arange(len(hints)), np.int64)
        add(ids, posterior, lambda row, oid: self.links.get(oid, (int(hints[row]),))
            if hints[row] >= 0 or oid in self.links else ())
        votes = defaultdict(lambda: defaultdict(float))
        for label, weight, mids in evidence.values():
            for mid in mids:
                votes[mid][label] += weight
        ids = np.asarray(sorted(votes), np.int64)
        ids = ids[store.is_live(ids)]
        owners = dict(zip(map(int, ids), map(int, store.select(ids, include_optimizer=False)['object_ids'])))
        eligible = set()
        grouped = defaultdict(list)
        for mid in ids:
            mid = int(mid)
            choice = votes[mid]
            new = max(choice, key=choice.get)
            if choice[new]/sum(choice.values()) < self.dominant_fraction:
                report['conflicts'] += 1
                self.pending.pop(mid, None)
                continue
            old = owners[mid]
            if old == new:
                self.pending.pop(mid, None)
                continue
            if registry.objects.get(new, {}).get('status', 'active') in {'retired', 'archived', 'provisional'}:
                continue
            previous = self.pending.get(mid)
            consecutive = previous is not None and previous['owner'] == new and previous['update'] == update-1
            count = previous['count']+1 if consecutive else 1
            reference_frame = previous['reference_frame'] if consecutive else int(frame)
            self.pending[mid] = dict(owner=new, count=count, update=update, reference_frame=reference_frame)
            eligible.add(mid)
            if count < self.confirmations:
                continue
            # Anchor the change at its first reliable observation, before the
            # confirmation delay. Otherwise a vehicle moving during that delay
            # inherits an already stale background position in its new frame.
            # Every candidate still uses both owner poses at one instant.
            pairs = [(int(frame), {oid: np.linalg.inv(camera) @ pose for oid, pose in current_poses.items()})]
            pairs.extend((key, value['object_to_camera']) for key, value in sorted(views.items(), reverse=True))
            pairs.sort(key=lambda value: value[0] != reference_frame)
            pair = next(((when, poses) for when, poses in pairs if old in poses and new in poses), None)
            if pair is None:
                report['skipped_without_pose'] += 1
                continue
            when, poses = pair
            transform = np.linalg.inv(poses[new]) @ poses[old]
            grouped[(old, new, when)].append((mid, transform))
        for (old, new, when), entries in grouped.items():
            mids = np.asarray([mid for mid, _ in entries], np.int64)
            event = store.reassign(mids, new, np.asarray([value for _, value in entries]),
                                   frame_id=when, reason='confident_window_ownership')
            report['events'].append(dict(revision=event['revision'], count=len(mids),
                                         old_object_id=old, new_object_id=new, reference_frame=when))
            report['transferred'] += len(mids)
            for mid in mids:
                self.pending.pop(int(mid), None)
        self.pending = {mid: value for mid, value in self.pending.items() if mid in eligible}
        report['pending'] = len(self.pending)
        return report

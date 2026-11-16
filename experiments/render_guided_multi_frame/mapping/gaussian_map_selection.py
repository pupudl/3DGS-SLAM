"""Spatially balanced map training selection, independent from map visibility.

The complete visible map stays on the CPU. This scheduler chooses a bounded
training subset across view tiles and object owners, and rotates deferred points
through later solves instead of repeatedly taking the same sorted map IDs.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _unique_rows(values, lookup):
    return np.asarray(list(dict.fromkeys(lookup[int(value)] for value in values
                                         if int(value) in lookup)), np.int64)


def select_active_gaussians(ids, owners, tile_keys, quality, budget, *, update=0,
                            last_selected=None, track_ids=(), recent_ids=(),
                            priority_ids=(), frozen=None, options=None):
    """Return selected rows and a JSON-safe coverage report.

    ``tile_keys``/``quality`` have shape [N,V], with -1 for invisible view tiles.
    Tile keys must be unique across views. Frozen points remain eligible as
    occluders; their optimization mask is handled by the backend.
    """
    ids, owners = np.asarray(ids, np.int64), np.asarray(owners, np.int64)
    tiles, quality = np.asarray(tile_keys, np.int64), np.asarray(quality, float)
    n, budget = len(ids), max(0, int(budget))
    if owners.shape != (n,) or tiles.ndim != 2 or tiles.shape[0] != n or quality.shape != tiles.shape:
        raise ValueError('Map selection expects owners [N] and tile/quality arrays [N,V]')
    if len(np.unique(ids)) != n or np.any(~np.isfinite(quality)):
        raise ValueError('Map selection IDs must be unique and quality finite')
    options, last = options or {}, last_selected or {}
    frozen = np.zeros(n, bool) if frozen is None else np.asarray(frozen, bool)
    if frozen.shape != (n,):
        raise ValueError('Map selection frozen mask must have shape [N]')
    budget = min(n, budget)
    if budget == n:
        # Most early submap solves fit the active budget. Do not construct
        # N*V tile queues or an N-entry Python lookup when every point fits.
        occupied = np.unique(tiles[tiles >= 0])
        unique_owners, owner_counts = np.unique(owners, return_counts=True)
        tracks = np.intersect1d(ids, np.asarray(track_ids, np.int64))
        priorities = np.intersect1d(ids, np.asarray(priority_ids, np.int64))
        report = dict(visible_candidates=n, training_gaussians=n, budget=budget,
                      deferred_visible_gaussians=0, occupied_view_tiles=len(occupied),
                      selected_view_tiles=len(occupied),
                      selected_id_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                      priority_requested=len(priorities), priority_selected=len(priorities),
                      track_candidates=len(tracks), track_selected=len(tracks),
                      recent_selected=len(np.intersect1d(ids, np.asarray(recent_ids, np.int64))),
                      frozen_selected=int(frozen.sum()), oldest_deferred_age=0,
                      selection_mode='all_visible',
                      per_object={str(int(owner)): dict(visible=int(count), selected=int(count))
                                  for owner, count in zip(unique_owners, owner_counts)})
        return np.arange(n, dtype=np.int64), report
    lookup = {int(value): i for i, value in enumerate(ids)}
    # A deterministic ID hash avoids persistent preference for early insertions.
    with np.errstate(over='ignore'):
        hashed = ids.astype(np.uint64)*np.uint64(11400714819323198485)
        hashed ^= np.uint64(int(update)+1)*np.uint64(14029467366897019727)
    tie = hashed.astype(np.float64)/float(np.iinfo(np.uint64).max)
    age = np.asarray([int(update)-int(last.get(int(value), -1)) for value in ids], float)
    qmax = np.max(quality, axis=1) if tiles.shape[1] else np.ones(n)
    picked, selected = [], np.zeros(n, bool)

    def add(rows, count):
        remaining = min(max(0, int(count)), budget-len(picked))
        if remaining <= 0:
            return
        for row in rows:
            row = int(row)
            if not selected[row]:
                selected[row] = True
                picked.append(row)
                remaining -= 1
                if remaining == 0:
                    break

    def spatial(mask, count, fair=False):
        """Round-robin ranked candidates from every occupied view tile."""
        candidate = np.flatnonzero(mask & ~selected)
        if not len(candidate) or count <= 0:
            return
        rr, vv = np.nonzero(tiles[candidate] >= 0)
        if not len(rr):
            add(candidate[np.lexsort((tie[candidate], -age[candidate]))], count)
            return
        rows, keys = candidate[rr], tiles[candidate[rr], vv]
        score = quality[rows, vv]
        if fair:
            order = np.lexsort((tie[rows], -score, -age[rows], keys))
        else:
            order = np.lexsort((tie[rows], -score, keys))
        rows, keys = rows[order], keys[order]
        starts = np.r_[0, np.flatnonzero(np.diff(keys))+1]
        rank = np.arange(len(rows))-np.repeat(starts, np.diff(np.r_[starts, len(rows)]))
        # Rotating tile order also keeps the last partial round fair.
        distinct, tile_index = np.unique(keys, return_inverse=True)
        rotated = (tile_index+int(update)) % max(1, len(distinct))
        interleaved = np.lexsort((tie[rows], rotated, rank))
        add(rows[interleaved], count)

    priority = _unique_rows(priority_ids, lookup)
    add(priority, budget)  # Refinement children must receive their first solve.
    tracks = _unique_rows(track_ids, lookup)
    track_quota = max(1, int(np.ceil(budget*float(options.get('track_fraction', .15)))))
    add(tracks[np.lexsort((tie[tracks], -age[tracks], -qmax[tracks]))], track_quota)
    dynamic = np.unique(owners[owners != 0])
    dynamic_quota = int(np.ceil(budget*float(options.get('dynamic_fraction', .2))))
    if len(dynamic):
        quota = max(1, dynamic_quota//len(dynamic))
        for owner in np.roll(dynamic, int(update) % len(dynamic)):
            spatial(owners == owner, quota, fair=True)
    spatial(frozen, int(np.ceil(budget*float(options.get('frozen_fraction', .1)))), fair=True)
    recent = np.zeros(n, bool)
    recent[_unique_rows(recent_ids, lookup)] = True
    spatial(recent, int(np.ceil(budget*float(options.get('recent_fraction', .2)))))
    fair_count = int(np.ceil(budget*float(options.get('fairness_fraction', .35))))
    spatial(np.ones(n, bool), fair_count, fair=True)
    spatial(np.ones(n, bool), budget-len(picked), fair=False)
    # Points can be outside every tile only through a caller's conservative
    # footprint margin. They still participate if spatial passes leave space.
    spatial(np.ones(n, bool), budget-len(picked), fair=True)
    rows = np.asarray(sorted(picked), np.int64)
    occupied = np.unique(tiles[tiles >= 0])
    used = np.unique(tiles[rows][tiles[rows] >= 0])
    digest = hashlib.sha256(ids[rows].astype('<i8').tobytes()).hexdigest()
    report = dict(visible_candidates=n, training_gaussians=len(rows), budget=budget,
                  deferred_visible_gaussians=n-len(rows), occupied_view_tiles=len(occupied),
                  selected_view_tiles=len(used), selected_id_sha256=digest,
                  priority_requested=len(priority), priority_selected=int(selected[priority].sum()),
                  track_candidates=len(tracks), track_selected=int(selected[tracks].sum()),
                  recent_selected=int((selected & recent).sum()), frozen_selected=int((selected & frozen).sum()),
                  per_object={str(int(owner)): dict(visible=int((owners == owner).sum()),
                              selected=int((selected & (owners == owner)).sum())) for owner in np.unique(owners)},
                  oldest_deferred_age=int(age[~selected].max()) if np.any(~selected) else 0,
                  selection_mode='balanced_tiles')
    return rows, report

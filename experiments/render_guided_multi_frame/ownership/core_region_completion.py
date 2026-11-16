"""Detached, per-source-frame surface targets grown from measured motion cores.

The region is a spatial prior, not new motion evidence or a segmentation mask.
Growth never uses the current posterior. Camera XYZ is used only within its
own snapshot; stable Gaussian IDs map the resulting targets into the window.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from ownership.object_connectivity_constraints import _snapshot_edges


def _motion_observations(snap):
    objects = np.asarray(snap['object_ids'])
    costs = np.asarray(snap['costs'], float)
    weights = np.asarray(snap['weights'], float)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights[:, None], costs.shape)
    valid = np.asarray(snap['payload'].get('object_model_valid', np.ones(len(objects))), bool)
    return objects, costs, weights, valid


def _incoming_observations(previous, current, options):
    """Verify current surfaces with the previous measured transform's inverse.

    No outgoing transform is fabricated and no historical posterior is used.
    The samples are fresh RGB-D comparisons, not new repeated motion cores.
    """
    if previous is None:
        return None
    before = previous.get('render_observations') or {}
    now = current.get('render_observations') or {}
    if (not {'source', 'source_depth', 'source_confidence'} <= before.keys() or
            'source_confidence' not in now or
            not {'H_object', 'K'} <= previous['payload'].keys()):
        return None
    from ownership.gaussian_object_backend import rgbd_motion_costs
    objects, _, _, valid = _motion_observations(previous)
    models = np.asarray(previous['payload']['H_object'], float)
    valid = valid & np.isfinite(models).all(axis=(1, 2))
    inverse = np.full_like(models, np.nan)
    if valid.any():
        inverse[valid] = np.linalg.inv(models[valid])
    costs, weights = rgbd_motion_costs(
        current['payload'], inverse, valid, previous['payload']['K'],
        before['source'], before['source_depth'], now['source_confidence'],
        before['source_confidence'], options)
    return objects, costs, weights, valid


def build_region_completion(window, state, options=None):
    options = {} if options is None else dict(options)
    empty = dict(rows=np.empty(0, np.int64), columns=np.empty(0, np.int64),
                 weights=np.empty(0, np.float32), by_pair={})
    report = dict(enabled=bool(options.get('enabled', False)), frames=[],
                  target_entries=0, unique_gaussians=0, conflicting_gaussians=0,
                  targets_from_current_posterior=False, supplies_motion_cores=False,
                  forces_assignment=False)
    if not report['enabled']:
        return dict(**empty, report=report)
    limits = dict(max_distance_m=float(options.get('max_distance_m', 3.)),
                  max_core_span_m=float(options.get('max_core_span_m', 8.)),
                  cost_max=float(options.get('cost_max', 1.)),
                  min_weight=float(options.get('min_weight', .15)),
                  background_margin=float(options.get('background_margin', .15)),
                  competitor_margin=float(options.get('competitor_margin', .15)),
                  min_color_affinity=float(options.get('min_color_affinity', .05)),
                  color_path_weight=float(options.get('color_path_weight', 2.)))
    if (not np.isfinite(list(limits.values())).all() or
            min(limits['max_distance_m'], limits['max_core_span_m'], limits['cost_max']) <= 0 or
            not 0 < limits['min_weight'] <= 1 or
            not 0 < limits['min_color_affinity'] <= 1 or
            min(limits['background_margin'], limits['competitor_margin'], limits['color_path_weight']) < 0):
        raise ValueError('Invalid region_completion growth bounds or evidence thresholds')
    report['options'] = limits
    gids = np.asarray(state['active_gids'])
    objects = np.asarray(state['object_ids'])
    columns = {int(oid): c for c, oid in enumerate(objects)}
    background = int(state['options'].get('background_object_id', 0))
    cores = state['core_by_object']
    all_core = np.unique(np.concatenate(list(cores.values()))) if cores else np.empty(0, np.int64)
    by_second = {int(s.get('second', pair[1])): s for pair, s in window._active.items()}
    parts = []
    for pair, snap in sorted(window._active.items()):
        local_gids = np.asarray(snap['gaussian_ids'])
        rows = np.searchsorted(gids, local_gids)
        n = len(rows)
        outgoing = _motion_observations(snap)
        outgoing_column = {int(oid): c for c, oid in enumerate(outgoing[0])}
        frame_report = dict(first=int(snap.get('first', pair[0])), objects={})
        report['frames'].append(frame_report)
        if not n:
            continue
        payload = snap['payload']
        xyz = np.asarray(payload['xyz'], float)
        source_valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0) & state['height_keep'][rows]
        proposals = []
        edges = affinities = None
        incoming = None
        incoming_checked = False
        for oid in sorted(cores):
            if oid == background:
                continue
            selected = np.isin(rows, cores[oid]) & source_valid
            info = dict(visible_core_count=int(selected.sum()), target_count=0)
            frame_report['objects'][str(oid)] = info
            if not selected.any():
                info['action'] = 'skip_no_visible_core_or_valid_motion'
                continue
            span = float(np.linalg.norm(np.ptp(xyz[selected], axis=0)))
            info['core_span_m'] = span
            if span > limits['max_core_span_m']:
                info['action'] = 'skip_spread_core'
                continue
            data = outgoing
            direction = 'outgoing'
            if oid not in outgoing_column or not outgoing[3][outgoing_column[oid]]:
                if not incoming_checked:
                    incoming = _incoming_observations(by_second.get(frame_report['first']), snap,
                                                      getattr(window, 'cfg', {}))
                    incoming_checked = True
                data, direction = incoming, 'incoming'
            local_column = {} if data is None else {int(value): c for c, value in enumerate(data[0])}
            if (oid not in local_column or background not in local_column or
                    not data[3][local_column[oid]]):
                info['action'] = 'skip_no_visible_core_or_valid_motion'
                continue
            local_objects, costs, weights, model_valid = data
            c, b = local_column[oid], local_column[background]
            measured = np.isfinite(costs) & np.isfinite(weights) & (weights >= limits['min_weight'])
            measured &= model_valid[None, :]
            info['observation_direction'] = direction
            own = costs[:, c]
            # A clear contrary observation stops growth; missing background
            # projection does not pretend to be a zero-cost background fit.
            background_stop = measured[:, b] & (costs[:, b]+limits['background_margin'] < own)
            other = [j for j, value in enumerate(local_objects) if value not in (background, oid)
                     and int(value) in columns]
            other_stop = np.zeros(n, bool)
            for j in other:
                other_stop |= measured[:, j] & (costs[:, j]+limits['competitor_margin'] < own)
            other_core = np.isin(rows, np.setdiff1d(all_core, cores[oid]))
            eligible = source_valid & measured[:, c] & (own < limits['cost_max'])
            eligible &= ~background_stop & ~other_stop & ~other_core
            # Compact, metric bounds cap both runtime and accidental expansion
            # along a weak road/body connection. They do not invent an ROI.
            low = xyz[selected].min(0)-limits['max_distance_m']
            high = xyz[selected].max(0)+limits['max_distance_m']
            eligible &= ((xyz >= low) & (xyz <= high)).all(1)
            seed = np.flatnonzero(selected & eligible & measured[:, b] &
                                  (own < float(state['options'].get('core_cost_max', .5))) &
                                  (costs[:, b]-own >= float(state['options'].get('core_advantage', .3))))
            info.update(seed_count=len(seed), background_stops=int(background_stop.sum()),
                        other_object_stops=int((other_stop | other_core).sum()))
            if not len(seed):
                info['action'] = 'skip_no_current_positive_core'
                continue
            if edges is None:
                edges, affinities, _ = _snapshot_edges(snap, np.arange(n), state['options'])
            keep = eligible[edges].all(1) & (affinities >= limits['min_color_affinity'])
            left, right = edges[keep].T
            length = np.maximum(np.linalg.norm(xyz[left]-xyz[right], axis=1), 1e-6)
            length *= 1+limits['color_path_weight']*(1-affinities[keep])
            graph = sparse.csr_matrix((np.r_[length, length],
                                      (np.r_[left, right], np.r_[right, left])), shape=(n, n))
            distance = dijkstra(graph, directed=False, indices=seed, min_only=True,
                                limit=limits['max_distance_m'])
            reached = np.flatnonzero(eligible & np.isfinite(distance))
            confidence = np.clip(weights[reached, c], 0, 1)
            fit = np.clip(1-own[reached]/limits['cost_max'], .1, 1)
            strength = confidence*fit*(1-.5*distance[reached]/limits['max_distance_m'])
            proposals.append(dict(oid=oid, local=reached, strength=strength,
                                  graph=graph, seed=seed, info=info))
            info['action'] = 'grow_from_current_core'
        # Ambiguous overlaps are boundaries. Recheck reachability after taking
        # them out, so an object cannot tunnel through another proposed object.
        occurrences = np.zeros(n, np.int32)
        for proposal in proposals:
            occurrences[proposal['local']] += 1
        for proposal in proposals:
            local = proposal['local']
            keep = occurrences[local] == 1
            retained = local[keep]
            _, labels = connected_components(proposal['graph'][retained][:, retained], directed=False)
            seeds = np.isin(retained, proposal['seed'])
            connected = np.isin(labels, np.unique(labels[seeds]))
            local = retained[connected]
            strength = proposal['strength'][keep][connected]
            proposal['info'].update(target_count=len(local), overlap_stops=int((~keep).sum()))
            if len(local):
                parts.append(dict(pair=pair, rows=rows[local], oid=proposal['oid'],
                                  weights=strength.astype(np.float32)))
    if not parts:
        return dict(**empty, report=report)
    # A stable GID claimed by different objects in different frames remains
    # uncertain. Repeated observations never become additional core evidence.
    claimed = np.concatenate([p['rows'] for p in parts])
    claims = np.concatenate([np.full(len(p['rows']), p['oid'], np.int64) for p in parts])
    low = np.full(len(gids), np.iinfo(np.int64).max, np.int64)
    high = np.full(len(gids), -1, np.int64)
    np.minimum.at(low, claimed, claims)
    np.maximum.at(high, claimed, claims)
    ambiguous = low < high
    report['conflicting_gaussians'] = int(ambiguous.sum())
    parts = [dict(p, rows=p['rows'][~ambiguous[p['rows']]],
                  weights=p['weights'][~ambiguous[p['rows']]]) for p in parts]
    parts = [p for p in parts if len(p['rows'])]
    frame_counts = {oid: sum(p['oid'] == oid for p in parts) for oid in cores}
    target_rows, target_columns, target_weights = [], [], []
    by_pair = {}
    for part in parts:
        oid = part['oid']
        strength = part['weights']/len(part['rows'])/frame_counts[oid]
        target_rows.append(part['rows'])
        target_columns.append(np.full(len(part['rows']), columns[oid], np.int64))
        target_weights.append(strength)
        by_pair.setdefault(part['pair'], []).append(dict(
            gaussian_ids=gids[part['rows']], object_id=oid, weights=part['weights']))
    if not parts:
        return dict(**empty, report=report)
    rows = np.concatenate(target_rows)
    report.update(target_entries=len(rows), unique_gaussians=len(np.unique(rows)),
                  object_ids=sorted({p['oid'] for p in parts}),
                  normalization='mean_within_source_frame_then_frames_then_objects')
    return dict(rows=rows, columns=np.concatenate(target_columns),
                weights=np.concatenate(target_weights), by_pair=by_pair, report=report)


def region_completion_loss(logp, rows, columns, weights):
    """Source-frame-normalized positive targets; gradients only to logits."""
    if not len(rows):
        return logp.sum()*0.
    loss = -weights*logp[rows, columns]
    sums = logp.new_zeros(logp.shape[1]).scatter_add(0, columns, loss)
    import torch
    present = torch.bincount(columns, minlength=logp.shape[1]) > 0
    return sums[present].mean()


def initialize_region_probabilities(probabilities, region, minimum=.1):
    """Give verified surfaces a recoverable soft start, keeping the first prior fixed."""
    if not np.isfinite(minimum) or not 0 <= minimum < 1:
        raise ValueError('region_completion.min_initial_probability must be in [0,1)')
    p = np.array(probabilities, copy=True)
    rows, first = np.unique(region['rows'], return_index=True)
    columns = region['columns'][first]
    selected = p[rows, columns] < minimum
    rows, columns = rows[selected], columns[selected]
    old = p[rows, columns].copy()
    p[rows] *= ((1-minimum)/(1-old))[:, None]
    p[rows, columns] = minimum
    return p, len(rows)

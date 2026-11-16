"""Core-guided connectivity targets for persistent Gaussian object logits.

The cached graph is built in each source camera separately. Only stable
Gaussian IDs join observations across cameras; their XYZ coordinates are never
concatenated into one metric space. Core selection consumes fresh motion costs,
not inherited track labels, old posteriors, component area, or current argmax.

Targets are detached CPU data. Multiplying the returned penalty by the current
soft probabilities supplies a differentiable loss for disconnected islands.
With no_core_to_background, objects lacking a reliable core instead have their
probability mass transferred to background and their winning rows forced to O0.
Object zero is exempt from the connectivity requirement.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _options(options):
    options = {} if options is None else options
    return dict(options.get('connectivity', options))


def _components(adjacency, selected):
    """Component IDs on an induced graph, -1 for excluded global rows."""
    rows = np.flatnonzero(selected)
    labels = np.full(len(selected), -1, np.int32)
    if len(rows):
        _, local = connected_components(adjacency[rows][:, rows], directed=False)
        labels[rows] = local
    return labels


def _snapshot_edges(snapshot, rows, options):
    payload = snapshot['payload']
    uv = np.asarray(payload['uv'], float)
    xyz = np.asarray(payload['xyz'], float)
    if uv.shape != (len(rows), 2) or xyz.shape != (len(rows), 3):
        raise ValueError('Connectivity requires aligned uv[N,2] and xyz[N,3]')
    valid = np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
    if not valid.any():
        return np.empty((0, 2), np.int64), np.empty(0, np.float32), 0
    xy = np.zeros_like(uv, np.int64)
    xy[valid] = np.rint(uv[valid]).astype(np.int64)
    valid &= (xy >= 0).all(1)
    if not valid.any():
        return np.empty((0, 2), np.int64), np.empty(0, np.float32), 0
    observations = snapshot.get('render_observations') or {}
    if 'source' in observations:
        height, width = np.asarray(observations['source']).shape[:2]
    else:
        width, height = np.max(xy[valid], axis=0)+1
    if height*width > int(options.get('max_image_pixels', 20_000_000)):
        raise ValueError('Connectivity source UV range is implausibly large')
    valid &= (xy[:, 0] < width) & (xy[:, 1] < height)
    local = np.flatnonzero(valid)
    pixels = xy[local, 1]*width+xy[local, 0]
    unique, first, counts = np.unique(pixels, return_index=True, return_counts=True)
    # Ambiguous pixel bindings do not create geometric identity shortcuts.
    clean = counts == 1
    grid = np.full(height*width, -1, np.int64)
    grid[unique[clean]] = local[first[clean]]
    grid = grid.reshape(height, width)
    neighbors = int(options.get('neighbors', 4))
    if neighbors not in (4, 8):
        raise ValueError('Connectivity neighbors must be 4 or 8')
    offsets = [(0, 1), (1, 0)]
    if neighbors == 8:
        offsets += [(1, 1), (1, -1)]
    color = np.asarray(payload.get('color', np.zeros((len(rows), 3))), float)
    if color.shape != (len(rows), 3) or not np.isfinite(color).all():
        raise ValueError('Connectivity color must be finite [N,3]')
    distance_base = float(options.get('edge_distance_m', .25))
    distance_relative = float(options.get('edge_distance_relative', .005))
    depth_base = float(options.get('edge_depth_m', .15))
    depth_relative = float(options.get('edge_depth_relative', .02))
    color_sigma = float(options.get('edge_color_sigma', .3))
    if min(distance_base, depth_base, color_sigma) <= 0 or min(distance_relative, depth_relative) < 0:
        raise ValueError('Connectivity edge scales must be positive and relative terms nonnegative')
    edges, weights = [], []
    for dy, dx in offsets:
        if dy >= height or abs(dx) >= width:
            continue
        y0, y1 = slice(0, height-dy), slice(dy, height)
        if dx >= 0:
            a, b = grid[y0, :width-dx], grid[y1, dx:]
        else:
            a, b = grid[y0, -dx:], grid[y1, :width+dx]
        seen = (a >= 0) & (b >= 0)
        left, right = a[seen], b[seen]
        z = np.minimum(xyz[left, 2], xyz[right, 2])
        keep = ((np.abs(xyz[left, 2]-xyz[right, 2]) <= depth_base+depth_relative*z) &
                (np.linalg.norm(xyz[left]-xyz[right], axis=1) <= distance_base+distance_relative*z))
        left, right = left[keep], right[keep]
        edge = np.sort(np.column_stack((rows[left], rows[right])), axis=1)
        distinct = edge[:, 0] != edge[:, 1]
        edges.append(edge[distinct])
        # Appearance is a soft affinity, never a hard edge rejection: windows
        # and bodywork can remain connected when their colors differ greatly.
        difference = np.linalg.norm(color[left]-color[right], axis=1)
        weights.append(np.exp(-.5*(difference[distinct]/color_sigma)**2).astype(np.float32))
    return (np.concatenate(edges) if edges else np.empty((0, 2), np.int64),
            np.concatenate(weights) if weights else np.empty(0, np.float32),
            int(np.count_nonzero(~clean)))


def _height_gate(height_info, count, options):
    if height_info is None:
        return np.ones(count, bool), dict(height_available=False)
    height = np.asarray(height_info['height'], float)
    valid = np.asarray(height_info.get('valid', np.isfinite(height)), bool)
    if height.shape != (count,) or valid.shape != (count,):
        raise ValueError('height_info height/valid must align with active Gaussian rows')
    minimum = float(options.get('core_height_min', -.15))
    maximum = float(options.get('core_height_max', 3.5))
    if minimum >= maximum:
        raise ValueError('core_height_min must be below core_height_max')
    # Unknown ground does not assert that a point has an impossible height.
    known = valid & np.isfinite(height)
    if 'allowed' in height_info:
        allowed = np.asarray(height_info['allowed'], bool)
        if allowed.shape != (count,):
            raise ValueError('height_info allowed must align with active Gaussian rows')
    else:
        allowed = (height >= minimum) & (height <= maximum)
    keep = ~known | allowed
    keep &= ~np.asarray(height_info.get('hard_background_mask', np.zeros(count, bool)), bool)
    return keep, dict(height_available=True, known_height_rows=int(known.sum()),
                      core_height_rejected=int(np.count_nonzero(~keep)))


def build_connectivity_state(window, options=None, height_info=None):
    """Build a cached CPU graph and reliable foreground core sets.

    Requires the ObjectWindow active snapshot protocol. ``height_info`` may
    contain signed above-ground ``height[Nactive]`` and ``valid[Nactive]``.
    Defaults require two distinct frame-pair observations with low object cost
    and a clear advantage over O0. A tracked core is preferred; an untracked
    core requires extra observations. No reliable core means no constraint.
    """
    options = _options(options)
    gids = np.asarray(window._active_gids, np.int64).copy()
    objects = np.asarray(window.object_ids, np.int64).copy()
    if len(np.unique(gids)) != len(gids) or np.any(gids[1:] <= gids[:-1]):
        raise ValueError('Active Gaussian IDs must be unique and sorted')
    n, k = len(gids), len(objects)
    column = {int(oid): j for j, oid in enumerate(objects)}
    background = int(options.get('background_object_id', 0))
    bg_column = column.get(background)
    counts = np.zeros((n, k), np.uint16)
    positives = np.zeros_like(counts)
    advantages = np.zeros((n, k), np.float32)
    evidence_weights = np.zeros((n, k), np.float32)
    fit_seen = np.zeros((n, k), bool)
    model_observed = np.zeros((n, k), bool)
    tracked = np.zeros(n, bool)
    raw_edges, edge_weights = [], []
    ambiguous_pixels = 0
    margin = float(options.get('core_advantage', .3))
    cost_max = float(options.get('core_cost_max', .5))
    minimum_weight = float(options.get('core_min_weight', .15))
    margin_cap = float(options.get('advantage_cap', 3.))
    if min(margin, cost_max, margin_cap) <= 0 or not 0 <= minimum_weight <= 1:
        raise ValueError('Invalid core advantage/cost/reliability configuration')
    for pair in sorted(window._active):
        snap = window._active[pair]
        local_gids = np.asarray(snap['gaussian_ids'], np.int64)
        rows = np.searchsorted(gids, local_gids)
        if (np.any(rows >= n) or not np.array_equal(gids[rows], local_gids) or
                len(np.unique(rows)) != len(rows)):
            raise ValueError('Connectivity snapshots must contain unique active Gaussian IDs')
        edges, weights, ambiguous = _snapshot_edges(snap, rows, options)
        raw_edges.append(edges)
        edge_weights.append(weights)
        ambiguous_pixels += ambiguous
        if snap.get('track_ids') is not None:
            tracks = np.asarray(snap['track_ids'])
            if tracks.shape != (len(rows),):
                raise ValueError('Snapshot track IDs must align with Gaussian rows')
            tracked[rows[tracks >= 0]] = True
        local_objects = np.asarray(snap['object_ids'], np.int64)
        bg = np.flatnonzero(local_objects == background)
        if len(bg) != 1:
            continue
        costs = np.asarray(snap['costs'], float)
        weight = np.asarray(snap['weights'], float)
        if costs.shape != (len(rows), len(local_objects)):
            raise ValueError('Snapshot costs must align with Gaussian/object rows')
        if weight.shape == (len(rows),):
            weight = np.broadcast_to(weight[:, None], costs.shape)
        if weight.shape != costs.shape:
            raise ValueError('Snapshot weights must have shape [N] or [N,K]')
        b = int(bg[0])
        for local, oid in enumerate(local_objects):
            if int(oid) == background or int(oid) not in column:
                continue
            c = column[int(oid)]
            own_fit = (np.isfinite(costs[:, local]) &
                       (costs[:, local] < float(options.get('bridge_cost_max', 1.))) &
                       np.isfinite(weight[:, local]) & (weight[:, local] >= minimum_weight))
            model_observed[rows[own_fit], c] = True
            confidence = np.minimum(weight[:, b], weight[:, local])
            observed = (np.isfinite(costs[:, b]) & np.isfinite(costs[:, local]) &
                        np.isfinite(confidence) & (confidence > 0))
            ids = np.flatnonzero(observed)
            target_rows = rows[ids]
            advantage = np.clip(costs[ids, b]-costs[ids, local], -margin_cap, margin_cap)
            reliability = np.clip(confidence[ids], 0, 1)
            counts[target_rows, c] += 1
            evidence_weights[target_rows, c] += reliability
            advantages[target_rows, c] += (reliability*advantage).astype(np.float32)
            good = ((costs[ids, local] < cost_max) & (advantage >= margin) &
                    (reliability >= minimum_weight))
            positives[target_rows[good], c] += 1
            fit_seen[target_rows[costs[ids, local] < float(options.get('bridge_cost_max', 1.))], c] = True
    edges = np.concatenate(raw_edges) if raw_edges else np.empty((0, 2), np.int64)
    weights = np.concatenate(edge_weights) if edge_weights else np.empty(0, np.float32)
    if len(edges):
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, weights = edges[order], weights[order]
        starts = np.flatnonzero(np.r_[True, np.any(edges[1:] != edges[:-1], axis=1)])
        weights = np.maximum.reduceat(weights, starts)
        edges = edges[starts]
        left = np.r_[edges[:, 0], edges[:, 1]]
        right = np.r_[edges[:, 1], edges[:, 0]]
        affinity = np.r_[weights, weights]
        # Prevent underflow from converting a color difference into a hard cut.
        affinity = np.maximum(affinity, np.finfo(np.float32).tiny)
        adjacency = sparse.csr_matrix((affinity, (left, right)), shape=(n, n))
    else:
        adjacency = sparse.csr_matrix((n, n), dtype=np.float32)
    observed = counts > 0
    mean_advantage = np.divide(advantages, evidence_weights, out=np.zeros_like(advantages),
                               where=evidence_weights > 0)
    fraction = np.divide(positives, counts, out=np.zeros_like(advantages), where=counts > 0)
    height_keep, height_report = _height_gate(height_info, n, options)
    minimum_frames = int(options.get('core_min_frames', 2))
    minimum_points = int(options.get('core_min_points', 3))
    untracked_frames = int(options.get('core_untracked_min_frames', minimum_frames+1))
    minimum_fraction = float(options.get('core_support_fraction', .6))
    bridge_margin = float(options.get('bridge_advantage_tolerance', .1))
    if min(minimum_frames, minimum_points) < 1 or not 0 < minimum_fraction <= 1 or bridge_margin < 0:
        raise ValueError('Invalid core support or bridge configuration')
    compatible = observed & fit_seen & (mean_advantage >= -bridge_margin)
    cores, object_reports = {}, {}
    fragmented_candidates = {}
    for c, oid_value in enumerate(objects):
        oid = int(oid_value)
        if oid == background:
            object_reports[str(oid)] = dict(action='background_exempt', core_count=0)
            continue
        candidate = ((positives[:, c] >= minimum_frames) & (fraction[:, c] >= minimum_fraction) &
                     (mean_advantage[:, c] >= margin) & height_keep)
        candidate &= tracked | (positives[:, c] >= untracked_frames)
        rows = np.flatnonzero(candidate)
        if not len(rows):
            object_reports[str(oid)] = dict(action='skip_no_reliable_core', core_count=0,
                                             observed_rows=int(observed[:, c].sum()))
            continue
        components = _components(adjacency, compatible[:, c] | candidate)
        candidates = []
        for component in np.unique(components[rows]):
            selected = rows[components[rows] == component]
            if len(selected) < minimum_points:
                continue
            tracks = selected[tracked[selected]]
            quality_rows = tracks if len(tracks) else selected
            quality = float(np.median(mean_advantage[quality_rows, c]))
            persistence = float(np.median(positives[quality_rows, c]))
            # Evidence quality wins, never component area/current posterior.
            # Count is only a capped final tie breaker after actual evidence.
            score = (bool(len(tracks)), quality, persistence,
                     float(np.mean(mean_advantage[quality_rows, c])), min(len(quality_rows), 8))
            candidates.append((score, selected))
        if not candidates:
            object_reports[str(oid)] = dict(action='skip_fragmented_core', core_count=0,
                                             candidate_count=int(len(rows)))
            fragmented_candidates[oid] = rows
            continue
        _, selected = max(candidates, key=lambda item: item[0])
        cores[oid] = selected
        object_reports[str(oid)] = dict(action='constrain', core_count=int(len(selected)),
                                         tracked_core_count=int(tracked[selected].sum()),
                                         rejected_core_candidates=int(len(rows)-len(selected)),
                                         median_motion_advantage=float(np.median(mean_advantage[selected, c])))
    adjacency_by_object, core_link_edges, object_core_groups = {}, {}, {}
    link_report = dict(enabled=bool(options.get('core_evidence_links', {}).get('enabled', False)),
                       objects={})
    if link_report['enabled'] and fragmented_candidates:
        from ownership.core_evidence_links import build_core_evidence_links
        linked = build_core_evidence_links(window, fragmented_candidates, options, height_keep)
        link_report.update(linked['report'])
        for oid, groups in linked['groups_by_object'].items():
            c = column[oid]
            eligible = []
            for selected in groups:
                tracks = selected[tracked[selected]]
                quality_rows = tracks if len(tracks) else selected
                score = (bool(len(tracks)), float(np.median(mean_advantage[quality_rows, c])),
                         float(np.median(positives[quality_rows, c])),
                         float(np.mean(mean_advantage[quality_rows, c])),
                         min(len(quality_rows), 8), -int(gids[selected].min()))
                eligible.append((score, selected))
            if not eligible:
                continue
            _, selected = max(eligible, key=lambda item: item[0])
            links = linked['edges_by_object'][oid]
            links = links[np.isin(links[:, 0], selected) & np.isin(links[:, 1], selected)]
            extra = sparse.csr_matrix((np.ones(2*len(links), np.float32),
                                       (np.r_[links[:, 0], links[:, 1]],
                                        np.r_[links[:, 1], links[:, 0]])), shape=(n, n))
            # Endpoint links certify only this object's independently supported
            # core. They never assign path interiors or alter another graph.
            adjacency_by_object[oid] = adjacency.maximum(extra)
            core_link_edges[oid] = links
            object_core_groups[oid] = [selected.copy()]
            cores[oid] = selected
            object_reports[str(oid)] = dict(
                action='constrain_linked_core', core_count=int(len(selected)),
                tracked_core_count=int(tracked[selected].sum()),
                rejected_core_candidates=int(len(fragmented_candidates[oid])-len(selected)),
                median_motion_advantage=float(np.median(mean_advantage[selected, c])),
                bounded_core_link_edges=int(len(links)))
    return dict(active_gids=gids, object_ids=objects, edges=edges, adjacency=adjacency,
                adjacency_by_object=adjacency_by_object, core_link_edges=core_link_edges,
                object_core_groups=object_core_groups,
                observed=observed, model_observed=model_observed,
                compatible=compatible, mean_advantage=mean_advantage,
                height_keep=height_keep,
                core_by_object=cores, options=options,
                diagnostics=dict(nodes=n, edges=int(len(edges)), source_snapshots=len(window._active),
                                 ambiguous_source_pixels=ambiguous_pixels, objects=object_reports,
                                 core_evidence_links=link_report,
                                 background_object_id=background, **height_report))


def condition_core_initialization(state, probabilities):
    """Condition certified-core winners on supported candidates, before fallback.

    This is an optimizer proposal, not evidence that a model is physically
    moving. Only original full-column winners with a certified endpoint group
    qualify. First priors, costs, energy, cores and input probabilities stay
    immutable; ordinary no-core and final connectivity rules still apply.
    """
    p = np.array(probabilities, dtype=np.float64, copy=True)
    objects = np.asarray(state['object_ids'])
    if (p.shape != (len(state['active_gids']), len(objects)) or
            not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(1), 1., atol=1e-5)):
        raise ValueError('Core initialization requires normalized active Gaussian/object probabilities')
    options = state['options']
    mode = options.get('initialization_conditioning', 'none')
    if mode not in ('none', 'certified_core_winner'):
        raise ValueError('Unknown connectivity.initialization_conditioning')
    report = dict(mode=mode, enabled=mode != 'none', eligible_object_ids=[],
                  selected_count=0, removed_probability_mass=0., zero_mass_count=0,
                  interpretation='Optimizer initialization only; not physical motion validation')
    if mode == 'none':
        return dict(probabilities=p, report=report)
    background = int(options.get('background_object_id', 0))
    if np.count_nonzero(objects == background) != 1:
        raise ValueError('Core initialization requires the configured background object ID')
    minimum = int(options.get('core_min_points', 3))
    eligible = []
    for oid_value, groups in state.get('object_core_groups', {}).items():
        oid = int(oid_value)
        if oid == background or oid not in objects:
            continue
        core = np.asarray(state['core_by_object'].get(oid, ()))
        for group in groups:
            group = np.asarray(group)
            if (group.ndim == 1 and np.issubdtype(group.dtype, np.integer) and
                    len(np.unique(group)) >= minimum and
                    np.all((group >= 0) & (group < len(p))) and
                    np.isin(group, core).all()):
                eligible.append(oid)
                break
    eligible = sorted(set(eligible))
    if not eligible:
        return dict(probabilities=p, report=report)
    # A probability tie is resolved by object ID, independent of column order.
    order = np.argsort(objects, kind='stable')
    winners = (objects[p.argmax(1)] if np.array_equal(order, np.arange(len(objects)))
               else objects[order[p[:, order].argmax(1)]])
    selected = np.isin(winners, eligible)
    disabled = np.asarray([int(oid) != background and
                           not len(state['core_by_object'].get(int(oid), ()))
                           for oid in objects])
    rows = np.flatnonzero(selected)
    removed = float(p[np.ix_(rows, np.flatnonzero(disabled))].sum())
    p[np.ix_(rows, np.flatnonzero(disabled))] = 0.
    mass = p[rows].sum(1)
    positive = mass > 0
    p[rows[positive]] /= mass[positive, None]
    zero = rows[~positive]
    p[zero] = 0.
    p[zero, int(np.flatnonzero(objects == background)[0])] = 1.
    report.update(eligible_object_ids=eligible, selected_count=int(selected.sum()),
                  removed_probability_mass=removed, zero_mass_count=int(len(zero)))
    return dict(probabilities=p, report=report)


def no_core_background_policy(state, probabilities):
    """Disable unsupported object columns without donating mass to other objects.

    Winners of a disabled object become exact background. For other rows, only
    the disabled probability mass is transferred to O0; supported object mass
    stays unchanged. This is evaluated once after the window's core selection.
    """
    p = np.array(probabilities, dtype=np.float64, copy=True)
    objects = np.asarray(state['object_ids'])
    if (p.shape != (len(state['active_gids']), len(objects)) or not np.isfinite(p).all() or
            np.any(p < 0) or not np.allclose(p.sum(1), 1., atol=1e-5)):
        raise ValueError('No-core fallback requires normalized active Gaussian/object probabilities')
    enabled = bool(state['options'].get('no_core_to_background', False))
    disabled = np.zeros(len(objects), bool)
    forced = np.zeros(len(p), bool)
    background = int(state['options'].get('background_object_id', 0))
    transferred = 0.
    if enabled:
        if background not in objects:
            raise ValueError('No-core fallback requires the configured background object ID')
        bg = int(np.flatnonzero(objects == background)[0])
        disabled = np.asarray([int(oid) != background and
                               not len(state['core_by_object'].get(int(oid), ())) for oid in objects])
        forced = disabled[p.argmax(1)]
        mass = p[:, disabled].sum(1)
        transferred = float(mass.sum())
        p[:, disabled] = 0.
        p[:, bg] += mass
        p[forced] = 0.
        p[forced, bg] = 1.
    return dict(probabilities=p, forced_background_mask=forced, disabled_columns=disabled,
                report=dict(enabled=enabled, background_object_id=background,
                            disabled_object_ids=objects[disabled].astype(int).tolist(),
                            forced_background_gaussians=int(forced.sum()),
                            transferred_probability_mass=transferred,
                            assignment='winning_disabled_object_to_exact_background',
                            other_rows='disabled_probability_mass_to_background'))


def update_connectivity_targets(state, probabilities):
    """Refresh detached penalties/anchors, e.g. every five Adam iterations.

    Only observed, currently high-probability disconnected regions and their
    immediate halo are penalized. Unobserved nodes are neither deleted nor
    assumed to be disconnected. Low-probability paths cannot connect a distant
    island. By default paths also require compatible raw motion costs. With
    ``bridge_require_cost_compatibility=False``, fresh observations, the current
    winning object, probability and height determine surface paths; reliable
    motion costs still select every core and remain part of the loss.
    """
    p = np.asarray(probabilities)
    objects = state['object_ids']
    n, k = len(state['active_gids']), len(objects)
    if (p.shape != (n, k) or not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(1), 1, atol=1e-5)):
        raise ValueError('Connectivity probabilities must be normalized [Nactive,Kglobal]')
    options, adjacency = state['options'], state['adjacency']
    high = float(options.get('high_probability', .35))
    halo_probability = float(options.get('halo_probability', .1))
    halo_hops = int(options.get('halo_hops', 1))
    core_target = float(options.get('core_target', .9))
    require_compatible = bool(options.get('bridge_require_cost_compatibility', True))
    if not 0 < high <= 1 or not 0 <= halo_probability <= high or not 0 < core_target <= 1 or halo_hops < 0:
        raise ValueError('Invalid connectivity probability/halo configuration')
    penalty = np.zeros((n, k), np.float32)
    seed_rows, seed_columns = [], []
    report = dict(objects={}, nodes=n, edges=len(state['edges']),
                  detached_targets=True, forces_background_assignment=False,
                  bridge_require_cost_compatibility=require_compatible,
                  bridge_require_background_observation=bool(options.get(
                      'bridge_require_background_observation', True)))
    winner = None if require_compatible else p.argmax(axis=1)
    if winner is not None:
        best = p[np.arange(n), winner]
        for c in np.argsort(objects):
            tied = (p[:, c] == best) & (objects[winner] > objects[c])
            winner[tied] = c
    height_keep = np.asarray(state.get('height_keep', np.ones(n, bool)), bool)
    for c, oid_value in enumerate(objects):
        oid = int(oid_value)
        adjacency = state.get('adjacency_by_object', {}).get(oid, state['adjacency'])
        core = state['core_by_object'].get(oid)
        if core is None or not len(core):
            report['objects'][str(oid)] = dict(action=state['diagnostics']['objects'][str(oid)]['action'],
                                                penalized_count=0, core_count=0)
            continue
        eligible = state['observed'][:, c]
        if require_compatible:
            # Keep the historical cost-gated path and halo unchanged unless
            # the explicit surface-path option is selected.
            high_rows = eligible & (p[:, c] >= high)
            path = high_rows & state['compatible'][:, c]
            path[core] = True
            halo_allowed = eligible & state['compatible'][:, c] & (p[:, c] >= halo_probability)
        else:
            if not options.get('bridge_require_background_observation', True):
                eligible = state.get('model_observed', state['observed'])[:, c]
            eligible = eligible & height_keep
            high_rows = eligible & (winner == c) & (p[:, c] >= high)
            path = high_rows.copy()
            # A stale/background core still receives its positive anchor loss,
            # but cannot connect surfaces until it wins with fresh support.
            path[core] &= state['compatible'][core, c]
            halo_allowed = path
        components = _components(adjacency, path)
        core_components = np.unique(components[core])
        core_components = core_components[core_components >= 0]
        protected = path & np.isin(components, core_components)
        # A short compatible boundary halo lets a partially uncertain body
        # attach to its anchors; color affinity controls only soft halo weight.
        frontier = protected.copy()
        for _ in range(halo_hops):
            adjacent = np.asarray(adjacency @ frontier.astype(np.float32)).reshape(-1) > 0
            frontier = adjacent & halo_allowed & ~protected
            protected |= frontier
        disconnected = high_rows & ~protected
        value = disconnected.astype(np.float32)
        frontier = disconnected.astype(np.float32)
        for _ in range(halo_hops):
            frontier = np.minimum(np.asarray(adjacency @ frontier).reshape(-1), 1.)
            frontier *= eligible & ~protected & (p[:, c] >= halo_probability)
            value = np.maximum(value, frontier)
        # More reliable contrary motion evidence strengthens an island's
        # penalty; no mass is explicitly assigned to any alternative object.
        contrary = np.clip(-state['mean_advantage'][:, c], 0, 3.)/3.
        penalty[:, c] = value*(1+contrary)
        seed_rows.extend(core.tolist())
        seed_columns.extend([c]*len(core))
        report['objects'][str(oid)] = dict(action='constrain', core_count=int(len(core)),
                                            protected_count=int(protected.sum()),
                                            disconnected_high_count=int(disconnected.sum()),
                                            penalized_count=int(np.count_nonzero(value)))
    seeds = np.asarray(seed_rows, np.int64)
    columns = np.asarray(seed_columns, np.int64)
    report['penalized_entries'] = int(np.count_nonzero(penalty))
    report['seed_entries'] = int(len(seeds))
    return dict(penalty=penalty, seed_rows=seeds, seed_columns=columns,
                seed_targets=np.full(len(seeds), core_target, np.float32), report=report)


def connectivity_losses(probabilities, targets):
    """Optional torch helper: detached targets, gradients only to probabilities.

    Returns separate ``disconnected`` and positive ``anchor`` terms so the
    caller explicitly chooses their weights. Torch is imported only on use.
    """
    import torch
    penalty = torch.as_tensor(targets['penalty'], dtype=probabilities.dtype, device=probabilities.device)
    if tuple(probabilities.shape) != tuple(penalty.shape):
        raise ValueError('Probability/penalty shapes differ')
    disconnected = (probabilities*penalty).sum()/torch.count_nonzero(penalty).clamp_min(1)
    rows = torch.as_tensor(targets['seed_rows'], dtype=torch.long, device=probabilities.device)
    columns = torch.as_tensor(targets['seed_columns'], dtype=torch.long, device=probabilities.device)
    if len(rows):
        target = torch.as_tensor(targets['seed_targets'], dtype=probabilities.dtype,
                                 device=probabilities.device)
        anchor = -(target*probabilities[rows, columns].clamp_min(1e-12).log()).mean()
    else:
        anchor = probabilities.sum()*0.
    return dict(disconnected=disconnected, anchor=anchor)

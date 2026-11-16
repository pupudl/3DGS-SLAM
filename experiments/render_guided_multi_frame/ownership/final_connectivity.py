"""Project final object assignments onto motion-supported connected branches.

This is a mandatory assignment projection, separate from the differentiable
connectivity loss. It does not evaluate or claim improvement of a render loss.
The window graph may join observations only through already established GIDs.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def _assign(probabilities, objects, support, minimum, margin, forced, background):
    """Apply the window's confidence rule without a full partition copy."""
    n, k = probabilities.shape
    rows = np.arange(n)
    winner = probabilities.argmax(axis=1)
    best = probabilities[rows, winner]
    # Equal probabilities must not depend on the storage order of columns.
    for c in np.argsort(objects):
        tied = (probabilities[:, c] == best) & (objects[winner] > objects[c])
        winner[tied] = c
    second = np.zeros(n, np.float64)
    if k > 1:
        for c in range(k):
            np.maximum(second, np.where(winner == c, 0., probabilities[:, c]), out=second)
    confident = (support[rows, winner] & (best >= minimum) & (best-second >= margin))
    labels = np.full(n, -1, np.int64)
    labels[confident] = objects[winner[confident]]
    labels[forced] = objects[background]
    return winner, labels


def finalize_connectivity(state, probabilities, *, assignment_support,
                          min_probability, min_margin, hard_background_mask=None):
    """Return normalized probabilities and hard labels after final connectivity.

    Anchors must still confidently belong to their motion-supported object.
    Low-confidence rows can connect them only while that same object wins and
    fresh observed motion is compatible by default. The optional posterior
    surface path skips that extra raw-cost gate only for intermediate surfaces;
    core anchors always require compatible motion. A branch needs ``core_min_points``
    owned anchors. Competing branches are selected by anchor motion advantage,
    never by the number of pixels in the branch. Independently certified local
    core groups retain their structural relation between at least three owned
    anchors, without inserting a background/unknown member or path interior.

    Decisions for all objects are made from the same posterior, then applied
    together. Background mass transfers may invalidate another object's core;
    the monotone projection repeats until its assignments are stable. Inputs
    and the cached graph are not mutated.
    """
    p = np.array(probabilities, dtype=np.float64, copy=True)
    objects = np.asarray(state['object_ids'], np.int64)
    gids = np.asarray(state['active_gids'], np.int64)
    n, k = len(gids), len(objects)
    if (p.shape != (n, k) or not k or not np.isfinite(p).all() or np.any(p < 0) or
            not np.allclose(p.sum(axis=1), 1., atol=1e-5)):
        raise ValueError('Final connectivity requires normalized active Gaussian/object probabilities')
    if objects.ndim != 1 or len(np.unique(objects)) != k or gids.shape != (n,):
        raise ValueError('Final connectivity requires unique object IDs and aligned Gaussian IDs')
    support = np.asarray(assignment_support, bool)
    observed = np.asarray(state['observed'], bool)
    compatible = np.asarray(state['compatible'], bool)
    if any(value.shape != (n, k) for value in (support, observed, compatible)):
        raise ValueError('Assignment support, observed and compatible must align with probabilities')
    adjacency = state['adjacency']
    if not sparse.issparse(adjacency) or adjacency.shape != (n, n):
        raise ValueError('Final connectivity requires an aligned sparse adjacency graph')
    adjacency = adjacency.tocsr(copy=False)
    options = state.get('options', {})
    background_id = int(options.get('background_object_id', 0))
    if background_id not in objects:
        raise ValueError('Final connectivity requires the configured background object ID')
    background = int(np.flatnonzero(objects == background_id)[0])
    minimum_points = int(options.get('core_min_points', 3))
    bridge_probability = float(options.get('final_bridge_probability', .35))
    require_compatible = bool(options.get('bridge_require_cost_compatibility', True))
    surface_observations = observed
    if not require_compatible and not options.get('bridge_require_background_observation', True):
        surface_observations = np.asarray(state.get('model_observed', observed), bool)
        if surface_observations.shape != (n, k):
            raise ValueError('Surface model observations must align with probabilities')
    if (minimum_points < 1 or not 0 <= bridge_probability <= 1 or
            not 0 <= min_probability <= 1 or not 0 <= min_margin <= 1):
        raise ValueError('Invalid final connectivity core, bridge or assignment thresholds')
    height_keep = np.asarray(state.get('height_keep', np.ones(n, bool)), bool)
    supplied_hard = (np.zeros(n, bool) if hard_background_mask is None else
                     np.asarray(hard_background_mask, bool))
    if height_keep.shape != (n,) or supplied_hard.shape != (n,):
        raise ValueError('Height and hard background masks must align with active Gaussian IDs')
    advantages = state.get('mean_advantage')
    if advantages is not None:
        advantages = np.asarray(advantages)
        if advantages.shape != (n, k):
            raise ValueError('Mean motion advantages must align with probabilities')
    cores, structural_groups = {}, {}
    for oid in objects:
        raw = np.asarray(state.get('core_by_object', {}).get(int(oid), ()), np.int64)
        if raw.ndim != 1 or np.any(raw < 0) or np.any(raw >= n):
            raise ValueError('Reliable core rows must be valid active Gaussian indices')
        cores[int(oid)] = np.unique(raw)
        structural_groups[int(oid)] = []
        for group in state.get('object_core_groups', {}).get(int(oid), ()):
            group = np.asarray(group, np.int64)
            if group.ndim != 1 or np.any(group < 0) or np.any(group >= n):
                raise ValueError('Certified structural core groups require valid active row indices')
            structural_groups[int(oid)].append(np.unique(group))

    forced = supplied_hard.copy()
    p[forced] = 0.
    p[forced, background] = 1.
    disabled = np.zeros(k, bool)
    rounds, object_reports = [], {}
    transferred_mass = 0.
    updates = 0
    while True:
        winner, labels = _assign(p, objects, support, min_probability, min_margin,
                                 forced, background)
        disable_next = np.zeros(k, bool)
        force_next = np.zeros(n, bool)
        decisions = {}
        for c, oid_value in enumerate(objects):
            oid = int(oid_value)
            if c == background:
                decisions[str(oid)] = dict(action='background_exempt')
                continue
            if disabled[c]:
                continue
            raw_core = cores[oid]
            valid_core = raw_core[(labels[raw_core] == oid) & height_keep[raw_core] &
                                  observed[raw_core, c] & compatible[raw_core, c] &
                                  ~forced[raw_core]]
            report = dict(input_core_count=int(len(raw_core)),
                          owned_core_count=int(len(valid_core)),
                          structural_owned_core_groups=[])
            if len(valid_core) < minimum_points:
                disable_next[c] = True
                force_next |= winner == c
                report.update(action='disable_no_final_core', retained_count=0)
                decisions[str(oid)] = report
                continue

            path = ((winner == c) & (p[:, c] >= bridge_probability) &
                    surface_observations[:, c] & height_keep & ~forced)
            if require_compatible:
                path &= compatible[:, c]
            else:
                # Even in the posterior surface mode, a cost-core rejected by
                # its current motion evidence cannot serve as an implicit
                # anchor or a path endpoint merely because its posterior wins.
                path[raw_core] &= compatible[raw_core, c]
            # These anchors have fresh motion evidence and a CONFIRMED final
            # label. This never restores an anchor assigned to O0/another object.
            path[valid_core] = True
            rows = np.flatnonzero(path)
            object_graph = state.get('adjacency_by_object', {}).get(oid, adjacency)
            if not sparse.issparse(object_graph) or object_graph.shape != (n, n):
                raise ValueError('Per-object core connectivity must align with active Gaussian IDs')
            selected_graph = object_graph[rows][:, rows]
            structural_left, structural_right = [], []
            for group in structural_groups[oid]:
                owned = np.intersect1d(group, valid_core, assume_unique=True)
                if len(owned) < minimum_points:
                    continue
                # The group was independently certified by bounded, single-
                # camera geometry paths. Its identity relation is not a tree
                # that depends on every endpoint winning the render posterior.
                # Connect only CURRENTLY CONFIRMED endpoints: background and
                # unknown members never participate in this structural edge.
                positions = np.searchsorted(rows, owned)
                hub = int(positions[np.argmin(gids[owned])])
                others = positions[positions != hub]
                structural_left.extend(np.full(len(others), hub, np.int64).tolist())
                structural_right.extend(others.tolist())
                report['structural_owned_core_groups'].append(dict(
                    certified_core_count=int(len(group)), owned_core_count=int(len(owned)),
                    owned_gaussian_ids=gids[owned].astype(int).tolist(),
                    added_endpoint_edges=int(len(others)), intermediate_rows_assigned=False))
            if structural_left:
                extra = sparse.csr_matrix((np.ones(2*len(structural_left), np.float32),
                                           (np.r_[structural_left, structural_right],
                                            np.r_[structural_right, structural_left])),
                                          shape=(len(rows), len(rows)))
                selected_graph = selected_graph.maximum(extra)
            _, components = connected_components(selected_graph, directed=False)
            core_components = components[np.searchsorted(rows, valid_core)]
            ids, counts = np.unique(core_components, return_counts=True)
            candidates = []
            for component in ids[counts >= minimum_points]:
                anchors = valid_core[core_components == component]
                quality = (np.zeros(len(anchors)) if advantages is None else
                           advantages[anchors, c])
                quality = quality[np.isfinite(quality)]
                median = float(np.median(quality)) if len(quality) else -np.inf
                mean = float(np.mean(quality)) if len(quality) else -np.inf
                score = (median, mean, min(len(anchors), 8), -int(gids[anchors].min()))
                candidates.append((score, int(component), len(anchors)))
            if not candidates:
                disable_next[c] = True
                force_next |= winner == c
                report.update(action='disable_fragmented_final_core', retained_count=0)
                decisions[str(oid)] = report
                continue
            _, selected, selected_core_count = max(candidates, key=lambda item: item[0])
            retained = rows[components == selected]
            disconnected = winner == c
            disconnected[retained] = False
            force_next |= disconnected
            report.update(action='retain_core_connected_branch',
                          selected_core_count=int(selected_core_count),
                          retained_count=int(len(retained)),
                          disconnected_winner_count=int(disconnected.sum()),
                          low_confidence_bridge_count=int(np.count_nonzero(labels[retained] != oid)),
                          core_supported_branch_count=int(len(candidates)))
            decisions[str(oid)] = report

        new_disabled = disable_next & ~disabled
        new_forced = force_next & ~forced
        rounds.append(dict(iteration=updates,
                           disabled_object_ids=objects[new_disabled].astype(int).tolist(),
                           forced_background_count=int(new_forced.sum()), objects=decisions))
        object_reports.update(decisions)
        if not np.any(new_disabled) and not np.any(new_forced):
            break
        disabled |= new_disabled
        mass = np.zeros(n, np.float64)
        for c in np.flatnonzero(new_disabled):
            mass += p[:, c]
            p[:, c] = 0.
        p[:, background] += mass
        transferred_mass += float(mass.sum())
        # A row whose foreground winner was displaced by a mandatory mass
        # transfer is an explicit background decision, even without RGB support.
        next_winner, _ = _assign(p, objects, support, min_probability, min_margin,
                                  forced, background)
        displaced = (winner != background) & (next_winner == background)
        forced |= new_forced | displaced
        p[forced] = 0.
        p[forced, background] = 1.
        updates += 1
        # Each update permanently disables a column or fixes at least one row.
        if updates > n+k:
            raise RuntimeError('Final connectivity projection failed to converge monotonically')

    report = dict(stage='final_hard_assignment_projection', projection_only=True,
                  objective_evaluated=False, loss_improvement_claimed=False,
                  background_object_id=background_id, core_min_points=minimum_points,
                  bridge_probability=bridge_probability, projection_updates=updates,
                  bridge_require_cost_compatibility=require_compatible,
                  bridge_require_background_observation=bool(options.get(
                      'bridge_require_background_observation', True)),
                  graph_scope='active_window_shared_gaussian_ids',
                  component_selection='core_motion_advantage_then_capped_core_count_then_min_gid',
                  disabled_object_ids=objects[disabled].astype(int).tolist(),
                  forced_background_gaussians=int(forced.sum()),
                  additional_forced_background_gaussians=int(np.count_nonzero(forced & ~supplied_hard)),
                  transferred_probability_mass=transferred_mass,
                  objects=object_reports, rounds=rounds)
    return dict(probabilities=p, labels=labels, forced_background_mask=forced,
                disabled_columns=disabled, report=report)

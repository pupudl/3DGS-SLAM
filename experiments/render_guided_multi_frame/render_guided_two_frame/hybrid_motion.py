"""Depth-aware multi-motion initialization from 2D and metric hypotheses.

This module is deliberately CPU-only: saved matches can be replayed without
loading the Gaussian renderer. Hypotheses never consume matches. Membership is
resolved jointly, after essential-matrix, PnP and rigid-fit proposals compete.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial import cKDTree


def motion_costs(anchors, models, K, cfg):
    """Return N x K dimensionless costs, using the observed pixel matches.

    ``cfg`` is the ``motion.hybrid`` mapping, rather than the whole motion cfg.
    Reprojection is symmetric; capped depth consistency provides extra metric
    discrimination without allowing a single noisy depth to dominate.
    """
    a, b = np.asarray(anchors['x0'], float), np.asarray(anchors['x1'], float)
    uv = np.asarray(anchors['matches'], float)
    result = np.empty((len(a), len(models)), np.float64)
    threshold = float(cfg.get('reprojection_threshold_px', 3.0))
    sigma = (float(cfg.get('depth_sigma_m', .15)) +
             float(cfg.get('depth_sigma_relative', .015)) * .5 * (a[:, 2] + b[:, 2]))
    sigma = np.maximum(sigma, 1e-6)
    weight = float(cfg.get('depth_weight', .20))
    cap = float(cfg.get('depth_cost_cap', 4.0))
    for k, H in enumerate(models):
        R, t = np.asarray(H)[:3, :3], np.asarray(H)[:3, 3]
        forward = a @ R.T + t
        backward = (b - t) @ R
        p1, p0 = _project(forward, K), _project(backward, K)
        reprojection = .5 * (np.sum((p1 - uv[:, 2:]) ** 2, axis=1) +
                            np.sum((p0 - uv[:, :2]) ** 2, axis=1)) / threshold**2
        depth = np.minimum(np.sum((forward-b)**2, axis=1) / sigma**2, cap)
        cost = (1-weight)*reprojection + weight*depth
        invalid = ((forward[:, 2] <= 1e-4) | (backward[:, 2] <= 1e-4) |
                   ~np.isfinite(cost))
        result[:, k] = np.where(invalid, 1e6, np.minimum(cost, 1e6))
    return result


def motion_probabilities(costs, cfg):
    """Conditional model probabilities and probability of a non-outlier.

    The explicit outlier component prevents an unexplained match from supplying
    strong ownership evidence merely because one wrong model is less wrong.
    """
    costs = np.asarray(costs, float)
    if costs.shape[1] == 0:
        return np.empty_like(costs), np.zeros(len(costs))
    temperature = float(cfg.get('temperature', .25))
    outlier = float(cfg.get('outlier_cost', 1.0))
    logits = -costs / temperature
    normalizer = np.maximum(logits.max(axis=1), -outlier / temperature)
    model_exp = np.exp(logits - normalizer[:, None])
    model_mass = model_exp.sum(axis=1)
    outlier_exp = np.exp(-outlier / temperature - normalizer)
    conditional = model_exp / np.maximum(model_mass[:, None], 1e-300)
    # Stable conditional probabilities even when every model is extremely poor.
    underflow = model_mass < 1e-250
    if np.any(underflow):
        stable = logits[underflow] - logits[underflow].max(axis=1, keepdims=True)
        stable = np.exp(stable)
        conditional[underflow] = stable / stable.sum(axis=1, keepdims=True)
    reliability = model_mass / (model_mass + outlier_exp)
    return conditional, reliability


def _project(points, K):
    z = points[:, 2:3]
    safe_z = np.where(np.abs(z) > 1e-8, z, 1e-8)
    return points[:, :2] / safe_z * np.asarray([K[0, 0], K[1, 1]]) + K[:2, 2]


def _rigid(a, b, weights=None):
    weights = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    weights = weights / weights.sum()
    ac, bc = np.sum(a*weights[:, None], axis=0), np.sum(b*weights[:, None], axis=0)
    u, _, vt = np.linalg.svd(((a-ac)*weights[:, None]).T @ (b-bc))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    H = np.eye(4)
    H[:3, :3] = vt.T @ correction @ u.T
    H[:3, 3] = bc - H[:3, :3] @ ac
    return H


def _pnp(a, uv1, K, ids, threshold, seed, iterations=150):
    if len(ids) < 6:
        return None
    cv2.setRNGSeed(int(seed % 2147483647))
    try:
        ok, r, t, _ = cv2.solvePnPRansac(
            np.ascontiguousarray(a[ids], np.float64),
            np.ascontiguousarray(uv1[ids], np.float64), K, None,
            iterationsCount=iterations, reprojectionError=threshold,
            confidence=.99, flags=cv2.SOLVEPNP_EPNP)
    except cv2.error:
        return None
    if not ok:
        return None
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = cv2.Rodrigues(r)[0], t.ravel()
    return H


def _metric_scale_pose(R, direction, a, b, uv, K, threshold):
    """Fit metric translation scale with observed first-depth-to-image rays."""
    direction = np.asarray(direction).ravel()
    rotated = a @ R.T
    target = uv[:, 2:]
    displacement = b - rotated
    initial_scale = np.median(displacement @ direction)
    upper = max(2., 4*float(np.median(np.linalg.norm(displacement, axis=1))),
                2*abs(float(initial_scale)))

    def objective(scale):
        moved = rotated + scale*direction
        residual = np.linalg.norm(_project(moved, K)-target, axis=1) / threshold
        return float(np.mean(2*(np.sqrt(1+residual**2)-1)))

    scale = minimize_scalar(objective, bounds=(0., upper), method='bounded',
                            options={'maxiter': 35, 'xatol': 1e-5}).x
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = R, scale*direction
    return H


def _essential(uv, a, b, K, threshold, seed, max_iterations):
    if len(uv) < 5:
        return []
    cv2.setRNGSeed(int(seed % 2147483647))
    try:
        E, mask = cv2.findEssentialMat(
            uv[:, :2], uv[:, 2:], K, method=cv2.RANSAC,
            prob=.995, threshold=threshold, maxIters=max_iterations)
    except cv2.error:
        return []
    if E is None:
        return []
    results = []
    for e in np.asarray(E).reshape(-1, 3, 3):
        try:
            _, R, direction, positive = cv2.recoverPose(
                e, uv[:, :2], uv[:, 2:], K,
                mask=mask.copy() if mask is not None else None)
        except cv2.error:
            continue
        keep = positive.ravel() > 0
        if np.count_nonzero(keep) < 3:
            keep = mask.ravel() > 0 if mask is not None else np.ones(len(a), bool)
        if not np.any(keep):
            continue
        # Rotation/direction are estimated without depth. Only this scalar fit
        # uses the first-frame depth and observed second-frame pixels.
        results.append(_metric_scale_pose(R, direction, a[keep], b[keep],
                                           uv[keep], K, threshold))
    return results


def _refit(anchors, H, ids, K, cfg):
    """Robust metric pose refinement on exclusive low-error core points."""
    if len(ids) < 4:
        return H
    a, b = anchors['x0'][ids], anchors['x1'][ids]
    uv = anchors['matches'][ids]
    confidence = np.asarray(anchors.get('confidence', np.ones(len(anchors['x0'])))).reshape(-1)[ids]
    weights = np.sqrt(np.maximum(confidence, .05))[:, None]
    threshold = float(cfg.get('reprojection_threshold_px', 3.0))
    depth_sigma = (float(cfg.get('depth_sigma_m', .15)) +
                   float(cfg.get('depth_sigma_relative', .015))*.5*(a[:, 2]+b[:, 2]))
    depth_weight = float(cfg.get('depth_weight', .20))
    cap = np.sqrt(float(cfg.get('depth_cost_cap', 4.0)))

    def residual(parameters):
        R = cv2.Rodrigues(parameters[:3])[0]
        t = parameters[3:]
        fwd, bwd = a @ R.T + t, (b-t) @ R
        forward = (_project(fwd, K)-uv[:, 2:]) / threshold
        backward = (_project(bwd, K)-uv[:, :2]) / threshold
        depth = (fwd-b) / np.maximum(depth_sigma[:, None], 1e-6)
        norm = np.linalg.norm(depth, axis=1, keepdims=True)
        depth = depth * np.minimum(1., cap/np.maximum(norm, 1e-8))
        return np.concatenate((np.sqrt(.5*(1-depth_weight))*forward,
                               np.sqrt(.5*(1-depth_weight))*backward,
                               np.sqrt(depth_weight)*depth), axis=1).ravel() * np.repeat(weights.ravel(), 7)

    r = cv2.Rodrigues(H[:3, :3])[0].ravel()
    fit = least_squares(residual, np.r_[r, H[:3, 3]], loss='soft_l1', f_scale=.5,
                        max_nfev=int(cfg.get('refit_max_nfev', 35)))
    refined = np.eye(4)
    refined[:3, :3], refined[:3, 3] = cv2.Rodrigues(fit.x[:3])[0], fit.x[3:]
    return refined


def _exclusive_labels(costs, cfg):
    if costs.shape[1] == 0:
        return np.full(len(costs), -1, np.int32)
    winner = np.argmin(costs, axis=1)
    best = costs[np.arange(len(costs)), winner]
    second = np.partition(costs, 1, axis=1)[:, 1] if costs.shape[1] > 1 else np.full(len(costs), np.inf)
    margin = float(cfg.get('core_margin', .10))
    keep = (best < float(cfg.get('outlier_cost', 1.))) & ((second-best) >= margin)
    return np.where(keep, winner, -1).astype(np.int32)


def _metric_consistency(anchors, models, cfg):
    """Uncapped depth-core support for metrically identifiable hypotheses.

    Capping depth residuals makes noisy observations harmless during scoring,
    but a whole model must not obtain scale support only from capped outliers.
    The existing depth noise/cap define this core; no new distance threshold.
    """
    a, b = anchors['x0'], anchors['x1']
    sigma = (float(cfg.get('depth_sigma_m', .15)) +
             float(cfg.get('depth_sigma_relative', .015))*.5*(a[:, 2]+b[:, 2]))
    cap = float(cfg.get('depth_cost_cap', 4.))
    return np.column_stack([
        np.sum((a @ H[:3, :3].T+H[:3, 3]-b)**2, axis=1) <= cap*sigma**2
        for H in models]) if len(models) else np.empty((len(a), 0), bool)


def _core_labels(anchors, models, K, cfg):
    costs = motion_costs(anchors, models, K, cfg)
    labels = _exclusive_labels(costs, cfg)
    valid = _metric_consistency(anchors, models, cfg)
    if len(models):
        consistent = valid[np.arange(len(labels)), np.maximum(labels, 0)]
        labels = np.where(consistent, labels, -1).astype(np.int32)
    return costs, labels


def _merge_aliases(anchors, models, sources, K, cfg, minimum):
    """Pool models indistinguishable on substantially shared observations.

    Noisy depths can produce several poses for the same background. Comparing
    parameter vectors alone is misleading (rotation and translation compensate
    at different depths), so compare predicted pixels and metric displacement
    on shared inliers, at the noise scales already used for fitting.
    """
    costs = motion_costs(anchors, models, K, cfg)
    inliers = costs < float(cfg.get('outlier_cost', 1.))
    a, b = anchors['x0'], anchors['x1']
    sigma = (float(cfg.get('depth_sigma_m', .15)) +
             float(cfg.get('depth_sigma_relative', .015))*.5*(a[:, 2]+b[:, 2]))
    pixel_threshold = float(cfg.get('alias_prediction_threshold_px',
                                   cfg.get('reprojection_threshold_px', 3.)))
    overlap_threshold = float(cfg.get('alias_support_overlap', .65))
    depth_threshold = float(cfg.get('alias_depth_sigma', 1.))
    parent = np.arange(len(models))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    forward = [a @ H[:3, :3].T + H[:3, 3] for H in models]
    backward = [(b-H[:3, 3]) @ H[:3, :3] for H in models]
    projected_fwd = [_project(points, K) for points in forward]
    projected_bwd = [_project(points, K) for points in backward]
    merged_pairs = []
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            shared = inliers[:, i] & inliers[:, j]
            count = int(np.count_nonzero(shared))
            overlap = count / max(1, min(np.count_nonzero(inliers[:, i]),
                                         np.count_nonzero(inliers[:, j])))
            if count < minimum or overlap < overlap_threshold:
                continue
            pixel = np.sqrt(.5*(np.sum((projected_fwd[i][shared]-projected_fwd[j][shared])**2, axis=1) +
                                  np.sum((projected_bwd[i][shared]-projected_bwd[j][shared])**2, axis=1)))
            metric = np.linalg.norm(forward[i][shared]-forward[j][shared], axis=1) / sigma[shared]
            if np.median(pixel) <= pixel_threshold and np.median(metric) <= depth_threshold:
                parent[find(j)] = find(i)
                merged_pairs.append({'models': [i, j], 'support_overlap': float(overlap),
                                     'median_prediction_difference_px': float(np.median(pixel)),
                                     'median_metric_difference_sigma': float(np.median(metric))})
    groups = {}
    for i in range(len(models)):
        groups.setdefault(find(i), []).append(i)
    result, result_sources = [], []
    for group in groups.values():
        if len(group) == 1:
            result.append(models[group[0]])
            result_sources.append(sources[group[0]])
            continue
        ids = np.flatnonzero(np.any(inliers[:, group], axis=1))
        local_cost = np.minimum(costs[np.ix_(ids, group)], float(cfg.get('outlier_cost', 1.))).sum(axis=0)
        best = group[int(np.argmin(local_cost))]
        result.append(_refit(anchors, models[best], ids, K, cfg))
        result_sources.append(sources[best])
    return result, result_sources, merged_pairs


def _recover_unassigned(anchors, models, labels, sources, prior_models,
                        prior_sources, K, cfg, minimum, maximum):
    """One terminal recovery pass, restricted to final unassigned matches.

    Pre-merge supports are overlapping associations, not unique labels. Refit
    each old hypothesis using only its still-gray supporters, then require it
    to win against retained motions by the existing core margin and gain.
    Retained motions, their core labels and the reference index stay fixed.
    Recovered motions are appended without another merge/competition cycle.
    unexplained_only excludes gray points with any retained-model cost below
    the same threshold used for yellow rings, for both refitting and labeling.
    """
    models, sources = list(models), list(sources)
    labels = np.asarray(labels, np.int32).copy()
    original_labels = labels.copy()
    original_models = np.stack(models).copy()
    initially_gray = labels < 0
    mode = cfg.get('gray_recovery_mode', 'all_unassigned')
    if mode not in ('all_unassigned', 'unexplained_only'):
        raise ValueError('gray_recovery_mode must be all_unassigned or unexplained_only')
    outlier = float(cfg.get('outlier_cost', 1.))
    current_costs = motion_costs(anchors, models, K, cfg)
    # Yellow rings on globally unassigned points mean that at least one of
    # these retained models has cost < outlier. Freeze this test now: neither
    # cached pre-merge hypotheses nor newly recovered models define the mask.
    yellow_gray = initially_gray & np.any(current_costs < outlier, axis=1)
    eligible_gray = initially_gray.copy()
    if mode == 'unexplained_only':
        eligible_gray &= ~yellow_gray
    report = dict(enabled=True, mode=mode, passes=1, prior_models=len(prior_models),
                  base_models=len(models), unassigned_before=int(initially_gray.sum()),
                  yellow_gray_points=int(yellow_gray.sum()),
                  excluded_yellow_gray_points=int((initially_gray & ~eligible_gray).sum()),
                  eligible_gray_points=int(eligible_gray.sum()),
                  selection_cost_threshold=outlier, selection_snapshot='before_recovery',
                  proposals=[], recovered=[])
    if eligible_gray.sum() < minimum or len(models) >= maximum:
        report['skip_reason'] = ('insufficient_gray_points' if initially_gray.sum() < minimum
                                 else 'insufficient_eligible_gray_points' if eligible_gray.sum() < minimum
                                 else 'component_limit')
    else:
        min_gain = float(cfg.get('model_penalty', minimum*.20))
        quality = np.sqrt(np.clip(np.asarray(anchors.get(
            'confidence', np.ones(len(labels)))).reshape(-1), .05, 1.))
        prior_costs = motion_costs(anchors, prior_models, K, cfg)
        prior_support = ((prior_costs < outlier) &
                         _metric_consistency(anchors, prior_models, cfg))
        proposals = []
        for old, H in enumerate(prior_models):
            ids = np.flatnonzero(eligible_gray & prior_support[:, old])
            row = dict(prior_model=old, source=prior_sources[old],
                       gray_support_before_refit=len(ids))
            report['proposals'].append(row)
            if len(ids) < minimum:
                row['status'] = 'insufficient_gray_support'
                continue
            refined = _refit(anchors, H, ids, K, cfg)
            proposals.append(dict(H=refined, cost=motion_costs(
                anchors, [refined], K, cfg)[:, 0],
                metric=_metric_consistency(anchors, [refined], cfg)[:, 0],
                origin_support=prior_support[:, old], row=row))
            row['status'] = 'pending'

        while proposals and len(models) < maximum:
            best = None
            best_gain = -np.inf
            for proposal in proposals:
                row = proposal['row']
                competition = _exclusive_labels(np.column_stack(
                    (current_costs, proposal['cost'])), cfg)
                accepted = (eligible_gray & (labels < 0) & proposal['origin_support'] &
                            proposal['metric'] & (competition == len(models)))
                count = int(accepted.sum())
                improvement = np.maximum(0., np.minimum(
                    current_costs.min(axis=1), outlier)-proposal['cost'])
                gain = float(np.sum(improvement[accepted]*quality[accepted]))
                row.update(eligible_gray_support=count, gain=gain,
                           status='insufficient_independent_support_or_gain')
                if count >= minimum and gain >= min_gain and gain > best_gain:
                    best, best_gain = (proposal, accepted), gain
            if best is None:
                break
            proposal, accepted = best
            new_index = len(models)
            models.append(proposal['H'])
            sources.append(proposal['row']['source'])
            labels[accepted] = new_index
            current_costs = np.column_stack((current_costs, proposal['cost']))
            proposal['row'].update(status='recovered', model=new_index,
                                   core_support=int(accepted.sum()),
                                   recovered_indices=np.flatnonzero(accepted).tolist())
            report['recovered'].append(dict(proposal['row']))
            proposals = [p for p in proposals if p is not proposal]
        if len(models) >= maximum:
            report['component_limit_reached'] = True

    # Recovery deliberately does not reassign any previously accepted core.
    assert np.array_equal(labels[~initially_gray], original_labels[~initially_gray])
    assert np.array_equal(labels[~eligible_gray], original_labels[~eligible_gray])
    assert np.array_equal(np.stack(models[:len(original_models)]), original_models)
    report.update(unassigned_after=int(np.count_nonzero(labels < 0)),
                  recovered_points=int(np.count_nonzero(initially_gray & (labels >= 0))),
                  added_models=len(models)-len(original_models),
                  existing_labels_unchanged=True, existing_models_unchanged=True,
                  ineligible_labels_unchanged=True)
    return models, labels, sources, report


def discover_hybrid(anchors, K, cfg, seed, matches_2d=None):
    """Discover competing metric motions without greedy removal of matches.

    ``cfg`` is the full motion mapping; optional ``matches_2d`` contains all 2D
    correspondences, including ones lacking two usable depths. Those contribute
    to global essential geometry; metric support/labels refer to ``anchors``.
    Returns models, exclusive labels (-1 means ambiguous/outlier), supports and
    a JSON-serializable report. The first model is the background/reference
    proposal selected by support and image coverage, rather than discovery order.
    """
    hcfg = cfg.get('hybrid', {})
    a = np.asarray(anchors['x0'], np.float64)
    b = np.asarray(anchors['x1'], np.float64)
    uv = np.asarray(anchors['matches'], np.float64)
    anchors = {**anchors, 'x0': a, 'x1': b, 'matches': uv}
    K = np.asarray(K, np.float64)
    n = len(a)
    minimum = int(hcfg.get('min_support', cfg.get('min_support', 6)))
    maximum = int(hcfg.get('max_components', cfg.get('max_components', 6)))
    if n < 3:
        raise RuntimeError(f'Need at least three metric anchors; got {n}')
    rng = np.random.default_rng(seed)
    quality = np.sqrt(np.clip(np.asarray(anchors.get('confidence', np.ones(n))).reshape(-1), .05, 1.))
    threshold = float(hcfg.get('reprojection_threshold_px', 3.0))
    outlier = float(hcfg.get('outlier_cost', 1.0))
    pool, pool_costs, pool_sources = [], [], []
    attempted = {'essential': 0, 'pnp': 0, 'rigid': 0}

    def add(H, source):
        attempted[source] += 1
        if H is None or not np.all(np.isfinite(H)):
            return
        cost = motion_costs(anchors, [H], K, hcfg)[:, 0]
        support = (cost < outlier) & _metric_consistency(anchors, [H], hcfg)[:, 0]
        if np.count_nonzero(support) < minimum:
            return
        # Only near-identical support AND cost profiles are deduplicated. A
        # small real model is never discarded because its support is a subset.
        for i, old_cost in enumerate(pool_costs):
            old = old_cost < outlier
            union = np.count_nonzero(old | support)
            overlap = np.count_nonzero(old & support) / max(union, 1)
            if overlap > .85 and np.median(np.abs(old_cost[old | support]-cost[old | support])) < .12:
                if np.sum(np.minimum(cost, outlier)*quality) < np.sum(np.minimum(old_cost, outlier)*quality):
                    pool[i], pool_costs[i], pool_sources[i] = H, cost, source
                return
        pool.append(H)
        pool_costs.append(cost)
        pool_sources.append(source)

    all_ids = np.arange(n)
    add(_rigid(a, b, quality), 'rigid')
    add(_pnp(a, uv[:, 2:], K, all_ids, threshold, seed, 1000), 'pnp')
    for H in _essential(uv, a, b, K, threshold, seed, 1000):
        add(H, 'essential')

    # Local seeds preferentially include small objects while global draws
    # continue to propose dispersed background structures. No labels are fixed.
    tree = cKDTree(uv[:, :2])
    seed_count = min(n, int(hcfg.get('local_seeds', 48)))
    centers = rng.choice(n, seed_count, replace=False)
    neighborhood_sizes = hcfg.get('neighborhood_sizes', [8, 16, 32])
    groups = []
    seen = set()
    for center in centers:
        for size in neighborhood_sizes:
            ids = np.atleast_1d(tree.query(uv[center, :2], k=min(int(size), n))[1]).astype(int)
            key = tuple(sorted(ids.tolist()))
            if len(ids) >= minimum and key not in seen:
                seen.add(key)
                groups.append(ids)
    for group_index, ids in enumerate(groups):
        proposal_seed = seed + 31*(group_index+1)
        add(_rigid(a[ids], b[ids], quality[ids]), 'rigid')
        add(_pnp(a, uv[:, 2:], K, ids, threshold, proposal_seed), 'pnp')
        for H in _essential(uv[ids], a[ids], b[ids], K, threshold, proposal_seed, 100):
            add(H, 'essential')
        # Three-point metric seeds can expose a motion that shares a local
        # neighborhood with background, without requiring eight/twenty supports.
        for _ in range(3):
            sample = rng.choice(ids, 3, replace=False)
            if np.linalg.matrix_rank(a[sample]-a[sample].mean(axis=0)) >= 2:
                add(_rigid(a[sample], b[sample]), 'rigid')
    for _ in range(int(hcfg.get('global_rigid_proposals', 256))):
        ids = rng.choice(n, 3, replace=False)
        if np.linalg.matrix_rank(a[ids]-a[ids].mean(axis=0)) >= 2:
            add(_rigid(a[ids], b[ids]), 'rigid')

    # Extra 2D-only matches may improve the global E. Transfer its geometry to
    # metric anchors and fit scale from them, never manufacture depth for them.
    if matches_2d is not None and len(matches_2d) > n:
        extra = np.asarray(matches_2d, np.float64)
        cv2.setRNGSeed(int(seed % 2147483647))
        try:
            E, mask = cv2.findEssentialMat(extra[:, :2], extra[:, 2:], K,
                                         method=cv2.RANSAC, threshold=threshold,
                                         prob=.995, maxIters=1000)
            if E is not None:
                for e in np.asarray(E).reshape(-1, 3, 3):
                    _, R, direction, _ = cv2.recoverPose(
                        e, extra[:, :2], extra[:, 2:], K, mask=mask.copy())
                    inverse_K = np.linalg.inv(K)
                    F = inverse_K.T @ e @ inverse_K
                    p0 = np.column_stack((uv[:, :2], np.ones(n)))
                    p1 = np.column_stack((uv[:, 2:], np.ones(n)))
                    line1, line0 = p0 @ F.T, p1 @ F
                    sampson = np.sum(p1*line1, axis=1)**2 / np.maximum(
                        np.sum(line1[:, :2]**2+line0[:, :2]**2, axis=1), 1e-12)
                    keep = sampson < threshold**2
                    if np.count_nonzero(keep) >= 3:
                        add(_metric_scale_pose(R, direction, a[keep], b[keep],
                                               uv[keep], K, threshold), 'essential')
        except cv2.error:
            pass
    if not pool:
        raise RuntimeError(f'No hybrid motion with {minimum} consistent anchors from {n} matches')

    matrix = np.column_stack(pool_costs)
    best_cost = np.full(n, outlier)
    selected = []
    selection_gain = []
    min_gain = float(hcfg.get('model_penalty', minimum*.20))
    improve = float(hcfg.get('independent_improvement', .10))
    # Each proposal is considered at most once. The final component budget is
    # applied after alias merging, so background variants cannot consume it.
    for _ in range(len(pool)):
        gains = np.maximum(0., best_cost[:, None]-matrix)*quality[:, None]
        gain = gains.sum(axis=0)
        # A small real motion may already fall within a background's broad
        # inlier threshold. Count meaningful relative improvement as independent
        # evidence; alias merging removes redundant variants of the background.
        independent = ((best_cost[:, None]-matrix > improve) &
                       (matrix < outlier)).sum(axis=0)
        eligible = (independent >= minimum) & (gain >= min_gain)
        eligible[selected] = False
        if not np.any(eligible):
            break
        j = int(np.argmax(np.where(eligible, gain, -np.inf)))
        selected.append(j)
        selection_gain.append(float(gain[j]))
        best_cost = np.minimum(best_cost, matrix[:, j])
    if not selected:
        # A model still must have geometric support, but extremely low global
        # confidence alone must not discard an otherwise valid initialization.
        selected = [int(np.argmax(np.sum(matrix < outlier, axis=0)))]
    models = [pool[j].copy() for j in selected]
    sources = [pool_sources[j] for j in selected]
    recover_gray = bool(hcfg.get('recover_unassigned_after_merge', False))
    if recover_gray:
        # Keep the selected hypotheses before the first alias merge. Their
        # support relations are evaluated only for gray-point recovery later.
        prior_models = [H.copy() for H in models]
        prior_sources = list(sources)
    models, sources, merged_aliases = _merge_aliases(anchors, models, sources, K, hcfg, minimum)
    capped_alias_groups = []
    while len(models) > maximum:
        # Apply the count budget to distinct models. Remove the one contributing
        # least beyond all other models, rather than the one found last.
        current_costs = np.minimum(motion_costs(anchors, models, K, hcfg), outlier)
        best_with_all = current_costs.min(axis=1)
        contributions = [float(np.sum((np.delete(current_costs, k, axis=1).min(axis=1)-
                                       best_with_all)*quality)) for k in range(len(models))]
        remove = int(np.argmin(contributions))
        capped_alias_groups.append({'source': sources[remove],
                                    'independent_gain': contributions[remove]})
        del models[remove]
        del sources[remove]
    refinement_log = []
    for iteration in range(int(hcfg.get('competition_iterations', 4))):
        costs, labels = _core_labels(anchors, models, K, hcfg)
        counts = np.bincount(labels[labels >= 0], minlength=len(models))
        keep = np.flatnonzero(counts >= minimum)
        if not len(keep):
            keep = np.array([int(np.argmax(counts))])
        models, sources = [models[k] for k in keep], [sources[k] for k in keep]
        costs, labels = _core_labels(anchors, models, K, hcfg)
        models = [_refit(anchors, H, np.flatnonzero(labels == k), K, hcfg)
                  for k, H in enumerate(models)]
        refinement_log.append({'iteration': iteration, 'core_supports':
                               [int(np.count_nonzero(labels == k)) for k in range(len(models))]})

    models, sources, final_aliases = _merge_aliases(anchors, models, sources, K, hcfg, minimum)
    costs, labels = _core_labels(anchors, models, K, hcfg)
    # Refining an exclusive core can move its boundary. Apply the same minimum
    # to the final supports, rather than retaining a now-unsupported component.
    while len(models) > 1:
        counts = np.bincount(labels[labels >= 0], minlength=len(models))
        keep = np.flatnonzero(counts >= minimum)
        if len(keep) == len(models):
            break
        if not len(keep):
            keep = np.array([int(np.argmax(counts))])
        models, sources = [models[k] for k in keep], [sources[k] for k in keep]
        costs, labels = _core_labels(anchors, models, K, hcfg)
    fallback = None
    if np.count_nonzero(labels >= 0) < minimum:
        best = int(np.argmax(np.sum(matrix < outlier, axis=0)))
        models, sources = [pool[best].copy()], [pool_sources[best]]
        _, labels = _core_labels(anchors, models, K, hcfg)
        fallback = 'best_supported_unmodified_proposal'
    # Prefer spatially dispersed support for the reference, not the order in
    # which a robust estimator happened to discover a model.
    span = np.maximum(np.ptp(uv[:, :2], axis=0), 1.)
    reference_scores = []
    for k in range(len(models)):
        points = uv[labels == k, :2]
        if len(points) >= 3:
            area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) / float(np.prod(span))
        else:
            area = 0.
        reference_scores.append(float(len(points))*(.25+area))
    reference = int(np.argmax(reference_scores))
    order = [reference] + [k for k in range(len(models)) if k != reference]
    inverse_order = np.argsort(order)
    labels = np.where(labels >= 0, inverse_order[np.maximum(labels, 0)], -1).astype(np.int32)
    models = [models[k] for k in order]
    ordered_sources = [sources[k] for k in order]
    ordered_reference_scores = [reference_scores[k] for k in order]
    if recover_gray:
        models, labels, ordered_sources, recovery_report = _recover_unassigned(
            anchors, models, labels, ordered_sources, prior_models, prior_sources,
            K, hcfg, minimum, maximum)
        # The ordinary initializer has already selected the reference. Append
        # scores for diagnostics without renumbering existing models/labels.
        for k in range(len(ordered_reference_scores), len(models)):
            points = uv[labels == k, :2]
            area = (float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))) /
                    float(np.prod(span))) if len(points) >= 3 else 0.
            ordered_reference_scores.append(float(len(points))*(.25+area))
    supports = [int(np.count_nonzero(labels == k)) for k in range(len(models))]
    report = {'method': 'hybrid', 'anchor_count': n,
              'matches_2d_count': n if matches_2d is None else len(matches_2d),
              'proposal_attempts': attempted, 'proposal_count': len(pool),
              'selected_proposal_gains': selection_gain,
              'sources': ordered_sources, 'supports': supports,
              'unassigned_count': int(np.count_nonzero(labels < 0)),
              'reference_index': 0, 'reference_scores': ordered_reference_scores,
              'competition': refinement_log,
              'merged_aliases': merged_aliases, 'final_merged_aliases': final_aliases,
              'removed_for_component_budget': capped_alias_groups,
              'fallback': fallback,
              'cost_definition': 'symmetric_observed_reprojection_plus_capped_depth',
              'min_support': minimum}
    if recover_gray:
        report['gray_recovery'] = recovery_report
    return models, labels, supports, report

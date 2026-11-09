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
        direction = direction.ravel()
        keep = positive.ravel() > 0
        if np.count_nonzero(keep) < 3:
            keep = mask.ravel() > 0 if mask is not None else np.ones(len(a), bool)
        if not np.any(keep):
            continue
        # Rotation/direction are estimated without depth. Only this scalar fit
        # uses the first-frame depth and observed second-frame pixels.
        rotated = a[keep] @ R.T
        target = uv[keep, 2:]
        metric_displacement = b[keep] - rotated
        initial_scale = np.median(metric_displacement @ direction)
        upper = max(2., 4*float(np.median(np.linalg.norm(metric_displacement, axis=1))),
                    2*abs(float(initial_scale)))

        def objective(scale):
            projected = _project(rotated + scale*direction, K)
            residual = np.linalg.norm(projected-target, axis=1) / threshold
            return float(np.mean(2*(np.sqrt(1+residual**2)-1)))

        scale_fit = minimize_scalar(objective, bounds=(0., upper), method='bounded',
                                    options={'maxiter': 35, 'xatol': 1e-5})
        H = np.eye(4)
        H[:3, :3], H[:3, 3] = R, scale_fit.x*direction
        results.append(H)
    return results


def _refit(anchors, H, ids, K, cfg):
    """Robust metric pose refinement on exclusive low-error core points."""
    if len(ids) < 4:
        return H
    a, b = anchors['x0'][ids], anchors['x1'][ids]
    uv = anchors['matches'][ids]
    confidence = np.asarray(anchors.get('confidence', np.ones(len(anchors['x0']))))[ids]
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
    quality = np.sqrt(np.clip(np.asarray(anchors.get('confidence', np.ones(n))), .05, 1.))
    threshold = float(hcfg.get('reprojection_threshold_px', 3.0))
    outlier = float(hcfg.get('outlier_cost', 1.0))
    pool, pool_costs, pool_sources = [], [], []
    attempted = {'essential': 0, 'pnp': 0, 'rigid': 0}

    def add(H, source):
        attempted[source] += 1
        if H is None or not np.all(np.isfinite(H)):
            return
        cost = motion_costs(anchors, [H], K, hcfg)[:, 0]
        support = cost < outlier
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
    for _ in range(maximum):
        gains = np.maximum(0., best_cost[:, None]-matrix)*quality[:, None]
        gain = gains.sum(axis=0)
        independent = ((best_cost[:, None]-matrix > improve) & (matrix < outlier)).sum(axis=0)
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
    refinement_log = []
    for iteration in range(int(hcfg.get('competition_iterations', 4))):
        costs = motion_costs(anchors, models, K, hcfg)
        labels = _exclusive_labels(costs, hcfg)
        counts = np.bincount(labels[labels >= 0], minlength=len(models))
        keep = np.flatnonzero(counts >= minimum)
        if not len(keep):
            keep = np.array([int(np.argmax(counts))])
        models, sources = [models[k] for k in keep], [sources[k] for k in keep]
        costs = motion_costs(anchors, models, K, hcfg)
        labels = _exclusive_labels(costs, hcfg)
        models = [_refit(anchors, H, np.flatnonzero(labels == k), K, hcfg)
                  for k, H in enumerate(models)]
        refinement_log.append({'iteration': iteration, 'core_supports':
                               [int(np.count_nonzero(labels == k)) for k in range(len(models))]})

    costs = motion_costs(anchors, models, K, hcfg)
    labels = _exclusive_labels(costs, hcfg)
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
    supports = [int(np.count_nonzero(labels == k)) for k in range(len(models))]
    report = {'method': 'hybrid', 'anchor_count': n,
              'matches_2d_count': n if matches_2d is None else len(matches_2d),
              'proposal_attempts': attempted, 'proposal_count': len(pool),
              'selected_proposal_gains': selection_gain,
              'sources': [sources[k] for k in order], 'supports': supports,
              'unassigned_count': int(np.count_nonzero(labels < 0)),
              'reference_index': 0, 'reference_scores': [reference_scores[k] for k in order],
              'competition': refinement_log,
              'cost_definition': 'symmetric_observed_reprojection_plus_capped_depth',
              'min_support': minimum}
    return models, labels, supports, report

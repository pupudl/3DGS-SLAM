"""Mask-guided evidence collection and explicit replacement of existing motions.

No instance labels or ground truth enter this module. Existing ownership is a
proposal prior, never a hard assignment of new matches. All alternatives in a
round see identical anchors, geometry, observations and iteration budgets.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pair_utils import add_unique, discover, errors, unexplained_clusters


def _pixel_problem_mask(score, valid, own_projection, cfg, allowed=None):
    """Threshold pixel signals; optional `allowed` exists for historical replay."""
    domain = valid if allowed is None else valid & allowed
    values, counts = own_projection
    sums = cv2.GaussianBlur(values, (5, 5), 0)
    mass = cv2.GaussianBlur(counts, (5, 5), 0)
    gradient = sums / np.maximum(mass, 1e-12)
    present = valid & (mass > 1e-6)
    quantile = float(cfg.get('gradient_scale_quantile', .95))
    scale = max(float(np.quantile(gradient[present], quantile)), 1e-12) if present.any() else 1.
    normalized_gradient = gradient / scale
    priority = score + float(cfg.get('gradient_weight', .25)) * gradient / scale
    residual_trigger = domain & (score >= float(cfg.get('mask_residual_threshold', .12)))
    gradient_trigger = domain & present & (
        normalized_gradient >= float(cfg.get('mask_gradient_threshold', .5)))
    selected = residual_trigger | gradient_trigger
    n, labels, stats, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8), 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(cfg.get('mask_min_area_px', 16))
    selected = keep[labels]
    radius = int(cfg.get('mask_dilation_px', 3))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
        selected = cv2.dilate(selected.astype(np.uint8), kernel).astype(bool)
    return selected & domain, priority, dict(residual_trigger=residual_trigger,
                                           gradient_trigger=gradient_trigger,
                                           normalized_gradient=normalized_gradient)


def global_problem_mask(score, valid, own_projection, cfg):
    """Build a full-image pixel mask with no windows, tiling or region ranking."""
    return _pixel_problem_mask(score, valid, own_projection, cfg)


def score_guided_mask(score, valid, own_projection, cfg, confidence=None):
    """Retain the additive discovery criterion with continuous neighborhoods.

    Residuals are locally averaged with missing observations contributing zero.
    Ownership gradients use all projected soft mass, including the original
    full-image normalization domain. A Gaussian neighborhood replaces the fixed
    grid; one score threshold and a pixel budget determine the irregular mask.
    """
    score = np.asarray(score, np.float32)
    valid = np.asarray(valid, bool) & np.isfinite(score)
    values, counts = (np.asarray(a, np.float32) for a in own_projection)
    if score.shape != valid.shape or values.shape != score.shape or counts.shape != score.shape:
        raise ValueError('Residual, validity and projected gradients must share an image shape')
    sigma = float(cfg.get('score_smoothing_sigma_px', 3.))
    if sigma < 0:
        raise ValueError('score_smoothing_sigma_px must be nonnegative')
    def smooth(array):
        return (cv2.GaussianBlur(array, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
                if sigma > 0 else array.copy())

    support = smooth(valid.astype(np.float32))
    residual_mean = smooth(np.where(valid, score, 0))
    present = np.isfinite(values) & np.isfinite(counts) & (counts > 0)
    gradient_mean = smooth(np.where(present, values, 0)) / np.maximum(
        smooth(np.where(present, counts, 0)), 1e-12)
    scale = 1.
    if cfg.get('normalize_gradient', True) and present.any():
        quantile = float(cfg.get('gradient_scale_quantile', .95))
        if not 0 < quantile <= 1:
            raise ValueError('gradient_scale_quantile must be in (0,1]')
        # Preserve the original normalization scale, before spatial averaging.
        scale = max(float(np.quantile(values[present]/counts[present], quantile)), 1e-12)
    priority = residual_mean + float(cfg.get('gradient_weight', .25))*gradient_mean/scale
    eligible = valid & (support >= float(cfg.get('min_observation_fraction', .6)))
    if confidence is not None:
        if confidence.shape != score.shape:
            raise ValueError('Stereo confidence must share the target image shape')
        eligible &= confidence >= float(cfg.get('mask_min_stereo_confidence', .15))
    threshold = float(cfg.get('min_score', .12))
    candidates = eligible & np.isfinite(priority) & (priority >= threshold)
    minimum_area = int(cfg.get('mask_min_area_px', 16))
    def remove_noise(mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
        return keep[labels]
    candidates = remove_noise(candidates)
    budget = int(cfg.get('mask_budget_pixels', 110592))
    if budget < 0:
        raise ValueError('mask_budget_pixels must be nonnegative')
    ids = np.flatnonzero(candidates)
    selected = np.zeros(score.size, bool)
    if budget and len(ids):
        order = np.argsort(-priority.ravel()[ids], kind='stable')
        selected[ids[order[:budget]]] = True
    # Cleanup can only remove pixels; it never grows beyond the score/budget.
    selected = remove_noise(selected.reshape(score.shape))
    stats = dict(gradient_scale=scale, eligible_pixels=int(eligible.sum()),
                 candidate_pixels=int(candidates.sum()), budget_pixels=budget,
                 selected_pixels=int(selected.sum()), score_threshold=threshold,
                 cutoff_score=float(priority[selected].min()) if selected.any() else threshold,
                 smoothing_sigma_px=sigma)
    return selected, priority.astype(np.float32), stats


def problem_mask(score, valid, regions, own_projection, cfg):
    """Historical patch-limited mask, retained only for old configurations."""
    windows = np.zeros(score.shape, bool)
    for x0, y0, x1, y1 in regions:
        windows[y0:y1, x0:x1] = True
    selected, priority, _ = _pixel_problem_mask(score, valid, own_projection, cfg, windows)
    return selected, priority


def spaced_indices(matches, confidence, priority, cfg):
    """Rank evidence and suppress nearby points by distance, without image cells."""
    if not len(matches):
        return np.empty(0, np.int64)
    xy = matches[:, 2:]
    uv = np.rint(xy).astype(int)
    h, w = priority.shape
    scores = confidence * (1 + priority[np.clip(uv[:, 1], 0, h-1),
                                         np.clip(uv[:, 0], 0, w-1)])
    radius = float(cfg.get('pending_min_distance_px', 8.))
    tree = cKDTree(xy)
    suppressed = np.zeros(len(matches), bool)
    selected = []
    for i in np.argsort(-scores, kind='stable'):
        if suppressed[i]:
            continue
        selected.append(int(i))
        suppressed[tree.query_ball_point(xy[i], radius)] = True
        if len(selected) >= int(cfg.get('pending_max_active', 600)):
            break
    return np.asarray(selected, np.int64)


def balanced_indices(matches, confidence, priority, cfg):
    """Bound evidence per target-image cell; do not let one dense patch dominate."""
    if not len(matches):
        return np.empty(0, np.int64)
    xy = np.rint(matches[:, 2:]).astype(int)
    h, w = priority.shape
    scores = confidence * (1 + priority[np.clip(xy[:, 1], 0, h-1),
                                         np.clip(xy[:, 0], 0, w-1)])
    cell = int(cfg.get('pending_cell_px', 24))
    keys = xy // cell
    groups = {}
    for i in np.argsort(-scores, kind='stable'):
        key = tuple(keys[i])
        group = groups.setdefault(key, [])
        if len(group) < int(cfg.get('pending_per_cell', 6)):
            group.append(int(i))
    # Round-robin across cells before spending a second point on any cell.
    selected = [group[j] for j in range(int(cfg.get('pending_per_cell', 6)))
                for group in groups.values() if j < len(group)]
    return np.asarray(selected[:int(cfg.get('pending_max_active', 600))], np.int64)


class PendingMatches:
    """Seen correspondences are cached until retained, including the 0.5--0.8m band."""
    def __init__(self):
        self.matches = np.empty((0, 4), np.float32)

    def add(self, matches):
        self.matches, _ = add_unique(self.matches, matches)

    def remove_retained(self, retained):
        if len(self.matches) and len(retained):
            distance, _ = cKDTree(retained).query(self.matches)
            self.matches = self.matches[distance > 3*np.sqrt(2)]


def region_gate(before, after, mask, cfg):
    """Require improvement on the triggering region and protect its surroundings."""
    common = before['coverage'] & after['coverage']
    delta = before['score'] - after['score']
    inside, outside = common & mask, common & ~mask
    gain = float(delta[common].mean()) if common.any() else -float('inf')
    local = float(delta[inside].mean()) if inside.any() else -float('inf')
    outer = float(delta[outside].mean()) if outside.any() else None
    coverage = int(after['coverage'].sum()) / max(int(before['coverage'].sum()), 1)
    local_coverage = int((after['coverage'] & mask).sum()) / max(
        int((before['coverage'] & mask).sum()), 1)
    accepted = (common.any() and inside.any() and
                gain >= float(cfg.get('feedback_min_render_improvement', 0)) and
                local >= float(cfg.get('mask_min_local_improvement', 0)) and
                coverage >= float(cfg.get('feedback_min_coverage_ratio', .98)) and
                local_coverage >= float(cfg.get('mask_min_coverage_ratio', .95)) and
                (outer is None or outer >= -float(cfg.get('mask_max_outside_regression', .002))))
    return dict(accepted=bool(accepted), render_gain=gain, mask_gain=local,
                outside_gain=outer, coverage_ratio=coverage,
                mask_coverage_ratio=local_coverage, common_pixels=int(common.sum()))


def minimum_errors(anchors, models):
    return np.stack([errors(anchors['x0'], anchors['x1'], H)
                     for H in models], axis=1).min(axis=1)


def fit_proposals(state, anchors, gmap, cfg, cycle):
    """Fit local evidence without first requiring agreement with a current H_k."""
    dcfg = cfg['discovery']
    minimum = int(dcfg.get('repair_min_anchors', 12))
    if len(anchors['x0']) < minimum:
        return []
    _, nearest = cKDTree(gmap['uv']).query(anchors['matches'][:, :2], workers=-1)
    priors = state['q'][nearest]
    proposals = []
    clusters = unexplained_clusters(anchors, dcfg)
    for cluster_id, ids in enumerate(clusters[:int(dcfg.get('repair_max_clusters', 6))]):
        if len(ids) < minimum:
            continue
        local = {key: value[ids] for key, value in anchors.items()}
        fit_cfg = {**cfg['motion'], 'max_components': 2, 'min_support': minimum,
                   'inlier_threshold_m': float(dcfg.get('repair_fit_threshold_m', .5))}
        try:
            fitted, labels, _ = discover(local, fit_cfg,
                                         int(cfg.get('seed', 7)) + 1009*(cycle+1) + cluster_id)
        except RuntimeError:
            continue
        for j, H in enumerate(fitted):
            support = ids[labels == j]
            confidence = anchors.get('confidence', np.ones(len(priors)))[support]
            if np.median(confidence) < float(dcfg.get('repair_min_confidence', .15)):
                continue
            weights = confidence / confidence.sum()
            association = (priors[support] * weights[:, None]).sum(axis=0)
            # A weak spatial ownership prior nominates components; geometry
            # and rendering decide. Never force the new points into that label.
            order = [k for k in np.argsort(-association)
                     if k != state['reference'] or dcfg.get('repair_reference', False)]
            for k in order[:int(dcfg.get('repair_components_per_fit', 2))]:
                if association[k] < float(dcfg.get('repair_min_prior', 0)):
                    continue
                replacement = list(state['H'].copy())
                replacement[int(k)] = H
                proposals.append(dict(kind='repair', component=int(k), models=replacement,
                                      support_ids=support, prior=float(association[k]),
                                      cluster=cluster_id))
            if (dcfg.get('enable_birth', True) and
                    len(state['H']) < int(dcfg.get('max_components_total', 8)) and
                    len(support) >= int(dcfg['birth_min_anchors']) and
                    np.median(confidence) >= float(dcfg['birth_min_median_stereo_confidence'])):
                uv = anchors['matches'][support, 2:]
                area = float(np.prod(uv.max(axis=0)-uv.min(axis=0)))
                if area >= float(dcfg['birth_min_area_px']):
                    proposals.append(dict(kind='birth', models=[*state['H'], H],
                                          support_ids=support, cluster=cluster_id,
                                          image_area_px=area))
    repairs = [p for p in proposals if p['kind'] == 'repair']
    births = [p for p in proposals if p['kind'] == 'birth']
    return (repairs[:int(dcfg.get('repair_max_proposals', 6))] +
            births[:int(dcfg.get('max_birth_clusters', 3))])


def warm_ownership(state, models, anchors, new_anchors, gmap, source, cfg):
    from joint_pair import anchor_probabilities, gaussian_probabilities

    q = state['q'].copy()
    if len(models) > q.shape[1]:
        q = np.column_stack((q * .975, np.full(len(q), .025, np.float32)))
    qa = anchor_probabilities(anchors, models, float(cfg['motion']['assignment_sigma_m']))
    fresh = gaussian_probabilities(gmap, anchors, qa, state['reference'],
                                   cfg['ownership'], source)
    distance, _ = cKDTree(new_anchors['x0']).query(gmap['xyz'], workers=-1)
    radius = float(cfg['ownership'].get('refresh_radius_m', 2.))
    mix = float(cfg['ownership'].get('refresh_fraction', .75)) * np.exp(-(distance/radius)**2)
    result = np.maximum(q*(1-mix[:, None])+fresh*mix[:, None], 1e-6)
    return (result / result.sum(axis=1, keepdims=True)).astype(np.float32)


def feedback_cycle(state, anchors, gmap, active, source, cfg, cycle, mask,
                   prepare, optimize, structural_gate):
    """Compare continuation, ownership refresh, motion replacement and birth.

    `prepare` lifts/binds matches, and `optimize` calls the existing optimizer.
    A second, matched-budget check on just the retained subset prevents
    unexplainable pending points from silently entering the retained state.
    """
    dcfg = cfg['discovery']
    record = dict(proposals=[], retained_3d_anchors=0,
                  existing_motion_accepted=False, comparison_3d_anchors=0)
    if not len(active['x0']):
        return state, anchors, gmap, record
    count = len(anchors['x0'])
    matches = np.concatenate((anchors['matches'], active['matches']))
    trial_map, trial = prepare(matches)
    record['comparison_3d_anchors'] = len(trial['x0'])
    baseline = optimize(trial_map, trial, list(state['H']), state['q'])
    specifications = [dict(kind='attach', models=list(state['H']))]
    specifications += fit_proposals(state, active, gmap, cfg, cycle)
    candidates = []
    # Continuation is also an option: new geometry constraints may repair a
    # motion without an explicit replacement or an ownership reset.
    guard = region_gate(state, baseline, mask, dcfg)
    row = dict(kind='attach', variant='continuation', accepted=False, **{
        key: value for key, value in guard.items() if key != 'accepted'})
    record['proposals'].append(row)
    if guard['accepted']:
        candidates.append((baseline, row))
    for spec in specifications:
        info = {key: value for key, value in spec.items()
                if key not in ('models', 'support_ids')}
        if 'support_ids' in spec:
            info['support'] = int(len(spec['support_ids']))
        try:
            q = warm_ownership(state, spec['models'], trial, active, trial_map,
                               source, cfg)
            candidate = optimize(trial_map, trial, spec['models'], q)
            guard = region_gate(state, candidate, mask, dcfg)
            info.update({key: value for key, value in guard.items() if key != 'accepted'})
            viable = guard['accepted']
            if spec['kind'] != 'attach':
                comparison = structural_gate(baseline, candidate, dcfg, spec['kind'])
                info['same_evidence_gate'] = comparison
                viable = viable and comparison['accepted']
            info['eligible'] = bool(viable)
            info['accepted'] = False
            record['proposals'].append(info)
            if viable:
                candidates.append((candidate, info))
        except RuntimeError as exc:
            record['proposals'].append(dict(**info, accepted=False, error=str(exc)))
    penalty = float(dcfg.get('complexity_penalty', .005))
    candidates.sort(key=lambda item: item[0]['loss'] + penalty*len(item[0]['H']))
    limit = float(dcfg.get('attach_threshold_m', .5))
    for winner, info in candidates:
        retained = minimum_errors(active, winner['H']) < limit
        if not retained.any():
            info['commit_rejection'] = 'no_new_points_explained'
            continue
        final_matches = np.concatenate((anchors['matches'], active['matches'][retained]))
        final_map, final_anchors = prepare(final_matches)
        # Both finalists received the same trial budget; refine both for the
        # same number of steps on the exact geometry/evidence being committed.
        base_final = optimize(final_map, final_anchors, list(baseline['H']), baseline['q'])
        candidate_final = optimize(final_map, final_anchors, list(winner['H']), winner['q'])
        guard = region_gate(state, candidate_final, mask, dcfg)
        info['commit_gate'] = guard
        viable = guard['accepted']
        if info['kind'] != 'attach':
            comparison = structural_gate(base_final, candidate_final, dcfg, info['kind'])
            info['commit_same_evidence_gate'] = comparison
            viable = viable and comparison['accepted']
        if not viable:
            continue
        # Attach updates do not change lineage. Structural viewers only see
        # accepted repair/birth records, while the attach decision is separate.
        info['accepted'] = info['kind'] != 'attach'
        info['committed'] = True
        record.update(retained_3d_anchors=len(final_anchors['x0'])-count,
                      existing_motion_accepted=True,
                      existing_motion_update=guard['render_gain'],
                      mask_improvement=guard['mask_gain'],
                      outside_improvement=guard['outside_gain'],
                      selected_update=info['kind'])
        return candidate_final, final_anchors, final_map, record
    return state, anchors, gmap, record

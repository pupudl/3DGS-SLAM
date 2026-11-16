"""CPU guard against destroying a motion model's fixed sparse observations.

This checks consistency on the same measured observations before and after a
pose update. It is not an estimate of matching precision or physical accuracy.
"""
from __future__ import annotations

import numpy as np

from .hybrid_motion import _core_labels, _metric_consistency, motion_costs


def _models(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        array = array.reshape(0, 4, 4)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f'{name} must have shape (M, 4, 4)')
    return array


def guard(anchors, initial_models, proposed_models, K, hybrid_cfg, options=None):
    """Accept/revert each proposed pose on its *initial* exclusive metric core.

    Options: ``min_core_count`` (hybrid min_support, otherwise 6),
    ``min_support_fraction`` (.8), ``max_clipped_cost_increase`` (.1), and
    ``enabled`` (True). Loss is the confidence-weighted mean of costs clipped
    at the hybrid outlier cost, over the entire fixed core. Retained support
    also requires the uncapped metric check. Models with too few initial core
    points remain free to update; no new core points replace lost ones.

    Returns models (M,4,4), accepted_mask (M,), and a JSON-serializable report.
    Inputs are never mutated. Confidence is measured anchor reliability;
    missing confidence means uniform weights, with no artificial weight floor.
    """
    options = dict(options or {})
    initial = _models(initial_models, 'initial_models')
    proposed = _models(proposed_models, 'proposed_models')
    if proposed.shape != initial.shape:
        raise ValueError('Initial and proposed models must have identical shapes')
    if not np.isfinite(initial).all():
        raise ValueError('initial_models must be finite')
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError('K must be a finite (3, 3) array')

    count = options.get('min_core_count', hybrid_cfg.get('min_support', 6))
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError('min_core_count must be a positive integer')
    fraction = float(options.get('min_support_fraction', .8))
    allowance = float(options.get('max_clipped_cost_increase', .1))
    outlier = float(hybrid_cfg.get('outlier_cost', 1.))
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('min_support_fraction must lie in [0, 1]')
    if not np.isfinite(allowance) or allowance < 0:
        raise ValueError('max_clipped_cost_increase must be finite and nonnegative')
    if not np.isfinite(outlier) or outlier <= 0:
        raise ValueError('outlier_cost must be finite and positive')
    enabled = options.get('enabled', True)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError('enabled must be boolean')

    observed = dict(anchors)
    for key, width in [('x0', 3), ('x1', 3), ('matches', 4)]:
        array = np.asarray(anchors[key], dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(f'anchors[{key!r}] must have shape (N, {width})')
        observed[key] = array
    n = len(observed['matches'])
    if len(observed['x0']) != n or len(observed['x1']) != n:
        raise ValueError('Anchor arrays must have the same number of rows')
    confidence = np.asarray(anchors.get('confidence', np.ones(n)), dtype=np.float64).reshape(-1)
    if len(confidence) != n or not np.isfinite(confidence).all() or np.any(confidence < 0):
        raise ValueError('Anchor confidence must have N finite nonnegative values')

    # Empty rows/models cannot supply a core, and need no argmin operation.
    if n and len(initial):
        initial_costs, labels = _core_labels(observed, initial, K, hybrid_cfg)
        proposed_costs = motion_costs(observed, proposed, K, hybrid_cfg)
        proposed_metric = _metric_consistency(observed, proposed, hybrid_cfg)
    else:
        initial_costs = proposed_costs = np.empty((n, len(initial)), np.float64)
        proposed_metric = np.empty((n, len(initial)), bool)
        labels = np.full(n, -1, np.int32)

    models = proposed.copy()
    accepted = np.ones(len(initial), bool)
    reports = []
    for model in range(len(initial)):
        ids = np.flatnonzero(labels == model)
        weights = confidence[ids]
        mass = float(weights.sum())
        finite_pose = bool(np.isfinite(proposed[model]).all())
        valid = ((proposed_costs[ids, model] < outlier) &
                 proposed_metric[ids, model])
        new_mass = float(weights[valid].sum())
        old_loss = float(np.average(np.minimum(initial_costs[ids, model], outlier), weights=weights)) if mass > 0 else None
        new_loss = float(np.average(np.minimum(proposed_costs[ids, model], outlier), weights=weights)) if mass > 0 else None
        retained = new_mass / mass if mass > 0 else None
        evaluated = bool(enabled and len(ids) >= count and mass > 0)
        reasons = []
        if enabled and not finite_pose:
            reasons.append('nonfinite_proposed_pose')
        if evaluated:
            if new_mass + 1e-12 * max(1., mass) < fraction * mass:
                reasons.append('insufficient_retained_support')
            if not np.isfinite(new_loss) or new_loss > old_loss + allowance + 1e-12:
                reasons.append('clipped_cost_increased')
        if reasons:
            accepted[model] = False
            models[model] = initial[model]
        if not enabled:
            reason = 'disabled'
        elif reasons:
            reason = ','.join(reasons)
        elif len(ids) < count:
            reason = 'insufficient_initial_core'
        elif mass <= 0:
            reason = 'zero_initial_support_mass'
        else:
            reason = 'accepted'
        reports.append({
            'model_index': model,
            'initial_core_count': int(len(ids)),
            'evaluated': evaluated,
            'proposed_cost_inlier_count': int(np.sum(proposed_costs[ids, model] < outlier)),
            'proposed_metric_inlier_count': int(np.sum(proposed_metric[ids, model])),
            'proposed_valid_support_count': int(np.sum(valid)),
            'initial_support_mass': mass,
            'proposed_support_mass': new_mass,
            'retained_support_fraction': retained,
            'initial_clipped_cost': old_loss,
            'proposed_clipped_cost': new_loss,
            'clipped_cost_increase': new_loss - old_loss if mass > 0 else None,
            'accepted': bool(accepted[model]),
            'reason': reason,
            'fallback_reasons': reasons,
        })
    report = {
        'enabled': bool(enabled),
        'core_definition': 'fixed initial exclusive low-cost, metric-consistent anchors',
        'interpretation': 'Same-observation consistency check; not matching precision or physical accuracy.',
        'min_core_count': int(count),
        'min_support_fraction': fraction,
        'max_clipped_cost_increase': allowance,
        'outlier_cost': outlier,
        'model_count': int(len(initial)),
        'anchor_count': n,
        'fixed_core_count': int(np.sum(labels >= 0)),
        'fallback_count': int(np.sum(~accepted)),
        'models': reports,
    }
    return {'models': models, 'accepted_mask': accepted, 'report': report}


guard_motion_poses = guard

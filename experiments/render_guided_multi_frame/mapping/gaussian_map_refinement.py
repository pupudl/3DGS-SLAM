"""Bounded, evidence-driven Gaussian topology plans in owner coordinates.

This module never changes ownership or mutates a map. Apply children first with
``store.append_from_parent(children['parent_ids'], child_parameters)`` and only
then remove ``split_ids`` and ``prune_ids``. All children inherit the parent's
owner, and the store must initialize new geometry anchors and zero Adam state.
"""
from __future__ import annotations

import numpy as np


PARAMETER_WIDTHS = dict(xyz=3, colors=3, log_scales=3, rotations=4, opacity_logits=1)


def _vector(value, n, name, dtype=float):
    array = np.asarray(value, dtype=dtype)
    if array.shape not in ((n,), (n, 1)) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite per-Gaussian vector')
    return array.reshape(-1)


def _rotation(quaternion):
    w, x, y, z = quaternion/np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], np.float64)


def plan_gaussian_refinement(batch, stats, cfg, total_map_count=None):
    """Return clone/split/prune operations justified by render observations.

    ``stats`` comes from optimize_gaussian_map(collect_refinement_stats=True).
    Screen gradients are loss derivatives per original image pixel, radii are
    three-sigma original-image footprint radii. A large projected Gaussian with
    no nonzero image gradient never splits. Frozen owners, protected landmarks,
    invisible points, and poorly observed points cannot be modified. Insertion
    limits count created children; active/global limits count net live growth.
    ``max_growth_fraction`` separately limits per-update net topology growth.
    """
    options = dict(cfg)
    options.update(cfg.get('refinement', {}))
    ids = np.asarray(batch['map_ids'], np.int64)
    n = len(ids)
    if ids.ndim != 1 or len(np.unique(ids)) != n:
        raise ValueError('Refinement map IDs must be unique [N]')
    arrays = {}
    for name, width in PARAMETER_WIDTHS.items():
        value = np.asarray(batch[name], np.float32)
        if value.shape != (n, width) or not np.isfinite(value).all():
            raise ValueError(f'Refinement {name} must be finite [N,{width}]')
        arrays[name] = value
    if np.any(np.linalg.norm(arrays['rotations'], axis=1) < 1e-8):
        raise ValueError('Refinement rotations must have nonzero norm')
    owners = _vector(batch.get('object_ids', np.zeros(n, np.int64)), n, 'object_ids', np.int64)
    trainable = _vector(batch.get('trainable_mask', np.ones(n, bool)), n, 'trainable_mask', bool)
    protected = _vector(batch.get('refinement_protected_mask', np.zeros(n, bool)), n,
                        'refinement_protected_mask', bool)
    observation_count = _vector(batch.get('observation_count', np.zeros(n, np.int64)), n,
                                'observation_count', np.int64)
    stats_ids = np.asarray(stats.get('map_ids', []), np.int64)
    if stats_ids.ndim != 1 or len(np.unique(stats_ids)) != len(stats_ids):
        raise ValueError('Refinement statistics IDs must be unique [N]')
    index = {int(value): row for row, value in enumerate(stats_ids)}
    matched = np.array([row for row, value in enumerate(ids) if int(value) in index], np.int64)
    source = np.array([index[int(ids[row])] for row in matched], np.int64)
    aligned = {}
    for key in ('screen_gradient_sum', 'screen_gradient_count', 'max_radius_px', 'visible_steps'):
        raw = _vector(stats.get(key, np.zeros(len(stats_ids))), len(stats_ids), key)
        if np.any(raw < 0):
            raise ValueError(f'Refinement {key} must be nonnegative')
        aligned[key] = np.zeros(n, np.float64)
        aligned[key][matched] = raw[source]
    limits = dict(max_new_per_update=1024, max_splits_per_update=256, max_clones_per_update=128,
                  max_prunes_per_update=512, max_children_per_object=512, max_active_gaussians=100000,
                  max_total_gaussians=1000000, min_points_per_object=4, min_observation_count=2,
                  prune_min_observation_count=3, min_visible_steps=3)
    for key, default in tuple(limits.items()):
        limits[key] = int(options.get(key, default))
        if limits[key] < 0:
            raise ValueError(f'Refinement {key} must be nonnegative')
    threshold = float(options.get('gradient_threshold', 1e-6))
    split_radius = float(options.get('split_radius_px', 6.))
    min_opacity = float(options.get('prune_opacity_threshold', .02))
    growth_fraction = float(options.get('max_growth_fraction', .05))
    shrink = float(options.get('split_scale_divisor', 1.6))
    min_scale = float(options.get('min_scale_m', .002))
    if (not np.isfinite([threshold, split_radius, min_opacity, growth_fraction, shrink, min_scale]).all() or
            threshold <= 0 or split_radius <= 0 or not 0 <= min_opacity < 1 or
            growth_fraction < 0 or shrink <= 1 or min_scale <= 0):
        raise ValueError('Invalid refinement gradient, radius, opacity, growth, or scale options')
    total = n if total_map_count is None else int(total_map_count)
    if total < n:
        raise ValueError('total_map_count cannot be smaller than the selected map batch')
    eligible = trainable & ~protected & (aligned['visible_steps'] >= limits['min_visible_steps'])
    opacity = 1/(1+np.exp(-np.clip(arrays['opacity_logits'][:, 0].astype(np.float64), -60, 60)))
    # Select lowest-alpha points first, protecting a small support set per owner.
    owner_remaining = {int(owner): int(np.count_nonzero(owners == owner)) for owner in np.unique(owners)}
    prune_rows = []
    prune_candidates = np.flatnonzero(eligible & (observation_count >= limits['prune_min_observation_count']) &
                                      (opacity < min_opacity))
    for row in sorted(prune_candidates, key=lambda i: (opacity[i], int(ids[i]))):
        owner = int(owners[row])
        if len(prune_rows) >= limits['max_prunes_per_update']:
            break
        if owner_remaining[owner] <= limits['min_points_per_object']:
            continue
        prune_rows.append(int(row))
        owner_remaining[owner] -= 1
    pruned = np.zeros(n, bool)
    pruned[prune_rows] = True
    mean_gradient = np.divide(aligned['screen_gradient_sum'], aligned['screen_gradient_count'],
                              out=np.zeros(n), where=aligned['screen_gradient_count'] > 0)
    grow = (eligible & ~pruned & (observation_count >= limits['min_observation_count']) &
            (aligned['screen_gradient_count'] >= limits['min_visible_steps']) &
            (mean_gradient >= threshold) & (opacity >= min_opacity))
    candidate_rows = sorted(np.flatnonzero(grow), key=lambda i: (-mean_gradient[i], int(ids[i])))
    # Budgets are enforced on the final live map, including concurrent pruning.
    net_budget = min(max(0, limits['max_active_gaussians']-(n-len(prune_rows))),
                     max(0, limits['max_total_gaussians']-(total-len(prune_rows))),
                     int(np.floor(n*growth_fraction)))
    new_budget = limits['max_new_per_update']
    split_rows, clone_rows, child_rows = [], [], []
    children = {name: [] for name in PARAMETER_WIDTHS}
    owner_children = {}
    for row in candidate_rows:
        if len(split_rows)+len(clone_rows) >= net_budget:
            break
        owner = int(owners[row])
        split = bool(aligned['max_radius_px'][row] >= split_radius)
        count = 2 if split else 1
        if (len(child_rows)+count > new_budget or
                owner_children.get(owner, 0)+count > limits['max_children_per_object']):
            continue
        if split and len(split_rows) >= limits['max_splits_per_update']:
            continue
        if not split and len(clone_rows) >= limits['max_clones_per_update']:
            continue
        scales = np.exp(arrays['log_scales'][row].astype(np.float64))
        if split and scales.max()/shrink < min_scale:
            continue
        axis = int(np.argmax(scales))
        direction = _rotation(arrays['rotations'][row])[:, axis]*scales[axis]
        offsets = (-.5*direction, .5*direction) if split else (.25*direction,)
        for offset in offsets:
            child_rows.append(int(row))
            for name in PARAMETER_WIDTHS:
                value = arrays[name][row].copy()
                if name == 'xyz':
                    value = value+offset
                elif name == 'log_scales' and split:
                    value = np.log(np.maximum(scales/shrink, min_scale))
                # Split children distribute optical thickness; clones start
                # with lower opacity and are fitted in a subsequent map pass.
                elif name == 'opacity_logits':
                    alpha = 1-np.sqrt(1-opacity[row]) if split else opacity[row]*.5
                    alpha = np.clip(alpha, 1e-6, 1-1e-6)
                    value = np.array([np.log(alpha/(1-alpha))])
                children[name].append(value)
        (split_rows if split else clone_rows).append(int(row))
        owner_children[owner] = owner_children.get(owner, 0)+count
    children = {name: np.asarray(value, np.float32).reshape(-1, width)
                for (name, width), value in zip(PARAMETER_WIDTHS.items(), children.values())}
    children.update(parent_ids=ids[np.asarray(child_rows, np.int64)].copy(),
                    object_ids=owners[np.asarray(child_rows, np.int64)].copy())
    result = dict(prune_ids=ids[np.asarray(prune_rows, np.int64)].copy(),
                  split_ids=ids[np.asarray(split_rows, np.int64)].copy(),
                  clone_ids=ids[np.asarray(clone_rows, np.int64)].copy(), children=children)
    growth = len(child_rows)-len(split_rows)-len(prune_rows)
    result['report'] = dict(pruned_count=len(prune_rows), split_count=len(split_rows), clone_count=len(clone_rows),
                          child_count=len(child_rows), net_growth=growth, candidate_count=len(candidate_rows),
                          active_count_before=n, active_count_after=n+growth,
                          total_count_before=total, total_count_after=total+growth,
                          gradient_threshold=threshold, gradient_units=stats.get('gradient_units', 'unknown'),
                          split_radius_px=split_radius, protected_count=int(protected.sum()),
                          frozen_count=int((~trainable).sum()), limits=limits)
    return result

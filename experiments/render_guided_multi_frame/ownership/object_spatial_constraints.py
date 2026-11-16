"""Fixed-window spatial observations shared by the differentiable object solver.

Ground is estimated in each source camera. Heights and graph edges are mapped
to persistent Gaussian IDs, without mixing camera-coordinate XYZ across frames.
Only object probabilities receive gradients from the returned fixed costs.
"""
from __future__ import annotations

import json

import numpy as np


def prepare_spatial_constraints(window, cfg):
    height_options = dict(cfg.get('height', {}))
    connectivity_options = dict(cfg.get('connectivity', {}))
    height_enabled = bool(height_options.get('enabled', False))
    connectivity_enabled = bool(connectivity_options.get('enabled', False))
    n = len(window._active_gids)
    result = dict(height_cost=np.zeros(n, np.float32), height_reliability=np.zeros(n, np.float32),
                  hard_background_mask=np.zeros(n, bool),
                  height_background_mask=np.zeros(n, bool), connectivity_background_mask=np.zeros(n, bool),
                  no_core_disabled_columns=np.zeros(len(getattr(window, 'object_ids', ())), bool),
                  no_core_initial_probabilities=None,
                  height_info=None, connectivity_state=None, connectivity_targets=None,
                  region_completion=None,
                  report=dict(height=dict(enabled=height_enabled),
                              connectivity=dict(enabled=connectivity_enabled)))
    if height_enabled:
        from ownership.object_height_constraints import estimate_ground_plane, height_penalty, above_height_limit

        numerator = np.zeros(n)
        height_sum = np.zeros(n)
        weight_sum = np.zeros(n)
        occurrences = np.zeros(n)
        allowed_sum = np.zeros(n)
        above_sum, below_sum = np.zeros(n), np.zeros(n)
        cache = getattr(window, '_ground_plane_cache', {})
        cache_key = json.dumps(height_options, sort_keys=True)
        retained_cache, planes = {}, {}
        for pair in sorted(window._active):
            snap = window._active[pair]
            payload = snap['payload']
            observations = snap.get('render_observations') or {}
            if not {'xyz', 'uv'} <= payload.keys() or 'source' not in observations:
                planes[str(snap['first'])] = dict(valid=False, reason='missing_source_geometry_or_observations')
                continue
            xyz, uv = np.asarray(payload['xyz']), np.asarray(payload['uv'])
            # Replacing an interval with changed observations must not reuse
            # its predecessor's plane just because its point count is equal.
            key = (int(snap['first']), id(snap), len(xyz), cache_key)
            if key not in cache:
                cached_plane = None
                if ('ground_plane_report_json' in payload and
                        not height_options.get('refit_cached_plane', False)):
                    try:
                        candidate = json.loads(str(np.asarray(payload['ground_plane_report_json']).item()))
                        normal = np.asarray(candidate['normal'], float)
                        if (normal.shape == (3,) and np.isfinite(normal).all() and
                                np.linalg.norm(normal) > 1e-8 and
                                np.isfinite([candidate['offset'], candidate['camera_height'],
                                             candidate['confidence']]).all()):
                            cached_plane = candidate
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                        pass
                cache[key] = cached_plane if cached_plane is not None else estimate_ground_plane(
                    xyz, uv, observations['source'].shape,
                    confidence=observations.get('source_confidence'), options=height_options)
            plane = cache[key]
            retained_cache[key] = plane
            planes[str(snap['first'])] = plane
            payload.update(ground_plane_normal=np.asarray(plane['normal'], np.float64),
                           ground_plane_offset=np.asarray(plane['offset']),
                           ground_plane_confidence=np.asarray(plane['confidence']),
                           ground_plane_valid=np.asarray(plane['valid']),
                           ground_plane_report_json=np.asarray(json.dumps(plane)))
            if not plane['valid']:
                continue
            rows = np.searchsorted(window._active_gids, snap['gaussian_ids'])
            if height_options.get('hard_upper_bound', False):
                # One shared ID has one assignment: any measured source-camera
                # violation overrides a majority of in-band observations.
                excluded = above_height_limit(xyz, plane, height_options)
                np.logical_or.at(result['hard_background_mask'], rows, excluded)
                payload['height_above_upper_bound'] = excluded
            penalty, reliability = height_penalty(xyz, plane, height_options)
            if 'source_confidence' in observations:
                confidence = np.asarray(observations['source_confidence'])
                finite = np.isfinite(uv).all(1)
                pixels = np.zeros_like(uv, np.int64)
                pixels[finite] = np.rint(uv[finite]).astype(np.int64)
                valid = (finite & (pixels[:, 0] >= 0) & (pixels[:, 0] < confidence.shape[1]) &
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < confidence.shape[0]))
                source_quality = np.zeros(len(xyz))
                source_quality[valid] = np.clip(np.nan_to_num(
                    confidence[pixels[valid, 1], pixels[valid, 0]], nan=0.), 0, 1)
                reliability *= source_quality
            height = xyz @ np.asarray(plane['normal']) + plane['offset']
            lower = float(height_options.get('ground_min_m', -.2))
            upper = float(height_options.get('max_height_m',
                          plane['camera_height'] + float(height_options.get('above_camera_m', 2.))))
            valid = np.isfinite(height) & (reliability > 0)
            weight = np.where(valid, reliability, 0)
            numerator[rows] += penalty*weight
            height_sum[rows] += np.where(valid, height, 0)*weight
            weight_sum[rows] += weight
            occurrences[rows] += valid
            allowed_sum[rows] += weight*((height >= lower) & (height <= upper))
            above_sum[rows] += weight*(height > upper)
            below_sum[rows] += weight*(height < lower)
        window._ground_plane_cache = retained_cache
        valid = weight_sum > 0
        cost = np.divide(numerator, weight_sum, out=np.zeros(n), where=valid)
        reliability = np.divide(weight_sum, occurrences, out=np.zeros(n), where=occurrences > 0)
        height = np.divide(height_sum, weight_sum, out=np.full(n, np.nan), where=valid)
        allowed = ~valid | (allowed_sum >= .5*weight_sum)
        allowed &= ~result['hard_background_mask']
        result.update(height_cost=cost.astype(np.float32), height_reliability=reliability.astype(np.float32),
                      height_info=dict(height=height, valid=valid, allowed=allowed,
                                       penalty=cost, reliability=reliability,
                                       hard_background_mask=result['hard_background_mask']))
        result['report']['height'].update(
            planes=planes, valid_planes=sum(bool(plane['valid']) for plane in planes.values()),
            known_gaussians=int(valid.sum()), outside_band_gaussians=int(np.count_nonzero(valid & ~allowed)),
            above_band_gaussians=int(np.count_nonzero(valid & (above_sum >= .5*weight_sum))),
            below_band_gaussians=int(np.count_nonzero(valid & (below_sum >= .5*weight_sum))),
            ground_min_m=float(height_options.get('ground_min_m', -.2)),
            above_camera_m=float(height_options.get('above_camera_m', 2.)),
            weight=float(height_options.get('weight', .15)),
            hard_upper_bound=bool(height_options.get('hard_upper_bound', False)),
            hard_background_gaussians=int(result['hard_background_mask'].sum()),
            hard_height_aggregation='any_valid_source_observation',
            background_object_id=int(height_options.get('background_object_id', 0)))
    result['height_background_mask'] = result['hard_background_mask'].copy()
    if connectivity_enabled:
        from ownership.object_connectivity_constraints import (build_connectivity_state,
                                                               condition_core_initialization,
                                                               no_core_background_policy,
                                                               update_connectivity_targets)

        state = build_connectivity_state(window, connectivity_options, result['height_info'])
        initial = window._render_initial_probabilities
        mode = connectivity_options.get('initialization_conditioning', 'none')
        render_options = cfg.get('render', {})
        conditioning_allowed = (render_options.get('enabled', False) and
                                render_options.get('initialization', 'previous') == 'measurement' and
                                connectivity_options.get('no_core_to_background', False) and
                                connectivity_options.get('final_hard_assignment', False))
        conditioning_report = dict(mode=mode, enabled=False,
                                   reason='disabled' if mode == 'none' else 'missing_preconditions',
                                   eligible_object_ids=[], selected_count=0,
                                   removed_probability_mass=0., zero_mass_count=0)
        if mode != 'none' and conditioning_allowed:
            conditioned = condition_core_initialization(state, initial)
            initial = conditioned['probabilities']
            conditioning_report = conditioned['report']
        fallback = no_core_background_policy(state, initial)
        result['connectivity_background_mask'] = fallback['forced_background_mask']
        result['hard_background_mask'] |= fallback['forced_background_mask']
        result['no_core_disabled_columns'] = fallback['disabled_columns']
        result['no_core_initial_probabilities'] = fallback['probabilities']
        targets = update_connectivity_targets(state, fallback['probabilities'])
        result.update(connectivity_state=state, connectivity_targets=targets)
        result['report']['connectivity'].update(
            graph=state['diagnostics'], initial_targets=targets['report'], no_core_fallback=fallback['report'],
            initialization_conditioning=conditioning_report,
            weight=float(connectivity_options.get('weight', .08)),
            seed_weight=float(connectivity_options.get('seed_weight', .03)),
            update_interval=int(connectivity_options.get('update_interval', 5)))
        region_options = dict(cfg.get('region_completion', {}))
        if region_options.get('enabled', False):
            from ownership.core_region_completion import (build_region_completion,
                                                          initialize_region_probabilities)
            region = build_region_completion(window, state, region_options)
            result['region_completion'] = region
            result['report']['region_completion'] = dict(
                region['report'], weight=float(region_options.get('weight', .1)))
            selected = np.unique(region['rows'])
            release = selected[result['connectivity_background_mask'][selected] &
                               ~result['height_background_mask'][selected]]
            # A former no-core winner cannot veto fresh, core-connected surface
            # evidence for a DIFFERENT supported object. Its old object column
            # stays disabled; height exclusions are never released.
            result['connectivity_background_mask'][release] = False
            result['hard_background_mask'][release] = False
            initial_min = float(region_options.get('min_initial_probability', .1))
            initial, changed = initialize_region_probabilities(
                result['no_core_initial_probabilities'], region, initial_min)
            result['no_core_initial_probabilities'] = initial
            result['report']['region_completion'].update(
                released_no_core_winner_rows=len(release), initialized_gaussian_count=changed,
                min_initial_probability=initial_min)
            # Each proposed surface has a current RGB-D fit (outgoing, or
            # incoming when the outgoing model is absent). Admit it only as
            # an observed surface; strict repeated core evidence is unchanged.
            state['model_observed'][region['rows'], region['columns']] = True
            for pair, snap in window._active.items():
                entries = region['by_pair'].get(pair, [])
                snap['payload']['region_completion_gaussian_ids'] = (
                    np.concatenate([e['gaussian_ids'] for e in entries]) if entries else np.empty(0, np.int64))
                snap['payload']['region_completion_object_ids'] = (
                    np.concatenate([np.full(len(e['gaussian_ids']), e['object_id'], np.int64)
                                    for e in entries]) if entries else np.empty(0, np.int64))
                snap['payload']['region_completion_weights'] = (
                    np.concatenate([e['weights'] for e in entries]) if entries else np.empty(0, np.float32))
    if cfg.get('region_completion', {}).get('enabled', False) and not connectivity_enabled:
        result['report']['region_completion'] = dict(enabled=False, reason='connectivity_disabled')
    return result

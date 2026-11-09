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
                  height_info=None, connectivity_state=None, connectivity_targets=None,
                  report=dict(height=dict(enabled=height_enabled),
                              connectivity=dict(enabled=connectivity_enabled)))
    if height_enabled:
        from object_height_constraints import estimate_ground_plane, height_penalty

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
        result.update(height_cost=cost.astype(np.float32), height_reliability=reliability.astype(np.float32),
                      height_info=dict(height=height, valid=valid, allowed=allowed,
                                       penalty=cost, reliability=reliability))
        result['report']['height'].update(
            planes=planes, valid_planes=sum(bool(plane['valid']) for plane in planes.values()),
            known_gaussians=int(valid.sum()), outside_band_gaussians=int(np.count_nonzero(valid & ~allowed)),
            above_band_gaussians=int(np.count_nonzero(valid & (above_sum >= .5*weight_sum))),
            below_band_gaussians=int(np.count_nonzero(valid & (below_sum >= .5*weight_sum))),
            ground_min_m=float(height_options.get('ground_min_m', -.2)),
            above_camera_m=float(height_options.get('above_camera_m', 2.)),
            weight=float(height_options.get('weight', .15)), background_object_id=0)
    if connectivity_enabled:
        from object_connectivity_constraints import build_connectivity_state, update_connectivity_targets

        state = build_connectivity_state(window, connectivity_options, result['height_info'])
        targets = update_connectivity_targets(state, window._render_initial_probabilities)
        result.update(connectivity_state=state, connectivity_targets=targets)
        result['report']['connectivity'].update(
            graph=state['diagnostics'], initial_targets=targets['report'],
            weight=float(connectivity_options.get('weight', .08)),
            seed_weight=float(connectivity_options.get('seed_weight', .03)),
            update_interval=int(connectivity_options.get('update_interval', 5)))
    return result

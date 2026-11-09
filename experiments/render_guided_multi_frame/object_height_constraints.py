"""Robust camera-coordinate ground estimates and soft object-height costs.

Coordinates are X right, Y down, Z forward. Camera Y is only an orientation
hint: the fitted up normal determines gravity-relative height h = n·X + d.
No object labels, semantic masks, ground-truth poses, or object probabilities
participate in estimation. Insufficiently supported planes disable the cost.
"""
from __future__ import annotations

import numpy as np


def _invalid(reason, **statistics):
    return dict(valid=False, reason=str(reason), normal=[0., -1., 0.], offset=0.,
                camera_height=0., confidence=0., **statistics)


def _orientation(options):
    up = np.asarray(options.get('config_up', options.get('calibration_up',
                    options.get('up', [0., -1., 0.]))), dtype=float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError('Ground up direction must be a finite nonzero 3-vector')
    return up/np.linalg.norm(up)


def _weighted_plane(points, weights, up):
    mass = float(weights.sum())
    if mass <= 0:
        return None
    center = np.sum(points*weights[:, None], axis=0)/mass
    centered = points-center
    covariance = (centered*weights[:, None]).T @ centered/mass
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= 1e-8:
        return None
    normal = vectors[:, 0]
    if normal @ up < 0:
        normal = -normal
    return normal, -float(normal @ center)


def estimate_ground_plane(xyz, uv, image_shape, confidence=None, options=None):
    """Fit a reliable broad ground plane to lower-image RGB-D surface points.

    The plane is oriented toward ``config_up`` (default camera -Y), allowing
    ``max_tilt_degrees`` of pitch/roll. RANSAC hypotheses must have a plausible
    positive camera height. Final support must span image X/Y and several
    depth bins, rejecting walls and short local vehicle-body planes.

    All returned values are JSON serializable. ``confidence`` may be [N] or
    an image-sized stereo-confidence map. Low-quality fits return valid=False.
    """
    options = {} if options is None else dict(options)
    xyz, uv = np.asarray(xyz, float), np.asarray(uv, float)
    n = len(xyz)
    if xyz.shape != (n, 3) or uv.shape != (n, 2):
        raise ValueError('Ground estimation requires xyz[N,3] and uv[N,2]')
    h, w = map(int, image_shape[:2])
    if h < 1 or w < 1:
        raise ValueError('Ground estimation image dimensions must be positive')
    up = _orientation(options)
    minimum = int(options.get('minimum_points', 300))
    minimum_inliers = int(options.get('minimum_inliers', 200))
    maximum = int(options.get('max_points', 20000))
    iterations = int(options.get('ransac_iterations', 512))
    min_depth = float(options.get('min_depth_m', 2.))
    max_depth = float(options.get('max_ground_depth_m', 35.))
    threshold_base = float(options.get('distance_threshold_m', .08))
    threshold_relative = float(options.get('distance_threshold_relative', .005))
    minimum_confidence = float(options.get('min_confidence', .15))
    min_height = float(options.get('min_camera_height_m', .3))
    max_height = float(options.get('max_camera_height_m', 3.))
    angle = float(options.get('max_tilt_degrees', 25.))
    if (min(minimum, minimum_inliers, maximum, iterations) < 3 or maximum < minimum or
            not 0 <= min_depth < max_depth or threshold_base <= 0 or threshold_relative < 0 or
            not 0 < min_height < max_height or not 0 < angle < 90):
        raise ValueError('Invalid ground-plane sampling, depth, residual or orientation options')
    finite_uv = np.isfinite(uv).all(axis=1)
    in_image = (finite_uv & (uv[:, 0] >= 0) & (uv[:, 0] < w) &
                (uv[:, 1] >= 0) & (uv[:, 1] < h))
    if confidence is None:
        point_confidence = np.ones(n)
    else:
        raw = np.asarray(confidence, float)
        if raw.shape == (n,):
            point_confidence = raw.copy()
        elif raw.shape == (h, w):
            point_confidence = np.zeros(n)
            ids = np.flatnonzero(in_image)
            xy = np.rint(uv[ids]).astype(int)
            xy[:, 0], xy[:, 1] = np.minimum(xy[:, 0], w-1), np.minimum(xy[:, 1], h-1)
            point_confidence[ids] = raw[xy[:, 1], xy[:, 0]]
        else:
            raise ValueError('Ground confidence must have shape [N] or image [H,W]')
    lower_y = float(options.get('min_image_y_fraction', .55))
    horizontal_margin = float(options.get('horizontal_margin_fraction', .05))
    if not 0 <= lower_y < 1 or not 0 <= horizontal_margin < .5:
        raise ValueError('Invalid lower-image ground candidate region')
    candidates = (in_image & np.isfinite(xyz).all(axis=1) &
                  (xyz[:, 2] >= min_depth) & (xyz[:, 2] <= max_depth) &
                  (uv[:, 1] >= lower_y*h) &
                  (uv[:, 0] >= horizontal_margin*w) & (uv[:, 0] < (1-horizontal_margin)*w) &
                  np.isfinite(point_confidence) & (point_confidence >= minimum_confidence))
    candidate_ids = np.flatnonzero(candidates)
    common = dict(total_points=n, candidate_points=int(len(candidate_ids)),
                  up_hint=up.tolist(), max_tilt_degrees=angle)
    if len(candidate_ids) < minimum:
        return _invalid('insufficient_reliable_lower_image_points', **common)
    rng = np.random.default_rng(int(options.get('seed', 17)))
    if len(candidate_ids) > maximum:
        candidate_ids = np.sort(rng.choice(candidate_ids, maximum, replace=False))
    points, pixels = xyz[candidate_ids], uv[candidate_ids]
    quality = np.clip(point_confidence[candidate_ids], 0, 1)
    threshold = threshold_base+threshold_relative*points[:, 2]
    common['sampled_points'] = len(points)
    samples = rng.integers(len(points), size=(iterations, 3))
    seeds = points[samples]
    normals = np.cross(seeds[:, 1]-seeds[:, 0], seeds[:, 2]-seeds[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)
    normals[normals @ up < 0] *= -1
    offsets = -np.sum(normals*seeds[:, 0], axis=1)
    orientation = normals @ up >= np.cos(np.deg2rad(angle))
    plausible = ((lengths > 1e-7) & orientation &
                 (offsets >= min_height) & (offsets <= max_height))
    hypotheses = np.flatnonzero(plausible)
    common['orientation_valid_hypotheses'] = len(hypotheses)
    if not len(hypotheses):
        return _invalid('no_upward_ground_hypothesis', **common)
    # Score in small batches: do not allocate Npoints × all hypotheses for
    # large user-supplied RANSAC budgets.
    scores = np.empty(len(hypotheses))
    for start in range(0, len(hypotheses), 64):
        ids = hypotheses[start:start+64]
        residual = np.abs(points @ normals[ids].T+offsets[ids])
        influence = np.maximum(0., 1-(residual/threshold[:, None])**2)
        scores[start:start+64] = (influence*quality[:, None]).sum(axis=0)
    order = np.argsort(-scores, kind='stable')[:int(options.get('refine_hypotheses', 24))]
    best, best_score, rejection = None, -np.inf, 'insufficient_ground_support'
    for proposal in order:
        normal = normals[hypotheses[proposal]].copy()
        offset = float(offsets[hypotheses[proposal]])
        for _ in range(3):
            residual = np.abs(points @ normal+offset)
            inliers = residual <= threshold
            if inliers.sum() < minimum_inliers:
                break
            robust = np.maximum(.05, 1-(residual[inliers]/threshold[inliers])**2)
            fitted = _weighted_plane(points[inliers], quality[inliers]*robust, up)
            if fitted is None:
                break
            normal, offset = fitted
        residual = np.abs(points @ normal+offset)
        inliers = residual <= threshold
        count = int(inliers.sum())
        fraction = float(quality[inliers].sum()/max(quality.sum(), 1e-8))
        if (count < minimum_inliers or fraction < float(options.get('min_inlier_fraction', .25)) or
                normal @ up < np.cos(np.deg2rad(angle)) or not min_height <= offset <= max_height):
            continue
        retained, retained_pixels = points[inliers], pixels[inliers]
        xspan = float(np.ptp(np.quantile(retained_pixels[:, 0], [.05, .95]))/w)
        yspan = float(np.ptp(np.quantile(retained_pixels[:, 1], [.05, .95]))/h)
        depth_span = float(np.ptp(np.quantile(retained[:, 2], [.1, .9])))
        bins = np.asarray(options.get('depth_bin_edges_m', [min_depth, 5., 10., 20., max_depth]), float)
        bins = np.unique(np.clip(bins, min_depth, max_depth))
        occupied = np.histogram(retained[:, 2], bins=bins)[0]
        occupied_bins = int(np.count_nonzero(occupied >= max(5, int(.02*count))))
        if (xspan < float(options.get('min_horizontal_span_fraction', .3)) or
                yspan < float(options.get('min_vertical_span_fraction', .08)) or
                depth_span < float(options.get('min_depth_span_m', 5.)) or
                occupied_bins < int(options.get('min_depth_bins', 3))):
            rejection = 'insufficient_spatial_or_depth_coverage'
            continue
        rmse = float(np.sqrt(np.average(residual[inliers]**2, weights=quality[inliers])))
        if rmse > float(options.get('max_fit_rmse_m', .16)):
            rejection = 'ground_residual_too_large'
            continue
        residual_quality = np.exp(-.5*(rmse/float(np.mean(threshold[inliers])))**2)
        coverage_quality = (min(1., xspan/.5)*min(1., yspan/.15)*min(1., depth_span/10.))
        confidence_value = float(np.clip(min(1., fraction/.5)*residual_quality*
                                         coverage_quality*np.mean(quality[inliers]), 0, 1))
        if confidence_value < float(options.get('min_plane_confidence', .4)):
            rejection = 'low_ground_plane_confidence'
            continue
        final_score = float(np.sum(quality[inliers]*np.maximum(
            0., 1-(residual[inliers]/threshold[inliers])**2))*coverage_quality)
        if final_score > best_score:
            best_score = final_score
            best = dict(valid=True, reason='reliable_lower_image_ground', normal=normal.tolist(),
                        offset=float(offset), camera_height=float(offset), confidence=confidence_value,
                        inliers=count, weighted_inlier_fraction=fraction, fit_rmse_m=rmse,
                        horizontal_span_fraction=xspan, vertical_span_fraction=yspan,
                        depth_span_m=depth_span, occupied_depth_bins=occupied_bins,
                        tilt_degrees=float(np.degrees(np.arccos(np.clip(normal @ up, -1, 1)))),
                        **common)
    return best if best is not None else _invalid(rejection, **common)


def height_penalty(xyz, plane, options=None):
    """Return bounded smooth outside-band cost and per-point reliability.

    The feasible band is ground_min_m=-0.2 through camera_height+above_camera_m
    (default +2m), or explicit max_height_m. Cost is exactly zero inside it.
    Outside distance is normalized by soft_margin_m=0.3 (also accepts
    soft_boundary_sigma_m), then passed
    through a smooth Huber-like function bounded by penalty_cap=3. Reliability
    combines fitted-plane confidence and depth uncertainty. The caller applies
    the object factor (1-p_background); background is never penalized here.
    """
    options = {} if options is None else dict(options)
    xyz = np.asarray(xyz, float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Height constraints require xyz[N,3]')
    penalty, reliability = np.zeros(len(xyz)), np.zeros(len(xyz))
    if not plane.get('valid', False):
        return penalty, reliability
    normal = np.asarray(plane.get('normal'), float)
    offset = float(plane.get('offset', plane.get('camera_height', 0.)))
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset):
        return penalty, reliability
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return penalty, reliability
    normal, offset = normal/norm, offset/norm
    camera_height = float(plane.get('camera_height', offset))
    lower = float(options.get('ground_min_m', options.get('ground_min', -.2)))
    upper = float(options.get('max_height_m', camera_height+float(options.get('above_camera_m', 2.))))
    sigma = float(options.get('soft_margin_m', options.get(
        'soft_boundary_sigma_m', options.get('sigma_m', .3))))
    cap = float(options.get('penalty_cap', 3.))
    if not np.isfinite([lower, upper, sigma, cap]).all() or upper <= lower or sigma <= 0 or cap <= 0:
        raise ValueError('Height band, soft boundary and penalty cap must be finite and ordered')
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    ids = np.flatnonzero(valid)
    height = xyz[ids] @ normal+offset
    outside = np.maximum(lower-height, 0)+np.maximum(height-upper, 0)
    normalized = np.minimum(outside, sigma*1e6)/sigma
    # hypot avoids overflow on extreme but finite bad-depth observations.
    smooth = np.hypot(1., normalized)-1.
    penalty[ids] = cap*(smooth/(cap+smooth))
    max_depth = float(options.get('max_constraint_depth_m', 80.))
    reliable_ids = ids[xyz[ids, 2] <= max_depth]
    depth_base = float(options.get('point_depth_sigma_m', .05))
    depth_relative = float(options.get('point_depth_sigma_relative', .005))
    depth_noise = depth_base+depth_relative*xyz[reliable_ids, 2]
    plane_noise = float(plane.get('fit_rmse_m', 0.))
    confidence = float(np.clip(plane.get('confidence', 0.), 0, 1))
    reliability[reliable_ids] = confidence/(1+(depth_noise**2+plane_noise**2)/sigma**2)
    return penalty, reliability

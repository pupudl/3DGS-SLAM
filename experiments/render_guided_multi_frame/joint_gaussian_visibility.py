"""A joint nearest-effective-splat Z buffer for object visualization only.

The native rasterizer does not expose a per-pixel Gaussian ID. This pass uses
its camera projection, projected covariance (+0.3 pixel low-pass variance),
near clipping and Gaussian alpha falloff, without rebuilding the extension.
It chooses the smallest camera-centre Z among splats with alpha >= threshold.
This is a hard visibility convention, not a differentiable training loss.
"""
from __future__ import annotations

import math
import numpy as np


def projected_footprints(xyz, covariance, K, image_shape, opacity=.9, threshold=.05):
    """Return visible centres, conics and inclusive pixel bounds (CPU only)."""
    xyz = np.asarray(xyz, np.float32)
    covariance = np.asarray(covariance, np.float32)
    K = np.asarray(K, np.float32)
    h, w = map(int, image_shape[:2])
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or covariance.shape != (len(xyz), 3, 3):
        raise ValueError('Expected XYZ [N,3] and covariance [N,3,3]')
    if K.shape != (3, 3) or h <= 0 or w <= 0 or not np.isfinite(K).all():
        raise ValueError('Expected finite intrinsics and positive image dimensions')
    if not (0 < threshold <= 1 and 0 <= opacity <= 1):
        raise ValueError('Invalid Gaussian opacity or visibility alpha threshold')
    valid = np.isfinite(xyz).all(1) & np.isfinite(covariance).all((1, 2)) & (xyz[:, 2] > .2)
    indices = np.flatnonzero(valid)
    empty = dict(indices=np.empty(0, np.int64), centres=np.empty((0, 2), np.float32),
                 conics=np.empty((0, 3), np.float32), bounds=np.empty((0, 4), np.int64))
    effective = max(float(threshold), 1 / 255.)
    if not len(indices) or opacity < effective or min(opacity, .99) < effective:
        return empty
    p, cov = xyz[indices], covariance[indices]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = p[:, 2]
    # setup_camera + ndc2Pix puts an integer pixel centre at K projection - .5.
    centres = p[:, :2] / (z[:, None] + 1e-7) * [fx, fy] + [cx - .5, cy - .5]
    limited = np.clip(p[:, :2] / z[:, None],
                      [-1.3 * w / (2 * fx), -1.3 * h / (2 * fy)],
                      [1.3 * w / (2 * fx), 1.3 * h / (2 * fy)])
    jac = np.zeros((len(p), 2, 3), np.float32)
    jac[:, 0, 0], jac[:, 1, 1] = fx / z, fy / z
    jac[:, 0, 2], jac[:, 1, 2] = -fx * limited[:, 0] / z, -fy * limited[:, 1] / z
    screen = jac @ cov @ np.transpose(jac, (0, 2, 1))
    a, b, c = screen[:, 0, 0] + .3, screen[:, 0, 1], screen[:, 1, 1] + .3
    det = a * c - b * b
    good = np.isfinite(det) & (det > 0) & (a > 0) & (c > 0)
    indices, centres, a, b, c, det = (value[good] for value in (indices, centres, a, b, c, det))
    if not len(indices):
        return empty
    conics = np.stack((c / det, -b / det, a / det), 1)
    q = -2 * math.log(effective / opacity)
    extent = np.sqrt(np.maximum(0, q * np.stack((a, c), 1)))
    lower = np.ceil(centres - extent).astype(np.int64)
    upper = np.floor(centres + extent).astype(np.int64)
    # Match the native preprocess tile rectangle as well as its alpha ellipse.
    mid = .5 * (a + c)
    radius = np.ceil(3 * np.sqrt(mid + np.sqrt(np.maximum(.1, mid * mid - det))))
    tile_lo = np.trunc((centres - radius[:, None]) / 16).astype(np.int64)
    tile_hi = np.trunc((centres + radius[:, None] + 15) / 16).astype(np.int64)
    lower = np.maximum(lower, np.maximum(tile_lo * 16, [0, 0]))
    upper = np.minimum(upper, np.minimum(tile_hi * 16 - 1, [w - 1, h - 1]))
    good = np.all(lower <= upper, 1)
    return dict(indices=indices[good], centres=centres[good].astype(np.float32),
                conics=conics[good].astype(np.float32),
                bounds=np.concatenate((lower[good], upper[good]), 1))


def nearest_gaussian_owners(xyz, covariance, labels, gaussian_ids, K, image_shape,
                            opacity=.9, threshold=.05, device='cuda',
                            max_candidate_pixels=1_000_000):
    """Choose one owner per pixel; -1 is unknown occluder, -2 is uncovered.

    Candidate expansion is bounded by max_candidate_pixels even for a single
    huge projected Gaussian. Equal Z is resolved by Gaussian ID, then original
    input order. CPU execution is supported for small deterministic tests.
    """
    import torch
    xyz = np.asarray(xyz, np.float32)
    labels = np.asarray(labels, np.int64)
    gids = np.asarray(gaussian_ids, np.int64)
    if labels.shape != (len(xyz),) or gids.shape != (len(xyz),):
        raise ValueError('Labels and Gaussian IDs must match XYZ')
    if np.any(labels < -1) or max_candidate_pixels <= 0:
        raise ValueError('Use -1 for unknown labels and a positive candidate budget')
    h, w = map(int, image_shape[:2])
    projected = projected_footprints(xyz, covariance, K, image_shape, opacity, threshold)
    selected = projected['indices']
    order = np.lexsort((selected, gids[selected], xyz[selected, 2]))
    selected = selected[order]
    projected = {key: value[order] for key, value in projected.items()}
    bounds = projected['bounds']
    widths = bounds[:, 2] - bounds[:, 0] + 1
    counts = widths * (bounds[:, 3] - bounds[:, 1] + 1)
    ends = np.cumsum(counts, dtype=np.int64)
    total = int(ends[-1]) if len(ends) else 0
    with torch.inference_mode():
        winner = torch.full((h * w,), len(selected), dtype=torch.int64, device=device)
        tensor = lambda value, dtype=None: torch.as_tensor(value, dtype=dtype, device=device)
        all_ends = tensor(ends)
        starts = tensor(np.r_[0, ends[:-1]])
        widths_t, bounds_t = tensor(widths), tensor(bounds)
        centres, conics = tensor(projected['centres']), tensor(projected['conics'])
        effective = max(float(threshold), 1 / 255.)
        for first in range(0, total, int(max_candidate_pixels)):
            linear = torch.arange(first, min(total, first + max_candidate_pixels), device=device)
            rank = torch.searchsorted(all_ends, linear, right=True)
            local = linear - starts[rank]
            x = bounds_t[rank, 0] + local % widths_t[rank]
            y = bounds_t[rank, 1] + torch.div(local, widths_t[rank], rounding_mode='floor')
            dx, dy = centres[rank, 0] - x, centres[rank, 1] - y
            power = -.5 * (conics[rank, 0] * dx * dx + conics[rank, 2] * dy * dy)
            power -= conics[rank, 1] * dx * dy
            alpha = (float(opacity) * torch.exp(power)).clamp(max=.99)
            keep = (power <= 0) & (alpha >= effective)
            winner.scatter_reduce_(0, (y[keep] * w + x[keep]).long(), rank[keep],
                                   reduce='amin', include_self=True)
        rank_image = winner.reshape(h, w).cpu().numpy()
    covered = rank_image < len(selected)
    owner = np.full((h, w), -2, np.int32)
    depth = np.full((h, w), np.inf, np.float32)
    winner_gid = np.full((h, w), -1, np.int64)
    if covered.any():
        indices = selected[rank_image[covered]]
        owner[covered], depth[covered], winner_gid[covered] = labels[indices], xyz[indices, 2], gids[indices]
    return dict(owner=owner, depth=depth, gaussian_ids=winner_gid,
                metadata=dict(method='nearest_effective_gaussian_camera_z',
                              alpha_threshold=max(float(threshold), 1 / 255.),
                              depth_order='gaussian_centre_camera_z',
                              tie_break='gaussian_id_then_scene_input_index',
                              input_gaussians=len(xyz), projected_gaussians=len(selected),
                              candidate_pixels=total, covered_pixels=int(covered.sum()),
                              unknown_occluder_pixels=int((owner == -1).sum()),
                              uncovered_pixels=int((owner == -2).sum()), overlap_pixels=0))

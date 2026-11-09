"""Shared image, geometry and visualization helpers for the two-frame experiment."""
from __future__ import annotations

import cv2
import importlib.util
import numpy as np
import sys
import torch
from pathlib import Path
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
_stereo_path = ROOT / "experiments" / "stereo_rigid_flow_mask" / "run_experiment.py"
_stereo_spec = importlib.util.spec_from_file_location("stereo_rigid_flow_io", _stereo_path)
if _stereo_spec is None or _stereo_spec.loader is None:
    raise ImportError(f"Cannot load stereo helper: {_stereo_path}")
stereo_io = importlib.util.module_from_spec(_stereo_spec)
_stereo_spec.loader.exec_module(stereo_io)

def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_depth(seq_root, frame, image, right, K, baseline, cfg):
    if cfg["source"] == "precomputed_igev":
        path = seq_root / cfg.get("subdir", "depth_sceneflow") / f"{frame:010d}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        depth = np.load(path).astype(np.float32)
    elif cfg["source"] == "sgbm":
        if right is None:
            raise FileNotFoundError(f"Missing right image for frame {frame}")
        _, depth = stereo_io.sgbm_depth(image, right, K, baseline, cfg.get("sgbm", {}))
    else:
        raise ValueError("depth.source must be precomputed_igev or sgbm")
    if depth.shape != image.shape[:2]:
        raise ValueError(f"Depth shape {depth.shape} differs from image {image.shape[:2]}")
    valid = np.isfinite(depth) & (depth >= cfg.get("min_m", 1)) & (depth <= cfg.get("max_m", 80))
    return np.where(valid, depth, np.nan)


def image_tensor(image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)[None] / 255.0


def masked_features(features, mask, limit):
    """Select target keypoints before matching/top-k without masking image content."""
    points = features['keypoints'][0]
    xy = points.round().long()
    h, w = mask.shape
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = torch.where(inside)[0]
    allowed = torch.as_tensor(mask, device=points.device, dtype=torch.bool)
    ids = ids[allowed[xy[ids, 1], xy[ids, 0]]]
    if limit >= 0 and len(ids) > limit:
        order = features['keypoint_scores'][0, ids].topk(limit).indices
        ids = ids[order]
    return {key: value[:, ids] for key, value in features.items()}


def match_images(image0, image1, extractor, matcher, device, origin0=(0, 0), origin1=(0, 0),
                 target_mask=None):
    with torch.inference_mode():
        im0, im1 = image_tensor(image0, device), image_tensor(image1, device)
        f0, _ = extractor({"image": im0})
        if target_mask is None:
            f1, _ = extractor({"image": im1})
        else:
            # The repository's SuperPoint has no mask input. Extract all target
            # detections, restrict their centers, then spend top-k inside the mask.
            limit = extractor.config['max_num_keypoints']
            extractor.config['max_num_keypoints'] = -1
            try:
                f1, _ = extractor({"image": im1})
            finally:
                extractor.config['max_num_keypoints'] = limit
            f1 = masked_features(f1, target_mask, limit)
        if min(f0["keypoints"].shape[1], f1["keypoints"].shape[1]) == 0:
            return np.empty((0, 4), np.float32)
        data = {**{k + "0": v for k, v in f0.items()},
                **{k + "1": v for k, v in f1.items()}, "image0": im0, "image1": im1}
        indices = matcher(data)["matches0"][0]
        good = indices >= 0
        if not good.any():
            return np.empty((0, 4), np.float32)
        p0 = f0["keypoints"][0, good].cpu().numpy() + np.asarray(origin0)
        p1 = f1["keypoints"][0, indices[good]].cpu().numpy() + np.asarray(origin1)
    return np.column_stack((p0, p1)).astype(np.float32)


def lift(matches, depth0, depth1, K):
    if not len(matches):
        return {"matches": np.empty((0, 4), np.float32),
                "x0": np.empty((0, 3), np.float32), "x1": np.empty((0, 3), np.float32)}
    h, w = depth0.shape
    xy0 = np.rint(matches[:, :2]).astype(int)
    xy1 = np.rint(matches[:, 2:]).astype(int)
    inside = ((xy0[:, 0] >= 0) & (xy0[:, 0] < w) & (xy0[:, 1] >= 0) & (xy0[:, 1] < h) &
              (xy1[:, 0] >= 0) & (xy1[:, 0] < w) & (xy1[:, 1] >= 0) & (xy1[:, 1] < h))
    xy0[:, 0] = np.clip(xy0[:, 0], 0, w-1)
    xy0[:, 1] = np.clip(xy0[:, 1], 0, h-1)
    xy1[:, 0] = np.clip(xy1[:, 0], 0, w-1)
    xy1[:, 1] = np.clip(xy1[:, 1], 0, h-1)
    z0 = depth0[xy0[:, 1], xy0[:, 0]]
    z1 = depth1[xy1[:, 1], xy1[:, 0]]
    valid = inside & np.isfinite(z0) & np.isfinite(z1)
    m, z0, z1 = matches[valid], z0[valid], z1[valid]

    def backproject(p, z):
        return np.column_stack(((p[:, 0] - K[0, 2]) * z / K[0, 0],
                                (p[:, 1] - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
    return {"matches": m, "x0": backproject(m[:, :2], z0), "x1": backproject(m[:, 2:], z1)}


def rigid_fit(a, b, weights=None):
    a, b = a.astype(np.float64), b.astype(np.float64)
    if weights is None:
        weights = np.ones(len(a), np.float64)
    weights = np.asarray(weights, np.float64)
    weights = weights / weights.sum()
    ac, bc = np.sum(a*weights[:, None], axis=0), np.sum(b*weights[:, None], axis=0)
    u, _, vt = np.linalg.svd(((a-ac)*weights[:, None]).T @ (b-bc))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    R = vt.T @ correction @ u.T
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = R, bc - R @ ac
    return H


def errors(a, b, H):
    return np.linalg.norm(a @ H[:3, :3].T + H[:3, 3] - b, axis=1)


def discover(anchors, cfg, seed):
    a, b = anchors["x0"], anchors["x1"]
    labels = np.full(len(a), -1, np.int32)
    remaining = np.arange(len(a))
    models, supports = [], []
    rng = np.random.default_rng(seed)
    # An independent stream preserves the original minimal-sample proposals
    # when optional larger-sample refinement is enabled.
    refinement_rng = np.random.default_rng(seed + 104729)
    threshold = float(cfg.get("inlier_threshold_m", 0.5))
    depth_scale = float(cfg.get('adaptive_depth_threshold_m', 0))
    if depth_scale > 0:
        depth = np.maximum(a[:, 2], b[:, 2])
        thresholds = threshold*(1+(depth/depth_scale)**2)
    else:
        thresholds = np.full(len(a), threshold)
    use_quality = depth_scale > 0 or bool(cfg.get('quality_weighted_ransac', False))
    if use_quality:
        quality = np.asarray(anchors.get('confidence', np.ones(len(a))), np.float64)
        quality = np.maximum(quality, 0.01)
    else:
        quality = np.ones(len(a), np.float64)
    use_fit_quality = use_quality or bool(cfg.get('quality_weighted_rigid_fit', False))
    fit_quality = (np.maximum(np.asarray(
        anchors.get('confidence', np.ones(len(a))), np.float64), 0.01)
                   if use_fit_quality else None)
    min_support = int(cfg.get("min_support", 12))
    # Keep the minimal three-point draw to find small motion components.  An
    # optional larger sample is fitted only after that seed has gathered a
    # geometrically consistent provisional consensus.
    seed_size = int(cfg.get("ransac_seed_size", 3))
    sample_size = int(cfg.get("ransac_sample_size", seed_size))
    if not 3 <= seed_size <= sample_size <= min_support:
        raise ValueError("Require 3 <= ransac_seed_size <= ransac_sample_size <= min_support")
    for _ in range(int(cfg.get("max_components", 3))):
        if len(remaining) < min_support:
            break
        best = np.empty(0, np.int32)
        best_score = -1.0
        best_median_error = float('inf')
        for _ in range(int(cfg.get("ransac_iterations", 350))):
            ids = (rng.choice(remaining, seed_size, replace=False,
                              p=quality[remaining]/quality[remaining].sum())
                   if use_quality else rng.choice(remaining, seed_size, replace=False))
            if np.linalg.matrix_rank(a[ids] - a[ids].mean(axis=0)) < 2:
                continue
            H = rigid_fit(a[ids], b[ids])
            provisional_error = errors(a[remaining], b[remaining], H)
            provisional = remaining[provisional_error < thresholds[remaining]]
            score = float(quality[provisional].sum())
            median_error = (float(np.median(provisional_error[
                provisional_error < thresholds[remaining]])) if len(provisional) else float('inf'))
            if score > best_score or (sample_size > seed_size and
                                      score == best_score and median_error < best_median_error):
                best = provisional
                best_score = score
                best_median_error = median_error
            if sample_size > seed_size:
                if len(provisional) < sample_size:
                    continue
                extra_pool = provisional[~np.isin(provisional, ids)]
                needed = sample_size - seed_size
                if len(extra_pool) < needed:
                    continue
                extra = (refinement_rng.choice(extra_pool, needed, replace=False,
                                               p=quality[extra_pool]/quality[extra_pool].sum())
                         if use_quality else refinement_rng.choice(extra_pool, needed, replace=False))
                fit_ids = np.concatenate((ids, extra))
                H = rigid_fit(a[fit_ids], b[fit_ids],
                              fit_quality[fit_ids] if use_fit_quality else None)
                refined_error = errors(a[remaining], b[remaining], H)
                refined_mask = refined_error < thresholds[remaining]
                inliers = remaining[refined_mask]
                score = float(quality[inliers].sum())
                median_error = (float(np.median(refined_error[refined_mask]))
                                if len(inliers) else float('inf'))
                if score > best_score or (score == best_score and median_error < best_median_error):
                    best = inliers
                    best_score = score
                    best_median_error = median_error
        if len(best) < min_support:
            break
        H = rigid_fit(a[best], b[best], fit_quality[best] if use_fit_quality else None)
        best = remaining[errors(a[remaining], b[remaining], H) < thresholds[remaining]]
        if len(best) < min_support:
            break
        H = rigid_fit(a[best], b[best], fit_quality[best] if use_fit_quality else None)
        labels[best] = len(models)
        models.append(H)
        supports.append(len(best))
        remaining = remaining[labels[remaining] < 0]
    if not models:
        raise RuntimeError(f"No rigid model from {len(a)} 3D anchors; check depth and thresholds")
    return models, labels, supports



def unexplained_clusters(anchors, cfg):
    """Group new anchors by target-image proximity, 3D proximity and motion similarity."""
    count = len(anchors["x0"])
    if not count:
        return []
    uv = anchors["matches"][:, 2:]
    xyz = anchors["x0"]
    displacement = anchors["x1"]-xyz
    radius_px = float(cfg.get("cluster_radius_px", 96))
    radius_m = float(cfg.get("cluster_radius_m", 3))
    motion_radius_m = float(cfg.get("cluster_motion_radius_m", 1.5))
    parent = np.arange(count)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(uv).query_pairs(radius_px):
        if (np.linalg.norm(xyz[i]-xyz[j]) <= radius_m and
                np.linalg.norm(displacement[i]-displacement[j]) <= motion_radius_m):
            parent[find(i)] = find(j)
    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    return sorted((np.asarray(ids, dtype=np.int32) for ids in groups.values()),
                  key=len, reverse=True)


def make_gaussians(image, depth, K, stride, max_depth_jump,
                   projected_sigma_px=1.6, initialization="surface_aligned"):
    """Initialize frozen first-frame Gaussians from RGB and stereo depth.

    The default projective mode matches lsgslam_windows. The optional surface
    mode estimates tangent directions and a thin normal axis from local depth.
    """
    if stride < 1:
        raise ValueError("Gaussian sampling stride must be positive")
    if initialization not in ("surface_aligned", "lsgslam_projective"):
        raise ValueError(f"Unknown Gaussian initialization: {initialization}")
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]

    if initialization == "lsgslam_projective":
        # Match lsgslam_windows: one Gaussian per valid depth pixel, RGB from
        # that pixel, isotropic scale z / mean(fx, fy), identity orientation.
        # The resulting map is still frozen during pairwise motion discovery.
        valid = np.isfinite(z) & (z > 0)
        centers = np.stack(((xx-K[0, 2])*z/K[0, 0],
                            (yy-K[1, 2])*z/K[1, 1], z), axis=-1)
        xyz = np.ascontiguousarray(centers[valid].astype(np.float32))
        radius = (xyz[:, 2]/((K[0, 0]+K[1, 1])*0.5)).astype(np.float32)
        scale = np.repeat(radius[:, None], 3, axis=1)
        basis = np.broadcast_to(np.eye(3, dtype=np.float32),
                                (len(xyz), 3, 3)).copy()
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[::stride, ::stride]
        return {"initialization": initialization, "xyz": xyz,
                "uv": np.column_stack((xx[valid], yy[valid])).astype(np.float32),
                "color": np.ascontiguousarray(color[valid].astype(np.float32)/255),
                "scale": scale, "basis": basis}

    left_x, right_x = np.maximum(xx-1, 0), np.minimum(xx+1, w-1)
    up_y, down_y = np.maximum(yy-1, 0), np.minimum(yy+1, h-1)
    zl, zr = depth[yy, left_x], depth[yy, right_x]
    zu, zd = depth[up_y, xx], depth[down_y, xx]
    neighbors = np.stack((zl, zr, zu, zd), axis=-1)
    valid = (np.isfinite(z) & np.isfinite(neighbors).all(axis=-1) &
             (xx > 0) & (xx < w-1) & (yy > 0) & (yy < h-1))
    valid &= np.max(np.abs(neighbors-np.where(np.isfinite(z), z, 0)[..., None]), axis=-1) < max_depth_jump

    def backproject(u, v, distance):
        return np.stack(((u-K[0, 2])*distance/K[0, 0],
                         (v-K[1, 2])*distance/K[1, 1], distance), axis=-1)

    centers = backproject(xx, yy, z)
    dx = backproject(right_x, yy, zr)-backproject(left_x, yy, zl)
    dy = backproject(xx, down_y, zd)-backproject(xx, up_y, zu)
    dx_norm = np.linalg.norm(dx, axis=-1)
    dy_norm = np.linalg.norm(dy, axis=-1)
    t1 = dx/np.maximum(dx_norm[..., None], 1e-6)
    normal = np.cross(t1, dy)
    normal_norm = np.linalg.norm(normal, axis=-1)
    normal /= np.maximum(normal_norm[..., None], 1e-6)
    t2 = np.cross(normal, t1)
    valid &= (dx_norm > 1e-6) & (dy_norm > 1e-6) & (normal_norm > 1e-6)
    # dx/dy span two image pixels. Set the surface footprint in image pixels
    # independently of the sampling stride, so splats preserve RGB detail.
    if projected_sigma_px <= 0:
        raise ValueError("projected_sigma_px must be positive")
    tangent_x = 0.5*dx_norm*projected_sigma_px
    tangent_y = 0.5*np.abs(np.sum(dy*t2, axis=-1))*projected_sigma_px
    depth_spread = np.median(np.abs(neighbors-z[..., None]), axis=-1)
    normal_scale = np.minimum(np.maximum(depth_spread, 0.01),
                              0.3*np.minimum(tangent_x, tangent_y))
    scales = np.stack((tangent_x, tangent_y, normal_scale), axis=-1)
    valid &= np.isfinite(centers).all(axis=-1) & np.isfinite(scales).all(axis=-1)
    valid &= np.min(scales, axis=-1) > 0
    basis = np.stack((t1, t2, normal), axis=-1)
    color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[::stride, ::stride]
    return {"initialization": initialization,
            "xyz": np.ascontiguousarray(centers[valid].astype(np.float32)),
            "uv": np.column_stack((xx[valid], yy[valid])).astype(np.float32),
            "color": np.ascontiguousarray(color[valid].astype(np.float32)/255),
            "scale": np.ascontiguousarray(scales[valid].astype(np.float32)),
            "basis": np.ascontiguousarray(basis[valid].astype(np.float32))}


def bind_anchor_gaussians(base_gmap, anchors, source_bgr, K):
    """Reserve one existing source-pixel Gaussian for every stereo landmark.

    A landmark replaces its nearest unused ordinary splat, so it is rendered
    exactly once. Rebuilding from the same base map and ordered anchors keeps
    Gaussian indices stable when feedback supplies additional matches.
    """
    if not len(anchors['x0']):
        raise ValueError('Cannot bind an empty landmark set')
    if base_gmap['initialization'] != 'lsgslam_projective':
        raise ValueError('Anchor Gaussians currently require lsgslam_projective initialization')
    uv = np.asarray(base_gmap['uv'])
    matches = np.asarray(anchors['matches'])
    k = min(32, len(uv))
    distance, nearest = cKDTree(uv).query(matches[:, :2], k=k, workers=-1)
    distance = np.asarray(distance).reshape(len(matches), k)
    nearest = np.asarray(nearest).reshape(len(matches), k)
    used = np.zeros(len(uv), dtype=bool)
    selected = np.empty(len(matches), np.int64)
    for j in range(len(matches)):
        options = nearest[j][(distance[j] <= 2.5) & ~used[nearest[j]]]
        if not len(options):
            raise ValueError(f'No unique source-pixel Gaussian within 2.5 px of landmark {j}')
        selected[j] = options[0]
        used[selected[j]] = True
    out = dict(base_gmap)
    for name in ('xyz', 'uv', 'color', 'scale'):
        out[name] = base_gmap[name].copy()
    out['xyz'][selected] = anchors['x0']
    out['uv'][selected] = matches[:, :2]
    rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    map_x = matches[:, 0].astype(np.float32)[None]
    map_y = matches[:, 1].astype(np.float32)[None]
    out['color'][selected] = cv2.remap(rgb, map_x, map_y,
                                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0]
    radius = anchors['x0'][:, 2]/float((K[0, 0]+K[1, 1])*0.5)
    out['scale'][selected] = radius[:, None]
    out['anchor_gaussian_indices'] = selected
    bound = {**anchors, 'gaussian_index': selected}
    return out, bound


def residual(rendering, target, depth, cfg, confidence=None):
    observed = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    observed_depth = np.isfinite(depth)
    covered = (rendering["sil"] >= float(cfg.get("min_silhouette", 0.65))) & observed_depth
    color = np.abs(observed-rendering["rgb"]).mean(axis=2)
    depth_error = np.abs(depth-rendering["depth"])
    if confidence is None:
        raise ValueError("Stereo-derived depth confidence is required")
    depth_score = confidence*np.clip(depth_error/float(cfg.get("depth_clip_m", 3)), 0, 1)
    score = (float(cfg.get("color_weight", 0.5))*color +
             float(cfg.get("depth_weight", 0.5))*depth_score)
    score[~observed_depth] = 0
    metrics = {"coverage_pixels": int(covered.sum()),
               "color_l1": float(color[covered].mean()) if covered.any() else None,
               "depth_abs_m": float(depth_error[covered].mean()) if covered.any() else None,
               "score": float(score[covered].mean()) if covered.any() else None}
    return score, covered, metrics


def select_regions(score, valid_observation, cfg, gradient=None, motion_gradient=None):
    """HTML §10.3: mean pixel residual plus mean projected Gaussian gradient."""
    h, w = score.shape
    size = int(cfg.get("patch_size_px", 96))
    border = int(cfg.get("border_margin_px", 48))
    eta = float(cfg.get("gradient_weight", 0.25))
    eta_motion = float(cfg.get("motion_gradient_weight", 0.0))
    empty = np.zeros_like(score)
    own_sum, own_count = gradient if gradient is not None else (empty, empty)
    motion_sum, motion_count = (
        motion_gradient if motion_gradient is not None else (empty, empty))
    if cfg.get('normalize_gradient', False):
        quantile = float(cfg.get('gradient_scale_quantile', 0.95))
        if not 0 < quantile <= 1:
            raise ValueError('gradient_scale_quantile must be in (0,1]')
        def scale(values, counts):
            present = counts > 0
            return max(float(np.quantile(values[present]/counts[present], quantile)), 1e-12) if present.any() else 1.0
        own_scale = scale(own_sum, own_count)
        motion_scale = scale(motion_sum, motion_count)
    else:
        own_scale = motion_scale = 1.0
    patch_score = np.zeros_like(score)
    found = []
    for y in range(0, h, size):
        for x in range(0, w, size):
            x2, y2 = min(w, x + size), min(h, y + size)
            if x < border or y < border or x2 > w - border or y2 > h - border:
                continue
            observed = valid_observation[y:y2, x:x2]
            if observed.mean() < float(cfg.get("min_observation_fraction", 0.6)):
                continue
            residual_mean = float(score[y:y2, x:x2].sum()) / ((y2 - y) * (x2 - x))
            own_n = float(own_count[y:y2, x:x2].sum())
            own_mean = float(own_sum[y:y2, x:x2].sum()) / max(own_n, 1)
            motion_n = float(motion_count[y:y2, x:x2].sum())
            motion_mean = float(motion_sum[y:y2, x:x2].sum()) / max(motion_n, 1)
            value = residual_mean + eta * own_mean/own_scale + eta_motion * motion_mean/motion_scale
            patch_score[y:y2, x:x2] = value
            if value >= float(cfg.get("min_score", 0.12)):
                found.append((value, (x, y, x2, y2)))
    found.sort(reverse=True)
    return [box for _, box in found[:int(cfg.get("max_regions", 4))]], patch_score


def supplemental_matches(im0, im1, regions, extractor, matcher, margin, target_margin,
                         target_mask=None):
    h, w = im0.shape[:2]
    matches = []
    for x0, y0, x1, y1 in regions:
        tx0, ty0 = max(0, x0-target_margin), max(0, y0-target_margin)
        tx1, ty1 = min(w, x1+target_margin), min(h, y1+target_margin)
        sx0, sy0 = max(0, tx0-margin), max(0, ty0-margin)
        sx1, sy1 = min(w, tx1+margin), min(h, ty1+margin)
        mask = None if target_mask is None else target_mask[ty0:ty1, tx0:tx1]
        if mask is not None and not mask.any():
            continue
        m = match_images(im0[sy0:sy1, sx0:sx1], im1[ty0:ty1, tx0:tx1],
                         extractor, matcher, "cuda", (sx0, sy0), (tx0, ty0), mask)
        if len(m):
            matches.append(m)
    return np.concatenate(matches) if matches else np.empty((0, 4), np.float32)


def full_image_masked_matches(im0, im1, mask, extractor, matcher):
    """Match full frames with target centers restricted by the pixel mask."""
    if mask.shape != im1.shape[:2]:
        raise ValueError('Target mask must match the complete target image')
    if not mask.any():
        return np.empty((0, 4), np.float32)
    return match_images(im0, im1, extractor, matcher, 'cuda', target_mask=mask)


def add_unique(initial, added):
    """Deduplicate 4D correspondences against history and within this batch."""
    if not len(added):
        return initial.copy(), added
    radius = 3 * np.sqrt(2)
    if len(initial):
        distance, _ = cKDTree(initial).query(added)
        candidates = added[distance > radius]
    else:
        candidates = added
    if not len(candidates):
        return initial.copy(), candidates
    neighborhoods = cKDTree(candidates).query_ball_point(candidates, radius)
    consumed = np.zeros(len(candidates), dtype=bool)
    selected = []
    for i, neighbors in enumerate(neighborhoods):
        if consumed[i]:
            continue
        selected.append(i)
        consumed[neighbors] = True
    novel = candidates[np.asarray(selected, dtype=np.int64)]
    return np.concatenate((initial, novel)), novel


def pose_error(H, gt):
    relative = H[:3, :3] @ gt[:3, :3].T
    angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
    return {"rotation_deg": float(np.degrees(angle)),
            "translation_m": float(np.linalg.norm(H[:3, 3]-gt[:3, 3]))}


def choose_reference(anchors, models, image_shape, sigma_m):
    """Pairwise part of HTML §12: hard geometric support, coverage and 3D extent.

    Soft probabilities may be broad before convergence. World-reference choice
    therefore uses thresholded geometric support and requires the support term
    to dominate pairwise area/extent, which lack temporal persistence.
    """
    h, w = image_shape
    residuals = np.stack([errors(anchors["x0"], anchors["x1"], H)
                          for H in models], axis=1)
    best = residuals.argmin(axis=1)
    threshold = sigma_m
    count, area, extent = [], [], []
    for k in range(len(models)):
        selected = (best == k) & (residuals[:, k] < threshold)
        count.append(float(selected.sum()))
        if selected.sum() >= 3:
            pixels = anchors["matches"][selected, :2]
            xyz = anchors["x0"][selected]
            area.append(float(np.prod(pixels.max(axis=0)-pixels.min(axis=0)))/(w*h))
            extent.append(min(float(np.linalg.norm(xyz.max(axis=0)-xyz.min(axis=0)))/30, 1))
        else:
            area.append(0.)
            extent.append(0.)
    support_max = max(max(count), 1)
    area_max = max(max(area), 1e-8)
    extent_max = max(max(extent), 1e-8)
    scores = [0.75*(count[k]/support_max)+0.15*(area[k]/area_max)+
              0.10*(extent[k]/extent_max) for k in range(len(models))]
    return int(np.argmax(scores)), scores


def save_rgb(path, rendered):
    rgb = np.clip(rendered["rgb"]*255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_heat(path, values, valid, maximum=0.5):
    image = cv2.applyColorMap(np.clip(values/maximum*255, 0, 255).astype(np.uint8),
                              cv2.COLORMAP_TURBO)
    image[~valid] = 0
    cv2.imwrite(str(path), image)



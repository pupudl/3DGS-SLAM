#!/usr/bin/env python3
"""Two-frame portion of motion_rigid_4dgs.html, without persistent SLAM state.

The first-frame Gaussian geometry stays fixed. Every structural proposal is
optimized and scored on the same anchors and target image as its alternative.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from pair_utils import (
    ROOT, stereo_io, resolve, load_depth, match_images, lift, discover, errors,
    rigid_fit, unexplained_clusters, make_gaussians, bind_anchor_gaussians,
    residual, select_regions,
    supplemental_matches, full_image_masked_matches, add_unique, choose_reference, pose_error, save_rgb,
    save_heat,
)
from joint_pair import PairwiseModel, anchor_probabilities, gaussian_probabilities, gradient_projection
from sp_lg.lightglue import LightGlue
from sp_lg.superpoint import SuperPoint
from masked_feedback import (PendingMatches, balanced_indices, feedback_cycle,
                             global_problem_mask, problem_mask, region_gate, score_guided_mask,
                             spaced_indices)


def stereo_confidence(left: np.ndarray, right: np.ndarray, depth: np.ndarray,
                      K: np.ndarray, baseline: float, sigma_rgb: float) -> np.ndarray:
    """Depth confidence from left/right stereo photometric agreement.

    This is measured from the actual right image, not from local smoothness of
    a single depth map. It is a proxy when the stereo network has no native
    uncertainty output.
    """
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    disparity = K[0, 0] * baseline / np.maximum(np.nan_to_num(depth, nan=0), 1e-5)
    right_x = xx - disparity.astype(np.float32)
    warped = cv2.remap(right, right_x, yy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT).astype(np.float32) / 255
    difference = np.abs(left.astype(np.float32) / 255 - warped).mean(axis=2)
    confidence = np.exp(-difference / sigma_rgb)
    confidence[~finite | (right_x < 1) | (right_x >= w - 2)] = 0
    return confidence.astype(np.float32)


def subset_anchors(anchors: dict, indices: np.ndarray) -> dict:
    return {key: value[indices] for key, value in anchors.items()}


def motion_thresholds(anchors: dict, base: float, motion_cfg: dict) -> np.ndarray:
    scale = float(motion_cfg.get('adaptive_depth_threshold_m', 0))
    if scale <= 0:
        return np.full(len(anchors['x0']), base, np.float32)
    depth = np.maximum(anchors['x0'][:, 2], anchors['x1'][:, 2])
    return (base*(1+(depth/scale)**2)).astype(np.float32)


def with_anchor_confidence(anchors: dict, source_confidence: np.ndarray,
                           target_confidence: np.ndarray, cfg: dict) -> dict:
    """Attach measured stereo quality to each 3D observation."""
    # The map may use a broad depth range, while motion estimation should
    # exclude matches whose stereo depth is unreliable in either frame.
    features = cfg.get('features', {})
    min_depth = float(features.get('match_depth_min_m', cfg['depth'].get('min_m', 0)))
    max_depth = float(features.get('match_depth_max_m', cfg['depth'].get('max_m', float('inf'))))
    if not 0 <= min_depth < max_depth:
        raise ValueError('Require 0 <= match_depth_min_m < match_depth_max_m')
    keep = ((anchors['x0'][:, 2] >= min_depth) & (anchors['x0'][:, 2] <= max_depth) &
            (anchors['x1'][:, 2] >= min_depth) & (anchors['x1'][:, 2] <= max_depth))
    anchors = {key: value[keep] for key, value in anchors.items()}
    if not cfg['ownership'].get('stereo_weighted_anchors', False):
        return anchors
    uv = np.rint(anchors['matches']).astype(np.int32)
    h, w = source_confidence.shape
    uv[:, (0, 2)] = np.clip(uv[:, (0, 2)], 0, w-1)
    uv[:, (1, 3)] = np.clip(uv[:, (1, 3)], 0, h-1)
    c0 = source_confidence[uv[:, 1], uv[:, 0]]
    c1 = target_confidence[uv[:, 3], uv[:, 2]]
    # Stereo depth uncertainty grows quadratically with distance. Retain a
    # small floor so imperfect photometric agreement does not erase evidence.
    z = np.maximum(anchors['x0'][:, 2], anchors['x1'][:, 2])
    depth_weight = 1/(1+(z/float(cfg['ownership'].get('reliable_depth_m', 35)))**2)
    confidence = np.clip(np.sqrt(c0*c1)*depth_weight, 0.05, 1).astype(np.float32)
    near_reliable = features.get('match_depth_near_reliable_m')
    far_reliable = features.get('match_depth_far_reliable_m')
    if near_reliable is not None or far_reliable is not None:
        if near_reliable is None or far_reliable is None:
            raise ValueError('Set both near and far reliable sparse-match depths')
        near_reliable, far_reliable = float(near_reliable), float(far_reliable)
        if not min_depth < near_reliable <= far_reliable < max_depth:
            raise ValueError('Require min < near_reliable <= far_reliable < max')
        edge_weight = float(features.get('match_depth_edge_weight', 0.05))
        if not 0 <= edge_weight <= 1:
            raise ValueError('match_depth_edge_weight must be in [0, 1]')
        near = np.minimum(anchors['x0'][:, 2], anchors['x1'][:, 2])
        near_factor = np.clip((near-min_depth)/(near_reliable-min_depth), 0, 1)
        far_factor = np.clip((max_depth-z)/(max_depth-far_reliable), 0, 1)
        reliability = edge_weight+(1-edge_weight)*near_factor*far_factor
        confidence = np.clip(confidence*reliability, 0.01, 1).astype(np.float32)
    return {**anchors, 'confidence': confidence}


def refresh_ownership(state: dict, anchors: dict, gmap: dict, source: np.ndarray,
                      cfg: dict, new_anchors: dict) -> np.ndarray:
    """Feed newly matched landmarks into local Gaussian ownership."""
    old = state['q']
    if not len(new_anchors['x0']) or not cfg['ownership'].get('refresh_after_feedback', False):
        return old
    sigma = float(cfg['motion']['assignment_sigma_m'])
    evidence = new_anchors if cfg['ownership'].get('refresh_new_only', False) else anchors
    qa = anchor_probabilities(evidence, list(state['H']), sigma)
    if cfg['ownership'].get('refresh_new_only', False):
        # Ambiguous residuals must not erase a confident existing assignment.
        certainty = (np.ones(len(qa), np.float32) if len(state['H']) == 1 else
                     (qa.max(axis=1)-1/len(state['H']))/(1-1/len(state['H'])))
        evidence = {**evidence, 'confidence':
                    evidence.get('confidence', np.ones(len(qa), np.float32))*certainty}
    proposed, mass = gaussian_probabilities(
        gmap, evidence, qa, state['reference'], cfg['ownership'], source,
        return_mass=True)
    distance, _ = cKDTree(new_anchors['x0']).query(gmap['xyz'], workers=-1)
    radius = float(cfg['ownership'].get('refresh_radius_m', 2.0))
    mix = float(cfg['ownership'].get('refresh_fraction', 0.75))*np.exp(-(distance/radius)**2)
    if cfg['ownership'].get('refresh_new_only', False):
        minimum = float(cfg['ownership'].get('min_anchor_weight', 0.02))
        mix *= np.clip((mass-minimum)/max(minimum, 1e-6), 0, 1)
    q = old*(1-mix[:, None])+proposed*mix[:, None]
    floor = float(cfg['ownership'].get('feedback_probability_floor', 0.025))
    q = q*(1-len(state['H'])*floor)+floor
    return (q/q.sum(axis=1, keepdims=True)).astype(np.float32)


def candidate_ownership(state: dict, proposal: list[np.ndarray], kind: str,
                        gmap: dict, anchors: dict, source: np.ndarray, cfg: dict,
                        components: tuple[int, ...], local_points=None) -> np.ndarray:
    """Map current responsibilities into a structural candidate's columns."""
    q = state['q']
    k = len(q[0])
    if kind == 'birth':
        qa = anchor_probabilities(anchors, proposal,
                                  float(cfg['motion']['assignment_sigma_m']))
        fresh = gaussian_probabilities(gmap, anchors, qa, state['reference'],
                                       cfg['ownership'], source)
        radius = float(cfg['ownership'].get('birth_radius_m', 2.5))
        distance, _ = cKDTree(local_points).query(gmap['xyz'], workers=-1)
        new_mass = np.clip(fresh[:, -1]*np.exp(-(distance/radius)**2), 0, 0.8)
        result = np.column_stack((q*(1-new_mass[:, None]), new_mass))
    elif kind == 'merge':
        a, b = components
        keep = [j for j in range(k) if j not in (a, b)]
        result = np.column_stack((q[:, keep], q[:, a]+q[:, b]))
    elif kind == 'prune':
        (remove,) = components
        keep = [j for j in range(k) if j != remove]
        result = q[:, keep].copy()
        result[:, keep.index(state['reference']) if state['reference'] in keep else 0] += q[:, remove]
    elif kind == 'split':
        (old,) = components
        keep = [j for j in range(k) if j != old]
        # Geometric evidence divides only the old component's responsibility.
        x = gmap['xyz']
        prediction = np.stack([x @ H[:3, :3].T + H[:3, 3] for H in proposal[-2:]], axis=1)
        old_prediction = x @ state['H'][old, :3, :3].T + state['H'][old, :3, 3]
        e2 = np.sum((prediction-old_prediction[:, None])**2, axis=2)
        logits = -e2/max(float(cfg['motion']['assignment_sigma_m'])**2, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        split = np.exp(logits)
        split /= split.sum(axis=1, keepdims=True)
        result = np.column_stack((q[:, keep], q[:, old, None]*split))
    else:
        raise ValueError(kind)
    floor = float(cfg['ownership'].get('feedback_probability_floor', 0.025))
    result = result*(1-len(proposal)*floor)+floor
    return (result/result.sum(axis=1, keepdims=True)).astype(np.float32)


def support_labels(anchors: dict, models: list[np.ndarray], threshold: float,
                   motion_cfg: dict | None = None) -> np.ndarray:
    e = np.stack([errors(anchors['x0'], anchors['x1'], H) for H in models], axis=1)
    best = e.argmin(axis=1)
    limits = (motion_thresholds(anchors, threshold, motion_cfg)
              if motion_cfg is not None else threshold)
    best[e[np.arange(len(e)), best] >= limits] = -1
    return best


def motion_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rotation = a[:3, :3] @ b[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1)))
    return float(angle), float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def score_state(gmap, anchors, models, K, target, depth, confidence,
                source, cfg, iterations, initial_q=None, hybrid_initial=False):
    motion_cfg = cfg['motion']
    sigma = float(motion_cfg['assignment_sigma_m'])
    reference, reference_scores = choose_reference(anchors, models, depth.shape, sigma)
    q_anchor = anchor_probabilities(anchors, models, sigma)
    q_gaussian = (initial_q if initial_q is not None else
                  gaussian_probabilities(gmap, anchors, q_anchor, reference,
                                         cfg['ownership'], source))
    joint_cfg = {**cfg['joint'], 'assignment_sigma_m': sigma}
    if hybrid_initial:
        joint_cfg['hybrid_geometry'] = motion_cfg.get('hybrid', {})
    engine = PairwiseModel(gmap, anchors, models, q_gaussian, K, target, depth,
                           joint_cfg, target_confidence=confidence)
    optimized = engine.optimize(iterations, int(cfg['optimization']['downsample']),
                                float(cfg['optimization']['pose_lr']),
                                float(cfg['optimization']['ownership_lr']))
    rendering = engine.render_numpy(optimized['H'], optimized['q'])
    score, coverage, metrics = residual(rendering, target, depth,
                                         cfg['residual'], confidence)
    H = optimized['H'].cpu().numpy()
    reference, reference_scores = choose_reference(anchors, list(H), depth.shape, sigma)
    if hybrid_initial:
        # Hybrid discovery orders the spatially broad reference candidate first.
        reference = 0
    return dict(H=H, q=optimized['q'].cpu().numpy(), q_anchor=optimized['q_anchor'],
                gradient_own=optimized['grad_q'], gradient_motion=optimized['grad_motion'],
                loss=optimized['loss'], terms=optimized['terms'], history=optimized['history'],
                rendering=rendering, score=score, coverage=coverage, metrics=metrics,
                reference=reference, reference_scores=reference_scores)


def same_coverage_gain(before, after) -> tuple[float, int]:
    common = before['coverage'] & after['coverage']
    if not common.any():
        return -float('inf'), 0
    return float((before['score'][common] - after['score'][common]).mean()), int(common.sum())


def candidate_gate(base, candidate, cfg, kind, local_box=None, focus_mask=None):
    gain, pixels = same_coverage_gain(base, candidate)
    old_k, new_k = len(base['H']), len(candidate['H'])
    penalty = float(cfg.get('complexity_penalty', 0.005))
    objective_gain = base['loss'] + penalty * old_k - candidate['loss'] - penalty * new_k
    coverage_ratio = float(candidate['coverage'].sum()) / max(int(base['coverage'].sum()), 1)
    local_gain = None
    if local_box is not None:
        x0, y0, x1, y1 = local_box
        common = base['coverage'][y0:y1, x0:x1] & candidate['coverage'][y0:y1, x0:x1]
        local_gain = (float((base['score'][y0:y1, x0:x1][common] -
                             candidate['score'][y0:y1, x0:x1][common]).mean())
                      if common.any() else -float('inf'))
    accepted = bool(
        pixels and objective_gain > float(cfg.get(f'{kind}_min_objective_improvement', 0))
        and gain > float(cfg.get(f'{kind}_min_render_improvement', 0))
        and coverage_ratio >= float(cfg.get(f'{kind}_min_coverage_ratio', 0.95))
        and (local_gain is None or
             local_gain > float(cfg.get(f'{kind}_min_local_improvement', 0)))
    )
    mask_check = region_gate(base, candidate, focus_mask, cfg) if focus_mask is not None else None
    if mask_check is not None:
        accepted = accepted and mask_check['accepted']
    return dict(accepted=accepted, objective_gain=objective_gain,
                render_gain=gain, common_pixels=pixels,
                coverage_ratio=coverage_ratio, local_gain=local_gain,
                **({'mask_gate': mask_check} if mask_check is not None else {}))


def try_structural_updates(state, anchors, gmap, K, source, target, depth,
                           confidence, cfg, cycle, seed, focus_mask=None):
    """Pairwise birth, split, merge and support-based prune from HTML §11.

    Temporal persistence conditions in §11 are outside a two-frame kernel.
    """
    updates = []
    motion_cfg, dcfg = cfg['motion'], cfg['discovery']
    iters = int(cfg['optimization']['feedback_iterations'])
    threshold = float(motion_cfg['inlier_threshold_m'])
    max_components = int(dcfg.get('max_components_total', 6))

    def evaluate(models, initial_q=None):
        return score_state(gmap, anchors, models, K, target, depth, confidence,
                           source, cfg, iters, initial_q=initial_q)

    # Split a component only when two supported rigid fits reduce its own
    # sparse geometry loss and the complete penalized objective also improves.
    labels = support_labels(anchors, list(state['H']), threshold, motion_cfg)
    for k in range(len(state['H']) if dcfg.get('enable_split', True) else 0):
        if len(state['H']) >= max_components:
            break
        ids = np.flatnonzero(labels == k)
        minimum = int(dcfg.get('split_min_anchors', motion_cfg['min_support']))
        if len(ids) < 2 * minimum:
            continue
        local = subset_anchors(anchors, ids)
        proposal_cfg = {**motion_cfg, 'max_components': 2, 'min_support': minimum}
        try:
            fitted, _, support = discover(local, proposal_cfg, seed + cycle * 31 + k)
        except RuntimeError:
            continue
        if len(fitted) != 2 or min(support) < minimum:
            continue
        old_e = errors(local['x0'], local['x1'], state['H'][k])
        new_e = np.minimum(*(errors(local['x0'], local['x1'], H) for H in fitted))
        robust_old = np.sqrt(old_e * old_e + 0.05**2) - 0.05
        robust_new = np.sqrt(new_e * new_e + 0.05**2) - 0.05
        geometry_gain = float(robust_old.sum() - robust_new.sum())
        if geometry_gain <= float(dcfg.get('split_min_geometry_gain', 0.5)):
            continue
        proposal = [H for j, H in enumerate(state['H']) if j != k] + fitted
        try:
            warm = candidate_ownership(state, proposal, 'split', gmap, anchors,
                                       source, cfg, (k,))
            candidate = evaluate(proposal, warm)
            gate = candidate_gate(state, candidate, dcfg, 'split', focus_mask=focus_mask)
        except RuntimeError as exc:
            updates.append(dict(kind='split', component=k, error=str(exc)))
            continue
        record = dict(kind='split', component=k, support=support,
                      geometry_gain=geometry_gain, **gate)
        updates.append(record)
        if gate['accepted']:
            state = candidate
            break

    # A two-frame merge tests pairwise SE(3) proximity and objective gain.
    # Long-term trajectory agreement is intentionally a multi-frame condition.
    if dcfg.get('enable_merge', True) and len(state['H']) > 1:
        labels = support_labels(anchors, list(state['H']), threshold, motion_cfg)
        for a in range(len(state['H'])):
            for b in range(a + 1, len(state['H'])):
                angle, distance = motion_distance(state['H'][a], state['H'][b])
                if (angle > float(dcfg.get('merge_max_rotation_deg', 2)) or
                        distance > float(dcfg.get('merge_max_translation_m', 0.2))):
                    continue
                ids = np.flatnonzero((labels == a) | (labels == b))
                if len(ids) < 3:
                    continue
                merged = rigid_fit(anchors['x0'][ids], anchors['x1'][ids])
                proposal = [H for j, H in enumerate(state['H']) if j not in (a, b)] + [merged]
                try:
                    warm = candidate_ownership(state, proposal, 'merge', gmap, anchors,
                                               source, cfg, (a, b))
                    candidate = evaluate(proposal, warm)
                    gate = candidate_gate(state, candidate, dcfg, 'merge', focus_mask=focus_mask)
                except RuntimeError as exc:
                    updates.append(dict(kind='merge', components=[a, b], error=str(exc)))
                    continue
                updates.append(dict(kind='merge', components=[a, b],
                                    rotation_deg=angle, translation_m=distance, **gate))
                if gate['accepted']:
                    state = candidate
                    break
            else:
                continue
            break

    # Large, weakly supported false motions can own much of the image, so
    # optionally test their removal too. The full objective/render gate still
    # decides whether deletion is justified by the two observations.
    if dcfg.get('enable_prune', True) and len(state['H']) > 1:
        labels = support_labels(anchors, list(state['H']), threshold, motion_cfg)
        mass = state['q'].sum(axis=0)
        min_mass = float(dcfg.get('prune_min_gaussian_mass', 0.005)) * len(state['q'])
        test_all = bool(dcfg.get('prune_test_all_nonreference', False))
        for k in np.argsort(mass):
            if k == state['reference'] or (mass[k] >= min_mass and not test_all):
                continue
            if len(state['H']) <= 1:
                break
            proposal = [H for j, H in enumerate(state['H']) if j != k]
            try:
                warm = candidate_ownership(state, proposal, 'prune', gmap, anchors,
                                           source, cfg, (int(k),))
                candidate = evaluate(proposal, warm)
                gate = candidate_gate(state, candidate, dcfg, 'prune', focus_mask=focus_mask)
            except RuntimeError as exc:
                updates.append(dict(kind='prune', component=int(k), error=str(exc)))
                continue
            updates.append(dict(kind='prune', component=int(k), mass=float(mass[k]),
                                anchor_support=int((labels == k).sum()), **gate))
            if gate['accepted']:
                state = candidate
                break
    return state, updates


def run(cfg: dict, holdout: tuple[int, int, int, int] | None = None, tag: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the Gaussian rasterizer')
    seed = int(cfg.get('seed', 7))
    np.random.seed(seed)
    torch.manual_seed(seed)
    first, second = int(cfg['frames']['first']), int(cfg['frames']['second'])
    if first == second:
        raise ValueError('Two distinct frames are required')
    sequence = cfg['dataset']['sequence']
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / sequence
    left_dir = sequence_root / 'image_00' / 'data_rect'
    right_dir = sequence_root / 'image_01' / 'data_rect'
    im0 = stereo_io.read_image(left_dir / f'{first:010d}.png')
    im1 = stereo_io.read_image(left_dir / f'{second:010d}.png')
    right0 = stereo_io.read_image(right_dir / f'{first:010d}.png')
    right1 = stereo_io.read_image(right_dir / f'{second:010d}.png')
    if im0.shape != im1.shape or im0.shape != right0.shape or im1.shape != right1.shape:
        raise ValueError('Stereo images must have the same shape')
    K, baseline = stereo_io.read_calibration(data_root / 'calibration' / 'perspective.txt')
    d0 = load_depth(sequence_root, first, im0, right0, K, baseline, cfg['depth'])
    d1 = load_depth(sequence_root, second, im1, right1, K, baseline, cfg['depth'])
    for frame, depth in ((first, d0), (second, d1)):
        sky = sequence_root / 'sky_masks' / f'{frame:010d}.png'
        if sky.is_file():
            depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
    # Match the legacy rasterizer's data-dependent depth threshold while
    # keeping one value fixed throughout this pairwise experiment.
    gamma = cfg['joint'].get('render_depth_gamma')
    if gamma is not None:
        valid_source_depth = d0[np.isfinite(d0)]
        if not len(valid_source_depth):
            raise RuntimeError('No valid first-frame depth for Gaussian initialization')
        cfg['joint']['render_depth_threshold_m'] = float(gamma) * float(valid_source_depth.max())
    confidence = stereo_confidence(im1, right1, d1, K, baseline,
                                   float(cfg['depth'].get('confidence_rgb_sigma', 0.15)))
    source_confidence = stereo_confidence(im0, right0, d0, K, baseline,
                                          float(cfg['depth'].get('confidence_rgb_sigma', 0.15)))
    if not np.any(confidence > 0):
        raise RuntimeError('No valid stereo depth confidence for target frame')
    name = tag or cfg.get('run_tag', 'strict_pairwise')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run tag must be one directory name')
    out = resolve(cfg['output_dir']) / f'{sequence}_{first:010d}_{second:010d}' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    started = time.perf_counter()

    extractor = SuperPoint(max_num_keypoints=int(cfg['features']['initial_keypoints'])).eval().cuda()
    matcher = LightGlue(pretrained='superpoint', depth_confidence=0.95).eval().cuda()
    full_matches = match_images(im0, im1, extractor, matcher, 'cuda')
    initial = full_matches
    if holdout is None:
        holdout = cfg.get('holdout_roi_second')
    removed_roi = 0
    if holdout:
        x0, y0, x1, y1 = map(int, holdout)
        if not (0 <= x0 < x1 <= im1.shape[1] and 0 <= y0 < y1 <= im1.shape[0]):
            raise ValueError('Holdout ROI must lie inside the target image')
        initial = full_matches[~((full_matches[:, 2] >= x0) & (full_matches[:, 2] < x1) &
                                 (full_matches[:, 3] >= y0) & (full_matches[:, 3] < y1))]
        removed_roi = len(full_matches) - len(initial)
    fraction = float(cfg['features'].get('initial_keep_fraction', 1))
    if not 0 < fraction <= 1:
        raise ValueError('initial_keep_fraction must be in (0,1]')
    if fraction < 1:
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(initial), max(3, int(len(initial) * fraction)), replace=False)
        initial = initial[np.sort(ids)]
    anchors = with_anchor_confidence(lift(initial, d0, d1, K),
                                     source_confidence, confidence, cfg)
    initial_3d_count = len(anchors['x0'])
    if len(anchors['x0']) < 3:
        raise RuntimeError('Fewer than three valid 3D anchors')
    hybrid_initial = cfg['motion'].get('initialization_method') == 'hybrid'
    initialization_report = {}
    if hybrid_initial:
        from hybrid_motion import discover_hybrid, motion_costs, motion_probabilities
        initial_models, initial_labels, initial_supports, initialization_report = discover_hybrid(
            anchors, K, cfg['motion'], seed, matches_2d=initial)
        initial_costs = motion_costs(anchors, initial_models, K, cfg['motion'].get('hybrid', {}))
        initial_q_anchor, initial_inlier_mass = motion_probabilities(
            initial_costs, cfg['motion'].get('hybrid', {}))
    else:
        initial_models, initial_labels, initial_supports = discover(anchors, cfg['motion'], seed)
        initial_costs = np.empty((len(anchors['x0']), 0), np.float32)
        initial_inlier_mass = np.ones(len(anchors['x0']), np.float32)
        initial_q_anchor = anchor_probabilities(anchors, initial_models,
                                               float(cfg['motion']['assignment_sigma_m']))
    anchors_initial = anchors['matches'].copy()
    base_gmap = make_gaussians(im0, d0, K, int(cfg['gaussians']['stride']),
                               float(cfg['gaussians']['max_depth_jump_m']),
                               float(cfg['gaussians'].get('projected_sigma_px', 1.6)),
                               cfg['gaussians'].get('initialization', 'surface_aligned'))
    if not len(base_gmap['xyz']):
        raise RuntimeError('No first-frame Gaussians')
    anchor_gaussians = bool(cfg['joint'].get('anchor_gaussians', False))
    if anchor_gaussians:
        if int(cfg['gaussians']['stride']) != 1:
            raise ValueError('Anchor Gaussians require Gaussian stride 1')
        gmap, anchors = bind_anchor_gaussians(base_gmap, anchors, im0, K)
    else:
        gmap = base_gmap
    first_ref, _ = choose_reference(anchors, initial_models, d0.shape,
                                    float(cfg['motion']['assignment_sigma_m']))
    if hybrid_initial:
        first_ref = 0
    propagation_anchors = {**anchors, 'confidence':
                           anchors.get('confidence', np.ones(initial_3d_count))*initial_inlier_mass}
    initial_q = gaussian_probabilities(gmap, propagation_anchors, initial_q_anchor,
                                       first_ref, cfg['ownership'], im0)
    initial_joint_cfg = {**cfg['joint'], 'assignment_sigma_m': cfg['motion']['assignment_sigma_m']}
    if hybrid_initial:
        initial_joint_cfg['hybrid_geometry'] = cfg['motion'].get('hybrid', {})
    engine = PairwiseModel(gmap, anchors, initial_models, initial_q, K, im1, d1,
                           initial_joint_cfg,
                           target_confidence=confidence)
    rendering = engine.render_numpy(engine.base_H, engine.q_initial)
    initial_residual, initial_coverage, initial_metrics = residual(
        rendering, im1, d1, cfg['residual'], confidence)
    state = score_state(gmap, anchors, initial_models, K, im1, d1, confidence,
                        im0, cfg, int(cfg['optimization']['initial_iterations']), initial_q,
                        hybrid_initial=hybrid_initial)
    initial_state = state
    if hybrid_initial and int(cfg['discovery'].get('max_cycles', 3)) > 0:
        # Feedback/structure candidates retain their original objective. Only
        # reevaluate that scalar here; do not change the accepted H or q.
        feedback_engine = PairwiseModel(
            gmap, anchors, list(state['H']), state['q'], K, im1, d1,
            {**cfg['joint'], 'assignment_sigma_m': cfg['motion']['assignment_sigma_m']},
            target_confidence=confidence)
        with torch.no_grad():
            feedback_loss, feedback_terms = feedback_engine.loss(
                feedback_engine.base_H, feedback_engine.q_initial,
                int(cfg['optimization']['downsample']))
        state = {**state, 'loss': float(feedback_loss),
                 'terms': {key: float(value) for key, value in feedback_terms.items()}}
        del feedback_engine
    all_matches = (anchors['matches'].copy()
                   if ('match_depth_min_m' in cfg['features'] or
                       'match_depth_max_m' in cfg['features']) else initial.copy())
    seen_matches = initial.copy()
    cycles = []
    score_mask_mode = cfg['discovery'].get('feedback_mode') == 'score_mask_repair'
    global_mask_mode = score_mask_mode or cfg['discovery'].get('feedback_mode') == 'global_mask_repair'
    masked_mode = global_mask_mode or cfg['discovery'].get('feedback_mode') == 'masked_repair'
    compact_output = bool(cfg.get('output', {}).get('compact', False))
    pending = PendingMatches()
    first_problem_mask = np.zeros(d1.shape, bool)
    first_discovery = np.zeros_like(d1, dtype=np.float32)
    first_own_sum = np.zeros_like(d1, dtype=np.float32)
    first_own_count = np.zeros_like(d1, dtype=np.float32)
    first_motion_sum = np.zeros_like(d1, dtype=np.float32)
    first_motion_count = np.zeros_like(d1, dtype=np.float32)
    max_cycles = int(cfg['discovery'].get('max_cycles', 3))
    extractor.config['max_num_keypoints'] = int(cfg['features']['supplemental_keypoints'])
    extractor.config['detection_threshold'] = float(cfg['features'].get('supplemental_threshold', 1e-5))
    for cycle in range(max_cycles):
        own_projection = gradient_projection(
            gmap, state['H'], state['q'], state['gradient_own'], K, d1.shape)
        motion_projection = gradient_projection(
            gmap, state['H'], state['q'], state['gradient_motion'], K, d1.shape)
        dcfg = cfg['discovery']
        observed_pixels = np.isfinite(d1)
        mask_signals = {}
        selection_stats = {}
        if score_mask_mode:
            focus_mask, pixel_priority, selection_stats = score_guided_mask(
                state['score'], observed_pixels, own_projection, dcfg, confidence)
            discovery_score = pixel_priority
            regions = []
        elif global_mask_mode:
            # No patch selector, bounding boxes or cropped matching in this path.
            focus_mask, pixel_priority, mask_signals = global_problem_mask(
                state['score'], observed_pixels, own_projection, dcfg)
            discovery_score = pixel_priority
            regions = []  # Only legacy/holdout bookkeeping uses this variable.
        else:
            regions, discovery_score = select_regions(
                state['score'], observed_pixels, dcfg,
                own_projection, motion_projection)
            focus_mask = None
        if masked_mode:
            if not global_mask_mode:
                focus_mask, pixel_priority = problem_mask(
                    state['score'], observed_pixels, regions, own_projection, dcfg)
            # Photometrically inconsistent stereo does not supply reliable 3D evidence.
            focus_mask &= confidence >= float(dcfg.get('mask_min_stereo_confidence', .15))
            if cycle == 0:
                first_problem_mask = focus_mask.copy()
            if not compact_output:
                cv2.imwrite(str(out / f'problem_mask_cycle_{cycle:02d}.png'),
                            focus_mask.astype(np.uint8)*255)
            if global_mask_mode and not score_mask_mode and not compact_output:
                for signal in ('residual_trigger', 'gradient_trigger'):
                    cv2.imwrite(str(out / f'{signal}_cycle_{cycle:02d}.png'),
                                mask_signals[signal].astype(np.uint8)*255)
        if global_mask_mode:
            new_matches = full_image_masked_matches(im0, im1, focus_mask, extractor, matcher)
        else:
            new_matches = supplemental_matches(im0, im1, regions, extractor, matcher,
                                                int(dcfg['source_margin_px']),
                                                int(dcfg['target_margin_px']), focus_mask)
        seen_matches, novel = add_unique(seen_matches, new_matches)
        new_anchors = with_anchor_confidence(lift(novel, d0, d1, K),
                                             source_confidence, confidence, cfg)
        if masked_mode:
            pending.add(new_anchors['matches'])
            pending.remove_retained(anchors['matches'])
            pending_anchors = with_anchor_confidence(lift(pending.matches, d0, d1, K),
                                                     source_confidence, confidence, cfg)
            eligible = pending_anchors['confidence'] >= float(dcfg.get('pending_min_confidence', .10))
            evidence = subset_anchors(pending_anchors, eligible)
            sampling = spaced_indices if global_mask_mode else balanced_indices
            ids = sampling(evidence['matches'], evidence['confidence'], pixel_priority, dcfg)
            active = subset_anchors(evidence, ids)

            def prepare_feedback(matches):
                trial = with_anchor_confidence(lift(matches, d0, d1, K),
                                               source_confidence, confidence, cfg)
                if anchor_gaussians:
                    return bind_anchor_gaussians(base_gmap, trial, im0, K)
                return base_gmap, trial

            def optimize_feedback(trial_map, trial_anchors, models, q):
                return score_state(trial_map, trial_anchors, models, K, im1, d1,
                                   confidence, im0, cfg,
                                   int(cfg['optimization']['feedback_iterations']), initial_q=q)

            before = state
            state, anchors, gmap, decision = feedback_cycle(
                state, anchors, gmap, active, im0, cfg, cycle, focus_mask,
                prepare_feedback, optimize_feedback, candidate_gate)
            all_matches = anchors['matches'].copy()
            pending.remove_retained(all_matches)
            cycle_record = dict(cycle=cycle,
                                **({'selection_type': 'composite_score_mask', 'selection': selection_stats}
                                   if score_mask_mode else
                                   {'selection_type': 'global_pixel_mask'} if global_mask_mode
                                   else {'regions': regions}),
                                supplemental_matches=int(len(new_matches)),
                                novel_3d_anchors=int(len(new_anchors['x0'])),
                                problem_mask_pixels=int(focus_mask.sum()),
                                problem_mask_fraction=float(focus_mask.sum()/max(observed_pixels.sum(), 1)),
                                active_pending_3d_anchors=int(len(active['x0'])),
                                pending_3d_anchors=int(len(pending.matches)), **decision)
            if before['q'].shape == state['q'].shape:
                cycle_record['ownership_changed_fraction'] = float(np.mean(
                    before['q'].argmax(axis=1) != state['q'].argmax(axis=1)))
            np.savez_compressed(out / f'feedback_evidence_cycle_{cycle:02d}.npz',
                                problem_mask=focus_mask, novel_matches=new_anchors['matches'],
                                active_matches=active['matches'], pending_matches=pending.matches,
                                raw_matches=new_matches, **mask_signals)
            if holdout:
                x0, y0, x1, y1 = map(int, holdout)
                uv = new_anchors['matches'][:, 2:]
                cycle_record['new_3d_anchors_in_holdout'] = int(
                    ((uv[:, 0] >= x0) & (uv[:, 0] < x1) &
                     (uv[:, 1] >= y0) & (uv[:, 1] < y1)).sum())
        else:
            if len(new_anchors['x0']):
                e = np.stack([errors(new_anchors['x0'], new_anchors['x1'], H)
                              for H in state['H']], axis=1)
                minimum = e.min(axis=1)
                attach = minimum < motion_thresholds(
                    new_anchors, float(dcfg['attach_threshold_m']), cfg['motion'])
                unexplained = minimum > motion_thresholds(
                    new_anchors, float(dcfg['new_threshold_m']), cfg['motion'])
            else:
                attach = np.zeros(0, dtype=bool)
                unexplained = np.zeros(0, dtype=bool)
            # Attach only anchors that an existing motion explains. Unexplained
            # anchors enter the retained state only if a new component is accepted.
            saved_matches, saved_anchors, saved_gmap = all_matches, anchors, gmap
            if attach.any():
                all_matches = np.concatenate((all_matches, new_anchors['matches'][attach]))
            anchors = with_anchor_confidence(lift(all_matches, d0, d1, K),
                                             source_confidence, confidence, cfg)
            if anchor_gaussians:
                gmap, anchors = bind_anchor_gaussians(base_gmap, anchors, im0, K)
            cycle_record = dict(cycle=cycle, regions=regions,
                                supplemental_matches=int(len(new_matches)),
                                novel_3d_anchors=int(len(new_anchors['x0'])),
                                retained_3d_anchors=int(attach.sum()),
                                attached=int(attach.sum()), unexplained=int(unexplained.sum()),
                                proposals=[])
            if holdout:
                x0, y0, x1, y1 = map(int, holdout)
                uv = new_anchors['matches'][:, 2:]
                cycle_record['new_3d_anchors_in_holdout'] = int(
                    ((uv[:, 0] >= x0) & (uv[:, 0] < x1) &
                     (uv[:, 1] >= y0) & (uv[:, 1] < y1)).sum())
            before = state
            if attach.any():
                updated_q = refresh_ownership(state, anchors, gmap, im0, cfg,
                                              subset_anchors(new_anchors, attach))
                candidate_state = score_state(gmap, anchors, list(state['H']), K, im1, d1,
                                              confidence, im0, cfg,
                                              int(cfg['optimization']['feedback_iterations']),
                                              initial_q=updated_q)
                gain, _ = same_coverage_gain(before, candidate_state)
                ratio = candidate_state['coverage'].sum()/max(before['coverage'].sum(), 1)
                accepted_update = bool(
                    gain >= float(dcfg.get('feedback_min_render_improvement', -float('inf')))
                    and ratio >= float(dcfg.get('feedback_min_coverage_ratio', 0)))
                cycle_record['existing_motion_update'] = gain
                cycle_record['existing_motion_coverage_ratio'] = float(ratio)
                cycle_record['existing_motion_accepted'] = accepted_update
                cycle_record['ownership_changed_fraction'] = float(
                    np.mean(before['q'].argmax(axis=1) != candidate_state['q'].argmax(axis=1)))
                if accepted_update:
                    state = candidate_state
                else:
                    all_matches, anchors, gmap = saved_matches, saved_anchors, saved_gmap
                    cycle_record['retained_3d_anchors'] = 0
            unexplained_anchors = subset_anchors(new_anchors, unexplained)
            clusters = unexplained_clusters(unexplained_anchors, dcfg)
            comparison_matches = np.concatenate((all_matches, unexplained_anchors['matches']))
            comparison_anchors = with_anchor_confidence(lift(comparison_matches, d0, d1, K),
                                                         source_confidence, confidence, cfg)
            if anchor_gaussians:
                comparison_gmap, comparison_anchors = bind_anchor_gaussians(
                    base_gmap, comparison_anchors, im0, K)
            else:
                comparison_gmap = gmap
            comparison_baseline = None
            for cluster_id, ids in enumerate(clusters[:int(dcfg.get('max_birth_clusters', 3))]
                                             if dcfg.get('enable_birth', True) else []):
                if len(state['H']) >= int(dcfg.get('max_components_total', 6)):
                    break
                minimum = int(dcfg['birth_min_anchors'])
                if len(ids) < minimum:
                    continue
                cluster = subset_anchors(unexplained_anchors, ids)
                birth_cfg = {**cfg['motion'], 'max_components': 1,
                             'min_support': minimum,
                             'inlier_threshold_m': float(dcfg['birth_fit_threshold_m'])}
                try:
                    fitted, labels, support = discover(cluster, birth_cfg,
                                                        seed + 97 * cycle + cluster_id + 1)
                except RuntimeError:
                    continue
                pixels = cluster['matches'][labels == 0, 2:]
                if not len(pixels):
                    continue
                area = float(np.prod(pixels.max(axis=0) - pixels.min(axis=0)))
                box = [int(v) for v in (*pixels.min(axis=0), *pixels.max(axis=0))]
                record = dict(kind='birth', cluster=cluster_id, support=int(support[0]),
                              image_area_px=area, target_bbox=box, accepted=False,
                              median_depth_m=float(np.median(cluster['x0'][:, 2])),
                              median_stereo_confidence=float(np.median(
                                  cluster.get('confidence', np.ones(len(cluster['x0']))))))
                reliable_birth = (record['median_stereo_confidence'] >=
                                  float(dcfg.get('birth_min_median_stereo_confidence', 0)))
                if not reliable_birth:
                    record['rejection_reason'] = 'insufficient_stereo_confidence'
                if area >= float(dcfg['birth_min_area_px']) and reliable_birth:
                    try:
                        # Both alternatives see the same unexplained evidence.
                        if comparison_baseline is None:
                            comparison_baseline = score_state(
                                comparison_gmap, comparison_anchors, list(state['H']), K,
                                im1, d1, confidence, im0, cfg,
                                int(cfg['optimization']['feedback_iterations']),
                                initial_q=state['q'])
                        proposal = [*comparison_baseline['H'], fitted[0]]
                        warm = candidate_ownership(
                            comparison_baseline, proposal, 'birth', comparison_gmap,
                            comparison_anchors, im0, cfg, (), cluster['x0'][labels == 0])
                        candidate = score_state(
                            comparison_gmap, comparison_anchors, proposal, K, im1, d1,
                            confidence, im0, cfg,
                            int(cfg['optimization']['feedback_iterations']), initial_q=warm)
                        gate = candidate_gate(comparison_baseline, candidate, dcfg, 'birth', box)
                        record.update(gate)
                        if gate['accepted']:
                            state = candidate
                            gmap = comparison_gmap
                            all_matches = comparison_matches
                            anchors = comparison_anchors
                            comparison_baseline = candidate
                            cycle_record['retained_3d_anchors'] = (
                                int(attach.sum()) + len(unexplained_anchors['x0']))
                    except RuntimeError as exc:
                        record['error'] = str(exc)
                cycle_record['proposals'].append(record)
        state, structural = try_structural_updates(
            state, anchors, gmap, K, im0, im1, d1, confidence, cfg, cycle, seed,
            focus_mask=focus_mask)
        cycle_record['proposals'].extend(structural)
        cycle_record['model_count'] = len(state['H'])
        cycle_record['loss'] = state['loss']
        cycle_record['render'] = state['metrics']
        cycles.append(cycle_record)
        if cycle == 0:
            first_discovery = discovery_score.copy()
            first_own_sum, first_own_count = own_projection
            first_motion_sum, first_motion_count = motion_projection
            if not compact_output:
                save_heat(out / 'discovery_score.png', discovery_score, observed_pixels, 0.6)
            # Rescale only the images for viewing; patch selection uses raw sums.
            for name, projection in (
                ('ownership_gradient', own_projection),
                ('motion_gradient', motion_projection),
            ):
                values, counts = projection
                mean_image = values / np.maximum(counts, 1)
                scale = np.quantile(mean_image[counts > 0], 0.95) if np.any(counts > 0) else 1
                if not compact_output:
                    save_heat(out / f'{name}.png', mean_image / max(float(scale), 1e-8),
                              observed_pixels, 1.0)
            selected = im1.copy()
            if global_mask_mode:
                selected[focus_mask] = (.65*selected[focus_mask] +
                                        .35*np.array([0, 255, 255])).astype(np.uint8)
            else:
                for box in regions:
                    cv2.rectangle(selected, tuple(box[:2]), tuple(box[2:]), (0, 255, 255), 2)
            if holdout:
                cv2.rectangle(selected, tuple(holdout[:2]), tuple(holdout[2:]),
                              (255, 0, 255), 2)
            if not compact_output:
                cv2.imwrite(str(out / ('selected_mask.png' if global_mask_mode else 'selected_regions.png')),
                            selected)
        if (not (masked_mode and len(pending.matches)) and
                not cycle_record['retained_3d_anchors'] and not any(
                p.get('accepted', False) for p in cycle_record['proposals'])):
            break

    # Ground truth is loaded only after all estimation decisions are complete.
    gt = None
    pose_path = data_root / 'data_poses' / sequence / 'cam0_to_world.txt'
    if pose_path.is_file():
        poses = stereo_io.read_poses(pose_path)
        if first in poses and second in poses:
            gt = np.linalg.inv(poses[second]) @ poses[first]
    common = initial_coverage & initial_state['coverage'] & state['coverage']
    summary = dict(
        frames=[first, second], depth_source=cfg['depth']['source'],
        initialization_method='hybrid' if hybrid_initial else 'ransac3d',
        initialization_report=initialization_report,
        feedback_mode=cfg['discovery'].get('feedback_mode', 'patch'),
        compact_output=compact_output,
        pending_3d_matches_final=int(len(pending.matches)) if masked_mode else 0,
        gaussian_initialization=cfg['gaussians'].get('initialization', 'surface_aligned'),
        gaussian_stride=int(cfg['gaussians']['stride']),
        gaussian_opacity=float(cfg['joint'].get('gaussian_opacity', 0.9)),
        anchor_gaussians_enabled=anchor_gaussians,
        anchor_gaussians_final=int(len(anchors['gaussian_index'])) if anchor_gaussians else 0,
        render_depth_threshold_m=float(cfg['joint'].get('render_depth_threshold_m', 1.0)),
        depth_confidence='left_right_photometric', gt_used_for_estimation=False,
        initial_matches=int(len(initial)), initial_3d_anchors=int(initial_3d_count),
        sparse_match_depth_range_m=[
            float(cfg['features'].get('match_depth_min_m', cfg['depth'].get('min_m', 0))),
            float(cfg['features'].get('match_depth_max_m', cfg['depth'].get('max_m', float('inf'))))],
        final_3d_anchors=int(len(anchors['x0'])), gaussians=int(len(gmap['xyz'])),
        initial_model_supports=[int(x) for x in initial_supports],
        cycles=cycles, final_models=len(state['H']),
        reference_sparse=int(first_ref), reference_final=int(state['reference']),
        relative_camera_transform=state['H'][state['reference']].tolist(),
        camera_c2w_second_from_identity_first=np.linalg.inv(state['H'][state['reference']]).tolist(),
        render_sparse=initial_metrics, render_joint=initial_state['metrics'],
        render_final=state['metrics'], final_loss_terms=state['terms'],
        common_coverage=dict(pixels=int(common.sum()),
                             sparse=float(initial_residual[common].mean()) if common.any() else None,
                             joint=float(initial_state['score'][common].mean()) if common.any() else None,
                             final=float(state['score'][common].mean()) if common.any() else None),
        pose_error_sparse=pose_error(initial_models[first_ref], gt) if gt is not None else None,
        pose_error_final=pose_error(state['H'][state['reference']], gt) if gt is not None else None,
        elapsed_seconds=time.perf_counter() - started)
    if holdout:
        x0, y0, x1, y1 = map(int, holdout)
        summary['holdout_evaluation'] = dict(
            initial_matches_removed=int(removed_roi),
            new_3d_anchors_in_holdout=sum(
                row.get('new_3d_anchors_in_holdout', 0) for row in cycles))
        if global_mask_mode:
            summary['holdout_evaluation']['first_mask_pixels_in_holdout'] = int(
                first_problem_mask[y0:y1, x0:x1].sum())
        else:
            summary['holdout_evaluation']['selected_patch_hits'] = sum(
                max(0, min(x1, bx1) - max(x0, bx0)) *
                max(0, min(y1, by1) - max(y0, by0)) > 0
                for row in cycles for bx0, by0, bx1, by1 in row['regions'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (out / 'cycles.json').write_text(json.dumps(cycles, indent=2), encoding='utf-8')
    (out / 'joint_loss_history.json').write_text(json.dumps(initial_state['history'], indent=2))
    if hybrid_initial:
        (out / 'initialization_report.json').write_text(json.dumps(initialization_report, indent=2))
    anchor_state = ({
        'anchor_gaussian_indices_final': gmap['anchor_gaussian_indices'],
        'anchor_gaussian_xyz_final': gmap['xyz'][gmap['anchor_gaussian_indices']],
        'anchor_gaussian_uv_final': gmap['uv'][gmap['anchor_gaussian_indices']],
        'anchor_gaussian_color_final': gmap['color'][gmap['anchor_gaussian_indices']],
        'anchor_gaussian_scale_final': gmap['scale'][gmap['anchor_gaussian_indices']],
    } if anchor_gaussians else {})
    np.savez_compressed(out / 'pairwise_state.npz',
                        H_sparse=np.stack(initial_models), H_joint=initial_state['H'], H_final=state['H'],
                        q_anchor_sparse=initial_q_anchor, q_anchor_final=state['q_anchor'],
                        q_gaussian_sparse=initial_q, q_gaussian_joint=initial_state['q'],
                        q_gaussian_final=state['q'], anchors_final=anchors['matches'],
                        anchors_initial=anchors_initial, initial_matches_2d=initial,
                        initial_motion_labels=initial_labels, initial_motion_costs=initial_costs,
                        initial_motion_inlier_probability=initial_inlier_mass,
                        ownership_gradient=state['gradient_own'],
                        motion_gradient=state['gradient_motion'],
                        pending_matches=pending.matches, **anchor_state)
    np.savez_compressed(out / 'score_maps.npz', sparse=initial_residual,
                        joint=initial_state['score'], final=state['score'],
                        discovery_first=first_discovery, problem_mask_first=first_problem_mask,
                        ownership_gradient_sum_first=first_own_sum,
                        ownership_gradient_count_first=first_own_count,
                        motion_gradient_sum_first=first_motion_sum,
                        motion_gradient_count_first=first_motion_count,
                        coverage_sparse=initial_coverage,
                        coverage_joint=initial_state['coverage'], coverage_final=state['coverage'],
                        rendered_depth_final=state['rendering']['depth'],
                        rendered_rgb_final=np.clip(state['rendering']['rgb']*255, 0, 255).astype(np.uint8))
    if not compact_output:
        for stage, scene in [('sparse', rendering), ('joint', initial_state['rendering']),
                             ('final', state['rendering'])]:
            save_rgb(out / f'render_{stage}.png', scene)
        for stage, values, coverage in [('sparse', initial_residual, initial_coverage),
                                        ('joint', initial_state['score'], initial_state['coverage']),
                                        ('final', state['score'], state['coverage'])]:
            save_heat(out / f'residual_{stage}.png', values, coverage)
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {out}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parent / 'configs/kitti360_full_feedback_four_pairs_hybrid_multibg_early.yaml')
    parser.add_argument('--holdout-roi', nargs=4, type=int,
                        metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    parser.add_argument('--tag')
    parser.add_argument('--frames', nargs=2, type=int, metavar=('FIRST', 'SECOND'),
                        help='Override the frame pair in the config')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.frames:
        cfg['frames'] = dict(first=args.frames[0], second=args.frames[1])
    run(cfg, tuple(args.holdout_roi) if args.holdout_roi else None, args.tag)


if __name__ == '__main__':
    main()

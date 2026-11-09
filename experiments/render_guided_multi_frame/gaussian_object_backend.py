"""CPU observation adapter for the persistent Gaussian object window.

Fresh RGB-D surface samples are fixed observations of shared Gaussian IDs.
Identity links are validated by tracks or mutual geometry/appearance matches
under every available motion, never just the previous winning object. The
objective combines measured track/centre errors with optional differentiable
Gaussian RGB-D rendering over the window. Geometry stays fixed in both modes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE / 'render_guided_two_frame'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hybrid_motion import _core_labels, _refit, _rigid, motion_costs
from object_window import ObjectWindow


def _project(xyz, K):
    homogeneous = np.asarray(xyz) @ np.asarray(K).T
    return homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8)


def _sample(image, uv, linear=False):
    """Sample without substituting border values for missing observations."""
    uv = np.asarray(uv, float)
    h, w = image.shape[:2]
    valid = (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= 0) &
             (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1))
    out = np.full((len(uv),) + image.shape[2:], np.nan, float)
    ids = np.flatnonzero(valid)
    # OpenCV remap requires each map dimension < SHRT_MAX. A source cloud
    # routinely has hundreds of thousands of samples, so use short rows.
    for start in range(0, len(ids), 16000):
        batch = ids[start:start+16000]
        xy = uv[batch].astype(np.float32)
        sampled = cv2.remap(np.asarray(image, np.float32), xy[:, 0][None],
                            xy[:, 1][None], cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=float('nan'))[0]
        out[batch] = sampled
    return out, valid


class SequenceObjectBackend:
    """Evaluate current observations and maintain fixed-lag shared attributes."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.options = cfg.get('object_window', {})
        self.render_enabled = bool(self.options.get('render', {}).get('enabled', False))
        motion = cfg.get('motion', {})
        self.hybrid = motion.get('hybrid', motion)
        self.minimum = int(self.options.get('model_min_support',
                           self.hybrid.get('min_support', motion.get('min_support', 6))))
        self.window = ObjectWindow(cfg)
        self._next_gid = 0
        self._previous = None
        self._track_gids = {}
        self._history = {}
        self._track_snapshots = {}

    def _anchor_indices(self, gmap, anchors):
        count = len(anchors['x0'])
        bound = gmap.get('anchor_gaussian_indices', anchors.get('gaussian_index'))
        if bound is not None:
            result = np.asarray(bound, np.int64).copy()
            if result.shape != (count,):
                raise ValueError('anchor Gaussian indices must match anchor rows')
            return result
        result = np.full(count, -1, np.int64)
        if count and len(gmap['uv']):
            distance, nearest = cKDTree(gmap['uv']).query(anchors['matches'][:, :2])
            valid = distance <= float(self.options.get('anchor_link_radius_px', 2.5))
            result[valid] = nearest[valid]
        return result

    def _feature_height_gate(self, gmap, anchors, source, source_confidence):
        """Estimate a measured ground plane and gate nonbackground motion support.

        Matches, tracks and initial probabilities remain intact. The resulting
        weights are consumed only inside the per-object motion proposal fit.
        """
        options = self.options.get('height', {})
        report = dict(enabled=bool(options.get('enabled', False)), applied=False,
                      excluded_anchors=0, downweighted_anchors=0, reason='disabled')
        if not report['enabled']:
            return None, report
        from object_height_constraints import estimate_ground_plane, height_penalty
        confidence, _ = _sample(source_confidence, gmap['uv'])
        plane = estimate_ground_plane(gmap['xyz'], gmap['uv'], source.shape,
                                      confidence=confidence, options=options)
        report['plane'] = plane
        report['reason'] = str(plane.get('reason', 'invalid_ground_plane'))
        if not plane.get('valid', False):
            return None, report
        xyz = np.asarray(anchors['x0'], float)
        penalty, reliability = height_penalty(xyz, plane, options)
        penalty = np.asarray(penalty, float)
        reliability = np.asarray(reliability, float)
        if penalty.shape != (len(xyz),) or reliability.shape != (len(xyz),):
            raise ValueError('Feature height penalty/reliability must match anchor rows')
        measured = (np.isfinite(penalty) & np.isfinite(reliability) &
                    np.isfinite(xyz).all(axis=1))
        reliability = np.where(measured, np.clip(reliability, 0, 1), 0.)
        penalty = np.where(measured, np.maximum(penalty, 0), 0.)
        signed_height = xyz @ np.asarray(plane['normal'], float) + float(plane['offset'])
        lower = float(options.get('ground_min_m', -.2))
        upper = float(options.get('max_height_m', float(plane['camera_height']) +
                                   float(options.get('above_camera_m', 2.))))
        margin = float(options.get('soft_margin_m', .3))
        minimum_reliability = float(options.get('feature_min_reliability', .5))
        if (not np.isfinite([lower, upper, margin, minimum_reliability]).all() or
                margin <= 0 or upper <= lower or not 0 <= minimum_reliability <= 1):
            raise ValueError('Invalid feature height band, margin or reliability threshold')
        excluded = ((reliability >= minimum_reliability) &
                    ((signed_height < lower-margin) | (signed_height > upper+margin)))
        weights = 1-reliability*np.minimum(penalty, 1.)
        weights[excluded] = 0.
        report.update(applied=True, reason='measured_ground_plane_height_gate',
                      excluded_anchors=int(excluded.sum()),
                      downweighted_anchors=int(np.count_nonzero((weights < 1-1e-8) & ~excluded)),
                      allowed_ground_height_m=[lower, upper], hard_margin_m=margin,
                      feature_min_reliability=minimum_reliability,
                      background_object_id=0, background_filtered=False)
        return dict(allowed=~excluded, weights=weights), report

    def _models(self, state, anchors, track_ids, mapping, registry, K, height_gate=None):
        """Choose one currently supported rigid transform per persistent object."""
        objects = sorted(set(map(int, registry.objects)) |
                         set(map(int, mapping.values())) | set(self.window.object_ids))
        models = np.full((len(objects), 4, 4), np.nan)
        valid = np.zeros(len(objects), bool)
        reports = []
        history = np.asarray([self._history.get(int(t), -1) for t in track_ids])
        confidence = np.asarray(anchors.get('confidence', np.ones(len(track_ids)))).reshape(-1)
        outlier = float(self.hybrid.get('outlier_cost', 1.))
        raw_models = np.asarray(state['H'], float)
        for column, oid in enumerate(objects):
            gated = height_gate is not None and oid != 0
            allowed = (np.asarray(height_gate['allowed'], bool) if gated else
                       np.ones(len(track_ids), bool))
            feature_weight = (np.asarray(height_gate['weights'], float) if gated else
                              np.ones(len(track_ids)))
            if allowed.shape != (len(track_ids),) or feature_weight.shape != (len(track_ids),):
                raise ValueError('Height gate must have one entry per motion anchor')
            fit_confidence = confidence*feature_weight
            fit_anchors = {**anchors, 'confidence': fit_confidence} if gated else anchors
            historical = np.flatnonzero((history == oid) & allowed & (fit_confidence > 0))
            other_cost = (motion_costs(anchors, list(models[valid]), K, self.hybrid).min(axis=1)
                          if valid.any() else np.full(len(track_ids), np.inf))
            candidates = [(raw_models[local], f'local_{local}')
                          for local in range(len(raw_models))
                          if int(mapping[str(local)]) == oid]
            # A disappeared local hypothesis is reconstructed from *this*
            # interval's measured correspondences, never copied from history.
            if len(historical) >= self.minimum:
                proposals = [historical]
                rng = np.random.default_rng(oid + 137)
                if len(historical) > 3:
                    proposals += [rng.choice(historical, 3, replace=False)
                                  for _ in range(min(32, 2*len(historical)))]
                best = None
                best_score = None
                for ids in proposals:
                    if len(ids) < 3 or np.linalg.matrix_rank(
                            anchors['x0'][ids] - anchors['x0'][ids].mean(axis=0)) < 2:
                        continue
                    H = _rigid(anchors['x0'][ids], anchors['x1'][ids], fit_confidence[ids])
                    costs = motion_costs(anchors, [H], K, self.hybrid)[:, 0]
                    core = historical[costs[historical] < outlier]
                    score = (float(feature_weight[core].sum()),
                             -float((feature_weight[historical]*np.minimum(costs[historical], outlier)).sum()))
                    if len(core) >= self.minimum and (best_score is None or score > best_score):
                        best = (H, core)
                        best_score = score
                if best is not None:
                    candidates.append((_refit(fit_anchors, best[0], best[1], K, self.hybrid),
                                       'current_historical_track_refit'))
                # Contaminating background can also dominate the historical
                # fit itself. Preserve a proposal from currently unexplained
                # tracks instead of relying on rare all-object RANSAC draws.
                independent_ids = historical[other_cost[historical] >= outlier]
                if (valid.any() and len(independent_ids) >= self.minimum and
                        np.linalg.matrix_rank(anchors['x0'][independent_ids] -
                                              anchors['x0'][independent_ids].mean(axis=0)) >= 2):
                    H = _rigid(anchors['x0'][independent_ids], anchors['x1'][independent_ids],
                               fit_confidence[independent_ids])
                    candidates.append((_refit(fit_anchors, H, independent_ids, K, self.hybrid),
                                       'current_historical_track_refit'))
            winner = None
            winner_score = None
            independent_improvement = float(self.hybrid.get('independent_improvement', .10))
            for H, source in candidates:
                if not np.isfinite(H).all():
                    continue
                costs, labels = _core_labels(anchors, [H], K, self.hybrid)
                core = (labels == 0) & allowed & (fit_confidence > 0)
                height_gated_refit = False
                if gated:
                    eligible = np.flatnonzero(core)
                    if len(eligible) < self.minimum:
                        continue
                    # A local proposal may itself have been fitted using
                    # forbidden high features. Refit from permitted evidence,
                    # rather than merely omitting those features from scoring.
                    seed = np.asarray(H).copy()
                    if len(eligible) < 4:
                        # _refit returns its seed unchanged below four points.
                        # A supported noncollinear three-point fit still must
                        # discard the potentially contaminated original pose.
                        centered = anchors['x0'][eligible] - anchors['x0'][eligible].mean(axis=0)
                        if len(eligible) < 3 or np.linalg.matrix_rank(centered) < 2:
                            continue
                        seed = _rigid(anchors['x0'][eligible], anchors['x1'][eligible],
                                      fit_confidence[eligible])
                    H = _refit(fit_anchors, seed, eligible, K, self.hybrid)
                    if not np.isfinite(H).all():
                        continue
                    costs, labels = _core_labels(anchors, [H], K, self.hybrid)
                    core = (labels == 0) & allowed & (fit_confidence > 0)
                    height_gated_refit = True
                support = int(core.sum())
                effective_support = float(feature_weight[core].sum())
                if support < self.minimum:
                    continue
                continuing = int(np.count_nonzero(core & (history == oid)))
                effective_continuing = float(feature_weight[core & (history == oid)].sum())
                # A mixed object's larger background alias must not defeat a
                # real independent motion merely by inheriting more tracks.
                # If two objects genuinely move together, neither gains this
                # preference and their uncertainty is retained downstream.
                independent = core & (other_cost-costs[:, 0] > independent_improvement)
                independent_support = int(independent.sum()) if valid.any() else 0
                effective_independent = float(feature_weight[independent].sum()) if valid.any() else 0.
                reliable_independent = effective_independent >= self.minimum
                independent_continuing = int(np.count_nonzero(independent & (history == oid)))
                effective_independent_continuing = float(feature_weight[independent & (history == oid)].sum())
                score = (reliable_independent,
                         effective_independent_continuing if reliable_independent else 0,
                         effective_independent if reliable_independent else 0,
                         effective_continuing, effective_support,
                         -float((feature_weight[allowed]*np.minimum(costs[allowed, 0], outlier)).sum()))
                if winner_score is None or score > winner_score:
                    winner = (np.asarray(H).copy(), source, support, continuing,
                              independent_support, height_gated_refit)
                    winner_score = score
            row = dict(object_id=oid, valid=winner is not None,
                       historical_tracks=int(np.count_nonzero(history == oid)),
                       height_gating_applied=gated,
                       height_excluded_anchors=int(np.count_nonzero(~allowed)),
                       height_downweighted_anchors=int(np.count_nonzero(allowed & (feature_weight < 1-1e-8))),
                       eligible_historical_tracks=int(len(historical)), height_gated_refit=False)
            if winner is not None:
                models[column], source, support, continuing, independent, refitted = winner
                valid[column] = True
                row.update(source=source, core_support=support, continuing_support=continuing,
                           independent_support=independent, height_gated_refit=refitted)
            reports.append(row)
        return np.asarray(objects, np.int64), models, valid, reports

    def _raw_costs(self, gmap, models, valid, K, target, target_depth,
                   source_confidence, target_confidence):
        xyz = np.asarray(gmap['xyz'], float)
        color = np.asarray(gmap['color'], float)
        target_rgb = cv2.cvtColor(target, cv2.COLOR_BGR2RGB).astype(np.float32)/255.
        source_weight, _ = _sample(source_confidence, gmap['uv'])
        source_weight = np.nan_to_num(source_weight, nan=0.).clip(0, 1)
        costs = np.full((len(xyz), len(models)), np.nan)
        weights = np.zeros_like(costs)
        photo_sigma = float(self.options.get('color_sigma', .10))
        sigma_base = float(self.options.get('depth_sigma_m', .15))
        sigma_relative = float(self.options.get('depth_sigma_relative', .015))
        color_weight = float(self.options.get('color_weight', .5))
        depth_weight = float(self.options.get('depth_weight', .5))
        for column, H in enumerate(models):
            if not valid[column]:
                continue
            moved = xyz @ H[:3, :3].T + H[:3, 3]
            uv = _project(moved, K)
            rgb, visible = _sample(target_rgb, uv, linear=True)
            depth, _ = _sample(target_depth, uv)
            confidence, _ = _sample(target_confidence, uv)
            sigma = sigma_base + sigma_relative*np.maximum(moved[:, 2], 0)
            # A predicted point behind an observed surface is unobservable.
            visible &= (np.isfinite(depth) & (depth > 0) & (moved[:, 2] > 0) &
                        (moved[:, 2] <= depth + sigma) & np.isfinite(rgb).all(axis=1) &
                        np.isfinite(confidence) & (confidence > 0) & (source_weight > 0))
            photo = np.mean(np.abs(rgb-color), axis=1)/photo_sigma
            metric = np.abs(moved[:, 2]-depth)/np.maximum(sigma, 1e-6)
            costs[visible, column] = (color_weight*photo[visible]**2 +
                                      depth_weight*metric[visible]**2)
            weights[visible, column] = np.sqrt(source_weight[visible]*confidence[visible].clip(0, 1))
        return costs, weights

    def _geometry_links(self, gmap, first, source_depth, source_confidence, assigned, used):
        previous = self._previous
        if previous is None or previous['second'] != first or not len(gmap['uv']):
            return 0
        radius = float(self.options.get('link_radius_px', 2.5))
        metric_base = float(self.options.get('link_depth_m', .12))
        metric_relative = float(self.options.get('link_depth_relative', .015))
        color_limit = float(self.options.get('link_color_difference', .15))
        tree = cKDTree(gmap['uv'])
        previous_indices = np.flatnonzero(~np.isin(previous['gaussian_ids'], list(used)))
        if not len(previous_indices):
            return 0
        edges = []
        for H, valid in zip(previous['models'], previous['model_valid']):
            if not valid:
                continue
            moved = previous['xyz'][previous_indices] @ H[:3, :3].T + H[:3, 3]
            uv = _project(moved, previous['K'])
            depth, in_image = _sample(source_depth, uv)
            confidence, _ = _sample(source_confidence, uv)
            metric_limit = metric_base + metric_relative*np.maximum(moved[:, 2], 0)
            visible = (in_image & (moved[:, 2] > 0) & np.isfinite(depth) & (depth > 0) &
                       (np.abs(moved[:, 2]-depth) <= metric_limit) & (confidence > 0))
            distance, index = tree.query(uv, k=min(2, len(gmap['uv'])))
            distance = np.asarray(distance).reshape(len(uv), -1)
            index = np.asarray(index).reshape(len(uv), -1)
            for choice in range(distance.shape[1]):
                current = index[:, choice]
                difference = np.linalg.norm(gmap['xyz'][current]-moved, axis=1)
                appearance = np.mean(np.abs(gmap['color'][current]-
                                           previous['color'][previous_indices]), axis=1)
                keep = (visible & (distance[:, choice] <= radius) & (difference <= metric_limit) &
                        (appearance <= color_limit) & (assigned[current] < 0))
                ids = np.flatnonzero(keep)
                if len(ids):
                    score = ((distance[ids, choice]/radius)**2 +
                             (difference[ids]/metric_limit[ids])**2 +
                             (appearance[ids]/color_limit)**2)
                    edges.append(np.column_stack((previous_indices[ids], current[ids], score)))
        if not edges:
            return 0
        edges = np.concatenate(edges)
        # Collapse identical (previous,current) edges proposed by several Hs.
        order = np.lexsort((edges[:, 2], edges[:, 1], edges[:, 0]))
        edges = edges[order]
        unique = np.r_[True, np.any(edges[1:, :2] != edges[:-1, :2], axis=1)]
        edges = edges[unique]
        best = []
        margin = float(self.options.get('link_uniqueness_margin', .05))
        for key in (0, 1):
            order = np.lexsort((edges[:, 2], edges[:, key]))
            sorted_edges = edges[order]
            starts = np.flatnonzero(np.r_[True, np.diff(sorted_edges[:, key]) != 0])
            ends = np.r_[starts[1:], len(sorted_edges)]
            acceptable = (ends-starts == 1)
            multiple = np.flatnonzero(~acceptable)
            acceptable[multiple] = (sorted_edges[starts[multiple]+1, 2] -
                                    sorted_edges[starts[multiple], 2] > margin)
            best.append(set(map(int, order[starts[acceptable]])))
        accepted = sorted(best[0] & best[1])
        for row in accepted:
            old, current = map(int, edges[row, :2])
            assigned[current] = previous['gaussian_ids'][old]
            used.add(int(assigned[current]))
        return len(accepted)

    def _identities(self, first, second, gmap, anchors, track_ids, anchor_indices,
                    source_depth, source_confidence, target_confidence):
        assigned = np.full(len(gmap['xyz']), -1, np.int64)
        gaussian_tracks = np.full(len(assigned), -1, np.int64)
        used = set()
        linked_tracks = 0
        pixel_limit = float(self.options.get('track_link_radius_px', 1.5))
        metric_base = float(self.options.get('link_depth_m', .12))
        color_limit = float(self.options.get('link_color_difference', .15))
        actual_source_depth, _ = _sample(source_depth, anchors['matches'][:, :2])
        actual_source_confidence, _ = _sample(source_confidence, anchors['matches'][:, :2])
        actual_target_confidence, _ = _sample(target_confidence, anchors['matches'][:, 2:])
        anchor_confidence = np.asarray(anchors.get('confidence', np.ones(len(track_ids)))).reshape(-1)
        seen_tracks = set()
        for row, (track, index) in enumerate(zip(track_ids, anchor_indices)):
            track, index = int(track), int(index)
            if track < 0 or index < 0 or track in seen_tracks or gaussian_tracks[index] >= 0:
                continue
            seen_tracks.add(track)
            gaussian_tracks[index] = track
            previous = self._track_gids.get(track)
            if previous is None or previous['frame'] != first or previous['gid'] in used:
                continue
            xyz = np.asarray(anchors['x0'][row])
            tolerance = metric_base + float(self.options.get('link_depth_relative', .015))*xyz[2]
            if (previous['confidence'] > 0 and anchor_confidence[row] > 0 and
                    actual_source_confidence[row] > 0 and np.isfinite(actual_source_depth[row]) and
                    abs(actual_source_depth[row]-xyz[2]) <= tolerance and
                    np.linalg.norm(anchors['matches'][row, :2]-previous['uv']) <= pixel_limit and
                    np.linalg.norm(xyz-previous['xyz']) <= tolerance and
                    np.linalg.norm(gmap['xyz'][index]-xyz) <= tolerance and
                    np.mean(np.abs(gmap['color'][index]-previous['color'])) <= color_limit):
                assigned[index] = previous['gid']
                used.add(previous['gid'])
                linked_tracks += 1
        linked_geometry = self._geometry_links(gmap, first, source_depth, source_confidence,
                                               assigned, used)
        fresh = np.flatnonzero(assigned < 0)
        assigned[fresh] = np.arange(self._next_gid, self._next_gid+len(fresh))
        self._next_gid += len(fresh)
        for row, (track, index) in enumerate(zip(track_ids, anchor_indices)):
            track, index = int(track), int(index)
            if track >= 0 and index >= 0 and gaussian_tracks[index] == track:
                self._track_gids[track] = dict(
                    gid=int(assigned[index]), frame=int(second),
                    xyz=np.asarray(anchors['x1'][row]).copy(),
                    uv=np.asarray(anchors['matches'][row, 2:]).copy(),
                    color=np.asarray(gmap['color'][index]).copy(),
                    confidence=float(np.nan_to_num(actual_target_confidence[row], nan=0.)) *
                               float(max(anchor_confidence[row], 0.)))
        return assigned, gaussian_tracks, dict(track_links=linked_tracks,
                                               geometry_links=linked_geometry,
                                               new_gaussians=len(fresh))

    def process(self, first, second, gmap, anchors, track_ids, mapping, state, K,
                source, target, source_depth, target_depth, source_confidence,
                target_confidence, output, registry):
        """Process a pair and rewrite all active snapshots with one posterior."""
        first, second = int(first), int(second)
        track_ids = np.asarray(track_ids, np.int64)
        anchors = {**anchors, 'x0': np.asarray(anchors['x0'], float),
                   'x1': np.asarray(anchors['x1'], float),
                   'matches': np.asarray(anchors['matches'], float)}
        height_gate, height_report = self._feature_height_gate(
            gmap, anchors, source, source_confidence)
        object_ids, models, valid, model_report = self._models(
            state, anchors, track_ids, mapping, registry, K, height_gate=height_gate)
        indices = self._anchor_indices(gmap, anchors)
        gids, gaussian_tracks, links = self._identities(
            first, second, gmap, anchors, track_ids, indices, source_depth, source_confidence,
            target_confidence)
        costs, weights = self._raw_costs(gmap, models, valid, K, target, target_depth,
                                          source_confidence, target_confidence)
        if len(track_ids) and valid.any():
            geometry = motion_costs(anchors, list(models[valid]), K, self.hybrid)
            confidence = np.asarray(anchors.get('confidence', np.ones(len(track_ids)))).reshape(-1)
            source_anchor, _ = _sample(source_confidence, anchors['matches'][:, :2])
            target_anchor, _ = _sample(target_confidence, anchors['matches'][:, 2:])
            measured_weight = np.sqrt(np.maximum(source_anchor, 0)*np.maximum(target_anchor, 0))
            measured_weight *= confidence.clip(0, 1)
            # A measured visible track is valid negative evidence even when
            # a wrong hypothesis projects it outside the target image.
            observed = np.isfinite(measured_weight) & (measured_weight > 0)
            for row, index in enumerate(indices):
                if index >= 0 and observed[row]:
                    costs[index, valid] = geometry[row]
                    weights[index, valid] = measured_weight[row]
        q_local = np.array(state['q'], copy=True)
        q_motion = np.array(state.get('q_motion_initial', state['q']), copy=True)
        initial = np.zeros((len(gids), len(object_ids)), float)
        object_column = {int(oid): column for column, oid in enumerate(object_ids)}
        for local in range(q_local.shape[1]):
            initial[:, object_column[int(mapping[str(local)])]] += q_local[:, local]
        payload = {name: np.asarray(gmap[name]) for name in ('xyz', 'uv', 'color', 'scale')}
        if 'basis' in gmap:
            payload['basis'] = np.asarray(gmap['basis'])
        payload.update(K=np.asarray(K), H_object=models, object_model_valid=valid,
                       matches=anchors['matches'], anchor_track_ids=track_ids)
        if 'plane' in height_report:
            plane = height_report['plane']
            payload.update(
                ground_plane_normal=np.asarray(plane.get('normal', [0., -1., 0.]), float),
                ground_plane_offset=np.asarray(plane.get('offset', 0.), float),
                ground_plane_confidence=np.asarray(plane.get('confidence', 0.), float),
                ground_plane_valid=np.asarray(bool(plane.get('valid', False))),
                ground_plane_report_json=np.asarray(json.dumps(plane, sort_keys=True)))
        render_arguments = {}
        if self.render_enabled:
            # The window owns copying/caching these raw observations. Keep them
            # separate from archive payloads so RGB-D arrays are not duplicated
            # into every per-pair object_state.npz file.
            render_arguments['render_observations'] = dict(
                source=source, target=target, source_depth=source_depth,
                target_depth=target_depth, source_confidence=source_confidence,
                target_confidence=target_confidence)
        posterior = self.window.update(
            first, second, gids, object_ids, initial, costs, weights=weights,
            output_dir=output, track_ids=gaussian_tracks, payload=payload,
            q_motion_initial=q_motion, **render_arguments)
        report = posterior['report']
        self._track_snapshots[(first, second)] = (gids.copy(), gaussian_tracks.copy())
        active = {tuple(pair) for pair in report['window_pairs']}
        self._track_snapshots = {pair: snapshot for pair, snapshot in self._track_snapshots.items()
                                 if pair in active}
        associations = sorted({(int(track), int(gid))
                               for ids, tracks in self._track_snapshots.values()
                               for gid, track in zip(ids, tracks) if track >= 0})
        if associations and hasattr(registry, 'commit_window_tracks'):
            unique_gids = np.asarray(sorted({gid for _, gid in associations}), np.int64)
            shared = self.window.get_state(unique_gids)
            lookup = {int(gid): index for index, gid in enumerate(unique_gids)}
            rows = np.asarray([lookup[gid] for _, gid in associations], np.int64)
            registry.commit_window_tracks(
                np.asarray([track for track, _ in associations], np.int64), shared['object_ids'],
                shared['object_probabilities'][rows], shared['object_labels'][rows])
        self._history = dict(registry.track_objects)
        self._previous = dict(first=first, second=second, gaussian_ids=gids.copy(),
                              xyz=np.asarray(gmap['xyz']).copy(), color=np.asarray(gmap['color']).copy(),
                              models=models.copy(), model_valid=valid.copy(), K=np.asarray(K).copy())
        return dict(**report, links=links, models=model_report, feature_height_gate=height_report,
                    evidence=('window_Gaussian_RGB_depth_raster_and_measured_track_costs'
                              if self.render_enabled else 'measured_track_and_RGB_depth_centre_errors'),
                    geometry_optimized=False, full_gaussian_raster_loss=self.render_enabled,
                    current_gaussians=len(gids), current_tracked_gaussians=int((gaussian_tracks >= 0).sum()))

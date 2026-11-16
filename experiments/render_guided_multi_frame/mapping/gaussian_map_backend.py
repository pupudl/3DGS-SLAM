"""Persistent object-local Gaussian maps behind the ownership frontend.

Object ownership is corrected from the measured observation window before
each geometry solve. Camera poses and rigid motions remain fixed measurements.
Observation IDs and persistent map IDs are independent.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from mapping.persistent_gaussian_map import PersistentGaussianMap
from mapping.gaussian_map_ownership import MapOwnershipLinks


def transform_points(xyz, transform):
    return np.asarray(xyz) @ transform[:3, :3].T + transform[:3, 3]


def rigid_transform(value):
    value = np.asarray(value, float)
    return (value.shape == (4, 4) and np.isfinite(value).all() and
            np.allclose(value[3], [0, 0, 0, 1], atol=1e-5) and
            np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-3) and
            np.linalg.det(value[:3, :3]) > 0)


class GaussianMapBackend:
    def __init__(self, cfg, output_root, map_output=None):
        self.cfg = dict(cfg.get('gaussian_map', {}))
        self.output = Path(map_output) if map_output is not None else Path(output_root) / 'gaussian_map'
        self.frames_dir = self.output / 'frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.store = PersistentGaussianMap(self.cfg)
        self.ownership = MapOwnershipLinks(self.cfg)
        self.window_size = int(self.cfg.get('window_frames', cfg.get('object_window', {}).get('window_frames', 5)))
        self.max_active = int(self.cfg.get('max_active_gaussians', 100000))
        self.max_samples = int(self.cfg.get('max_samples_per_frame', 30000))
        self.max_new = int(self.cfg.get('max_new_gaussians_per_frame', 20000))
        self.stride = int(self.cfg.get('observation_stride', 4))
        self.checkpoint_interval = int(self.cfg.get('checkpoint_interval', 5))
        if (self.window_size < 2 or min(self.max_active, self.max_samples,
                                       self.stride, self.checkpoint_interval) < 1 or self.max_new < 0):
            raise ValueError('Invalid map window, sampling, insertion or active-map budget')
        self._views = {}
        self._camera_to_world = None
        self._pose_frame = None
        self._object_to_world = {0: np.eye(4)}
        self._track_landmarks = {}
        self._updates = 0
        self._last_report = {}
        self._selection_last = {}
        self._selection_report = {}

    def retain_tracks(self, track_ids):
        keep = set(map(int, track_ids))
        self._track_landmarks = {track: value for track, value in self._track_landmarks.items() if track in keep}

    def _recover_pose(self, oid, xyz, tracks, camera_to_world):
        rows = [i for i, track in enumerate(tracks)
                if track >= 0 and self._track_landmarks.get(int(track), (-1,))[0] == oid and
                self.store.is_live(np.asarray([self._track_landmarks[int(track)][1]], np.int64))[0]]
        if len(rows) < 3:
            return None
        ids = np.asarray([self._track_landmarks[int(tracks[i])][1] for i in rows], np.int64)
        unique, select = np.unique(ids, return_index=True)
        if len(unique) < 3:
            return None
        a = self.store.select(unique, include_optimizer=False)['xyz'].astype(float)
        b = np.asarray(xyz)[np.asarray(rows)[select]].astype(float)
        ac, bc = a-a.mean(0), b-b.mean(0)
        if np.linalg.matrix_rank(ac) < 2:
            return None
        u, _, vt = np.linalg.svd(ac.T @ bc)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, b.mean(0)-rotation @ a.mean(0)
        error = np.sqrt(np.mean(np.sum((transform_points(a, pose)-b)**2, axis=1)))
        if error > float(self.cfg.get('pose_recovery_max_error_m', .3)):
            return None
        return camera_to_world @ pose

    def _poses(self, first, second, gmap, labels, tracks, object_ids, models, valid, source_seen, target_seen):
        if self._pose_frame is not None and first != self._pose_frame:
            raise ValueError('Map frame pairs must extend the previous target frame')
        lookup = {int(oid): j for j, oid in enumerate(object_ids)}
        background = lookup.get(0)
        if background is None or not valid[background] or not rigid_transform(models[background]):
            raise RuntimeError('Persistent mapping needs a currently supported background motion; '
                               'cannot safely integrate a map without camera localization')
        camera0 = np.eye(4) if self._camera_to_world is None else self._camera_to_world.copy()
        camera1 = camera0 @ np.linalg.inv(models[background])
        source_poses, target_poses = {}, {}
        skipped = []
        for oid in sorted(set(source_seen) | {0}):
            if oid == 0:
                pose = np.eye(4)
            elif oid in self._object_to_world:
                pose = self._object_to_world[oid].copy()
            elif oid not in self.store.object_ids:
                points = np.asarray(gmap['xyz'])[labels == oid]
                if not len(points):
                    continue
                pose = np.eye(4)
                pose[:3, 3] = transform_points(points, camera0).mean(0)
            else:
                owned = labels == oid
                pose = self._recover_pose(oid, np.asarray(gmap['xyz'])[owned], tracks[owned], camera0)
                if pose is None:
                    skipped.append(dict(object_id=oid, reason='no_current_pose_or_three_reliable_map_tracks'))
                    continue
            source_poses[oid] = pose
            j = lookup.get(oid)
            if oid == 0:
                target_poses[oid] = np.eye(4)
            elif oid in target_seen and j is not None and valid[j] and rigid_transform(models[j]):
                target_poses[oid] = camera1 @ models[j] @ np.linalg.inv(camera0) @ pose
        self._camera_to_world, self._pose_frame = camera1, int(second)
        self._object_to_world = target_poses
        return camera0, camera1, source_poses, target_poses, skipped

    def _sample(self, gmap, posterior, tracks, confidence):
        xyz, uv = np.asarray(gmap['xyz']), np.asarray(gmap['uv'])
        labels = np.asarray(posterior['object_labels'])
        h, w = confidence.shape
        finite = np.isfinite(uv).all(1) & np.isfinite(xyz).all(1) & (xyz[:, 2] > 0)
        xy = np.zeros_like(uv, dtype=np.int64)
        xy[finite] = np.rint(uv[finite]).astype(np.int64)
        valid = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        quality = np.zeros(len(xyz))
        quality[valid] = np.nan_to_num(confidence[xy[valid, 1], xy[valid, 0]], nan=0.)
        probability = np.asarray(posterior['object_probabilities']).max(1) if len(xyz) else np.empty(0)
        valid &= ((labels >= 0) & (quality >= float(self.cfg.get('min_confidence', .1))) &
                  (probability >= float(self.cfg.get('min_probability', .6))))
        sampled = (((xy[:, 0] % self.stride == 0) & (xy[:, 1] % self.stride == 0)) | (tracks >= 0))
        rows = np.flatnonzero(valid & sampled)
        if len(rows) > self.max_samples:
            tracked = rows[tracks[rows] >= 0][:self.max_samples]
            rest = rows[tracks[rows] < 0]
            count = self.max_samples-len(tracked)
            if count and len(rest):
                rest = rest[np.linspace(0, len(rest)-1, min(count, len(rest)), dtype=int)]
            else:
                rest = np.empty(0, np.int64)
            rows = np.sort(np.r_[tracked, rest])
        return rows, quality

    def _view(self, frame, image, depth, confidence, camera, poses, K, measured_tracks=None):
        world_to_camera = np.linalg.inv(camera)
        self._views[int(frame)] = dict(
            frame=int(frame), image=np.array(image, copy=True), depth=np.array(depth, copy=True),
            confidence=np.array(confidence, copy=True), K=np.array(K, copy=True),
            object_to_camera={oid: world_to_camera @ pose for oid, pose in poses.items()},
            measured_tracks=measured_tracks)

    def _active_batch(self, recent_ids, priority_ids=(), reserve=0):
        if self.cfg.get('active_selection', {}).get('enabled', False):
            return self._balanced_active_batch(recent_ids, priority_ids, reserve)
        candidates = []
        for oid in self.store.object_ids:
            views = [view for view in self._views.values() if oid in view['object_to_camera']]
            if not views:
                continue
            block = self.store.export_object(oid)
            visible = np.zeros(len(block['map_ids']), bool)
            for view in views:
                points = transform_points(block['xyz'], view['object_to_camera'][oid])
                z = points[:, 2]
                projected = points @ view['K'].T
                uv = projected[:, :2]/np.maximum(z[:, None], 1e-6)
                h, w = view['depth'].shape
                # A small margin retains splats whose footprint crosses the edge.
                visible |= ((z > .2) & (uv[:, 0] >= -16) & (uv[:, 0] < w+16) &
                            (uv[:, 1] >= -16) & (uv[:, 1] < h+16))
            candidates.append(block['map_ids'][visible])
        ids = np.unique(np.concatenate(candidates)) if candidates else np.empty(0, np.int64)
        total = len(ids)
        if total > self.max_active:
            priority = np.intersect1d(ids, np.asarray(recent_ids, np.int64))
            priority = priority[:self.max_active]
            others = np.setdiff1d(ids, priority)
            count = self.max_active-len(priority)
            if count:
                others = others[np.linspace(0, len(others)-1, count, dtype=int)]
                ids = np.sort(np.r_[priority, others])
            else:
                ids = priority
        batch = self.store.select(ids)
        frozen = {'retired', 'archived'}
        if self.cfg.get('freeze_when_missing', True):
            frozen.add('missing')
        batch['trainable_mask'] = np.asarray([self.store.get_status(int(oid)) not in frozen
                                               for oid in batch['object_ids']], bool)
        return batch, self._optimization_views(batch), total-len(ids)

    def _optimization_views(self, batch):
        views = []
        for saved in sorted(self._views.values(), key=lambda value: value['frame']):
            rows = np.flatnonzero(np.isin(batch['object_ids'], list(saved['object_to_camera'])))
            if not len(rows):
                continue
            transforms = np.stack([saved['object_to_camera'][int(oid)] for oid in batch['object_ids'][rows]])
            view = {key: saved[key] for key in ('frame', 'image', 'depth', 'confidence', 'K')}
            view.update(rows=rows, transforms=transforms)
            tracks = saved.get('measured_tracks')
            if tracks is not None:
                local_lookup = {int(gid): i for i, gid in enumerate(batch['map_ids'][rows])}
                selected = [i for i, gid in enumerate(tracks['map_ids']) if int(gid) in local_lookup]
                if selected:
                    view.update(track_indices=np.asarray([local_lookup[int(tracks['map_ids'][i])] for i in selected]),
                                track_uv=tracks['uv'][selected], track_xyz=tracks['xyz'][selected],
                                track_weights=tracks['weights'][selected])
            views.append(view)
        return views

    def _project_block(self, block, saved, oid):
        """Conservative per-view support and depth-aware training quality."""
        points = transform_points(block['xyz'], saved['object_to_camera'][oid])
        z = points[:, 2]
        projected = points @ saved['K'].T
        uv = projected[:, :2]/np.maximum(z[:, None], 1e-6)
        h, w = saved['depth'].shape
        radius = 3*np.exp(np.max(block['log_scales'], axis=1))*max(
            abs(saved['K'][0, 0]), abs(saved['K'][1, 1]))/np.maximum(z, .2)
        # The bound uses the largest covariance axis, so rotated elongated
        # splats crossing the image edge are retained without a fixed margin.
        radius += float(self.cfg.get('snapshot_margin_px', 1.))
        opacity = 1/(1+np.exp(-np.clip(block['opacity_logits'][:, 0], -60, 60)))
        visible = (np.isfinite(points).all(1) & np.isfinite(uv).all(1) & (z > .2) &
                   (uv[:, 0]+radius >= 0) & (uv[:, 0]-radius < w) &
                   (uv[:, 1]+radius >= 0) & (uv[:, 1]-radius < h) &
                   (opacity >= float(self.cfg.get('snapshot_min_opacity', .001))))
        xy = np.rint(np.nan_to_num(uv, nan=0., posinf=w, neginf=0.)).astype(np.int64)
        xy[:, 0], xy[:, 1] = np.clip(xy[:, 0], 0, w-1), np.clip(xy[:, 1], 0, h-1)
        measured = saved['depth'][xy[:, 1], xy[:, 0]]
        confidence = np.nan_to_num(saved['confidence'][xy[:, 1], xy[:, 0]], nan=0.)
        valid_depth = np.isfinite(measured) & (measured > .2)
        residual = np.zeros(len(z))
        residual[valid_depth] = np.abs(z[valid_depth]-measured[valid_depth])/np.maximum(
            .3, .05*measured[valid_depth])
        quality = opacity*(.05+np.clip(confidence, 0, 1))/(1+residual)
        # Keep occluders in the set, while preferring surfaces supported by
        # measured depth over many hidden points with identical screen tiles.
        quality[valid_depth & (z > measured+np.maximum(.3, .05*measured))] *= .1
        tile_size = max(1, int(self.cfg.get('active_selection', {}).get('tile_size_px', 64)))
        nx = (w+tile_size-1)//tile_size
        tile = (xy[:, 1]//tile_size)*nx+xy[:, 0]//tile_size
        return visible, tile, quality, nx*((h+tile_size-1)//tile_size)

    def _balanced_active_batch(self, recent_ids, priority_ids, reserve):
        from mapping.gaussian_map_selection import select_active_gaussians
        saved_views = sorted(self._views.values(), key=lambda value: value['frame'])
        groups = []
        for oid in self.store.object_ids:
            block = self.store.export_object(oid)
            n = len(block['map_ids'])
            tiles = np.full((n, len(saved_views)), -1, np.int64)
            quality = np.zeros(tiles.shape, np.float32)
            offset = 0
            for j, saved in enumerate(saved_views):
                h, w = saved['depth'].shape
                size = max(1, int(self.cfg.get('active_selection', {}).get('tile_size_px', 64)))
                count = ((w+size-1)//size)*((h+size-1)//size)
                if oid in saved['object_to_camera']:
                    visible, tile, scores, _ = self._project_block(block, saved, oid)
                    tiles[visible, j], quality[visible, j] = tile[visible]+offset, scores[visible]
                offset += count
            keep = np.any(tiles >= 0, axis=1)
            if np.any(keep):
                groups.append((block['map_ids'][keep], np.full(int(keep.sum()), oid, np.int64),
                               tiles[keep], quality[keep]))
        if groups:
            ids, owners, tiles, quality = [np.concatenate([group[i] for group in groups]) for i in range(4)]
        else:
            ids, owners = np.empty(0, np.int64), np.empty(0, np.int64)
            tiles, quality = np.empty((0, len(saved_views)), np.int64), np.empty((0, len(saved_views)))
        frozen_states = {'retired', 'archived'}
        if self.cfg.get('freeze_when_missing', True):
            frozen_states.add('missing')
        frozen_owners = [oid for oid in self.store.object_ids if self.store.get_status(oid) in frozen_states]
        frozen = np.isin(owners, frozen_owners)
        budget = max(1, self.max_active-int(reserve))
        rows, report = select_active_gaussians(
            ids, owners, tiles, quality, budget, update=self._updates,
            last_selected=self._selection_last, recent_ids=recent_ids,
            priority_ids=priority_ids, track_ids=[mid for _, mid in self._track_landmarks.values()],
            frozen=frozen, options=self.cfg.get('active_selection', {}))
        chosen = np.sort(ids[rows])
        batch = self.store.select(chosen)
        batch['trainable_mask'] = ~np.isin(batch['object_ids'], frozen_owners)
        self._selection_last.update({int(mid): self._updates for mid in chosen})
        # The dictionary belongs only to the current map; removed split parents
        # should not accumulate across thousands of topology updates.
        if len(self._selection_last) > len(self.store)*1.1+1024:
            old = np.asarray(list(self._selection_last), np.int64)
            self._selection_last = {int(mid): self._selection_last[int(mid)]
                                    for mid in old[self.store.is_live(old)]}
        report.update(refinement_reserved=int(reserve), max_active_gaussians=self.max_active,
                      frozen_trainable=False)
        self._selection_report = report
        return batch, self._optimization_views(batch), len(ids)-len(chosen)

    def _write_frames(self, batch, views):
        if self.cfg.get('snapshot_scope') == 'full_visible_map':
            return self._write_full_frames(batch.get('optimized_map_ids', batch['map_ids']))
        counts = {}
        for view in views:
            rows = view['rows']
            oids = np.unique(batch['object_ids'][rows])
            poses = np.stack([self._views[view['frame']]['object_to_camera'][int(oid)] for oid in oids])
            arrays = {name: batch[name][rows] for name in (
                'map_ids', 'xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits')}
            arrays.update(object_labels=batch['object_ids'][rows], object_ids=oids,
                          object_to_camera=poses, K=view['K'], frame=np.asarray(view['frame']),
                          map_revision=np.asarray(self.store.revision, np.int64))
            destination = self.frames_dir / f"frame_{view['frame']:010d}.npz"
            temporary = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
            counts[str(view['frame'])] = len(rows)
        return counts

    def _write_full_frames(self, optimized_ids):
        """Save all current CPU map splats visible in each individual frame.

        This does not allocate a full-scene GPU tensor. Frames with no current
        object support are saved empty rather than retaining an old ghost map.
        """
        counts = {}
        names = ('map_ids', 'xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits')
        for frame, saved in sorted(self._views.items()):
            blocks, labels = [], []
            for oid in self.store.object_ids:
                if oid not in saved['object_to_camera']:
                    continue
                block = self.store.export_object(oid)
                visible, _, _, _ = self._project_block(block, saved, oid)
                if np.any(visible):
                    blocks.append({name: block[name][visible] for name in names})
                    labels.append(np.full(int(visible.sum()), oid, np.int64))
            if blocks:
                arrays = {name: np.concatenate([block[name] for block in blocks]) for name in names}
                owner_labels = np.concatenate(labels)
            else:
                arrays = self.store.select(np.empty(0, np.int64), include_optimizer=False)
                arrays = {name: arrays[name] for name in names}
                owner_labels = np.empty(0, np.int64)
            oids = np.unique(owner_labels)
            poses = (np.stack([saved['object_to_camera'][int(oid)] for oid in oids])
                     if len(oids) else np.empty((0, 4, 4)))
            arrays.update(object_labels=owner_labels, object_ids=oids,
                          object_to_camera=poses, K=saved['K'], frame=np.asarray(frame),
                          map_revision=np.asarray(self.store.revision, np.int64),
                          optimized_in_last_solve=np.isin(arrays['map_ids'], optimized_ids),
                          snapshot_scope=np.asarray('full_visible_map'))
            destination = self.frames_dir / f'frame_{frame:010d}.npz'
            temporary = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **arrays)
            temporary.replace(destination)
            counts[str(frame)] = len(owner_labels)
        return counts

    def _refinement_reserve(self):
        if not self.cfg.get('active_selection', {}).get('enabled', False):
            return 0
        options = self.cfg.get('refinement', {})
        interval = int(options.get('interval', 3))
        warmup = int(options.get('warmup_pairs', options.get('warmup_updates', 2)))
        if (not options.get('enabled', False) or interval < 1 or self._updates < warmup or
                (self._updates+1) % interval):
            return 0
        requested = int(self.cfg.get('active_selection', {}).get(
            'refinement_reserve_gaussians', options.get('max_new_per_update', 1024)))
        return max(0, min(requested, max(1, self.max_active//4), self.max_active-1))

    def process(self, first, second, gmap, posterior, object_ids, models, valid, K,
                observations, registry, gaussian_tracks, presence=None,
                observation_ids=None, ownership_window=None):
        from mapping.gaussian_map_optimizer import optimize_gaussian_map

        labels = np.asarray(posterior['object_labels'])
        tracks = np.asarray(gaussian_tracks, np.int64)
        if observation_ids is not None:
            observation_ids = np.asarray(observation_ids, np.int64)
            if observation_ids.shape != labels.shape or len(np.unique(observation_ids)) != len(labels):
                raise ValueError('Map observation IDs must uniquely index source observations')
        presence = presence or {}
        # Only the ownership frontend can decide current presence. A persistent
        # surface remaining in the map never renews an object's lifecycle.
        source_seen = presence.get('source', {}).get('observed_object_ids', np.unique(labels[labels >= 0]).tolist())
        target_seen = presence.get('target', {}).get('observed_object_ids', [])
        camera0, camera1, source_poses, target_poses, skipped = self._poses(
            first, second, gmap, labels, tracks, object_ids, np.asarray(models), valid, source_seen, target_seen)
        rows, quality = self._sample(gmap, posterior, tracks, observations['source_confidence'])
        self.ownership.retain(self.store, min(self._views, default=first), ownership_window,
                              extra_map_ids=[mid for _, mid in self._track_landmarks.values()])
        hints = self.ownership.hints(observation_ids, tracks, self._track_landmarks, self.store)
        correction_posterior = dict(posterior)
        correction_labels = labels.copy()
        correction_labels[quality < float(self.cfg.get('min_confidence', .1))] = -1
        correction_posterior['object_labels'] = correction_labels
        correction = self.ownership.correct(
            self.store, first, self._updates, observation_ids, correction_posterior, hints,
            source_poses, camera0, self._views, registry, ownership_window)
        # Keep a point's track identity consistent after changing its canonical
        # owner. Geometry and measured UV/XYZ remain independent of the map.
        if correction['transferred']:
            self._refresh_track_owners()
        deferred_ownership = 0
        if self.ownership.enabled and np.any(hints[rows] >= 0):
            known = np.unique(hints[rows][hints[rows] >= 0])
            owners = dict(zip(map(int, known), map(int, self.store.select(known, include_optimizer=False)['object_ids'])))
            # A known observation cannot create a second copy in another
            # owner while its label is weak, conflicting or awaiting poses.
            wait = np.asarray([hints[row] >= 0 and owners[int(hints[row])] != labels[row] for row in rows])
            deferred_ownership = int(wait.sum())
            rows = rows[~wait]
        assigned = np.full(len(labels), -1, np.int64)
        total_limit = int(self.cfg.get('refinement', {}).get('max_total_gaussians', self.store.next_id+self.max_new))
        budget, inserted, matched, rejected = min(self.max_new, max(0, total_limit-len(self.store))), 0, 0, 0
        basis = np.asarray(gmap.get('basis', np.broadcast_to(np.eye(3), (len(labels), 3, 3))))
        scales = np.asarray(gmap['scale'])
        if scales.ndim == 1:
            scales = np.repeat(scales[:, None], 3, axis=1)
        if scales.shape[1] == 1:
            scales = np.repeat(scales, 3, axis=1)
        for oid in sorted(source_poses, key=lambda value: (value == 0, value)):
            selected = rows[labels[rows] == oid]
            if not len(selected):
                continue
            linked = np.zeros(len(selected), bool)
            if self.ownership.enabled and observation_ids is not None:
                for j, row in enumerate(selected):
                    if hints[row] >= 0 and int(hints[row]) in self.ownership.links.get(int(observation_ids[row]), ()):
                        linked[j] = True
                confirmed = selected[linked]
                if len(confirmed):
                    assigned[confirmed] = hints[confirmed]
                    self.store.record_observations(hints[confirmed], first)
                    matched += len(confirmed)
            selected = selected[~linked]
            if not len(selected):
                continue
            camera_to_object = np.linalg.inv(source_poses[oid]) @ camera0
            local_xyz = transform_points(np.asarray(gmap['xyz'])[selected], camera_to_object)
            local_basis = camera_to_object[:3, :3][None] @ basis[selected]
            quaternion_xyzw = Rotation.from_matrix(local_basis).as_quat()
            rotations = quaternion_xyzw[:, [3, 0, 1, 2]].astype(np.float32)
            integration = self.store.integrate(
                oid, local_xyz, np.asarray(gmap['color'])[selected],
                scales=scales[selected]*float(self.cfg.get('initial_scale_multiplier', self.stride)),
                rotations=rotations, frame_id=first, max_new=budget,
                existing_map_ids=hints[selected])
            assigned[selected] = integration['map_ids']
            budget -= integration['inserted']
            inserted += integration['inserted']
            matched += integration['matched']
            rejected += integration['skipped']
        for row in np.flatnonzero((assigned >= 0) & (tracks >= 0)):
            self._track_landmarks[int(tracks[row])] = (int(labels[row]), int(assigned[row]))
        self.ownership.remember(observation_ids, assigned, first)
        measured = np.flatnonzero((assigned >= 0) & (tracks >= 0))
        measurements = dict(map_ids=assigned[measured], uv=np.asarray(gmap['uv'])[measured].copy(),
                            xyz=np.asarray(gmap['xyz'])[measured].copy(), weights=quality[measured])
        self._view(first, observations['source'], observations['source_depth'], observations['source_confidence'],
                   camera0, source_poses, K, measurements)
        self._view(second, observations['target'], observations['target_depth'], observations['target_confidence'],
                   camera1, target_poses, K)
        frames = sorted(self._views)[-self.window_size:]
        self._views = {frame: self._views[frame] for frame in frames}
        for oid in self.store.object_ids:
            record = registry.objects.get(int(oid), {})
            self.store.set_status(oid, record.get('status', 'active'),
                                  last_seen_frame=record.get('last_observed_frame'))
        reserve = self._refinement_reserve()
        batch, views, deferred = self._active_batch(assigned[assigned >= 0], reserve=reserve)
        initial_selection = dict(self._selection_report)
        optimization = optimize_gaussian_map(batch, views, self.cfg)
        self.store.commit(batch['map_ids'], optimization['parameters'], optimization['optimizer_state'])
        optimized_ids = (batch['map_ids'][batch['trainable_mask']].copy()
                         if optimization['report']['iterations_completed'] else np.empty(0, np.int64))
        refinement = self._refine(batch, optimization, first)
        if refinement['changed']:
            batch, views, deferred = self._active_batch(assigned[assigned >= 0],
                                                       priority_ids=refinement.pop('_child_ids', ()))
            refinement['post_selection'] = dict(self._selection_report)
            refine_options = self.cfg.get('refinement', {})
            post_iterations = int(refine_options.get('refine_iterations', refine_options.get('post_iterations', 10)))
            if post_iterations > 0:
                post_cfg = dict(self.cfg)
                post_cfg['optimization'] = dict(self.cfg.get('optimization', {}),
                                                iterations=post_iterations,
                                                collect_refinement_stats=False)
                stages = post_cfg['optimization'].get('stages')
                if stages:
                    post_cfg['optimization']['downsample'] = stages[-1].get('downsample', 1)
                post_cfg['optimization'].pop('stages', None)
                post = optimize_gaussian_map(batch, views, post_cfg)
                self.store.commit(batch['map_ids'], post['parameters'], post['optimizer_state'])
                if post['report']['iterations_completed']:
                    optimized_ids = np.union1d(optimized_ids, batch['map_ids'][batch['trainable_mask']])
                refinement['post_optimization'] = post['report']
        learned = self.store.select(batch['map_ids'], include_optimizer=False)
        learned['optimized_map_ids'] = optimized_ids
        snapshot_counts = self._write_frames(learned, views)
        self._updates += 1
        trajectory = dict(first=int(first), second=int(second), camera_to_world_first=camera0.tolist(),
                          camera_to_world_second=camera1.tolist(),
                          object_to_world_first={str(oid): pose.tolist() for oid, pose in source_poses.items()},
                          object_to_world_second={str(oid): pose.tolist() for oid, pose in target_poses.items()})
        with (self.output / 'trajectory.jsonl').open('a') as stream:
            stream.write(json.dumps(trajectory)+'\n')
        if self._updates % self.checkpoint_interval == 0:
            self.store.save(self.output / 'map')
        self._last_report = dict(
            enabled=True, geometry_optimized=bool(optimization['report']['iterations_completed']),
            updates=self._updates, window_frames=frames, total_map_gaussians=len(self.store),
            active_map_gaussians=len(batch['map_ids']), deferred_visible_gaussians=deferred,
            sampled_observations=len(rows), inserted_gaussians=inserted,
            matched_observations=matched, insertion_budget_skipped=rejected,
            ownership_correction=correction, ownership_deferred_observations=deferred_ownership,
            ownership_links=len(self.ownership.links), map_revision=self.store.revision,
            allocated_map_gaussians=self.store.next_id, refinement=refinement,
            active_selection=initial_selection, final_active_selection=dict(self._selection_report),
            optimized_trainable_gaussians=len(optimized_ids),
            snapshot_gaussians_per_frame=snapshot_counts,
            fixed_object_ownership=True, camera_poses_optimized=False, object_poses_optimized=False,
            coordinate_system='background_world;dynamic_object_canonical',
            map_ids_distinct_from_observation_ids=True, skipped_objects=skipped,
            checkpoint='gaussian_map/map/manifest.json', snapshots='gaussian_map/frames',
            snapshot_scope=self.cfg.get('snapshot_scope', 'visible_map_within_active_budget'),
            optimization=optimization['report'])
        return self.to_dict()

    def _refresh_track_owners(self):
        ids = np.asarray(sorted({mid for _, mid in self._track_landmarks.values()}), np.int64)
        ids = ids[self.store.is_live(ids)]
        owners = dict(zip(map(int, ids), map(int, self.store.select(ids, include_optimizer=False)['object_ids'])))
        self._track_landmarks = {track: (owners[mid], mid) for track, (_, mid) in self._track_landmarks.items()
                                 if mid in owners}

    def _refine(self, batch, optimization, frame):
        options = self.cfg.get('refinement', {})
        report = dict(enabled=bool(options.get('enabled', False)), changed=False,
                      inserted=0, removed=0, split=0, cloned=0, pruned=0)
        interval = int(options.get('interval', 3))
        warmup = int(options.get('warmup_pairs', options.get('warmup_updates', 2)))
        if interval < 1 or warmup < 0:
            raise ValueError('Invalid map refinement interval or warmup')
        if not report['enabled'] or self._updates < warmup or (self._updates+1) % interval:
            return report
        from mapping.gaussian_map_refinement import plan_gaussian_refinement
        if 'refinement_stats' not in optimization:
            report['reason'] = 'no_render_gradient_statistics'
            return report
        learned = self.store.select(batch['map_ids'])
        learned['trainable_mask'] = batch['trainable_mask']
        protected = {mid for _, mid in self._track_landmarks.values()}
        learned['refinement_protected_mask'] = np.asarray([int(mid) in protected for mid in learned['map_ids']])
        plan_cfg = dict(self.cfg)
        plan_cfg['refinement'] = dict(options, max_active_gaussians=self.max_active)
        plan = plan_gaussian_refinement(learned, optimization.get('refinement_stats'), plan_cfg,
                                       total_map_count=len(self.store))
        children = plan['children']
        parents = np.asarray(children['parent_ids'], np.int64)
        child_ids = np.empty(0, np.int64)
        if len(parents):
            parameters = {name: children[name] for name in (
                'xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits')}
            child_ids = self.store.append_from_parent(parents, parameters=parameters,
                                                      frame_id=frame,
                                                      reason='adaptive_refinement')
        removed = np.unique(np.r_[plan['split_ids'], plan['prune_ids']]).astype(np.int64)
        if len(removed):
            self.store.remove(removed, frame_id=frame, reason='adaptive_refinement')
        if len(parents) or len(removed):
            self.ownership.topology(parents, child_ids, removed)
            # Sparse anchors are protected from removal by the planner. Any
            # external removal still invalidates stale links safely.
            self._refresh_track_owners()
            self.ownership.retain(self.store, min(self._views),
                                  extra_map_ids=[mid for _, mid in self._track_landmarks.values()])
        report.update(changed=bool(len(parents) or len(removed)), inserted=len(parents), removed=len(removed),
                      split=len(plan['split_ids']), cloned=len(plan['clone_ids']), pruned=len(plan['prune_ids']),
                      plan=plan['report'])
        if report['changed']:
            report['_child_ids'] = child_ids
        return report

    def finalize(self, registry):
        for oid in self.store.object_ids:
            record = registry.objects.get(int(oid), {})
            self.store.set_status(oid, record.get('status', 'active'),
                                  last_seen_frame=record.get('last_observed_frame'))
        self.store.save(self.output / 'map')
        self._last_report['finalized'] = True
        self._last_report['map'] = self.store.to_dict()
        return self.to_dict()

    def to_dict(self):
        return json.loads(json.dumps(self._last_report))

"""Render final accepted object Gaussians with the project's CUDA rasterizer.

Selection and geometry preparation are CPU-only. CUDA imports happen only for
a nonempty render. Local-motion probabilities and object probabilities never
select splats or scale their opacity here.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _empty(object_id, terminal, reason, model_valid=True, label_selected_count=0):
    return dict(xyz=np.empty((0, 3), np.float32), color=np.empty((0, 3), np.float32),
                scale=np.empty((0, 3), np.float32), basis=None, cov=None, cov6=None,
                indices=np.empty(0, np.int64), object_id=int(object_id),
                terminal=bool(terminal), model_valid=bool(model_valid), reason=reason,
                label_selected_count=int(label_selected_count))


def _scales(value, count):
    scales = np.asarray(value, np.float64)
    if scales.shape == (count,):
        scales = np.repeat(scales[:, None], 3, axis=1)
    elif scales.shape == (count, 1):
        scales = np.repeat(scales, 3, axis=1)
    if scales.shape != (count, 3):
        raise ValueError('Gaussian scale must have shape [N], [N,1] or [N,3]')
    return scales


def _covariance(gmap, count, indices):
    for key in ('cov', 'covariance', 'cov6', 'cov3D_precomp'):
        if key not in gmap:
            continue
        value = np.asarray(gmap[key], np.float64)
        if value.shape == (count, 3, 3):
            return value[indices].copy()
        if value.shape == (count, 6):
            packed = value[indices]
            result = np.zeros((len(indices), 3, 3), np.float64)
            result[:, 0, 0], result[:, 0, 1], result[:, 0, 2] = packed[:, 0], packed[:, 1], packed[:, 2]
            result[:, 1, 0], result[:, 1, 1], result[:, 1, 2] = packed[:, 1], packed[:, 3], packed[:, 4]
            result[:, 2, 0], result[:, 2, 1], result[:, 2, 2] = packed[:, 2], packed[:, 4], packed[:, 5]
            return result
        raise ValueError(f'{key} must have shape [N,3,3] or [N,6]')
    return None


def prepare_object_gaussians(gmap: dict, object_id: int, terminal: bool = False) -> dict:
    """Select saved final global labels and prepare source/terminal geometry.

    Unknown labels are excluded even when their probability argmax is strong.
    Terminal geometry uses the selected object's explicitly indexed H_object;
    missing or invalid models produce an empty render. Anisotropic Gaussians
    need a saved basis or covariance, otherwise their orientation is unknown.
    """
    object_id = int(object_id)
    if object_id < 0:
        raise ValueError('An object ID must be nonnegative; unknown is not an object')
    if 'object_labels' not in gmap:
        return _empty(object_id, terminal, 'missing_final_object_labels')
    xyz_all = np.asarray(gmap['xyz'], np.float64)
    count = len(xyz_all)
    labels = np.asarray(gmap['object_labels'], np.int64)
    if xyz_all.shape != (count, 3) or labels.shape != (count,):
        raise ValueError('xyz and final object_labels must have compatible [N,3] / [N] shapes')
    indices = np.flatnonzero(labels == object_id)
    if not len(indices):
        return _empty(object_id, terminal, 'no_accepted_gaussians')
    rotation = np.eye(3)
    translation = np.zeros(3)
    if terminal:
        columns = np.asarray(gmap.get('object_ids', []), np.int64).reshape(-1)
        found = np.flatnonzero(columns == object_id)
        if len(found) != 1 or 'H_object' not in gmap:
            return _empty(object_id, terminal, 'missing_object_motion', False, len(indices))
        column = int(found[0])
        models = np.asarray(gmap['H_object'], np.float64)
        valid = np.asarray(gmap.get('object_model_valid', np.ones(len(models), bool)), bool)
        if (column >= len(models) or column >= len(valid) or not valid[column] or
                models[column].shape != (4, 4) or not np.isfinite(models[column]).all()):
            return _empty(object_id, terminal, 'invalid_object_motion', False, len(indices))
        rotation, translation = models[column, :3, :3], models[column, :3, 3]
    color = np.asarray(gmap['color'], np.float64)[indices].copy()
    scales = _scales(gmap['scale'], count)[indices].copy()
    xyz = xyz_all[indices] @ rotation.T + translation
    basis = None
    covariance = _covariance(gmap, count, indices)
    if 'basis' in gmap:
        all_basis = np.asarray(gmap['basis'], np.float64)
        if all_basis.shape != (count, 3, 3):
            raise ValueError('Gaussian basis must have shape [N,3,3]')
        local_basis = all_basis[indices]
        basis = np.einsum('ij,njk->nik', rotation, local_basis)
        if covariance is None:
            scaled = local_basis*scales[:, None, :]
            covariance = scaled @ np.transpose(scaled, (0, 2, 1))
    if covariance is not None:
        covariance = rotation[None] @ covariance @ rotation.T[None]
        cov6 = np.stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
                         covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]), axis=1)
    else:
        if not np.allclose(scales, scales[:, :1], rtol=1e-5, atol=1e-8):
            raise ValueError('Anisotropic Gaussian scales require a saved basis or covariance')
        cov6 = None
    if (color.shape != (len(indices), 3) or not np.isfinite(color).all() or
            np.any(color < 0) or np.any(color > 1)):
        raise ValueError('Gaussian color must be finite RGB [N,3] in [0,1]')
    if (not np.isfinite(xyz).all() or not np.isfinite(scales).all() or np.any(scales <= 0) or
            (basis is not None and not np.isfinite(basis).all()) or
            (covariance is not None and not np.isfinite(covariance).all())):
        raise ValueError('Selected Gaussian geometry must be finite with positive scales')
    return dict(xyz=np.ascontiguousarray(xyz, np.float32),
                color=np.ascontiguousarray(color, np.float32),
                scale=np.ascontiguousarray(scales, np.float32),
                basis=None if basis is None else np.ascontiguousarray(basis, np.float32),
                cov=None if covariance is None else np.ascontiguousarray(covariance, np.float32),
                cov6=None if cov6 is None else np.ascontiguousarray(cov6, np.float32),
                indices=indices, object_id=object_id, terminal=bool(terminal), model_valid=True,
                reason='accepted_final_object_labels', label_selected_count=len(indices))


class GaussianObjectRenderer:
    """Actual RGB/alpha Gaussian rendering with lazy CUDA dependencies.

    ``cfg.object_render.silhouette_threshold`` defaults to 0.05. Base opacity
    comes from ``cfg.joint.gaussian_opacity`` and is identical in both passes.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self._torch = None
        self._rasterizer = None
        self._setup_camera = None
        self._cameras = {}

    def _load_cuda(self):
        if self._torch is not None:
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Actual object Gaussian rendering requires CUDA')
        from diff_gaussian_rasterization import GaussianRasterizer
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from utils.recon_helpers import setup_camera
        self._torch, self._rasterizer, self._setup_camera = torch, GaussianRasterizer, setup_camera

    def render_scene(self, sources, image_shape, cfg, object_ids=None):
        """Render requested objects with one Z buffer for all frame objects.

        Each source supplies gmap, object_ids and terminal. A disappearing
        object can use its last incoming estimate while other objects use the
        current outgoing map. Unknown source-frame Gaussians occlude but are
        never assigned a displayed object. Their terminal motion is unknown,
        so no unknown geometry is invented for the last frame.
        """
        from joint_gaussian_visibility import nearest_gaussian_owners
        prepared, object_sources = [], {}
        intrinsics = None
        unknown_count, terminal_unknown_skipped = 0, 0
        unknown_added = False
        for source in sources:
            gmap, terminal = source['gmap'], bool(source['terminal'])
            K = np.asarray(gmap['K'], np.float64)
            if intrinsics is None:
                intrinsics = K
            elif not np.allclose(K, intrinsics):
                raise ValueError('All Gaussian sources in one frame need the same intrinsics')
            for oid in source['object_ids']:
                oid = int(oid)
                if oid in object_sources:
                    raise ValueError('One frame must have exactly one geometry source per object')
                value = prepare_object_gaussians(gmap, oid, terminal)
                value['labels'] = np.full(len(value['indices']), oid, np.int64)
                value['gaussian_ids'] = np.asarray(gmap.get('gaussian_ids',
                    np.arange(len(gmap['xyz']))), np.int64)[value['indices']]
                prepared.append(value)
                object_sources[oid] = (gmap, terminal)
            unknown = np.asarray(gmap['object_labels']) < 0
            if terminal:
                terminal_unknown_skipped += int(unknown.sum())
            elif not unknown_added:
                # Reuse all geometry validation without modifying saved labels.
                temporary = dict(gmap, object_labels=np.where(unknown, 0, -1))
                value = prepare_object_gaussians(temporary, 0, terminal=False)
                value['labels'] = np.full(len(value['indices']), -1, np.int64)
                value['gaussian_ids'] = np.asarray(gmap.get('gaussian_ids',
                    np.arange(len(gmap['xyz']))), np.int64)[value['indices']]
                prepared.append(value)
                unknown_count = len(value['indices'])
                unknown_added = True
        if intrinsics is None:
            raise ValueError('Joint visibility needs at least one frame source')
        nonempty = [value for value in prepared if len(value['indices'])]
        xyz = np.concatenate([value['xyz'] for value in nonempty]) if nonempty else np.empty((0, 3))
        covariances = []
        for value in nonempty:
            if value['cov'] is None:
                covariance = np.zeros((len(value['indices']), 3, 3), np.float32)
                covariance[:, np.arange(3), np.arange(3)] = value['scale'] ** 2
            else:
                covariance = value['cov']
            covariances.append(covariance)
        cov = np.concatenate(covariances) if covariances else np.empty((0, 3, 3))
        labels = np.concatenate([value['labels'] for value in nonempty]) if nonempty else np.empty(0, np.int64)
        gids = np.concatenate([value['gaussian_ids'] for value in nonempty]) if nonempty else np.empty(0, np.int64)
        options = cfg.get('object_render', {})
        threshold = float(options.get('visibility_alpha_threshold', .05))
        visibility = nearest_gaussian_owners(
            xyz, cov, labels, gids, intrinsics, image_shape,
            opacity=float(cfg.get('joint', {}).get('gaussian_opacity', .9)),
            threshold=threshold, device=self.device,
            max_candidate_pixels=int(options.get('visibility_candidate_pixel_budget', 1_000_000)))
        visibility['metadata'].update(
            unknown_source_gaussians=unknown_count,
            terminal_unknown_gaussians_without_motion=terminal_unknown_skipped,
            unknown_policy='source_unknowns_occlude_without_color; terminal_unknowns_not_projected',
            object_pixel_counts={str(oid): int((visibility['owner'] == oid).sum())
                                 for oid in object_sources})
        requested = sorted(object_sources) if object_ids is None else list(map(int, object_ids))
        renders = {}
        independent_count = np.zeros(tuple(image_shape[:2]), np.uint16)
        visible_count = np.zeros(tuple(image_shape[:2]), np.uint16)
        for oid in sorted(object_sources):
            gmap, terminal = object_sources[oid]
            rgb, independent_mask, metadata = self.render(gmap, oid, image_shape, cfg, terminal)
            independent_count += np.asarray(independent_mask, bool)
            silhouette = visibility['owner'] == oid
            visible_count += silhouette
            rgb = rgb.copy()
            rgb[~silhouette] = 0
            metadata.update(
                independent_silhouette_pixels=int(independent_mask.sum()),
                silhouette_pixels=int(silhouette.sum()),
                visibility='joint_nearest_effective_gaussian_camera_z',
                visibility_alpha_threshold=threshold,
                rgb_mode='native_object_gaussian_rgb_clipped_to_joint_owner_mask',
                all_scene_objects_considered=True)
            if oid in requested:
                renders[oid] = (rgb, silhouette, metadata)
        visibility['metadata'].update(
            independent_object_overlap_pixels=int((independent_count > 1).sum()),
            visible_object_overlap_pixels=int((visible_count > 1).sum()),
            independent_max_objects_per_pixel=int(independent_count.max(initial=0)),
            visible_max_objects_per_pixel=int(visible_count.max(initial=0)))
        return renders, visibility

    def render(self, gmap, object_id, image_shape, cfg, terminal=False):
        prepared = prepare_object_gaussians(gmap, object_id, terminal)
        h, w = map(int, image_shape[:2])
        if h <= 0 or w <= 0:
            raise ValueError('Render dimensions must be positive')
        options = cfg.get('object_render', {})
        threshold = float(options.get('silhouette_threshold', .05))
        opacity = float(cfg.get('joint', {}).get('gaussian_opacity', .9))
        depth_threshold = float(cfg.get('joint', {}).get('render_depth_threshold_m', 1.))
        if not (0 < threshold <= 1 and 0 <= opacity <= 1 and np.isfinite(depth_threshold)):
            raise ValueError('Invalid silhouette threshold, opacity or render depth threshold')
        metadata = dict(object_id=int(object_id), terminal=bool(terminal),
                        selected_gaussians=len(prepared['indices']),
                        label_selected_gaussians=int(prepared['label_selected_count']),
                        model_valid=bool(prepared['model_valid']), reason=prepared['reason'],
                        opacity=opacity, probability_weighted_opacity=False,
                        silhouette_threshold=threshold, render_depth_threshold_m=depth_threshold,
                        silhouette_pixels=0, renderer='empty', width=w, height=h)
        if not len(prepared['indices']):
            return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), bool), metadata
        self._load_cuda()
        torch = self._torch
        K = np.asarray(gmap['K'], np.float64)
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError('Rendering requires finite 3x3 intrinsics K')
        with torch.cuda.device(self.device), torch.inference_mode():
            key = (h, w, K.tobytes(), depth_threshold)
            if key not in self._cameras:
                camera = self._setup_camera(w, h, K, np.eye(4), depth_threshold=depth_threshold)
                self._cameras[key] = self._rasterizer(raster_settings=camera)
            raster = self._cameras[key]
            tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device).contiguous()
            xyz = tensor(prepared['xyz'])
            colors = tensor(prepared['color'])
            alpha = torch.full((len(xyz), 1), opacity, dtype=torch.float32, device=self.device)
            if prepared['cov6'] is None:
                scales = tensor(prepared['scale'])
                rotations = torch.zeros((len(xyz), 4), dtype=torch.float32, device=self.device)
                rotations[:, 0] = 1.
                covariance = None
            else:
                scales, rotations = None, None
                covariance = tensor(prepared['cov6'])
            common = dict(means3D=xyz, means2D=torch.zeros_like(xyz), opacities=alpha,
                          scales=scales, rotations=rotations, cov3D_precomp=covariance, shs=None)
            rendered = raster(colors_precomp=colors, **common)[0]
            coverage = raster(colors_precomp=torch.ones_like(colors), **common)[0]
            rgb = rendered.permute(1, 2, 0).cpu().numpy()
            alpha_image = coverage[0].cpu().numpy()
        if not np.isfinite(rgb).all() or not np.isfinite(alpha_image).all():
            raise RuntimeError('Gaussian rasterizer returned non-finite RGB or alpha')
        bgr = np.rint(np.clip(rgb[:, :, ::-1], 0, 1)*255).astype(np.uint8)
        silhouette = alpha_image >= threshold
        metadata.update(renderer='cuda_diff_gaussian_rasterization',
                        silhouette_pixels=int(silhouette.sum()),
                        geometry_representation=('covariance' if prepared['cov6'] is not None else
                                                 'scale_identity_quaternion'))
        return bgr, silhouette, metadata

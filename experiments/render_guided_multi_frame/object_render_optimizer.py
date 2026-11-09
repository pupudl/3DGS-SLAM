"""Joint fixed-window object logits optimized through actual CUDA splatting.

Every active Gaussian ID indexes one shared logit vector. Images receive the
alpha-composited RGB/depth/coverage of all valid object-motion copies, with
opacity base_alpha * softmax(object_logits). Geometry and motion stay fixed.
The CUDA native depth output has no useful backward path; expected depth is
therefore reconstructed from a differentiable [z, 1, z²] *color* raster pass.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np


def render_gaussian_view(torch, raster, xyz, colors, scales, probabilities, models,
                         opacity=.5, covariance=None, pass_kind='both'):
    """Differentiable RGB/expected-depth/alpha CUDA passes for one view.

    Inputs are CUDA tensors. probabilities is [N,K], models [K,4,4]. All
    candidates participate; there is no hard ownership or probability cutoff.
    ``pass_kind`` lets the solver backward/free RGB before allocating depth's
    raster buffers. The default ``both`` is convenient for small test scenes.
    """
    if pass_kind not in ('rgb', 'depth', 'both'):
        raise ValueError('pass_kind must be rgb, depth, or both')
    k, n = len(models), len(xyz)
    transformed = (torch.einsum('kij,nj->kni', models[:, :3, :3], xyz) +
                   models[:, None, :3, 3]).reshape(k*n, 3)
    if covariance is None:
        repeated_scales = scales.repeat(k, 1)
        rotations = torch.zeros((k*n, 4), dtype=xyz.dtype, device=xyz.device)
        rotations[:, 0] = 1.
        packed_covariance = None
    else:
        rotation = models[:, :3, :3]
        cov = torch.einsum('kij,njl,kml->knim', rotation, covariance, rotation).reshape(k*n, 3, 3)
        packed_covariance = torch.stack((cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                                        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]), dim=1)
        repeated_scales, rotations = None, None
    common = dict(means3D=transformed, means2D=torch.zeros_like(transformed),
                  opacities=(float(opacity)*probabilities.T.reshape(-1, 1)).contiguous(),
                  scales=repeated_scales, rotations=rotations,
                  cov3D_precomp=packed_covariance, shs=None)
    result = dict(rgb=None, depth=None, alpha=None, radii=None)
    if pass_kind in ('rgb', 'both'):
        result['rgb'], result['radii'], _, _ = raster(colors_precomp=colors.repeat(k, 1), **common)
    if pass_kind in ('depth', 'both'):
        depth_colors = torch.stack((transformed[:, 2], torch.ones_like(transformed[:, 2]),
                                    transformed[:, 2].square()), dim=1)
        packed, result['radii'], _, _ = raster(colors_precomp=depth_colors, **common)
        result['alpha'] = packed[1]
        result['depth'] = packed[0]/packed[1].clamp_min(1e-6)
    return result


def build_render_views(window):
    """CPU-only view plan, averaged within target frame and across frames."""
    snapshots = [window._active[pair] for pair in sorted(window._active)]
    by_first = {int(snap['first']): snap for snap in snapshots}
    columns = {int(oid): j for j, oid in enumerate(window.object_ids)}
    views = []

    def observed(depth, confidence):
        depth, confidence = np.asarray(depth), np.asarray(confidence)
        return bool(np.any(np.isfinite(depth) & (depth > 0) &
                           np.isfinite(confidence) & (confidence > 0)))
    for snap in snapshots:
        observations = snap.get('render_observations')
        if observations is None:
            continue
        payload = snap['payload']
        models = np.asarray(payload.get('H_object', []), float)
        if models.shape != (len(snap['object_ids']), 4, 4):
            raise ValueError('Render snapshots require H_object aligned to snapshot object_ids')
        valid = (np.isfinite(models).all(axis=(1, 2)) &
                 np.asarray(payload.get('object_model_valid', np.ones(len(models), bool)), bool))
        retained = np.flatnonzero(valid)
        if not len(retained):
            continue
        common = dict(object_ids=np.asarray(snap['object_ids'])[retained],
                      columns=np.asarray([columns[int(oid)] for oid in np.asarray(snap['object_ids'])[retained]], np.int64),
                      K=np.asarray(payload['K'], float), transition=(int(snap['first']), int(snap['second'])))
        if observed(observations['target_depth'], observations['target_confidence']):
            views.append(dict(**common, direction='forward', frame=int(snap['second']),
                              geometry=snap, models=models[retained],
                              target=observations['target'], depth=observations['target_depth'],
                              confidence=observations['target_confidence']))
        next_source = by_first.get(int(snap['second']))
        if (next_source is not None and next_source.get('render_observations') is not None and
                observed(observations['source_depth'], observations['source_confidence'])):
            views.append(dict(**common, direction='backward', frame=int(snap['first']),
                              geometry=next_source, models=np.linalg.inv(models[retained]),
                              target=observations['source'], depth=observations['source_depth'],
                              confidence=observations['source_confidence']))
    counts = Counter(view['frame'] for view in views)
    for view in views:
        view['frame_weight'] = 1./(len(counts)*counts[view['frame']])
        view['rows'] = np.searchsorted(window._active_gids, view['geometry']['gaussian_ids'])
        if (np.any(view['rows'] >= len(window._active_gids)) or not np.array_equal(
                window._active_gids[view['rows']], view['geometry']['gaussian_ids'])):
            raise ValueError('Render Gaussian IDs must refer to the active shared parameter rows')
    return views


def _geometry(payload):
    xyz, color = np.asarray(payload['xyz'], np.float32), np.asarray(payload['color'], np.float32)
    scales = np.asarray(payload['scale'], np.float32)
    n = len(xyz)
    if scales.shape == (n,):
        scales = np.repeat(scales[:, None], 3, axis=1)
    elif scales.shape == (n, 1):
        scales = np.repeat(scales, 3, axis=1)
    if xyz.shape != (n, 3) or color.shape != (n, 3) or scales.shape != (n, 3):
        raise ValueError('Render geometry requires xyz/color/scale [N,3]')
    covariance = None
    if 'basis' in payload:
        basis = np.asarray(payload['basis'], np.float32)
        if basis.shape != (n, 3, 3):
            raise ValueError('Anisotropic Gaussian basis must have shape [N,3,3]')
        scaled = basis*scales[:, None, :]
        covariance = scaled @ np.transpose(scaled, (0, 2, 1))
    elif not np.allclose(scales, scales[:, :1], rtol=1e-5, atol=1e-8):
        raise ValueError('Anisotropic render Gaussians require their saved basis')
    if not (np.isfinite(xyz).all() and np.isfinite(color).all() and
            np.isfinite(scales).all() and np.all(scales > 0)):
        raise ValueError('Render geometry must be finite with positive scales')
    return xyz, color, scales, covariance


def _candidate_support(view, radii, downsample, options):
    """Visible depth/color fit for hard-label support, never a pixel-loss mask."""
    payload = view['geometry']['payload']
    xyz = np.asarray(payload['xyz'], float)
    models = view['models']
    moved = np.einsum('kij,nj->kni', models[:, :3, :3], xyz)+models[:, None, :3, 3]
    pixel = moved @ view['K'].T
    uv = pixel[:, :, :2]/np.maximum(pixel[:, :, 2:], 1e-8)
    h, w = view['depth'].shape
    finite = np.isfinite(uv).all(axis=2) & (moved[:, :, 2] > 0)
    xy = np.zeros_like(uv, np.int64)
    xy[finite] = np.rint(uv[finite]).astype(np.int64)
    visible = (finite & (xy[:, :, 0] >= 0) & (xy[:, :, 0] < w) &
               (xy[:, :, 1] >= 0) & (xy[:, :, 1] < h) & (radii.reshape(len(models), len(xyz)) > 0))
    rows, points = np.nonzero(visible)
    x, y = xy[rows, points, 0], xy[rows, points, 1]
    depth = np.asarray(view['depth'])[y, x]
    confidence = np.asarray(view['confidence'])[y, x]
    sigma_base = float(options.get('support_depth_sigma_m', .15))
    sigma_relative = float(options.get('support_depth_sigma_relative', .015))
    sigma = sigma_base + sigma_relative*np.maximum(depth, 0)
    target_rgb = np.asarray(view['target'])[y, x, ::-1].astype(np.float64)/255.
    source_rgb = np.asarray(payload['color'], np.float64)[points]
    color_error = np.mean(np.abs(source_rgb-target_rgb), axis=1)
    color_difference = float(options.get('support_color_difference', .25))
    keep = (np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0) &
            (np.abs(moved[rows, points, 2]-depth) <= sigma) &
            np.isfinite(color_error) & (color_error <= color_difference))
    visible[rows, points] = keep
    return visible.T


def optimize_render_window(window, cfg):
    """Optimize shared object logits using real window RGB/depth rasters.

    ``cfg`` is the object_window section; all optimizer controls live in its
    ``render`` mapping. The returned arrays have active_gids/global-object
    column order. The caller commits them and decides hard-label confidence.
    """
    options = dict(cfg.get('render', {}))
    iterations = int(options.get('iterations', 30))
    learning_rate = float(options.get('lr', options.get('learning_rate', .08)))
    downsample = int(options.get('downsample', 2))
    min_branch_probability = float(options.get('min_branch_probability', .01))
    if iterations < 1 or learning_rate <= 0 or downsample < 1:
        raise ValueError('Render iterations/lr/downsample must be positive')
    if not np.isfinite(min_branch_probability) or min_branch_probability < 0:
        raise ValueError('Render min_branch_probability must be finite and nonnegative')
    if float(options.get('support_color_difference', .25)) < 0:
        raise ValueError('Render support_color_difference must be nonnegative')
    names = ('rgb', 'depth', 'coverage', 'centre', 'prior', 'entropy')
    defaults = (1., .2, .1, .05, .005, 0.)
    weights = {name: float(options.get(name+'_weight', default)) for name, default in zip(names, defaults)}
    height_options = dict(cfg.get('height', {}))
    connectivity_options = dict(cfg.get('connectivity', {}))
    height_enabled = bool(height_options.get('enabled', False))
    connectivity_enabled = bool(connectivity_options.get('enabled', False))
    weights.update(height=float(height_options.get('weight', .15)) if height_enabled else 0.,
                   connectivity=float(connectivity_options.get('weight', .08)) if connectivity_enabled else 0.,
                   core=float(connectivity_options.get('seed_weight', .03)) if connectivity_enabled else 0.)
    names += ('height', 'connectivity', 'core')
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError('Render loss weights must be finite and nonnegative')
    initial = np.array(window._render_initial_probabilities, dtype=np.float64, copy=True)
    n, count = len(window._active_gids), len(window.object_ids)
    if initial.shape != (n, count) or np.any(~np.isfinite(initial)) or np.any(initial < 0):
        raise ValueError('Render warm start must be finite nonnegative [Nactive,Kglobal]')
    mass = initial.sum(axis=1, keepdims=True)
    initial = np.divide(initial, mass, out=np.full_like(initial, 1./max(count, 1)), where=mass > 0)
    views = build_render_views(window)
    if any(min_branch_probability >= 1./len(view['columns']) for view in views):
        raise ValueError('Render min_branch_probability must be smaller than 1 / valid model count')
    support = np.zeros((n, count), bool)
    frame_ids = sorted({view['frame'] for view in views})
    metadata_views = [dict(direction=view['direction'], frame=view['frame'],
                           transition=list(view['transition']), frame_weight=view['frame_weight'],
                           gaussian_count=len(view['rows']), object_ids=view['object_ids'].astype(int).tolist())
                      for view in views]
    report = dict(optimized_parameter='object_logits', motion_probabilities_optimized=False,
                  frames=frame_ids, views=metadata_views, iterations_completed=0,
                  nonzero_gradient_steps=0, render_passes=0, probability_changed_count=0,
                  gaussian_opacity=float(options.get('gaussian_opacity', getattr(window, 'gaussian_opacity', .5))),
                  downsample=downsample, weights=weights,
                  min_branch_probability=min_branch_probability,
                  exploration_opacity_lower_bound=(min_branch_probability*float(options.get(
                      'gaussian_opacity', getattr(window, 'gaussian_opacity', .5)))),
                  exploration_floor_changes_saved_probabilities=False,
                  hard_label_support='visible_depth_color_fit',
                  support_color_difference=float(options.get('support_color_difference', .25)),
                  source_identity_reconstruction_optimized=False,
                  source_identity_diagnostic='Not optimized or split into K coincident copies; no object-assignment information.',
                  source_identity_diagnostic_frames=sorted({int(s['first']) for s in window._active.values()}-
                                                           set(frame_ids)),
                  depth_output='differentiable_color_channels_z_and_alpha',
                  depth_loss='bounded_charbonnier', newest_frame_extra_weight=False)
    spatial = None
    if n and count and (height_enabled or connectivity_enabled):
        from object_spatial_constraints import prepare_spatial_constraints
        spatial = prepare_spatial_constraints(window, cfg)
        report['spatial_constraints'] = spatial['report']
    spatial_active = bool(spatial is not None and (
        (weights['height'] > 0 and np.any(spatial['height_cost']*spatial['height_reliability'] > 0)) or
        (spatial['connectivity_targets'] is not None and
         (np.any(spatial['connectivity_targets']['penalty'] > 0) or
          len(spatial['connectivity_targets']['seed_rows'])))))
    empty_losses = dict(total=0., **{name: 0. for name in names})
    if not n or not count or (not views and not spatial_active):
        report.update(initial_losses=empty_losses.copy(), final_losses=empty_losses.copy(),
                      reason='no_render_views', history=[])
        return dict(probabilities=initial, render_support=support, report=report)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Differentiable object-window rendering requires CUDA')
    from diff_gaussian_rasterization import GaussianRasterizer
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from utils.recon_helpers import setup_camera

    device = options.get('device', 'cuda')
    opacity = report['gaussian_opacity']
    if not 0 < opacity <= 1:
        raise ValueError('Render base opacity must be in (0,1]')
    epsilon = float(options.get('charbonnier_epsilon', .001))
    depth_sigma = float(options.get('depth_sigma_m', 1.))
    depth_cap = float(options.get('depth_error_cap', 3.))
    depth_min_alpha = float(options.get('depth_min_alpha', .01))
    if min(epsilon, depth_sigma, depth_cap) <= 0 or not 0 <= depth_min_alpha < 1:
        raise ValueError('Invalid robust render-loss configuration')
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(
        np.ascontiguousarray(value), dtype=dtype, device=device)
    with torch.cuda.device(device):
        logits = torch.nn.Parameter(tensor(np.log(np.maximum(initial, 1e-30))))
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        priors_np = np.stack([window._prior(int(gid)) for gid in window._active_gids])
        log_prior = tensor(np.log(np.maximum(priors_np, 1e-30)))
        energy_np = np.asarray(getattr(window, '_render_energy', np.zeros_like(initial)), float)
        if energy_np.shape != initial.shape or not np.isfinite(energy_np).all():
            raise ValueError('Auxiliary centre energy must be finite [Nactive,Kglobal]')
        energy = tensor(energy_np)
        ever_evidence = torch.zeros(n, dtype=torch.bool, device=device)
        if weights['centre'] > 0:
            ever_evidence |= energy.max(dim=1).values-energy.min(dim=1).values > 1e-12
        height_coefficients = tensor(np.zeros(n, np.float32) if spatial is None else
                                     spatial['height_cost']*spatial['height_reliability'])
        height_denominator = (logits.new_tensor(1.) if spatial is None else
                              tensor(spatial['height_reliability']).sum().clamp_min(1.))
        background_column = None
        if height_enabled:
            background_id = int(height_options.get('background_object_id', 0))
            if background_id not in window.object_ids:
                raise ValueError('Object height constraints require the configured background object ID')
            background_column = window.object_ids.index(background_id)
            if weights['height'] > 0:
                ever_evidence |= height_coefficients > 0
        connectivity_penalty = tensor(np.zeros((n, count), np.float32))
        connectivity_denominator = logits.new_tensor(1.)
        seed_rows = tensor(np.empty(0, np.int64), torch.long)
        seed_columns = tensor(np.empty(0, np.int64), torch.long)
        seed_targets = tensor(np.empty(0, np.float32))
        connectivity_refreshes = []

        def refresh_connectivity(probabilities, iteration):
            nonlocal connectivity_penalty, connectivity_denominator, seed_rows, seed_columns, seed_targets
            if spatial is None or spatial['connectivity_state'] is None:
                return
            from object_connectivity_constraints import update_connectivity_targets
            targets = update_connectivity_targets(spatial['connectivity_state'], probabilities)
            connectivity_penalty.copy_(tensor(targets['penalty']))
            connectivity_denominator = torch.count_nonzero(connectivity_penalty).clamp_min(1)
            seed_rows = tensor(targets['seed_rows'], torch.long)
            seed_columns = tensor(targets['seed_columns'], torch.long)
            seed_targets = tensor(targets['seed_targets'])
            if weights['connectivity'] > 0:
                ever_evidence.logical_or_((connectivity_penalty > 0).any(dim=1))
            if weights['core'] > 0 and len(seed_rows):
                ever_evidence[seed_rows] = True
            connectivity_refreshes.append(dict(iteration=iteration, **targets['report']))
            report['spatial_constraints']['connectivity']['final_targets'] = targets['report']

        if connectivity_enabled:
            refresh_connectivity(initial, 0)
        refresh_interval = int(connectivity_options.get('update_interval', 5))
        if refresh_interval < 1:
            raise ValueError('Connectivity update_interval must be positive')
        cameras = {}

        def stage(view):
            payload = view['geometry']['payload']
            xyz, color, scale, covariance = _geometry(payload)
            h, w = view['depth'].shape
            out_h, out_w = max(1, h//downsample), max(1, w//downsample)
            intrinsics = view['K'].copy()
            intrinsics[0] *= out_w/w
            intrinsics[1] *= out_h/h
            threshold = float(payload.get('render_depth_threshold_m', options.get('render_depth_threshold_m', 1.)))
            key = (out_h, out_w, intrinsics.tobytes(), threshold)
            if key not in cameras:
                cameras[key] = GaussianRasterizer(raster_settings=setup_camera(
                    out_w, out_h, intrinsics, np.eye(4), depth_threshold=threshold))
            target = cv2.cvtColor(np.asarray(view['target'], np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            target = cv2.resize(target, (out_w, out_h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(np.asarray(view['depth'], np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            confidence = cv2.resize(np.asarray(view['confidence'], np.float32), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            observed = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
            reliability = np.where(observed, np.clip(confidence, 0, 1), 0).astype(np.float32)
            return dict(xyz=tensor(xyz), colors=tensor(color), scales=tensor(scale),
                        covariance=None if covariance is None else tensor(covariance),
                        models=tensor(view['models']), rows=tensor(view['rows'], torch.long),
                        columns=tensor(view['columns'], torch.long), raster=cameras[key],
                        target=tensor(target).permute(2, 0, 1),
                        target_depth=tensor(np.nan_to_num(depth, nan=0.)),
                        confidence=tensor(reliability), observed=tensor(observed, torch.bool))

        def draw(staged, kind):
            local = logits[staged['rows']][:, staged['columns']]
            # Missing H columns never enter this conditional softmax; a single
            # valid model yields constant ones and cannot change object logits.
            probability = torch.softmax(local, dim=1)
            # This is a differentiable *render exploration* floor, not a new
            # prior or a change to the stored object probabilities. It keeps
            # a weak branch above the native raster opacity cutoff near its
            # centre while gradients still flow through its original p.
            probability = (min_branch_probability +
                           (1-len(staged['columns'])*min_branch_probability)*probability)
            return render_gaussian_view(
                torch, staged['raster'], staged['xyz'], staged['colors'], staged['scales'],
                probability, staged['models'], opacity=opacity,
                covariance=staged['covariance'], pass_kind=kind)

        def pixel_terms(staged, rendered, kind):
            denominator = staged['confidence'].sum().clamp_min(1.)
            zero = logits.sum()*0.
            values = dict(rgb=zero, depth=zero, coverage=zero)
            if kind == 'rgb':
                residual = rendered['rgb']-staged['target']
                robust = (residual.square()+epsilon**2).sqrt()-epsilon
                values['rgb'] = (robust.mean(dim=0)*staged['confidence']).sum()/denominator
            else:
                alpha = rendered['alpha']
                values['coverage'] = ((1-alpha).clamp(0, 1)*staged['confidence']).sum()/denominator
                depth_observed = staged['observed'] & (alpha.detach() > depth_min_alpha)
                depth_reliability = staged['confidence']*depth_observed
                residual = (rendered['depth']-staged['target_depth'])/depth_sigma
                charbonnier = (residual.square()+epsilon**2).sqrt()-epsilon
                bounded = depth_cap*charbonnier/(depth_cap+charbonnier)
                values['depth'] = ((bounded*depth_reliability).sum()/
                                   depth_reliability.sum().clamp_min(1.))
            return values

        def regularizers():
            p = torch.softmax(logits, dim=1)
            logp = torch.log_softmax(logits, dim=1)
            centre = (p*energy).sum(dim=1).mean()
            prior = (p*(logp-log_prior)).sum(dim=1).mean()
            entropy = -(p*logp).sum(dim=1).mean()
            height = ((1-p[:, background_column])*height_coefficients).sum()/height_denominator if (
                background_column is not None) else p.sum()*0.
            connectivity = (p*connectivity_penalty).sum()/connectivity_denominator
            core = -(seed_targets*logp[seed_rows, seed_columns]).mean() if len(seed_rows) else p.sum()*0.
            return dict(centre=centre, prior=prior, entropy=entropy,
                        height=height, connectivity=connectivity, core=core)

        def evaluate(collect_support=False):
            values = {name: 0. for name in names}
            with torch.no_grad():
                for view in views:
                    staged = stage(view)
                    for kind in ('rgb', 'depth'):
                        rendered = draw(staged, kind)
                        report['render_passes'] += 1
                        terms = pixel_terms(staged, rendered, kind)
                        for name in ('rgb', 'depth', 'coverage'):
                            values[name] += view['frame_weight']*float(terms[name])
                        if collect_support and kind == 'rgb':
                            visible = _candidate_support(view, rendered['radii'].cpu().numpy(), downsample, options)
                            support[np.ix_(view['rows'], view['columns'])] |= visible
                        del rendered, terms
                    del staged
                for name, value in regularizers().items():
                    values[name] = float(value)
            values['total'] = sum(weights[name]*values[name] for name in names)
            return values

        initial_losses = evaluate()
        initial_losses_before_refresh = initial_losses.copy()
        history = []
        for iteration in range(iterations):
            if connectivity_enabled and iteration and iteration % refresh_interval == 0:
                refresh_connectivity(torch.softmax(logits.detach(), dim=1).cpu().numpy(), iteration)
            optimizer.zero_grad(set_to_none=True)
            values = {name: 0. for name in names}
            for view in views:
                staged = stage(view)
                for kind in ('rgb', 'depth'):
                    if kind == 'rgb' and weights['rgb'] == 0:
                        continue
                    if kind == 'depth' and weights['depth'] == 0 and weights['coverage'] == 0:
                        continue
                    rendered = draw(staged, kind)
                    report['render_passes'] += 1
                    terms = pixel_terms(staged, rendered, kind)
                    objective = sum(weights[name]*terms[name] for name in ('rgb', 'depth', 'coverage'))
                    objective = view['frame_weight']*objective
                    if not torch.isfinite(objective):
                        raise RuntimeError('Non-finite object-window pixel-render loss')
                    objective.backward()
                    for name in ('rgb', 'depth', 'coverage'):
                        values[name] += view['frame_weight']*float(terms[name].detach())
                    del rendered, terms, objective
                del staged
            if logits.grad is not None:
                ever_evidence |= logits.grad.detach().abs().sum(dim=1) > 0
            terms = regularizers()
            regularization = sum(weights[name]*terms[name] for name in
                                 ('centre', 'prior', 'entropy', 'height', 'connectivity', 'core'))
            regularization.backward()
            for name, value in terms.items():
                values[name] = float(value.detach())
            del terms, regularization
            if logits.grad is None or not torch.isfinite(logits.grad).all():
                raise RuntimeError('Missing or non-finite shared object-logit gradient')
            # Unseen/unconstrained rows never get KL/entropy-only updates.
            logits.grad[~ever_evidence] = 0.
            nonzero = bool((logits.grad.abs() > 0).any())
            if nonzero:
                report['nonzero_gradient_steps'] += 1
            clip = float(options.get('gradient_clip_norm', 10.))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_([logits], clip)
            optimizer.step()
            values['total'] = sum(weights[name]*values[name] for name in names)
            history.append(dict(iteration=iteration, **values))
            report['iterations_completed'] += 1
            if options.get('verbose', True) and (iteration == 0 or (iteration+1) % 5 == 0 or iteration+1 == iterations):
                print(f'[Object render frames {frame_ids}] {iteration+1}/{iterations}: '
                      f'loss={values["total"]:.6f} (initial={initial_losses["total"]:.6f}), '
                      f'rgb={values["rgb"]:.6f}, depth={values["depth"]:.6f}, '
                      f'coverage={values["coverage"]:.6f}, height={values["height"]:.6f}, '
                      f'connectivity={values["connectivity"]:.6f}; '
                      f'nonzero_gradient_steps={report["nonzero_gradient_steps"]}', flush=True)
        if connectivity_enabled:
            probabilities_for_targets = torch.softmax(logits.detach(), dim=1).cpu().numpy()
            refresh_connectivity(probabilities_for_targets, iterations)
            # The detached graph targets change between outer iterations.
            # Compare initial/final parameters using identical FINAL targets,
            # so target refresh itself cannot count as an objective decrease.
            saved_logits = logits.detach().clone()
            with torch.no_grad():
                logits.copy_(tensor(np.log(np.maximum(initial, 1e-30))))
            initial_losses = evaluate()
            with torch.no_grad():
                logits.copy_(saved_logits)
            del saved_logits
            report.update(loss_comparison='same_final_detached_connectivity_targets',
                          initial_losses_before_target_refresh=initial_losses_before_refresh,
                          connectivity_target_refreshes=connectivity_refreshes)
        final_losses = evaluate(collect_support=True)
        probabilities = torch.softmax(logits.detach(), dim=1).cpu().numpy().astype(np.float64)
        if spatial is not None:
            if background_column is not None:
                reliability = spatial['height_reliability']
                outside = (spatial['height_cost'] > 0) & (reliability > 0)
                denominator = max(float(reliability[outside].sum()), 1.)
                report['spatial_constraints']['height'].update(
                    outside_nonbackground_mass_initial=float(np.sum(
                        (1-initial[outside, background_column])*reliability[outside])/denominator),
                    outside_nonbackground_mass_final=float(np.sum(
                        (1-probabilities[outside, background_column])*reliability[outside])/denominator))
            report['spatial_constraints']['nonzero_evidence_rows'] = int(ever_evidence.sum().item())
        report.update(initial_losses=initial_losses, final_losses=final_losses, history=history,
                      probability_changed_count=int(np.count_nonzero(
                          np.max(np.abs(probabilities-initial), axis=1) > 1e-6)),
                      nonzero_gradient_rows=int(ever_evidence.sum().item()),
                      reason='actual_window_pixel_render_optimization')
    return dict(probabilities=probabilities, render_support=support, report=report)

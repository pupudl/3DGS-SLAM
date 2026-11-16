"""Joint optimization of persistent Gaussian geometry with fixed ownership/poses.

The production renderer is the project's CUDA rasterizer. ``torch_reference``
is an explicitly selected, size-limited differentiable fixture renderer, never
an automatic fallback. Original image/depth observations are only read here.
Quaternions use wxyz order; colors are direct RGB, not spherical harmonics.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np


PARAMETER_WIDTHS = dict(xyz=3, colors=3, log_scales=3, rotations=4, opacity_logits=1)


def quaternion_matrix(quaternions):
    """Differentiable rotation matrices from normalized wxyz quaternions."""
    import torch
    q = quaternions / quaternions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), dim=-1).reshape(-1, 3, 3)


def camera_geometry(parameters, rows, transforms):
    """Transform a single owned copy of each map Gaussian to one camera."""
    import torch
    local_rotation = quaternion_matrix(parameters['rotations'][rows])
    scaled = local_rotation * parameters['log_scales'][rows].exp()[:, None, :]
    covariance = scaled @ scaled.transpose(1, 2)
    rotation = transforms[:, :3, :3]
    xyz = torch.einsum('nij,nj->ni', rotation, parameters['xyz'][rows]) + transforms[:, :3, 3]
    return xyz, rotation @ covariance @ rotation.transpose(1, 2)


class TorchReferenceRenderer:
    """Tiny analytical splatter for tests, with scene-wide depth compositing.

    This deliberately has O(N*H*W) cost. Limits prevent selecting it for a real
    sequence by accident. A 0.3 pixel covariance floor follows native splatting.
    """

    def __init__(self, max_gaussians=256, max_pixels=16384, max_elements=2000000):
        self.max_gaussians = int(max_gaussians)
        self.max_pixels = int(max_pixels)
        self.max_elements = int(max_elements)

    def __call__(self, xyz, covariance, colors, opacity, K, image_shape):
        import torch
        h, w = map(int, image_shape[:2])
        n = len(xyz)
        if n > self.max_gaussians or h*w > self.max_pixels or n*h*w > self.max_elements:
            raise ValueError('torch_reference is limited to tiny fixtures; use native CUDA for sequences')
        if not n:
            zero = xyz.sum()*0 + torch.zeros((h, w), dtype=xyz.dtype, device=xyz.device)
            return dict(rgb=zero[None].expand(3, -1, -1), depth=zero, alpha=zero)
        K = torch.as_tensor(K, device=xyz.device, dtype=xyz.dtype)
        z = xyz[:, 2].clamp_min(1e-4)
        # Native ndc2Pix maps the camera projection to pixel centers minus .5.
        u = K[0, 0]*xyz[:, 0]/z + K[0, 2] - .5
        v = K[1, 1]*xyz[:, 1]/z + K[1, 2] - .5
        zero = torch.zeros_like(z)
        jacobian = torch.stack((K[0, 0]/z, zero, -K[0, 0]*xyz[:, 0]/z.square(),
                                zero, K[1, 1]/z, -K[1, 1]*xyz[:, 1]/z.square()), -1).reshape(n, 2, 3)
        projected = jacobian @ covariance @ jacobian.transpose(1, 2)
        projected = projected + .3*torch.eye(2, dtype=xyz.dtype, device=xyz.device)[None]
        inverse = torch.linalg.inv(projected)
        yy, xx = torch.meshgrid(torch.arange(h, dtype=xyz.dtype, device=xyz.device),
                                torch.arange(w, dtype=xyz.dtype, device=xyz.device), indexing='ij')
        screen_points = torch.stack((u, v), -1)
        if screen_points.requires_grad:
            screen_points.retain_grad()
        offsets = torch.stack((xx[None]-screen_points[:, 0, None, None],
                               yy[None]-screen_points[:, 1, None, None]), -1)
        distance = torch.einsum('nhwi,nij,nhwj->nhw', offsets, inverse, offsets)
        alpha = (opacity.reshape(-1, 1, 1)*torch.exp(-.5*distance)).clamp(0., .99)
        alpha = alpha*(xyz[:, 2] > .2).to(xyz.dtype)[:, None, None]
        order = torch.argsort(xyz[:, 2].detach())
        alpha, colors, sorted_z = alpha[order], colors[order], xyz[order, 2]
        transmittance = torch.cumprod(torch.cat((torch.ones_like(alpha[:1]), 1-alpha[:-1]), 0), 0)
        weights = alpha*transmittance
        coverage = weights.sum(0)
        rgb = torch.einsum('nhw,nc->chw', weights, colors)
        depth = (weights*sorted_z[:, None, None]).sum(0)/coverage.clamp_min(1e-6)
        radii = torch.ceil(3*torch.linalg.eigvalsh(projected.detach())[:, -1].clamp_min(0).sqrt())
        visible = ((xyz[:, 2].detach() > .2) & (u.detach()+radii >= 0) &
                   (u.detach()-radii < w) & (v.detach()+radii >= 0) & (v.detach()-radii < h))
        radii = radii*visible
        return dict(rgb=rgb, depth=depth, alpha=coverage, radii=radii,
                    screen_points=screen_points, projected_centers=screen_points.detach(),
                    point_depths=xyz[:, 2].detach(), screen_gradient_units='render_pixels')


class CudaMapRenderer:
    """Differentiable native color and [z, 1, z²] passes with shared occlusion."""

    def __init__(self, device='cuda', depth_threshold=1., far=1000.):
        import torch
        if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
            raise RuntimeError('Joint Gaussian map optimization requires native CUDA; '
                               'torch_reference must be selected explicitly for tiny CPU fixtures')
        try:
            from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
        except ImportError as error:
            raise RuntimeError('Joint Gaussian map optimization requires diff_gaussian_rasterization') from error
        self.device = torch.device(device)
        self.rasterizer = GaussianRasterizer
        self.settings = GaussianRasterizationSettings
        self.depth_threshold = float(depth_threshold)
        if not np.isfinite(self.depth_threshold) or self.depth_threshold <= 0:
            raise ValueError('Native screen-gradient depth threshold must be positive')
        self.far = float(far)
        self.cameras = {}

    def __call__(self, xyz, covariance, colors, opacity, K, image_shape):
        import torch
        if not xyz.is_cuda:
            raise ValueError('Native map rasterization requires CUDA input tensors')
        # This extension allocates temporary buffers on the current CUDA
        # device. Tensor placement alone is insufficient when using cuda:1+.
        with torch.cuda.device(xyz.device):
            return self._render(xyz, covariance, colors, opacity, K, image_shape)

    def _render(self, xyz, covariance, colors, opacity, K, image_shape):
        import torch
        h, w = map(int, image_shape[:2])
        intrinsics = np.asarray(K, np.float64)
        key = (str(xyz.device), h, w, intrinsics.tobytes())
        if key not in self.cameras:
            fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
            near, far = .01, self.far
            projection = xyz.new_tensor([[2*fx/w, 0., -(w-2*cx)/w, 0.],
                                         [0., 2*fy/h, -(h-2*cy)/h, 0.],
                                         [0., 0., far/(far-near), -far*near/(far-near)],
                                         [0., 0., 1., 0.]])[None].transpose(1, 2)
            settings = self.settings(image_height=h, image_width=w, tanfovx=w/(2*fx),
                                     tanfovy=h/(2*fy), bg=xyz.new_zeros(3), scale_modifier=1.,
                                     depth_threshold=self.depth_threshold,
                                     viewmatrix=torch.eye(4, device=xyz.device)[None].transpose(1, 2),
                                     projmatrix=projection, sh_degree=0,
                                     campos=xyz.new_zeros(3), prefiltered=False)
            self.cameras[key] = self.rasterizer(raster_settings=settings)
        raster = self.cameras[key]
        cov6 = torch.stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2],
                            covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]), 1).contiguous()
        screen_points = torch.zeros_like(xyz, requires_grad=xyz.requires_grad)
        common = dict(means3D=xyz.contiguous(), means2D=screen_points,
                      opacities=opacity.contiguous(), scales=None, rotations=None,
                      cov3D_precomp=cov6, shs=None)
        rgb, radii, _, _ = raster(colors_precomp=colors.contiguous(), **common)
        # Native depth has no backward path. These color channels retain
        # derivatives through positions, covariance, compositing and opacity.
        depth_colors = torch.stack((xyz[:, 2], torch.ones_like(xyz[:, 2]), xyz[:, 2].square()), 1)
        packed, _, _, _ = raster(colors_precomp=depth_colors.contiguous(), **common)
        intrinsics_tensor = xyz.new_tensor(intrinsics)
        projected = xyz @ intrinsics_tensor.T
        centers = projected[:, :2]/projected[:, 2:].clamp_min(1e-4)-.5
        return dict(rgb=rgb, depth=packed[0]/packed[1].clamp_min(1e-6), alpha=packed[1], radii=radii,
                    screen_points=screen_points, projected_centers=centers.detach(),
                    point_depths=xyz[:, 2].detach(), screen_gradient_units='ndc',
                    # This project's Python raster backward attenuates only
                    # means2D by min(1, (camera_z/depth_threshold)^2). Undo
                    # that bookkeeping factor when comparing image evidence.
                    screen_gradient_scale=(xyz[:, 2].detach()/self.depth_threshold).square().clamp(min=1e-12, max=1.))


def _screen_gradient_original_pixels(gradient, units, shape, pixel_scale, view_weight=1., gradient_scale=None):
    """Normalize a renderer's screen derivative to one full-resolution view.

    CUDA backward.cu accumulates dL/d(mean_ndc), which is W/2 or H/2 times
    dL/d(mean_pixel). The local Python raster wrapper separately attenuates this
    surrogate gradient by camera-depth squared; gradient_scale removes it.
    Downsampled pixel derivatives then multiply low_resolution/full_resolution.
    """
    value = np.asarray(gradient, np.float64).copy()
    if units == 'ndc':
        value *= np.array([2./shape[1], 2./shape[0]])
    elif units != 'render_pixels':
        raise ValueError('Unknown renderer screen-gradient units')
    if gradient_scale is not None:
        scale = np.asarray(gradient_scale, np.float64).reshape(-1)
        if len(scale) != len(value) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Invalid renderer screen-gradient attenuation')
        value /= scale[:, None]
    if not np.isfinite(view_weight) or view_weight <= 0:
        raise ValueError('Screen-gradient view weight must be positive')
    return value*np.asarray(pixel_scale)/view_weight


def render_map_view(batch, view, cfg, renderer=None):
    """Forward-render learned map parameters, returning torch CHW RGB/depth/alpha.

    This uses the same joint compositor as optimization without downsampling.
    ``view`` needs rows, transforms, K, and ``image_shape=(H,W)`` (or an image
    or depth observation from which to read the shape). All five NumPy map
    arrays have the optimization schema. Returned tensors are detached; callers
    wanting gradients can use ``camera_geometry`` and either renderer directly.
    """
    import torch
    _, arrays = _validate_batch(batch)
    options = dict(cfg)
    options.update(cfg.get('optimization', {}))
    backend = options.get('backend', 'torch_reference' if isinstance(renderer, TorchReferenceRenderer) else 'cuda')
    if backend not in ('cuda', 'torch_reference'):
        raise ValueError('Map backend must be cuda or explicitly selected torch_reference')
    device = options.get('device', 'cpu' if backend == 'torch_reference' else 'cuda')
    if renderer is None:
        renderer = (TorchReferenceRenderer() if backend == 'torch_reference' else
                    CudaMapRenderer(device, options.get('render_depth_threshold_m', 1.), options.get('far', 1000.)))
    rows = np.asarray(view['rows'], np.int64)
    transforms = np.asarray(view['transforms'], np.float32)
    if (rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(arrays['xyz'])) or
            len(np.unique(rows)) != len(rows) or transforms.shape != (len(rows), 4, 4) or
            not np.isfinite(transforms).all()):
        raise ValueError('Render rows/transforms must index one fixed owned copy per map Gaussian')
    if 'image_shape' in view:
        shape = tuple(view['image_shape'][:2])
    elif 'depth' in view:
        shape = np.asarray(view['depth']).shape[:2]
    else:
        shape = np.asarray(view['image']).shape[:2]
    tensor = lambda value: torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device)
    with torch.no_grad():
        parameters = {name: tensor(value) for name, value in arrays.items()}
        rows = torch.as_tensor(rows, dtype=torch.long, device=device)
        xyz, covariance = camera_geometry(parameters, rows, tensor(transforms))
        rendered = renderer(xyz, covariance, parameters['colors'][rows],
                            parameters['opacity_logits'][rows].sigmoid(), view['K'], shape)
    return rendered


def _validate_batch(batch):
    ids = np.asarray(batch['map_ids'], np.int64).reshape(-1)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('Map IDs must be unique')
    parameters = {}
    for name, width in PARAMETER_WIDTHS.items():
        value = np.asarray(batch[name], np.float32)
        if value.shape != (len(ids), width) or not np.isfinite(value).all():
            raise ValueError(f'{name} must be finite [N,{width}]')
        parameters[name] = value.copy()
    if np.any(np.linalg.norm(parameters['rotations'], axis=1) < 1e-8):
        raise ValueError('Map quaternions must have nonzero norm')
    if np.any(parameters['colors'] < 0) or np.any(parameters['colors'] > 1):
        raise ValueError('Map colors must be RGB in [0,1]')
    return ids, parameters


def _aligned_state(ids, optimizer_state):
    """Load only matching map IDs, supporting changed active subsets/order."""
    state = dict(map_ids=ids.copy(), params={})
    old_ids = np.asarray((optimizer_state or {}).get('map_ids', []), np.int64)
    if old_ids.ndim != 1 or len(np.unique(old_ids)) != len(old_ids):
        raise ValueError('Optimizer state map IDs must be unique [N]')
    old_index = {int(gid): i for i, gid in enumerate(old_ids)}
    destination = np.array([i for i, gid in enumerate(ids) if int(gid) in old_index], np.int64)
    source = np.array([old_index[int(ids[i])] for i in destination], np.int64)
    for name, width in PARAMETER_WIDTHS.items():
        value = dict(exp_avg=np.zeros((len(ids), width), np.float32),
                     exp_avg_sq=np.zeros((len(ids), width), np.float32),
                     row_steps=np.zeros((len(ids), 1), np.int64))
        previous = (optimizer_state or {}).get('params', {}).get(name)
        if previous is not None:
            for key in value:
                old = np.asarray(previous[key])
                expected = (len(old_ids), 1 if key == 'row_steps' else width)
                if old.shape != expected or not np.isfinite(old).all():
                    raise ValueError(f'Invalid optimizer state {name}.{key}')
                if key in ('exp_avg_sq', 'row_steps') and np.any(old < 0):
                    raise ValueError(f'Negative optimizer state {name}.{key}')
                value[key][destination] = old[source]
        state['params'][name] = value
    return state


def sparse_track_loss(parameters, view, pixel_sigma=3., depth_sigma_m=.15,
                      depth_sigma_relative=.015, epsilon=.001):
    """Robust measured-track reprojection and optional camera-space 3D fit.

    ``view`` uses staged torch rows/transforms, track_indices indexing those
    rows, immutable measured track_uv and optional measured track_xyz. The
    pixel scale keeps residual normalization invariant to image downsampling.
    This never synthesizes a measurement from the current map prediction.
    """
    import torch
    indices = view['track_indices']
    if not len(indices):
        return parameters['xyz'].sum()*0.
    rows, transforms = view['rows'][indices], view['transforms'][indices]
    xyz = torch.einsum('nij,nj->ni', transforms[:, :3, :3], parameters['xyz'][rows]) + transforms[:, :3, 3]
    K = torch.as_tensor(view['K'], dtype=xyz.dtype, device=xyz.device)
    projected = xyz @ K.T
    uv = projected[:, :2]/projected[:, 2:].clamp_min(1e-4)
    robust = lambda difference: (difference.square()+epsilon**2).sqrt()-epsilon
    scale = torch.as_tensor(view.get('track_pixel_scale', [1., 1.]), dtype=xyz.dtype, device=xyz.device)
    residual = robust((uv-view['track_uv'])/(float(pixel_sigma)*scale)).mean(1)
    if view.get('track_xyz') is not None:
        measured = view['track_xyz']
        sigma = float(depth_sigma_m)+float(depth_sigma_relative)*measured[:, 2:3]
        residual = residual+robust((xyz-measured)/sigma).mean(1)
    reliability = view['track_weights']
    return (residual*reliability).sum()/reliability.sum().clamp_min(1e-8)


def _optimize_multiscale(batch, views, options, optimizer_state, renderer):
    """Carry parameters and row-wise Adam state through explicit image scales."""
    stages = options['stages']
    if not isinstance(stages, (list, tuple)) or not stages:
        raise ValueError('Map optimization stages must be a nonempty list')
    base = {key: value for key, value in options.items() if key not in ('stages', 'optimization')}
    current = dict(batch)
    reports, statistics = [], []
    state = optimizer_state if optimizer_state is not None else batch.get('optimizer_state')
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or 'stages' in stage or 'optimization' in stage:
            raise ValueError('Each map optimization stage must contain scalar optimization options')
        config = dict(base, **stage)
        result = optimize_gaussian_map(current, views, config, optimizer_state=state, renderer=renderer)
        current.update(result['parameters'])
        state = result['optimizer_state']
        reports.append(result['report'])
        if 'refinement_stats' in result:
            statistics.append(result['refinement_stats'])
    report = dict(reports[-1])
    report.update(stages=reports, downsample_schedule=[r['downsample'] for r in reports],
                  iterations_completed=sum(r['iterations_completed'] for r in reports),
                  initial_losses=reports[0]['initial_losses'], final_losses=reports[-1]['final_losses'],
                  loss_comparison='initial_and_final_use_their_respective_stage_image_scales',
                  history=[dict(row, stage=index) for index, r in enumerate(reports) for row in r['history']])
    report['gradients'] = {name: dict(
        max_abs=max(r['gradients'][name]['max_abs'] for r in reports),
        max_norm=max(r['gradients'][name]['max_norm'] for r in reports),
        nonzero_steps=sum(r['gradients'][name]['nonzero_steps'] for r in reports)) for name in PARAMETER_WIDTHS}
    output = dict(parameters=result['parameters'], optimizer_state=state, report=report)
    if statistics:
        combined = dict(statistics[-1])
        for key in ('screen_gradient_sum', 'screen_gradient_count', 'visible_steps'):
            combined[key] = sum((value[key] for value in statistics), np.zeros_like(statistics[0][key]))
        combined['max_radius_px'] = np.maximum.reduce([value['max_radius_px'] for value in statistics])
        output['refinement_stats'] = combined
    return output


def structural_dissimilarity(image, target, window=7, mask=None):
    """Differentiable per-pixel DSSIM, using local RGB means and covariances."""
    import torch.nn.functional as functional
    size = min(int(window), image.shape[-2], image.shape[-1])
    size = max(1, size if size % 2 else size-1)
    average = lambda value: functional.avg_pool2d(value[None], size, stride=1, padding=size//2)[0]
    if mask is None:
        pool = average
    else:
        valid = mask.to(image)[None]
        mass = average(valid).clamp_min(1e-8)
        pool = lambda value: average(value*valid)/mass
    mean_x, mean_y = pool(image), pool(target)
    var_x = (pool(image.square())-mean_x.square()).clamp_min(0)
    var_y = (pool(target.square())-mean_y.square()).clamp_min(0)
    covariance = pool(image*target)-mean_x*mean_y
    score = ((2*mean_x*mean_y+.01**2)*(2*covariance+.03**2) /
             ((mean_x.square()+mean_y.square()+.01**2)*(var_x+var_y+.03**2)))
    return ((1-score).clamp(0, 2)*.5).mean(0)


def optimize_gaussian_map(batch, views, cfg, optimizer_state=None, renderer=None):
    """Optimize shared map parameters, with fixed camera/object transforms.

    ``batch`` contains ``map_ids`` and the five PARAMETER_WIDTHS arrays in
    owner canonical coordinates. Each view contains ``rows`` [M] into batch,
    ``transforms`` [M,4,4] from owner coordinates to that camera, ``K``, uint8
    BGR ``image``, measured ``depth`` and ``confidence``. Optional immutable
    ``support_mask`` limits losses to explained regions. Optional track_indices
    index the view rows and select measured track_uv[J,2], track_weights[J],
    and optional camera-space track_xyz[J,3]. Returned CPU Adam
    state is keyed by stable map IDs and has per-row clocks: excluded map rows
    neither drift under old momentum nor decay while outside the window.
    """
    ids, initial = _validate_batch(batch)
    state = _aligned_state(ids, optimizer_state if optimizer_state is not None else batch.get('optimizer_state'))
    trainable = np.asarray(batch.get('trainable_mask', np.ones(len(ids), bool)), bool)
    if trainable.shape != (len(ids),):
        raise ValueError('Map trainable_mask must have shape [N]')
    options = dict(cfg)
    options.update(cfg.get('optimization', {}))
    if 'stages' in options:
        return _optimize_multiscale(batch, views, options, optimizer_state, renderer)
    iterations = int(options.get('iterations', 30))
    downsample = int(options.get('downsample', 2))
    if iterations < 0 or downsample < 1:
        raise ValueError('Map iterations must be nonnegative and downsample positive')
    defaults = dict(xyz=.0002, colors=.01, log_scales=.002, rotations=.001, opacity_logits=.01)
    rates = {name: float(options.get('lr_'+name, options.get('learning_rates', {}).get(name, default)))
             for name, default in defaults.items()}
    weights = {name: float(options.get(name+'_weight', default)) for name, default in
               dict(rgb=1., ssim=0., depth=.2, coverage=.01, track=.05, xyz_prior=.001, scale_prior=.001).items()}
    if any(not np.isfinite(x) or x < 0 for x in (*rates.values(), *weights.values())):
        raise ValueError('Map learning rates and loss weights must be finite and nonnegative')
    backend = options.get('backend', 'torch_reference' if isinstance(renderer, TorchReferenceRenderer) else 'cuda')
    report = dict(optimized_parameters=list(PARAMETER_WIDTHS), gaussian_count=len(ids),
                  backend=backend, iterations_completed=0,
                  frames=sorted({int(v['frame']) for v in views}),
                  ownership_optimized=False, camera_poses_optimized=False, object_poses_optimized=False,
                  depth_output='differentiable_color_channels_z_and_alpha', learning_rates=rates,
                  weights=weights, downsample=downsample, history=[],
                  reconstruction_mask='fixed_initial_alpha_and_valid_measured_depth_and_optional_support_mask',
                  track_count=0, track_3d_count=0,
                  track_loss='measured_2d_reprojection_plus_optional_measured_camera_3d',
                  gradients={name: dict(max_abs=0., max_norm=0., nonzero_steps=0) for name in PARAMETER_WIDTHS})
    if not len(ids) or not views or iterations == 0:
        report.update(reason='no_iterations' if iterations == 0 else 'no_map_or_views',
                      initial_losses={}, final_losses={})
        return dict(parameters=initial, optimizer_state=state, report=report)

    import cv2
    import torch
    if backend not in ('cuda', 'torch_reference'):
        raise ValueError('Map backend must be cuda or explicitly selected torch_reference')
    device = options.get('device', 'cpu' if backend == 'torch_reference' else 'cuda')
    if renderer is None:
        renderer = (TorchReferenceRenderer() if backend == 'torch_reference' else
                    CudaMapRenderer(device, options.get('render_depth_threshold_m', 1.), options.get('far', 1000.)))
    # Own tensor storage even on CPU: torch.as_tensor would otherwise alias
    # the initial NumPy arrays and make fixed geometry anchors move in-place.
    tensor = lambda value, dtype=torch.float32: torch.tensor(np.ascontiguousarray(value), dtype=dtype, device=device)
    parameters = {name: torch.nn.Parameter(tensor(value)) for name, value in initial.items()}
    moments = {name: {key: tensor(value, torch.long if key == 'row_steps' else torch.float32)
                      for key, value in state['params'][name].items()} for name in PARAMETER_WIDTHS}
    xyz_anchor = tensor(batch.get('initial_xyz', initial['xyz']))
    scale_anchor = tensor(batch.get('initial_log_scales', initial['log_scales']))
    if (xyz_anchor.shape != parameters['xyz'].shape or scale_anchor.shape != parameters['log_scales'].shape or
            not torch.isfinite(xyz_anchor).all() or not torch.isfinite(scale_anchor).all()):
        raise ValueError('Map initial geometry priors must match finite parameter arrays')
    epsilon = float(options.get('charbonnier_epsilon', .001))
    depth_sigma = float(options.get('depth_sigma_m', 1.))
    track_pixel_sigma = float(options.get('track_pixel_sigma', 3.))
    track_depth_sigma = float(options.get('track_depth_sigma_m', .15))
    track_depth_relative = float(options.get('track_depth_sigma_relative', .015))
    support_alpha = float(options.get('min_optimization_alpha', options.get('support_min_alpha', .01)))
    report['min_optimization_alpha'] = support_alpha
    if epsilon <= 0 or depth_sigma <= 0 or not 0 <= support_alpha < 1:
        raise ValueError('Invalid map robust-loss or support configuration')
    if (not np.isfinite([track_pixel_sigma, track_depth_sigma, track_depth_relative]).all() or
            track_pixel_sigma <= 0 or track_depth_sigma <= 0 or track_depth_relative < 0):
        raise ValueError('Invalid map sparse-track noise configuration')

    def draw(view):
        xyz, covariance = camera_geometry(parameters, view['rows'], view['transforms'])
        return renderer(xyz, covariance, parameters['colors'][view['rows']],
                        parameters['opacity_logits'][view['rows']].sigmoid(), view['K'], view['shape'])

    staged = []
    pixel_active = torch.zeros(len(ids), dtype=torch.bool, device=device)
    track_active = torch.zeros_like(pixel_active)
    pixel_enabled = any(weights[name] > 0 for name in ('rgb', 'ssim', 'depth', 'coverage'))
    collect_statistics = bool(options.get('collect_refinement_stats', False))
    refinement_stats = dict(map_ids=ids.copy(), screen_gradient_sum=np.zeros(len(ids), np.float64),
                            screen_gradient_count=np.zeros(len(ids), np.int64),
                            max_radius_px=np.zeros(len(ids), np.float32),
                            visible_steps=np.zeros(len(ids), np.int64),
                            gradient_units='loss_per_original_image_pixel', radius_units='original_image_pixels',
                            visibility='projected_center_with_measured_depth_agreement_and_fixed_support')
    for view in views:
        rows = np.asarray(view['rows'], np.int64)
        transforms = np.asarray(view['transforms'], np.float32)
        if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(ids)) or len(np.unique(rows)) != len(rows):
            raise ValueError('View rows must uniquely index active map parameters')
        if transforms.shape != (len(rows), 4, 4) or not np.isfinite(transforms).all():
            raise ValueError('View transforms must be finite [M,4,4]')
        image = np.asarray(view.get('image', view.get('target')))
        depth, confidence = np.asarray(view['depth'], np.float32), np.asarray(view['confidence'], np.float32)
        if image.ndim != 3 or image.shape[2] != 3 or depth.shape != image.shape[:2] or confidence.shape != depth.shape:
            raise ValueError('Map observations require BGR image and matching depth/confidence')
        h, w = depth.shape
        shape = (max(1, h//downsample), max(1, w//downsample))
        out_h, out_w = shape
        K = np.asarray(view['K'], np.float32).copy()
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError('Map view intrinsics must be finite [3,3] with positive focal lengths')
        K[0] *= out_w/w
        K[1] *= out_h/h
        target = cv2.resize(image[:, :, ::-1].astype(np.float32)/255., (out_w, out_h), interpolation=cv2.INTER_AREA)
        observed = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence) & (confidence > 0)
        depth = cv2.resize(np.where(observed, depth, 0), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        reliability = cv2.resize(np.where(observed, np.clip(confidence, 0, 1), 0),
                                 (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        value = dict(rows=tensor(rows, torch.long), transforms=tensor(transforms), K=K, shape=shape,
                     target=tensor(target).permute(2, 0, 1), depth=tensor(depth), reliability=tensor(reliability),
                     frame=int(view['frame']), pixel_scale=np.array([out_w/w, out_h/h], np.float32))
        indices_raw = np.asarray(view.get('track_indices', []))
        indices = indices_raw.astype(np.int64)
        track_count = len(indices) if indices.ndim == 1 else 0
        uv = np.asarray(view.get('track_uv', np.empty((0, 2))), np.float32)
        track_weights = np.asarray(view.get('track_weights', np.ones(track_count)), np.float32)
        if (indices.ndim != 1 or np.any(indices_raw != indices) or np.any(indices < 0) or np.any(indices >= len(rows)) or
                uv.shape != (track_count, 2) or not np.isfinite(uv).all() or track_weights.shape != (track_count,) or
                not np.isfinite(track_weights).all() or np.any(track_weights < 0)):
            raise ValueError('Sparse track indices, measured UV, and weights must have compatible finite shapes')
        measured_xyz = None
        if 'track_xyz' in view:
            measured_xyz = np.asarray(view['track_xyz'], np.float32)
            if (measured_xyz.shape != (track_count, 3) or not np.isfinite(measured_xyz).all() or
                    np.any(measured_xyz[:, 2] <= 0)):
                raise ValueError('Sparse track_xyz must be finite measured camera points [J,3] with positive depth')
        pixel_scale = np.array([out_w/w, out_h/h], np.float32)
        value.update(track_indices=tensor(indices, torch.long), track_uv=tensor(uv*pixel_scale),
                     track_weights=tensor(track_weights), track_pixel_scale=pixel_scale,
                     track_xyz=None if measured_xyz is None else tensor(measured_xyz))
        has_tracks = bool(np.any(track_weights > 0))
        if not len(rows):
            continue
        # Freeze the reconstruction domain to initially explained, measured
        # pixels. Coverage cannot expand sparse maps across unobserved images.
        with torch.no_grad():
            support = (draw(value)['alpha'] > support_alpha if np.any(reliability > 0) else
                       torch.zeros(shape, dtype=torch.bool, device=device))
        if 'support_mask' in view:
            mask = np.asarray(view['support_mask'], np.uint8)
            if mask.shape != (h, w):
                raise ValueError('Map support_mask must match the image')
            support &= tensor(cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0, torch.bool)
        value['reliability'] *= support
        has_pixels = bool(torch.any(value['reliability'] > 0))
        value['has_pixel_evidence'] = has_pixels and pixel_enabled
        if not has_pixels and not has_tracks:
            continue
        staged.append(value)
        if has_pixels and pixel_enabled:
            pixel_active[value['rows']] = True
        if has_tracks and weights['track'] > 0:
            track_active[value['rows'][value['track_indices'][value['track_weights'] > 0]]] = True
        report['track_count'] += int(np.count_nonzero(track_weights > 0))
        if measured_xyz is not None:
            report['track_3d_count'] += int(np.count_nonzero(track_weights > 0))
    pixel_active &= tensor(trainable, torch.bool)
    track_active &= tensor(trainable, torch.bool)
    active = pixel_active | track_active
    parameter_active = {name: active if name == 'xyz' else pixel_active for name in PARAMETER_WIDTHS}
    report['active_gaussian_count'] = int(active.sum())
    report['frozen_gaussian_count'] = int((~trainable).sum())
    report['view_count'] = len(staged)
    if not staged:
        report.update(reason='no_supported_measurements', initial_losses={}, final_losses={})
        return dict(parameters=initial, optimizer_state=state, report=report)
    if not torch.any(active):
        report.update(reason='no_trainable_gaussians', initial_losses={}, final_losses={})
        return dict(parameters=initial, optimizer_state=state, report=report)
    counts = {frame: sum(v['frame'] == frame for v in staged) for frame in {v['frame'] for v in staged}}
    for view in staged:
        view['weight'] = 1./(len(counts)*counts[view['frame']])

    def pixel_terms(view, rendered):
        reliability = view['reliability']
        denominator = reliability.sum().clamp_min(1e-8)
        robust = lambda difference: (difference.square()+epsilon**2).sqrt()-epsilon
        rgb = (robust(rendered['rgb']-view['target']).mean(0)*reliability).sum()/denominator
        # Masking low-alpha depth is detached; the differentiable ratio itself
        # keeps all geometry gradients where the surface is represented.
        depth_reliability = reliability*(rendered['alpha'].detach() > .001)
        depth = (robust((rendered['depth']-view['depth'])/depth_sigma)*depth_reliability).sum()/depth_reliability.sum().clamp_min(1e-8)
        coverage = ((1-rendered['alpha']).clamp(0, 1)*reliability).sum()/denominator
        terms = dict(rgb=rgb, depth=depth, coverage=coverage)
        if weights['ssim'] > 0:
            terms['ssim'] = (structural_dissimilarity(rendered['rgb'], view['target'],
                             options.get('ssim_window', 7), mask=reliability > 0)*reliability).sum()/denominator
        else:
            terms['ssim'] = rgb*0.
        return terms

    def priors():
        return dict(xyz_prior=(parameters['xyz'][active]-xyz_anchor[active]).square().mean(),
                    scale_prior=((parameters['log_scales'][pixel_active]-scale_anchor[pixel_active]).square().mean()
                                 if torch.any(pixel_active) else parameters['log_scales'].sum()*0.))

    def track_term(view):
        return sparse_track_loss(parameters, view, track_pixel_sigma, track_depth_sigma, track_depth_relative, epsilon)

    def evaluate():
        values = {name: 0. for name in weights}
        with torch.no_grad():
            for view in staged:
                for name, value in pixel_terms(view, draw(view)).items():
                    values[name] += view['weight']*float(value)
                values['track'] += view['weight']*float(track_term(view))
            for name, value in priors().items():
                values[name] = float(value)
        values['total'] = sum(weights[name]*values[name] for name in weights)
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError('Non-finite joint Gaussian map loss')
        return values

    beta1, beta2 = .9, .999
    adam_epsilon = float(options.get('adam_epsilon', 1e-8))
    min_scale, max_scale = float(options.get('min_scale_m', .002)), float(options.get('max_scale_m', 5.))
    if adam_epsilon <= 0 or min_scale <= 0 or max_scale < min_scale:
        raise ValueError('Invalid map Adam epsilon or scale limits')
    report['initial_losses'] = evaluate()
    for iteration in range(iterations):
        for value in parameters.values():
            value.grad = None
        losses = {name: 0. for name in weights}
        # Free each view's raster graph before staging the next graph.
        for view in staged:
            rendered = draw(view)
            terms = pixel_terms(view, rendered)
            terms['track'] = track_term(view)
            objective = view['weight']*sum(weights[name]*value for name, value in terms.items())
            if not torch.isfinite(objective):
                raise RuntimeError('Non-finite joint Gaussian map pixel loss')
            with torch.cuda.device(device) if parameters['xyz'].is_cuda else nullcontext():
                objective.backward()
            if collect_statistics and view['has_pixel_evidence']:
                rows_numpy = view['rows'].detach().cpu().numpy()
                radii = rendered.get('radii')
                points = rendered.get('screen_points')
                if radii is not None:
                    radius = radii.detach().cpu().numpy().astype(np.float32).reshape(-1)
                    visible = (radius > 0) & trainable[rows_numpy]
                    centers = rendered.get('projected_centers')
                    point_depths = rendered.get('point_depths')
                    if centers is None or point_depths is None:
                        visible[:] = False
                    else:
                        centers = centers.detach().cpu().numpy()
                        pixels = np.rint(centers).astype(np.int64)
                        in_image = ((pixels[:, 0] >= 0) & (pixels[:, 0] < view['shape'][1]) &
                                    (pixels[:, 1] >= 0) & (pixels[:, 1] < view['shape'][0]))
                        px = np.clip(pixels[:, 0], 0, view['shape'][1]-1)
                        py = np.clip(pixels[:, 1], 0, view['shape'][0]-1)
                        reliable = view['reliability'].detach().cpu().numpy()[py, px] > 0
                        measured = view['depth'].detach().cpu().numpy()[py, px]
                        projected_depth = point_depths.detach().cpu().numpy()
                        tolerance = float(options.get('visibility_depth_sigma_m', 1.)) + float(
                            options.get('visibility_depth_relative', .05))*measured
                        visible &= in_image & reliable & (np.abs(projected_depth-measured) <= tolerance)
                    original_radius = radius/float(np.min(view['pixel_scale']))
                    refinement_stats['max_radius_px'][rows_numpy] = np.maximum(
                        refinement_stats['max_radius_px'][rows_numpy], np.where(visible, original_radius, 0))
                    refinement_stats['visible_steps'][rows_numpy[visible]] += 1
                    if points is not None and points.grad is not None:
                        gradient = points.grad[:, :2].detach().cpu().numpy()
                        attenuation = rendered.get('screen_gradient_scale')
                        attenuation = None if attenuation is None else attenuation.detach().cpu().numpy()
                        gradient = _screen_gradient_original_pixels(gradient,
                            rendered.get('screen_gradient_units'), view['shape'], view['pixel_scale'],
                            view['weight'], attenuation)
                        norms = np.linalg.norm(gradient, axis=1)
                        valid = visible & np.isfinite(norms)
                        refinement_stats['screen_gradient_sum'][rows_numpy[valid]] += norms[valid]
                        refinement_stats['screen_gradient_count'][rows_numpy[valid]] += 1
            for name, value in terms.items():
                losses[name] += view['weight']*float(value.detach())
            del objective, terms, rendered
        terms = priors()
        regularizer = sum(weights[name]*value for name, value in terms.items())
        with torch.cuda.device(device) if parameters['xyz'].is_cuda else nullcontext():
            regularizer.backward()
        for name, value in terms.items():
            losses[name] = float(value.detach())
        for name, value in parameters.items():
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise RuntimeError(f'Missing or non-finite joint map gradient for {name}')
            value.grad[~parameter_active[name]] = 0
            gradient_report = report['gradients'][name]
            maximum = float(value.grad.abs().max())
            gradient_report['max_abs'] = max(gradient_report['max_abs'], maximum)
            gradient_report['max_norm'] = max(gradient_report['max_norm'], float(value.grad.norm()))
            gradient_report['nonzero_steps'] += int(maximum > 0)
        clip = float(options.get('gradient_clip_norm', 10.))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), clip)
        with torch.no_grad():
            for name, value in parameters.items():
                mask = parameter_active[name]
                if rates[name] == 0 or not torch.any(mask):
                    continue
                moment = moments[name]
                gradient = value.grad[mask]
                first = beta1*moment['exp_avg'][mask] + (1-beta1)*gradient
                second = beta2*moment['exp_avg_sq'][mask] + (1-beta2)*gradient.square()
                steps = moment['row_steps'][mask]+1
                moment['exp_avg'][mask] = first
                moment['exp_avg_sq'][mask] = second
                moment['row_steps'][mask] = steps
                corrected_first = first/(1-beta1**steps.to(first.dtype))
                corrected_second = second/(1-beta2**steps.to(second.dtype))
                value[mask] -= rates[name]*corrected_first/(corrected_second.sqrt()+adam_epsilon)
            if rates['colors'] > 0:
                parameters['colors'][pixel_active] = parameters['colors'][pixel_active].clamp(0, 1)
            if rates['log_scales'] > 0:
                parameters['log_scales'][pixel_active] = parameters['log_scales'][pixel_active].clamp(np.log(min_scale), np.log(max_scale))
            if rates['opacity_logits'] > 0:
                parameters['opacity_logits'][pixel_active] = parameters['opacity_logits'][pixel_active].clamp(-12, 12)
            if rates['rotations'] > 0:
                rotations = parameters['rotations'][pixel_active]
                parameters['rotations'][pixel_active] = rotations/rotations.norm(dim=1, keepdim=True).clamp_min(1e-12)
        losses['total'] = sum(weights[name]*losses[name] for name in weights)
        report['history'].append(dict(iteration=iteration, **losses))
        report['iterations_completed'] += 1
        if options.get('verbose', False) and (iteration == 0 or (iteration+1) % 10 == 0 or iteration+1 == iterations):
            print(f"[Gaussian map {report['frames']}] {iteration+1}/{iterations}: loss={losses['total']:.6f}", flush=True)
    report.update(reason='optimized', final_losses=evaluate())
    result = {name: value.detach().cpu().numpy().copy() for name, value in parameters.items()}
    state['params'] = {name: {key: value.detach().cpu().numpy().copy() for key, value in moment.items()}
                       for name, moment in moments.items()}
    output = dict(parameters=result, optimizer_state=state, report=report)
    if collect_statistics:
        output['refinement_stats'] = refinement_stats
    return output

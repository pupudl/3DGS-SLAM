import argparse
import os
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from copy import deepcopy
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette, quat_mult


META_KEYS = {
    'org_width',
    'org_height',
    'w2c',
    'intrinsics',
    'gt_w2c_all_frames',
    'cam_unnorm_rots',
    'cam_trans',
    'keyframe_time_indices',
}
STATIC_GAUSSIAN_KEYS = {
    'means3D',
    'rgb_colors',
    'unnorm_rotations',
    'logit_opacities',
    'log_scales',
    'timestep',
}
DYNAMIC_REQUIRED_KEYS = {
    'dyn_means3D_canon',
    'dyn_rgb_colors',
    'dyn_unnorm_rotations',
    'dyn_logit_opacities',
    'dyn_log_scales',
    'dyn_obj_ids',
    'dyn_birth_time',
    'dyn_obj_unnorm_rots',
    'dyn_obj_trans',
    'dyn_obj_visible',
}


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device=q.device, dtype=q.dtype)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def resolve_torch_device(cfg):
    requested = str(cfg.get('device', 'cuda:0'))
    if requested.startswith('cuda') and not torch.cuda.is_available():
        if cfg.get('render_mode') == 'centers':
            print("CUDA is unavailable; using CPU for centers-mode visualization.")
            return torch.device('cpu')
        raise RuntimeError(
            "CUDA is required for render_mode='color' or render_mode='depth' because "
            "diff_gaussian_rasterization runs on GPU. Use --render-mode centers or run on a CUDA machine."
        )
    return torch.device(requested)


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, device):
    # Load Scene Data
    raw_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {}
    for key, value in raw_params.items():
        if isinstance(value, np.ndarray) and value.dtype == object:
            all_params[key] = value
        else:
            all_params[key] = torch.as_tensor(value, device=device).float()
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4, device=device).float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.as_tensor(all_params[k], device=device).float()
        else:
            params[k] = all_params[k].to(device=device).float()

    return params, all_w2cs


def get_rendervars(params, w2c, curr_timestep):
    params_timesteps = params['timestep']
    selected_params_idx = params_timesteps <= curr_timestep
    keys = [k for k in STATIC_GAUSSIAN_KEYS if k in params]
    selected_params = {}
    for k in keys:
        selected_params[k] = params[k][selected_params_idx]
    if selected_params['log_scales'].shape[-1]  == 1:
        log_scales = torch.tile(selected_params['log_scales'], (1, 3))
    else:
        log_scales = selected_params['log_scales']
    device = selected_params['means3D'].device
    w2c = torch.as_tensor(w2c, device=device).float()
    rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': selected_params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'])
    }
    depth_rendervar = {
        'means3D': selected_params['means3D'],
        'colors_precomp': get_depth_and_silhouette(selected_params['means3D'], w2c),
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(selected_params['means3D'])
    }
    return rendervar, depth_rendervar


def has_dynamic_gaussians(params):
    return (
        DYNAMIC_REQUIRED_KEYS.issubset(params.keys())
        and params['dyn_means3D_canon'].numel() > 0
        and params['dyn_obj_trans'].numel() > 0
    )


def get_dynamic_rendervars(params, w2c, curr_timestep, cfg):
    if not has_dynamic_gaussians(params):
        return None, None, 0
    if curr_timestep >= params['dyn_obj_visible'].shape[-1]:
        return None, None, 0

    obj_ids_all = params['dyn_obj_ids'].long()
    birth = params['dyn_birth_time'].long()
    visible = params['dyn_obj_visible'].bool()
    active = (birth <= int(curr_timestep)) & visible[obj_ids_all, int(curr_timestep)]
    if not bool(active.any()):
        return None, None, 0

    obj_ids = obj_ids_all[active]
    local_pts = params['dyn_means3D_canon'][active].detach()
    local_rots = params['dyn_unnorm_rotations'][active].detach()
    obj_rots = F.normalize(params['dyn_obj_unnorm_rots'][obj_ids, :, int(curr_timestep)].detach())
    obj_trans = params['dyn_obj_trans'][obj_ids, :, int(curr_timestep)].detach()

    obj_rot_mats = build_rotation(obj_rots)
    world_pts = torch.bmm(obj_rot_mats, local_pts.unsqueeze(-1)).squeeze(-1) + obj_trans
    world_rots = quat_mult(obj_rots, F.normalize(local_rots))

    log_scales = params['dyn_log_scales'][active]
    if log_scales.shape[-1] == 1:
        log_scales = torch.tile(log_scales, (1, 3))

    if cfg.get('highlight_dynamic', False):
        dyn_color = torch.tensor([1.0, 0.22, 0.05], device=world_pts.device).float()
        colors = dyn_color.expand(world_pts.shape[0], 3)
    else:
        colors = params['dyn_rgb_colors'][active]

    w2c = torch.as_tensor(w2c, device=world_pts.device).float()
    rendervar = {
        'means3D': world_pts,
        'colors_precomp': colors,
        'rotations': F.normalize(world_rots),
        'opacities': torch.sigmoid(params['dyn_logit_opacities'][active]),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(world_pts),
    }
    depth_rendervar = {
        'means3D': world_pts,
        'colors_precomp': get_depth_and_silhouette(world_pts, w2c),
        'rotations': F.normalize(world_rots),
        'opacities': torch.sigmoid(params['dyn_logit_opacities'][active]),
        'scales': torch.exp(log_scales),
        'means2D': torch.zeros_like(world_pts),
    }
    return rendervar, depth_rendervar, int(active.sum().item())


def merge_rendervars(static_rendervar, dynamic_rendervar):
    if dynamic_rendervar is None:
        return static_rendervar
    return {
        key: torch.cat((static_rendervar[key], dynamic_rendervar[key]), dim=0)
        for key in static_rendervar.keys()
    }


def get_scene_rendervars(params, w2c, curr_timestep, cfg):
    static_rendervar, static_depth_rendervar = get_rendervars(params, w2c, curr_timestep)
    dynamic_rendervar, dynamic_depth_rendervar, dynamic_count = get_dynamic_rendervars(
        params,
        w2c,
        curr_timestep,
        cfg,
    )
    return (
        merge_rendervars(static_rendervar, dynamic_rendervar),
        merge_rendervars(static_depth_rendervar, dynamic_depth_rendervar),
        dynamic_count,
    )


def get_dynamic_centers(params, curr_timestep, cfg):
    if not has_dynamic_gaussians(params):
        return None, None, 0
    if curr_timestep >= params['dyn_obj_visible'].shape[-1]:
        return None, None, 0

    obj_ids_all = params['dyn_obj_ids'].long()
    birth = params['dyn_birth_time'].long()
    visible = params['dyn_obj_visible'].bool()
    active = (birth <= int(curr_timestep)) & visible[obj_ids_all, int(curr_timestep)]
    if not bool(active.any()):
        return None, None, 0

    obj_ids = obj_ids_all[active]
    local_pts = params['dyn_means3D_canon'][active].detach()
    obj_rots = F.normalize(params['dyn_obj_unnorm_rots'][obj_ids, :, int(curr_timestep)].detach())
    obj_trans = params['dyn_obj_trans'][obj_ids, :, int(curr_timestep)].detach()
    obj_rot_mats = build_rotation(obj_rots)
    world_pts = torch.bmm(obj_rot_mats, local_pts.unsqueeze(-1)).squeeze(-1) + obj_trans

    if cfg.get('highlight_dynamic', False):
        dyn_color = torch.tensor([1.0, 0.22, 0.05], device=world_pts.device).float()
        colors = dyn_color.expand(world_pts.shape[0], 3)
    else:
        colors = params['dyn_rgb_colors'][active]
    return world_pts, colors, int(active.sum().item())


def get_scene_centers(params, curr_timestep, cfg):
    selected = params['timestep'] <= curr_timestep
    means = params['means3D'][selected]
    colors = params['rgb_colors'][selected]
    dynamic_means, dynamic_colors, dynamic_count = get_dynamic_centers(params, curr_timestep, cfg)
    if dynamic_means is not None:
        means = torch.cat((means, dynamic_means), dim=0)
        colors = torch.cat((colors, dynamic_colors), dim=0)
    return means, colors, dynamic_count


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg):
    with torch.no_grad():
        depth_threshold = cfg.get('depth_threshold', 105.50137862302245)
        cam = setup_camera(
            cfg['viz_w'],
            cfg['viz_h'],
            k,
            w2c,
            cfg['viz_near'],
            cfg['viz_far'],
            depth_threshold=depth_threshold,
        )
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            depth_threshold=depth_threshold,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, _ = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        depth_sil, _, _, _ = Renderer(raster_settings=cam)(**timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())
    
    # Colorize point cloud
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def annotate_frame(frame_rgb, timestep, num_timesteps, dynamic_count, cfg):
    if cfg.get('no_time_label', False):
        return frame_rgb
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame_bgr.shape[:2]
    label = f"t={timestep:04d}/{num_timesteps - 1:04d} | dynamic GS: {dynamic_count}"
    cv2.rectangle(frame_bgr, (0, 0), (min(w, 470), 38), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    bar_width = int(w * float(timestep + 1) / max(1, num_timesteps))
    cv2.rectangle(frame_bgr, (0, h - 8), (w, h), (20, 20, 20), -1)
    cv2.rectangle(frame_bgr, (0, h - 8), (bar_width, h), (0, 185, 255), -1)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def capture_frame(vis, output_path, timestep, num_timesteps, dynamic_count, cfg):
    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    frame = annotate_frame(frame, timestep, num_timesteps, dynamic_count, cfg)
    cv2.imwrite(output_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def write_video_from_frames(frame_paths, output_path, fps):
    if not frame_paths:
        return
    first = cv2.imread(frame_paths[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        writer.write(frame)
    writer.release()


def write_gif_from_frames(frame_paths, output_path, fps):
    if not frame_paths:
        return
    duration = 1.0 / max(1, fps)
    frames = [imageio.imread(frame_path) for frame_path in frame_paths]
    imageio.mimsave(output_path, frames, duration=duration)


def select_export_timesteps(num_timesteps, frame_stride, max_frames):
    stride = max(1, int(frame_stride))
    timesteps = list(range(0, num_timesteps, stride))
    if timesteps[-1] != num_timesteps - 1:
        timesteps.append(num_timesteps - 1)
    if max_frames and max_frames > 0:
        timesteps = timesteps[:max_frames]
    return timesteps


def subsample_tensor_rows(values, max_rows):
    if max_rows <= 0 or values.shape[0] <= max_rows:
        return values
    step = int(np.ceil(values.shape[0] / float(max_rows)))
    return values[::step][:max_rows]


def render_centers_projection(means, colors, view_w2c, k, cfg):
    width = int(cfg['viz_w'])
    height = int(cfg['viz_h'])
    near = float(cfg.get('viz_near', 0.01))
    far = float(cfg.get('viz_far', 100.0))

    points = means.detach().float().cpu().numpy()
    rgb = np.clip(colors.detach().float().cpu().numpy(), 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    points_h = np.concatenate((points, np.ones((points.shape[0], 1), dtype=points.dtype)), axis=1)
    cam_points = (view_w2c @ points_h.T).T[:, :3]
    z = cam_points[:, 2]
    valid = (z > near) & (z < far)
    if not np.any(valid):
        return np.full((height, width, 3), 255, dtype=np.uint8)

    cam_points = cam_points[valid]
    rgb = rgb[valid]
    z = z[valid]
    px = (k[0, 0] * cam_points[:, 0] / z + k[0, 2]).astype(np.int32)
    py = (k[1, 1] * cam_points[:, 1] / z + k[1, 2]).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(inside):
        return np.full((height, width, 3), 255, dtype=np.uint8)

    px = px[inside]
    py = py[inside]
    rgb = rgb[inside]
    z = z[inside]

    image = np.full((height, width, 3), 255, dtype=np.uint8)
    order = np.argsort(z)[::-1]
    image[py[order], px[order]] = rgb[order]

    point_size = int(cfg.get('headless_point_size', 1))
    if point_size > 1:
        kernel = np.ones((point_size, point_size), dtype=np.uint8)
        mask = np.any(image != 255, axis=2).astype(np.uint8)
        grown_mask = cv2.dilate(mask, kernel)
        image = cv2.dilate(image, kernel)
        image[grown_mask == 0] = 255
    return image


def export_centers_headless(params, all_w2cs, first_frame_w2c, k, cfg, export_cfg):
    output_dir = export_cfg['output_dir']
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    first_view_w2c = first_frame_w2c.copy()
    if cfg.get('offset_first_viz_cam', True):
        first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])

    frame_paths = []
    timesteps = select_export_timesteps(
        len(all_w2cs),
        export_cfg.get('frame_stride', 1),
        export_cfg.get('max_frames', 0),
    )
    max_static_points = int(cfg.get('headless_max_static_points', 250000))
    max_dynamic_points = int(cfg.get('headless_max_dynamic_points', 100000))

    for export_idx, curr_timestep in enumerate(timesteps):
        selected = params['timestep'] <= curr_timestep
        static_means = subsample_tensor_rows(params['means3D'][selected], max_static_points)
        static_colors = subsample_tensor_rows(params['rgb_colors'][selected], max_static_points)

        dynamic_means, dynamic_colors, dynamic_count = get_dynamic_centers(params, curr_timestep, cfg)
        if dynamic_means is not None:
            dynamic_means = subsample_tensor_rows(dynamic_means, max_dynamic_points)
            dynamic_colors = subsample_tensor_rows(dynamic_colors, max_dynamic_points)
            means = torch.cat((static_means, dynamic_means), dim=0)
            colors = torch.cat((static_colors, dynamic_colors), dim=0)
        else:
            means = static_means
            colors = static_colors

        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        frame = render_centers_projection(means, colors, view_w2c, k, cfg)
        frame = annotate_frame(frame, curr_timestep, len(all_w2cs), dynamic_count, cfg)
        frame_path = os.path.join(frames_dir, f"{export_idx:04d}_t{curr_timestep:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
        print(f"Saved frame {export_idx + 1}/{len(timesteps)}: t={curr_timestep}")

    fps = int(export_cfg.get('fps', cfg.get('viz_fps', 5)))
    if export_cfg.get('write_video', False):
        video_path = os.path.join(output_dir, 'mapping_4dgs_online.mp4')
        write_video_from_frames(frame_paths, video_path, fps)
        print(f"Saved video: {video_path}")
    if export_cfg.get('write_gif', False):
        gif_path = os.path.join(output_dir, 'mapping_4dgs_online.gif')
        write_gif_from_frames(frame_paths, gif_path, fps)
        print(f"Saved GIF: {gif_path}")
    print(f"Saved frames: {frames_dir}")


def render_tensor_to_rgb_frame(image, depth, sil, cfg):
    if cfg.get('show_sil', False):
        image = (1 - sil).repeat(3, 1, 1)

    if cfg['render_mode'] == 'depth':
        depth_np = depth[0].detach().cpu().numpy()
        valid = depth_np > 0
        valid_values = depth_np[valid]
        vmax = float(np.percentile(valid_values, 98)) if valid_values.size else 1.0
        normalized = np.clip(depth_np / max(vmax, 1e-6), 0.0, 1.0)
        frame_bgr = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        frame_bgr[~valid] = 255
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    frame = torch.clamp(image, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
    return (frame * 255.0).astype(np.uint8)


def export_gaussian_headless(params, all_w2cs, first_frame_w2c, k, cfg, export_cfg):
    output_dir = export_cfg['output_dir']
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    first_view_w2c = first_frame_w2c.copy()
    if cfg.get('offset_first_viz_cam', True):
        first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])

    frame_paths = []
    timesteps = select_export_timesteps(
        len(all_w2cs),
        export_cfg.get('frame_stride', 1),
        export_cfg.get('max_frames', 0),
    )
    for export_idx, curr_timestep in enumerate(timesteps):
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        scene_data, scene_depth_data, dynamic_count = get_scene_rendervars(
            params,
            view_w2c,
            curr_timestep=curr_timestep,
            cfg=cfg,
        )
        image, depth, sil = render(view_w2c, k, scene_data, scene_depth_data, cfg)
        frame = render_tensor_to_rgb_frame(image, depth, sil, cfg)
        frame = annotate_frame(frame, curr_timestep, len(all_w2cs), dynamic_count, cfg)
        frame_path = os.path.join(frames_dir, f"{export_idx:04d}_t{curr_timestep:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_paths.append(frame_path)
        print(f"Saved Gaussian render frame {export_idx + 1}/{len(timesteps)}: t={curr_timestep}")

    fps = int(export_cfg.get('fps', cfg.get('viz_fps', 5)))
    if export_cfg.get('write_video', False):
        video_path = os.path.join(output_dir, 'mapping_4dgs_online.mp4')
        write_video_from_frames(frame_paths, video_path, fps)
        print(f"Saved video: {video_path}")
    if export_cfg.get('write_gif', False):
        gif_path = os.path.join(output_dir, 'mapping_4dgs_online.gif')
        write_gif_from_frames(frame_paths, gif_path, fps)
        print(f"Saved GIF: {gif_path}")
    print(f"Saved frames: {frames_dir}")


def visualize(scene_path, cfg, export_cfg=None):
    export_cfg = export_cfg or {}
    exporting = export_cfg.get('enabled', False)
    # Load Scene Data
    device = resolve_torch_device(cfg)
    first_frame_w2c, k = load_camera(cfg, scene_path)

    params, all_w2cs = load_scene_data(scene_path, device)
    print(f"Static Gaussians: {params['means3D'].shape[0]}")
    if has_dynamic_gaussians(params):
        print(
            "Dynamic 4DGS:",
            int(params['dyn_means3D_canon'].shape[0]),
            "gaussians,",
            int(params['dyn_obj_trans'].shape[0]),
            "objects",
        )
    else:
        print("Dynamic 4DGS: no dynamic gaussians found in params.npz")
    if exporting and cfg['render_mode'] == 'centers' and export_cfg.get('headless_centers', False):
        export_centers_headless(params, all_w2cs, first_frame_w2c, k, cfg, export_cfg)
        return
    if exporting and cfg['render_mode'] in ['color', 'depth'] and export_cfg.get('headless_render', True):
        export_gaussian_headless(params, all_w2cs, first_frame_w2c, k, cfg, export_cfg)
        return

    vis = o3d.visualization.Visualizer()
    created = vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                                height=int(cfg['viz_h'] * cfg['view_scale']),
                                visible=export_cfg.get('visible', True))
    if not created:
        if exporting and cfg['render_mode'] == 'centers':
            print("Open3D window is unavailable; falling back to headless centers export.")
            export_centers_headless(params, all_w2cs, first_frame_w2c, k, cfg, export_cfg)
            return
        raise RuntimeError(
            "Open3D failed to create a visualization window. "
            "Run from a desktop session, use xvfb-run on a server, or try without --hidden-window."
        )

    if cfg['render_mode'] == 'centers':
        means, colors, dynamic_count = get_scene_centers(params, curr_timestep=0, cfg=cfg)
        scene_data = {'means3D': means, 'colors_precomp': colors}
        scene_depth_data = None
        init_pts = o3d.utility.Vector3dVector(means.contiguous().double().cpu().numpy())
        init_cols = o3d.utility.Vector3dVector(colors.contiguous().double().cpu().numpy())
    else:
        scene_data, scene_depth_data, dynamic_count = get_scene_rendervars(
            params,
            first_frame_w2c,
            curr_timestep=0,
            cfg=cfg,
        )
        im, depth, sil = render(first_frame_w2c, k, scene_data, scene_depth_data, cfg)
        init_pts, init_cols = rgbd2pcd(im, depth, first_frame_w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(all_w2cs)
    cam_centers = []
    cam_colormap = plt.get_cmap('cool')
    norm_factor = 0.5
    total_num_lines = max(1, num_t - 1)
    line_colormap = plt.get_cmap('cool')
    
    # Initialize View Control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    first_view_w2c = first_frame_w2c.copy()
    if cfg.get('offset_first_viz_cam', True):
        first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    viz_start = True
    prev_timestep = None
    prev_frustum = None
    prev_lines = None
    last_scene_data = scene_data
    last_scene_depth_data = scene_depth_data
    last_dynamic_count = dynamic_count

    def update_timestep(curr_timestep):
        nonlocal cam_centers
        nonlocal k
        nonlocal last_scene_data
        nonlocal last_scene_depth_data
        nonlocal last_dynamic_count
        nonlocal prev_frustum
        nonlocal prev_lines
        nonlocal viz_start

        # Update Camera Frustum
        if cfg.get('visualize_cams', True):
            if curr_timestep == 0:
                cam_centers = []
                if prev_lines is not None:
                    vis.remove_geometry(prev_lines)
                    prev_lines = None
            if prev_frustum is not None:
                vis.remove_geometry(prev_frustum)
                prev_frustum = None
            new_frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[curr_timestep], 0.045)
            new_frustum.paint_uniform_color(np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3]))
            vis.add_geometry(new_frustum)
            prev_frustum = new_frustum
            cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])

            # Update Camera Trajectory
            if len(cam_centers) > 1 and curr_timestep > 0:
                num_lines = [1]
                cols = []
                for line_t in range(curr_timestep):
                    cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines)+norm_factor)[:3]))
                cols = np.array(cols)
                all_cols = [cols]
                out_pts = [np.array(cam_centers)]
                linesets = make_lineset(out_pts, all_cols, num_lines)
                lines = o3d.geometry.LineSet()
                lines.points = linesets[0].points
                lines.colors = linesets[0].colors
                lines.lines = linesets[0].lines
                vis.add_geometry(lines)
                prev_lines = lines
            elif prev_lines is not None:
                vis.remove_geometry(prev_lines)
                prev_lines = None

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        view_w2c = cam_params.extrinsic
        view_w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        cam_params.extrinsic = view_w2c
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)

        if cfg['render_mode'] == 'centers':
            means, colors, dynamic_count = get_scene_centers(params, curr_timestep, cfg)
            scene_data = {'means3D': means, 'colors_precomp': colors}
            scene_depth_data = None
            pts = o3d.utility.Vector3dVector(means.contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(colors.contiguous().double().cpu().numpy())
        else:
            scene_data, scene_depth_data, dynamic_count = get_scene_rendervars(
                params,
                view_w2c,
                curr_timestep=curr_timestep,
                cfg=cfg,
            )
            im, depth, sil = render(view_w2c, k, scene_data, scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, view_w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            return False
        vis.update_renderer()
        last_scene_data = scene_data
        last_scene_depth_data = scene_depth_data
        last_dynamic_count = dynamic_count
        viz_start = False
        return True

    window_open = True
    if exporting:
        output_dir = export_cfg['output_dir']
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        frame_paths = []
        timesteps = select_export_timesteps(
            num_t,
            export_cfg.get('frame_stride', 1),
            export_cfg.get('max_frames', 0),
        )
        for export_idx, curr_timestep in enumerate(timesteps):
            window_open = update_timestep(curr_timestep)
            if not window_open:
                break
            frame_path = os.path.join(frames_dir, f"{export_idx:04d}_t{curr_timestep:04d}.png")
            capture_frame(vis, frame_path, curr_timestep, num_t, last_dynamic_count, cfg)
            frame_paths.append(frame_path)

        fps = int(export_cfg.get('fps', cfg.get('viz_fps', 5)))
        if export_cfg.get('write_video', False):
            video_path = os.path.join(output_dir, 'mapping_4dgs_online.mp4')
            write_video_from_frames(frame_paths, video_path, fps)
            print(f"Saved video: {video_path}")
        if export_cfg.get('write_gif', False):
            gif_path = os.path.join(output_dir, 'mapping_4dgs_online.gif')
            write_gif_from_frames(frame_paths, gif_path, fps)
            print(f"Saved GIF: {gif_path}")
        print(f"Saved frames: {frames_dir}")
    else:
        # Rendering of Online Reconstruction
        start_time = time.time()
        num_timesteps = num_t
        curr_timestep = 0
        while curr_timestep < (num_timesteps-1) or not cfg['enter_interactive_post_online']:
            passed_time = time.time() - start_time
            passed_frames = passed_time * cfg['viz_fps']
            curr_timestep = int(passed_frames % num_timesteps)
            if prev_timestep is not None and curr_timestep == prev_timestep:
                continue
            window_open = update_timestep(curr_timestep)
            if not window_open:
                break
            prev_timestep = curr_timestep

    # Enter Interactive Mode once all frames have been visualized
    while window_open and (not exporting or export_cfg.get('interactive_after', False)):
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(last_scene_data['means3D'].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(last_scene_data['colors_precomp'].contiguous().double().cpu().numpy())
        else:
            im, depth, sil = render(w2c, k, last_scene_data, last_scene_depth_data, cfg)
            if cfg['show_sil']:
                im = (1-sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)
        
        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
    
    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


def resolve_input_paths(input_path, config, explicit_scene_path):
    if explicit_scene_path:
        return os.path.abspath(explicit_scene_path)

    if config.get("scene_path", ""):
        return os.path.abspath(config["scene_path"])

    config_dir = os.path.dirname(os.path.abspath(input_path))
    local_scene_path = os.path.join(config_dir, "params.npz")
    if os.path.exists(local_scene_path):
        return local_scene_path

    results_dir = os.path.join(config["workdir"], config["run_name"])
    return os.path.abspath(os.path.join(results_dir, "params.npz"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment config or result directory")
    parser.add_argument("--scene-path", type=str, default="", help="Path to params.npz")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for exported frames/video/GIF")
    parser.add_argument("--export-video", action="store_true", help="Export online mapping visualization as mp4")
    parser.add_argument("--export-gif", action="store_true", help="Export online mapping visualization as GIF")
    parser.add_argument("--save-frames", action="store_true", help="Export PNG frames without requiring mp4/GIF")
    parser.add_argument("--fps", type=int, default=None, help="Playback FPS for export and online viewer")
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every N timesteps")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of timesteps to export; 0 means all")
    parser.add_argument("--viz-w", type=int, default=None, help="Override visualization render width")
    parser.add_argument("--viz-h", type=int, default=None, help="Override visualization render height")
    parser.add_argument("--view-scale", type=float, default=None, help="Override Open3D window scale")
    parser.add_argument("--render-mode", choices=["centers", "color", "depth"], default=None, help="Override cfg['viz']['render_mode']")
    parser.add_argument("--highlight-dynamic", action="store_true", help="Color dynamic 4DGS Gaussians orange")
    parser.add_argument("--no-time-label", action="store_true", help="Do not draw timestep label/progress bar on exported frames")
    parser.add_argument("--hidden-window", action="store_true", help="Create the Open3D window as invisible")
    parser.add_argument("--headless-centers", action="store_true", help="Export centers mode without creating an Open3D window")
    parser.add_argument("--no-headless-render", action="store_true", help="Use Open3D window export for color/depth instead of direct rasterizer export")
    parser.add_argument("--headless-max-points", type=int, default=250000, help="Maximum static points per headless centers frame")
    parser.add_argument("--headless-max-dynamic-points", type=int, default=100000, help="Maximum dynamic points per headless centers frame")
    parser.add_argument("--headless-point-size", type=int, default=1, help="Point size for headless centers export")
    parser.add_argument("--interactive-after-export", action="store_true", help="Keep the viewer interactive after exporting")

    args = parser.parse_args()

    experiment_path = os.path.abspath(args.experiment)
    if os.path.isdir(experiment_path):
        experiment_path = os.path.join(experiment_path, "config.py")

    experiment = SourceFileLoader(
        os.path.basename(experiment_path), experiment_path
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    scene_path = resolve_input_paths(experiment_path, experiment.config, args.scene_path)
    viz_cfg = deepcopy(experiment.config["viz"])
    viz_cfg["device"] = experiment.config.get("primary_device", "cuda:0")
    if args.fps is not None:
        viz_cfg["viz_fps"] = args.fps
    if args.viz_w is not None:
        viz_cfg["viz_w"] = args.viz_w
    if args.viz_h is not None:
        viz_cfg["viz_h"] = args.viz_h
    if args.view_scale is not None:
        viz_cfg["view_scale"] = args.view_scale
    if args.render_mode is not None:
        viz_cfg["render_mode"] = args.render_mode
    viz_cfg["highlight_dynamic"] = bool(args.highlight_dynamic)
    viz_cfg["no_time_label"] = bool(args.no_time_label)
    viz_cfg["headless_max_static_points"] = args.headless_max_points
    viz_cfg["headless_max_dynamic_points"] = args.headless_max_dynamic_points
    viz_cfg["headless_point_size"] = args.headless_point_size

    export_enabled = args.export_video or args.export_gif or args.save_frames
    output_dir = args.output_dir or os.path.join(os.path.dirname(scene_path), "viz_online_4dgs")
    export_cfg = {
        "enabled": export_enabled,
        "output_dir": os.path.abspath(output_dir),
        "write_video": bool(args.export_video),
        "write_gif": bool(args.export_gif),
        "fps": args.fps or viz_cfg.get("viz_fps", 5),
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "visible": not args.hidden_window,
        "headless_centers": bool(args.headless_centers),
        "headless_render": not args.no_headless_render,
        "interactive_after": bool(args.interactive_after_export),
    }

    # Visualize Final Reconstruction
    visualize(scene_path, viz_cfg, export_cfg=export_cfg)

import argparse
import os
import sys

import numpy as np


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _BASE_DIR)

from sequence_submaps import (
    choose_trajectory,
    draw_overlay,
    dynamic_points_at,
    find_chunks,
    load_local_w2cs,
    make_video_writer,
    quat_wxyz_to_rot,
    require_cv2,
    resolve_base_folder_and_scene,
    subsample_rows,
    subsample_triplet,
    transform_static_chunk,
)


def require_torch_renderer():
    try:
        import torch
        import torch.nn.functional as F
        from diff_gaussian_rasterization import GaussianRasterizer as Renderer
        from utils.recon_helpers import setup_camera
        from utils.slam_helpers import matrix_to_quaternion, quat_mult
        from utils.slam_external import build_rotation
    except ImportError as exc:
        raise ImportError(
            "Gaussian render mode requires torch and diff_gaussian_rasterization. "
            "Run this script inside the lsgslam CUDA environment."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Gaussian render mode requires CUDA. Use --render-mode centers for CPU fallback.")
    return torch, F, Renderer, setup_camera, matrix_to_quaternion, quat_mult, build_rotation


def scaled_intrinsics(params, width, height):
    intrinsics = np.asarray(params["intrinsics"], dtype=np.float64).copy()
    if intrinsics.shape[0] > 3 or intrinsics.shape[1] > 3:
        intrinsics = intrinsics[:3, :3]

    org_width = float(np.asarray(params["org_width"]).reshape(-1)[0])
    org_height = float(np.asarray(params["org_height"]).reshape(-1)[0])
    intrinsics[0, :] *= float(width) / org_width
    intrinsics[1, :] *= float(height) / org_height
    intrinsics[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return intrinsics


def project_first_person(points, colors, w2c, intrinsics, width, height, near, far, point_size):
    cv2 = require_cv2()
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if points.size == 0:
        return image

    points_h = np.concatenate(
        (points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    cam_points = (w2c @ points_h.T).T[:, :3]
    z = cam_points[:, 2]
    valid = (z > near) & (z < far)
    if not np.any(valid):
        return image

    cam_points = cam_points[valid]
    colors = colors[valid]
    z = z[valid]
    px = (intrinsics[0, 0] * cam_points[:, 0] / z + intrinsics[0, 2]).astype(np.int32)
    py = (intrinsics[1, 1] * cam_points[:, 1] / z + intrinsics[1, 2]).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(inside):
        return image

    px = px[inside]
    py = py[inside]
    z = z[inside]
    rgb = (np.clip(colors[inside], 0.0, 1.0) * 255.0).astype(np.uint8)

    order = np.argsort(z)[::-1]
    image[py[order], px[order]] = rgb[order]

    if point_size > 1:
        kernel = np.ones((point_size, point_size), dtype=np.uint8)
        mask = np.any(image != 255, axis=2).astype(np.uint8)
        grown = cv2.dilate(mask, kernel)
        image = cv2.dilate(image, kernel)
        image[grown == 0] = 255
    return image


def rotation_transforms(local_w2cs, global_c2ws):
    return np.einsum("nij,njk->nik", global_c2ws[:, :3, :3], local_w2cs[:, :3, :3])


def build_rotation_np(quats):
    q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norms, 1e-12)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float64)
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


def matrix_to_quaternion_np(mats):
    mats = np.asarray(mats, dtype=np.float64).reshape(-1, 3, 3)
    quats = np.empty((mats.shape[0], 4), dtype=np.float64)
    for idx, mat in enumerate(mats):
        trace = float(np.trace(mat))
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2.0
            quats[idx] = [
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            ]
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            quats[idx] = [
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            ]
        elif mat[1, 1] > mat[2, 2]:
            s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            quats[idx] = [
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            ]
        else:
            s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            quats[idx] = [
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            ]
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    return (quats / np.maximum(norms, 1e-12)).astype(np.float32)


def transform_rotations_by_birth(quats, birth_steps, local_w2cs, global_c2ws):
    birth_steps = np.asarray(birth_steps, dtype=np.int64).reshape(-1)
    birth_steps = np.clip(birth_steps, 0, len(local_w2cs) - 1)
    transform_rots = rotation_transforms(local_w2cs, global_c2ws)[birth_steps]
    local_rots = build_rotation_np(quats)
    return matrix_to_quaternion_np(np.einsum("nij,njk->nik", transform_rots, local_rots))


def transform_static_gaussians(params, local_w2cs, global_c2ws, max_rows):
    means, colors, birth = transform_static_chunk(params, local_w2cs, global_c2ws)
    rotations = np.asarray(params["unnorm_rotations"], dtype=np.float32)
    log_scales = np.asarray(params["log_scales"], dtype=np.float32)
    opacities = np.asarray(params["logit_opacities"], dtype=np.float32)

    if max_rows > 0 and means.shape[0] > max_rows:
        step = int(np.ceil(means.shape[0] / float(max_rows)))
        rows = np.arange(0, means.shape[0], step, dtype=np.int64)[:max_rows]
        means = means[rows]
        colors = colors[rows]
        birth = birth[rows]
        rotations = rotations[rows]
        log_scales = log_scales[rows]
        opacities = opacities[rows]

    global_rotations = transform_rotations_by_birth(rotations, birth, local_w2cs, global_c2ws)
    return {
        "means3D": means.astype(np.float32, copy=False),
        "colors": np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
        "rotations": global_rotations,
        "log_scales": log_scales.astype(np.float32, copy=False),
        "logit_opacities": opacities.astype(np.float32, copy=False),
        "birth": birth.astype(np.int64, copy=False),
    }


def gaussian_slice(cache, mask=None):
    if mask is None:
        return {
            "means3D": cache["means3D"],
            "colors": cache["colors"],
            "rotations": cache["rotations"],
            "log_scales": cache["log_scales"],
            "logit_opacities": cache["logit_opacities"],
        }
    return {
        "means3D": cache["means3D"][mask],
        "colors": cache["colors"][mask],
        "rotations": cache["rotations"][mask],
        "log_scales": cache["log_scales"][mask],
        "logit_opacities": cache["logit_opacities"][mask],
    }


def gaussian_count(block):
    if block is None:
        return 0
    return int(block["means3D"].shape[0])


def cull_gaussians_to_camera(block, c2w, intrinsics, width, height, near, far, margin, max_rows):
    if block is None or block["means3D"].shape[0] == 0:
        return None

    points = block["means3D"]
    if not isinstance(points, np.ndarray):
        return block

    w2c = np.linalg.inv(c2w)
    points_h = np.concatenate(
        (points.astype(np.float64, copy=False), np.ones((points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    cam_points = (w2c @ points_h.T).T[:, :3]
    z = cam_points[:, 2]
    valid = (z > near) & (z < far)
    if not np.any(valid):
        return None

    px = intrinsics[0, 0] * cam_points[:, 0] / np.maximum(z, 1e-8) + intrinsics[0, 2]
    py = intrinsics[1, 1] * cam_points[:, 1] / np.maximum(z, 1e-8) + intrinsics[1, 2]
    valid &= (px >= -margin) & (px < width + margin) & (py >= -margin) & (py < height + margin)
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return None
    if max_rows > 0 and rows.size > max_rows:
        step = int(np.ceil(rows.size / float(max_rows)))
        rows = rows[::step][:max_rows]
    return gaussian_slice(block, rows)


def dynamic_gaussians_at(params, local_t, local_w2cs, global_c2ws, highlight_dynamic):
    torch, F, _Renderer, _setup_camera, matrix_to_quaternion, quat_mult, build_rotation = require_torch_renderer()
    required = {
        "dyn_means3D_canon",
        "dyn_rgb_colors",
        "dyn_unnorm_rotations",
        "dyn_logit_opacities",
        "dyn_log_scales",
        "dyn_obj_ids",
        "dyn_birth_time",
        "dyn_obj_unnorm_rots",
        "dyn_obj_trans",
        "dyn_obj_visible",
    }
    if not required.issubset(params.files) or params["dyn_means3D_canon"].size == 0:
        return None
    if local_t >= params["dyn_obj_visible"].shape[-1]:
        return None

    obj_ids_all = torch.as_tensor(params["dyn_obj_ids"], dtype=torch.long, device="cuda").reshape(-1)
    birth = torch.as_tensor(params["dyn_birth_time"], dtype=torch.long, device="cuda").reshape(-1)
    visible = torch.as_tensor(params["dyn_obj_visible"], dtype=torch.bool, device="cuda")
    active = (birth <= int(local_t)) & visible[obj_ids_all, int(local_t)]
    if not bool(active.any()):
        return None

    obj_ids = obj_ids_all[active]
    local_pts = torch.as_tensor(params["dyn_means3D_canon"], dtype=torch.float32, device="cuda")[active]
    local_rots = torch.as_tensor(params["dyn_unnorm_rotations"], dtype=torch.float32, device="cuda")[active]
    obj_rots = F.normalize(
        torch.as_tensor(params["dyn_obj_unnorm_rots"], dtype=torch.float32, device="cuda")[obj_ids, :, int(local_t)]
    )
    obj_trans = torch.as_tensor(params["dyn_obj_trans"], dtype=torch.float32, device="cuda")[obj_ids, :, int(local_t)]

    chunk_pts = torch.bmm(build_rotation(obj_rots), local_pts.unsqueeze(-1)).squeeze(-1) + obj_trans
    chunk_rots = quat_mult(obj_rots, F.normalize(local_rots))

    transform_rot = rotation_transforms(local_w2cs, global_c2ws)[int(local_t)]
    transform_rot = torch.as_tensor(transform_rot, dtype=torch.float32, device="cuda")
    chunk_pts_h = torch.cat((chunk_pts, torch.ones_like(chunk_pts[:, :1])), dim=1)
    local_w2c = torch.as_tensor(local_w2cs[int(local_t)], dtype=torch.float32, device="cuda")
    global_c2w = torch.as_tensor(global_c2ws[int(local_t)], dtype=torch.float32, device="cuda")
    means = (global_c2w @ (local_w2c @ chunk_pts_h.T)).T[:, :3]
    rotations = F.normalize(matrix_to_quaternion(torch.bmm(
        transform_rot.expand(chunk_rots.shape[0], 3, 3),
        build_rotation(F.normalize(chunk_rots)),
    )))

    if highlight_dynamic:
        colors = torch.tensor([1.0, 0.22, 0.05], dtype=torch.float32, device="cuda").expand(means.shape[0], 3)
    else:
        colors = torch.as_tensor(params["dyn_rgb_colors"], dtype=torch.float32, device="cuda")[active]

    return {
        "means3D": means,
        "colors": torch.clamp(colors, 0.0, 1.0),
        "rotations": rotations,
        "log_scales": torch.as_tensor(params["dyn_log_scales"], dtype=torch.float32, device="cuda")[active],
        "logit_opacities": torch.as_tensor(params["dyn_logit_opacities"], dtype=torch.float32, device="cuda")[active],
    }


def concat_gaussian_blocks(blocks, max_gaussians):
    torch, F, _Renderer, _setup_camera, _matrix_to_quaternion, _quat_mult, _build_rotation = require_torch_renderer()
    blocks = [block for block in blocks if block is not None and block["means3D"].shape[0] > 0]
    if not blocks:
        empty = torch.empty((0, 3), dtype=torch.float32, device="cuda")
        return {
            "means3D": empty,
            "colors_precomp": empty,
            "rotations": torch.empty((0, 4), dtype=torch.float32, device="cuda"),
            "opacities": torch.empty((0, 1), dtype=torch.float32, device="cuda"),
            "scales": empty,
            "means2D": empty,
        }

    def as_cuda(value, dtype=torch.float32):
        if isinstance(value, torch.Tensor):
            return value.to(device="cuda", dtype=dtype)
        return torch.as_tensor(value, dtype=dtype, device="cuda")

    means = torch.cat([as_cuda(block["means3D"]) for block in blocks], dim=0)
    colors = torch.cat([as_cuda(block["colors"]) for block in blocks], dim=0)
    rotations = torch.cat([as_cuda(block["rotations"]) for block in blocks], dim=0)
    scale_blocks = []
    for block in blocks:
        log_scales_i = as_cuda(block["log_scales"])
        if log_scales_i.shape[-1] == 1:
            log_scales_i = log_scales_i.expand(-1, 3)
        scale_blocks.append(log_scales_i)
    log_scales = torch.cat(scale_blocks, dim=0)
    logit_opacities = torch.cat([as_cuda(block["logit_opacities"]) for block in blocks], dim=0)

    if max_gaussians > 0 and means.shape[0] > max_gaussians:
        step = int(np.ceil(means.shape[0] / float(max_gaussians)))
        rows = torch.arange(0, means.shape[0], step, dtype=torch.long, device="cuda")[:max_gaussians]
        means = means[rows]
        colors = colors[rows]
        rotations = rotations[rows]
        log_scales = log_scales[rows]
        logit_opacities = logit_opacities[rows]

    return {
        "means3D": means,
        "colors_precomp": colors,
        "rotations": F.normalize(rotations),
        "opacities": torch.sigmoid(logit_opacities),
        "scales": torch.exp(log_scales),
        "means2D": torch.zeros_like(means),
    }


def render_gaussians(blocks, c2w, intrinsics, width, height, near, far, max_gaussians):
    torch, _F, Renderer, setup_camera, _matrix_to_quaternion, _quat_mult, _build_rotation = require_torch_renderer()
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

    cv2 = require_cv2()
    rendervars = concat_gaussian_blocks(blocks, max_gaussians)
    if rendervars["means3D"].shape[0] == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    w2c = np.linalg.inv(c2w)
    with torch.no_grad():
        cam = setup_camera(width, height, intrinsics, w2c, near=near, far=far, depth_threshold=far)
        white_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            depth_threshold=cam.depth_threshold,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered,
        )
        image, _radius, _depth, _ = Renderer(raster_settings=white_cam)(**rendervars)
        frame = torch.clamp(image, 0.0, 1.0).detach().cpu().permute(1, 2, 0).numpy()
    return (frame * 255.0).astype(np.uint8)


def maybe_add_camera_horizon(frame, enabled):
    if not enabled:
        return frame
    cv2 = require_cv2()
    h, w = frame.shape[:2]
    y = h // 2
    cv2.line(frame, (0, y), (w, y), (210, 210, 210), 1, cv2.LINE_AA)
    return frame


def export_first_person(args):
    base_folder, scene_name = resolve_base_folder_and_scene(args)
    chunks = find_chunks(base_folder, scene_name)
    c2ws, traj_source = choose_trajectory(base_folder, scene_name, chunks, args.trajectory)
    total_processed = c2ws.shape[0]
    if args.render_mode == "gaussian":
        require_torch_renderer()

    video_path = args.video_path or os.path.abspath("sequence_submaps_first_person.mp4")
    video_writer = make_video_writer(video_path, args.fps, args.width, args.height)
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")

    print(f"Base folder: {base_folder}")
    print(f"Scene: {scene_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Processed frames: {total_processed}")
    print(f"Trajectory: {traj_source}")
    print(f"Output video: {video_path}")

    cached_static_points = []
    cached_static_colors = []
    cached_static_gaussians = []
    processed_offset = 0
    export_idx = 0
    intrinsics = None

    for chunk_idx, chunk in enumerate(chunks):
        params = np.load(chunk.params_path, allow_pickle=True)
        if intrinsics is None:
            intrinsics = scaled_intrinsics(params, args.width, args.height)

        local_w2cs = load_local_w2cs(params)
        local_start = 0 if chunk_idx == 0 else 1
        count = min(len(local_w2cs) - local_start, total_processed - processed_offset)
        if count <= 0:
            break
        if local_start == 0:
            global_c2ws = c2ws[processed_offset : processed_offset + count]
        else:
            global_c2ws = np.concatenate(
                (
                    c2ws[processed_offset - 1 : processed_offset],
                    c2ws[processed_offset : processed_offset + count],
                ),
                axis=0,
            )
        local_w2cs_for_transform = local_w2cs[: local_start + count]

        if args.render_mode == "gaussian":
            static_gaussians = transform_static_gaussians(
                params,
                local_w2cs_for_transform,
                global_c2ws,
                args.max_gaussians_per_submap,
            )
            static_points = static_colors = static_birth = None
        else:
            static_points, static_colors, static_birth = transform_static_chunk(
                params,
                local_w2cs_for_transform,
                global_c2ws,
            )
            static_points, static_colors, static_birth = subsample_triplet(
                static_points,
                static_colors,
                static_birth,
                args.max_points_per_submap,
            )

        for local_t in range(local_start, local_start + count):
            processed_idx = processed_offset + (local_t - local_start)
            if processed_idx % args.frame_stride != 0:
                continue
            if args.max_frames > 0 and export_idx >= args.max_frames:
                break

            if args.render_mode == "gaussian":
                active = static_gaussians["birth"] <= local_t
                candidate_blocks = cached_static_gaussians + [gaussian_slice(static_gaussians, active)]
                gaussian_blocks = [
                    cull_gaussians_to_camera(
                        block,
                        c2ws[processed_idx],
                        intrinsics,
                        args.width,
                        args.height,
                        args.near,
                        args.far,
                        args.frustum_margin,
                        args.max_visible_per_submap,
                    )
                    for block in candidate_blocks
                ]
                gaussian_blocks.append(
                    dynamic_gaussians_at(
                        params,
                        local_t,
                        local_w2cs_for_transform,
                        global_c2ws,
                        args.highlight_dynamic,
                    )
                )
                frame = render_gaussians(
                    gaussian_blocks,
                    c2ws[processed_idx],
                    intrinsics,
                    args.width,
                    args.height,
                    args.near,
                    args.far,
                    args.max_gaussians,
                )
            else:
                active = static_birth <= local_t
                point_blocks = cached_static_points + [static_points[active]]
                color_blocks = cached_static_colors + [static_colors[active]]

                dyn_points, dyn_colors = dynamic_points_at(
                    params,
                    local_t,
                    local_w2cs_for_transform,
                    global_c2ws,
                    args.highlight_dynamic,
                )
                if dyn_points is not None:
                    point_blocks.append(dyn_points)
                    color_blocks.append(dyn_colors)

                points = np.concatenate(point_blocks, axis=0) if point_blocks else np.empty((0, 3), dtype=np.float32)
                colors = np.concatenate(color_blocks, axis=0) if color_blocks else np.empty((0, 3), dtype=np.float32)
                points = subsample_rows(points, args.max_points)
                colors = subsample_rows(colors, args.max_points)

                w2c = np.linalg.inv(c2ws[processed_idx])
                frame = project_first_person(
                    points,
                    colors,
                    w2c,
                    intrinsics,
                    args.width,
                    args.height,
                    args.near,
                    args.far,
                    args.point_size,
                )
            frame = maybe_add_camera_horizon(frame, args.horizon)

            raw_frame_id = chunk.start + local_t * chunk.stride
            frame = draw_overlay(
                frame,
                raw_frame_id,
                processed_idx,
                total_processed,
                os.path.basename(str(traj_source)),
                draw_label=not args.no_label,
            )

            cv2 = require_cv2()
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            export_idx += 1

            if export_idx % 25 == 0:
                print(f"Saved {export_idx} video frames, latest raw frame {raw_frame_id}")
                if args.render_mode == "gaussian":
                    visible_count = sum(gaussian_count(block) for block in gaussian_blocks)
                    print(f"Visible static/dynamic Gaussians this frame: {visible_count}")

        if args.render_mode == "gaussian":
            cached_static_gaussians.append(gaussian_slice(static_gaussians))
        else:
            cached_static_points.append(static_points)
            cached_static_colors.append(static_colors)
        processed_offset += count
        print(f"Finished chunk {chunk_idx + 1}/{len(chunks)}: {os.path.basename(chunk.path)}")

        if args.max_frames > 0 and export_idx >= args.max_frames:
            break

    video_writer.release()
    print(f"Saved video: {video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="First-person visualization for a long LSG-SLAM sequence stored as many submap params.npz files."
    )
    parser.add_argument("--config", default="", help="Experiment config. Used to derive workdir and scene_name.")
    parser.add_argument("--base-folder", default="", help="Result group folder, e.g. results/kitti360-0000-all.")
    parser.add_argument("--scene-name", default="", help="Scene prefix used by submap result folders.")
    parser.add_argument("--video-path", default="", help="Exact MP4 output path.")
    parser.add_argument("--trajectory", choices=["auto", "optimized", "odom"], default="auto")
    parser.add_argument("--render-mode", choices=["gaussian", "centers"], default="gaussian")
    parser.add_argument("--width", type=int, default=1408)
    parser.add_argument("--height", type=int, default=376)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every N processed frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all exported frames.")
    parser.add_argument("--max-points", type=int, default=800000, help="Maximum points projected per frame; 0 means no cap.")
    parser.add_argument("--max-points-per-submap", type=int, default=50000, help="Static points kept from each submap; 0 means no cap.")
    parser.add_argument("--max-gaussians", type=int, default=0, help="Maximum Gaussians rendered per frame; 0 means no cap.")
    parser.add_argument("--max-gaussians-per-submap", type=int, default=0, help="Static Gaussians kept from each submap; 0 means no cap.")
    parser.add_argument("--max-visible-per-submap", type=int, default=0, help="Visible static Gaussians kept from each submap per frame; 0 means no cap.")
    parser.add_argument("--frustum-margin", type=int, default=64, help="Pixel margin around the image for center-based frustum culling.")
    parser.add_argument("--point-size", type=int, default=2)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=80.0)
    parser.set_defaults(highlight_dynamic=True)
    parser.add_argument("--highlight-dynamic", dest="highlight_dynamic", action="store_true")
    parser.add_argument("--no-highlight-dynamic", dest="highlight_dynamic", action="store_false")
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--horizon", action="store_true", help="Draw a subtle image-center horizon guide.")
    args = parser.parse_args()

    export_first_person(args)


if __name__ == "__main__":
    main()

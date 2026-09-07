import argparse
import glob
import os
import sys
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader

import numpy as np


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for exporting frames/video. "
            "Run this script inside the lsgslam environment."
        ) from exc
    return cv2


def require_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError(
            "imageio is required for GIF export. Run this script inside the lsgslam environment."
        ) from exc
    return imageio


@dataclass
class Chunk:
    start: int
    end: int
    stride: int
    path: str
    params_path: str


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_wxyz_to_rot(q):
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_chunk_name(name):
    parts = name.split("_")
    try:
        start, end, stride = map(int, parts[-3:])
    except ValueError:
        return None
    return start, end, stride


def find_chunks(base_folder, scene_name):
    chunks = []
    for name in os.listdir(base_folder):
        path = os.path.join(base_folder, name)
        if not os.path.isdir(path):
            continue
        if "loop" in name or not name.startswith(scene_name):
            continue
        parsed = parse_chunk_name(name)
        if parsed is None:
            continue
        params_path = os.path.join(path, "params.npz")
        if os.path.exists(params_path):
            chunks.append(Chunk(*parsed, path=path, params_path=params_path))
    chunks = sorted(chunks, key=lambda item: item.start)
    if not chunks:
        raise FileNotFoundError(
            f"No submap params.npz found under {base_folder} for scene {scene_name}"
        )
    return chunks


def load_local_w2cs(params):
    rots = params["cam_unnorm_rots"]
    trans = params["cam_trans"]
    w2cs = []
    for idx in range(rots.shape[-1]):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = quat_wxyz_to_rot(rots[..., idx])
        w2c[:3, 3] = np.asarray(trans[..., idx], dtype=np.float64).reshape(-1)[:3]
        w2cs.append(w2c)
    return np.asarray(w2cs)


def stitch_odom_c2ws(chunks):
    all_w2cs = []
    for chunk in chunks:
        params = np.load(chunk.params_path, allow_pickle=True)
        local_w2cs = load_local_w2cs(params)
        if len(all_w2cs) == 0:
            all_w2cs.extend(local_w2cs)
        else:
            last_global_w2c = all_w2cs[-1]
            all_w2cs.extend([pose @ last_global_w2c for pose in local_w2cs[1:]])
    return np.linalg.inv(np.asarray(all_w2cs))


def latest_pose_graph_csv(base_folder, scene_name):
    csv_dir = os.path.join(base_folder, "PoseGraphResult", "csvs")
    if not os.path.isdir(csv_dir):
        return None
    optimized = [
        path
        for path in glob.glob(os.path.join(csv_dir, f"pose{scene_name}optimized_*.csv"))
        if "unoptimized" not in os.path.basename(path)
    ]
    unoptimized = glob.glob(os.path.join(csv_dir, f"pose{scene_name}unoptimized_*.csv"))
    candidates = optimized or unoptimized
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_pose_graph_c2ws(csv_path, odom_c2ws):
    poses = np.loadtxt(csv_path, delimiter=",")
    poses = np.atleast_2d(poses).reshape(-1, 4, 4).astype(np.float64)
    poses[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    if poses.shape[0] >= odom_c2ws.shape[0]:
        return poses[: odom_c2ws.shape[0]]

    if poses.shape[0] == 0:
        return odom_c2ws

    padded = np.empty_like(odom_c2ws)
    padded[: poses.shape[0]] = poses
    anchor = poses.shape[0] - 1
    odom_to_pose = poses[anchor] @ np.linalg.inv(odom_c2ws[anchor])
    for idx in range(poses.shape[0], odom_c2ws.shape[0]):
        padded[idx] = odom_to_pose @ odom_c2ws[idx]
    return padded


def resolve_base_folder_and_scene(args):
    if args.base_folder and args.scene_name:
        return os.path.abspath(args.base_folder), args.scene_name

    if not args.config:
        raise ValueError("Use --config or pass both --base-folder and --scene-name.")

    config_path = os.path.abspath(args.config)
    experiment = SourceFileLoader(os.path.basename(config_path), config_path).load_module()
    config = experiment.config
    config_dir = os.path.dirname(config_path)
    workdir = config["workdir"]
    if not os.path.isabs(workdir):
        workdir = os.path.abspath(os.path.join(config_dir, "..", "..", workdir))
        if not os.path.isdir(workdir):
            workdir = os.path.abspath(config["workdir"])
    return workdir, config["scene_name"]


def choose_trajectory(base_folder, scene_name, chunks, mode):
    odom_c2ws = stitch_odom_c2ws(chunks)
    if mode == "odom":
        return odom_c2ws, "stitched odometry"

    csv_path = latest_pose_graph_csv(base_folder, scene_name)
    if csv_path is None:
        if mode == "optimized":
            raise FileNotFoundError(
                f"No pose graph CSV found under {base_folder}/PoseGraphResult/csvs"
            )
        return odom_c2ws, "stitched odometry"

    return load_pose_graph_c2ws(csv_path, odom_c2ws), csv_path


def transform_points_by_birth(points, birth_steps, local_w2cs, global_c2ws):
    birth_steps = np.asarray(birth_steps, dtype=np.int64).reshape(-1)
    birth_steps = np.clip(birth_steps, 0, len(local_w2cs) - 1)
    points_h = np.concatenate(
        (points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)),
        axis=1,
    )
    camera_points = np.einsum("nij,nj->ni", local_w2cs[birth_steps], points_h)
    world_points = np.einsum("nij,nj->ni", global_c2ws[birth_steps], camera_points)
    return world_points[:, :3].astype(np.float32)


def subsample_rows(values, max_rows):
    if max_rows <= 0 or values.shape[0] <= max_rows:
        return values
    step = int(np.ceil(values.shape[0] / float(max_rows)))
    return values[::step][:max_rows]


def subsample_triplet(points, colors, birth, max_rows):
    if max_rows <= 0 or points.shape[0] <= max_rows:
        return points, colors, birth
    step = int(np.ceil(points.shape[0] / float(max_rows)))
    rows = np.arange(0, points.shape[0], step, dtype=np.int64)[:max_rows]
    return points[rows], colors[rows], birth[rows]


def transform_static_chunk(params, local_w2cs, global_c2ws):
    means = np.asarray(params["means3D"], dtype=np.float32)
    colors = np.asarray(params["rgb_colors"], dtype=np.float32)
    birth = np.asarray(params["timestep"], dtype=np.int64)
    transformed = transform_points_by_birth(means, birth, local_w2cs, global_c2ws)
    return transformed, np.clip(colors, 0.0, 1.0), birth


def has_dynamic(params):
    required = {
        "dyn_means3D_canon",
        "dyn_rgb_colors",
        "dyn_obj_ids",
        "dyn_birth_time",
        "dyn_obj_unnorm_rots",
        "dyn_obj_trans",
        "dyn_obj_visible",
    }
    return required.issubset(params.files) and params["dyn_means3D_canon"].size > 0


def dynamic_points_at(params, local_t, local_w2cs, global_c2ws, highlight_dynamic):
    if not has_dynamic(params) or local_t >= params["dyn_obj_visible"].shape[-1]:
        return None, None

    obj_ids_all = np.asarray(params["dyn_obj_ids"], dtype=np.int64).reshape(-1)
    birth = np.asarray(params["dyn_birth_time"], dtype=np.int64).reshape(-1)
    visible = np.asarray(params["dyn_obj_visible"]).astype(bool)
    active = (birth <= local_t) & visible[obj_ids_all, local_t]
    if not np.any(active):
        return None, None

    obj_ids = obj_ids_all[active]
    local_pts = np.asarray(params["dyn_means3D_canon"], dtype=np.float64)[active]
    obj_rots = np.asarray(params["dyn_obj_unnorm_rots"], dtype=np.float64)[obj_ids, :, local_t]
    obj_trans = np.asarray(params["dyn_obj_trans"], dtype=np.float64)[obj_ids, :, local_t]
    rot_mats = np.stack([quat_wxyz_to_rot(q) for q in obj_rots], axis=0)
    chunk_world = np.einsum("nij,nj->ni", rot_mats, local_pts) + obj_trans
    points = transform_points_by_birth(
        chunk_world.astype(np.float32),
        np.full(chunk_world.shape[0], local_t, dtype=np.int64),
        local_w2cs,
        global_c2ws,
    )

    if highlight_dynamic:
        colors = np.tile(np.array([[1.0, 0.22, 0.05]], dtype=np.float32), (points.shape[0], 1))
    else:
        colors = np.asarray(params["dyn_rgb_colors"], dtype=np.float32)[active]
    return points, np.clip(colors, 0.0, 1.0)


def compute_view_from_trajectory(c2ws, zoom):
    centers = c2ws[:, :3, 3]
    center = np.median(centers, axis=0)
    span = np.percentile(centers, 95, axis=0) - np.percentile(centers, 5, axis=0)
    extent = max(float(np.linalg.norm(span[[0, 2]])), 1.0)
    scale = extent * float(zoom)
    return center, scale


def project_topdown(points, colors, center, scale, width, height, point_size):
    cv2 = require_cv2()
    if points.size == 0:
        return np.full((height, width, 3), 255, dtype=np.uint8)

    x = points[:, 0]
    z = points[:, 2]
    px = ((x - center[0]) / scale * min(width, height) + width * 0.5).astype(np.int32)
    py = ((z - center[2]) / scale * min(width, height) + height * 0.5).astype(np.int32)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if np.any(inside):
        rgb = (np.clip(colors[inside], 0.0, 1.0) * 255.0).astype(np.uint8)
        image[py[inside], px[inside]] = rgb

    if point_size > 1:
        kernel = np.ones((point_size, point_size), dtype=np.uint8)
        mask = np.any(image != 255, axis=2).astype(np.uint8)
        grown = cv2.dilate(mask, kernel)
        image = cv2.dilate(image, kernel)
        image[grown == 0] = 255
    return image


def draw_overlay(frame, raw_frame_id, processed_idx, total_processed, source_name, draw_label=True):
    cv2 = require_cv2()
    if not draw_label:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    label = f"frame={raw_frame_id} | step={processed_idx + 1}/{total_processed} | {source_name}"
    cv2.rectangle(out, (0, 0), (min(w, 760), 38), (0, 0, 0), -1)
    cv2.putText(
        out,
        label,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    bar_width = int(w * float(processed_idx + 1) / max(1, total_processed))
    cv2.rectangle(out, (0, h - 8), (w, h), (20, 20, 20), -1)
    cv2.rectangle(out, (0, h - 8), (bar_width, h), (0, 185, 255), -1)
    return out


def write_video(frame_paths, output_path, fps):
    cv2 = require_cv2()
    if not frame_paths:
        return
    first = cv2.imread(frame_paths[0])
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first.shape[1], first.shape[0]),
    )
    for frame_path in frame_paths:
        writer.write(cv2.imread(frame_path))
    writer.release()


def make_video_writer(output_path, fps, width, height):
    cv2 = require_cv2()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def write_gif(frame_paths, output_path, fps):
    imageio = require_imageio()
    if not frame_paths:
        return
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(output_path, frames, duration=1.0 / max(1, fps))


def export_sequence(args):
    base_folder, scene_name = resolve_base_folder_and_scene(args)
    chunks = find_chunks(base_folder, scene_name)
    c2ws, traj_source = choose_trajectory(base_folder, scene_name, chunks, args.trajectory)
    total_processed = c2ws.shape[0]
    view_center, view_scale = compute_view_from_trajectory(c2ws, args.zoom)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    video_path = args.video_path or os.path.join(output_dir, "sequence_submaps_topdown.mp4")
    video_writer = None
    if args.export_video or args.video_only:
        video_writer = make_video_writer(video_path, args.fps, args.width, args.height)
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")
    if args.save_frames or args.export_gif:
        os.makedirs(frames_dir, exist_ok=True)

    print(f"Base folder: {base_folder}")
    print(f"Scene: {scene_name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Processed frames: {total_processed}")
    print(f"Trajectory: {traj_source}")
    print(f"Output video: {video_path if video_writer is not None else 'disabled'}")

    cached_static_points = []
    cached_static_colors = []
    frame_paths = []
    processed_offset = 0
    export_idx = 0

    for chunk_idx, chunk in enumerate(chunks):
        params = np.load(chunk.params_path, allow_pickle=True)
        local_w2cs = load_local_w2cs(params)
        count = min(len(local_w2cs), total_processed - processed_offset)
        if count <= 0:
            break
        global_c2ws = c2ws[processed_offset : processed_offset + count]
        static_points, static_colors, static_birth = transform_static_chunk(
            params,
            local_w2cs[:count],
            global_c2ws,
        )
        static_points, static_colors, static_birth = subsample_triplet(
            static_points,
            static_colors,
            static_birth,
            args.max_points_per_submap,
        )

        for local_t in range(count):
            processed_idx = processed_offset + local_t
            if processed_idx % args.frame_stride != 0:
                continue
            if args.max_frames > 0 and export_idx >= args.max_frames:
                break

            active = static_birth <= local_t
            point_blocks = cached_static_points + [static_points[active]]
            color_blocks = cached_static_colors + [static_colors[active]]

            dyn_points, dyn_colors = dynamic_points_at(
                params,
                local_t,
                local_w2cs[:count],
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

            frame = project_topdown(
                points,
                colors,
                view_center,
                view_scale,
                args.width,
                args.height,
                args.point_size,
            )

            cam_pos = c2ws[processed_idx, :3, 3]
            cam_frame = project_topdown(
                cam_pos.reshape(1, 3),
                np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
                view_center,
                view_scale,
                args.width,
                args.height,
                max(args.point_size + 3, 5),
            )
            frame[np.any(cam_frame != 255, axis=2)] = cam_frame[np.any(cam_frame != 255, axis=2)]

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
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if video_writer is not None:
                video_writer.write(frame_bgr)
            if args.save_frames or args.export_gif:
                frame_path = os.path.join(frames_dir, f"{export_idx:06d}_frame{raw_frame_id:06d}.png")
                cv2.imwrite(frame_path, frame_bgr)
                frame_paths.append(frame_path)
            export_idx += 1

            if export_idx % 25 == 0:
                print(f"Saved {export_idx} frames, latest raw frame {raw_frame_id}")

        cached_static_points.append(static_points)
        cached_static_colors.append(static_colors)
        processed_offset += count
        print(f"Finished chunk {chunk_idx + 1}/{len(chunks)}: {os.path.basename(chunk.path)}")

        if args.max_frames > 0 and export_idx >= args.max_frames:
            break

    if video_writer is not None:
        video_writer.release()
        print(f"Saved video: {video_path}")
    if args.export_gif:
        gif_path = os.path.join(output_dir, "sequence_submaps_topdown.gif")
        write_gif(frame_paths, gif_path, args.fps)
        print(f"Saved GIF: {gif_path}")
    if args.save_frames or args.export_gif:
        print(f"Saved frames: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a long sequence stored as many LSG-SLAM submap params.npz files."
    )
    parser.add_argument("--config", default="", help="Experiment config. Used to derive workdir and scene_name.")
    parser.add_argument("--base-folder", default="", help="Result group folder, e.g. results/kitti360-0000-all.")
    parser.add_argument("--scene-name", default="", help="Scene prefix used by submap result folders.")
    parser.add_argument("--output-dir", default="viz_sequence_submaps", help="Output directory.")
    parser.add_argument("--video-path", default="", help="Exact MP4 output path. Overrides --output-dir filename.")
    parser.add_argument("--trajectory", choices=["auto", "optimized", "odom"], default="auto")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frame-stride", type=int, default=1, help="Export every N processed frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all exported frames.")
    parser.add_argument("--max-points", type=int, default=600000, help="Maximum points drawn per frame; 0 means no cap.")
    parser.add_argument("--max-points-per-submap", type=int, default=50000, help="Static points kept from each submap; 0 means no cap.")
    parser.add_argument("--point-size", type=int, default=1)
    parser.add_argument("--zoom", type=float, default=1.25, help="Larger values zoom out.")
    parser.set_defaults(highlight_dynamic=True)
    parser.add_argument("--highlight-dynamic", dest="highlight_dynamic", action="store_true")
    parser.add_argument("--no-highlight-dynamic", dest="highlight_dynamic", action="store_false")
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--save-frames", action="store_true", help="Also save PNG frames.")
    parser.add_argument("--export-video", action="store_true")
    parser.add_argument("--export-gif", action="store_true")
    parser.set_defaults(video_only=True)
    parser.add_argument("--video-only", dest="video_only", action="store_true", help="Write MP4 directly without PNG/GIF side files.")
    parser.add_argument("--with-side-files", dest="video_only", action="store_false", help="Allow requested PNG/GIF side files.")
    args = parser.parse_args()

    export_sequence(args)


if __name__ == "__main__":
    main()

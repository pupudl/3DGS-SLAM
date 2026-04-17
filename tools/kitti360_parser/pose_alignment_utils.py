#!/usr/bin/env python3
"""
Pose helpers shared by KITTI-360 LiDAR ICP scripts.

Design goal: follow the original project pipeline as closely as possible.

- Original pipeline pose storage: `w2c` (world -> cam0)
- Original pipeline plotting: convert to `c2w`, then plot the `x-z` plane
- Recommended fusion reference frame: `cam0`
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ORIGINAL_PIPELINE_POSE_CONVENTION = "w2c"
ORIGINAL_PIPELINE_SENSOR_FRAME = "cam0"
ORIGINAL_PIPELINE_PLOT_AXES = ("x", "z")
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
def load_velo_to_cam(calib_path: str) -> np.ndarray:
    """Load KITTI-360 `calib_cam_to_velo.txt` as Velodyne -> cam0."""
    with open(calib_path, "r") as f:
        line = f.readline().strip().split()
    vals = [float(x) for x in line]
    if len(vals) != 12:
        raise ValueError(f"Expected 12 values in calibration file, got {len(vals)}: {calib_path}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
    return T


def read_cam0_to_world_map(path: str) -> Dict[int, np.ndarray]:
    """
    Read KITTI-360 `cam0_to_world.txt`.

    Each line is: frame_id + 16 values (row-major 4x4), i.e. `cam0 c2w`.
    """
    mp: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 17:
                continue
            try:
                fid = int(vals[0])
            except ValueError:
                continue
            mat = np.array([float(x) for x in vals[1:]], dtype=np.float64).reshape(4, 4)
            mp[fid] = mat
    return mp


def pose_to_kitti_line(T: np.ndarray) -> str:
    return " ".join(f"{v:.12g}" for v in T[:3, :].reshape(-1))


def read_kitti_pose_file(path: str) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals = [float(x) for x in s.split()]
            if len(vals) != 12:
                raise ValueError(f"File {path} contains a non-12-value line: {s}")
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            poses.append(T)
    return poses


def write_pose_file(path: str, poses: Sequence[np.ndarray]) -> None:
    with open(path, "w") as f:
        for T in poses:
            f.write(pose_to_kitti_line(T) + "\n")


def normalize_frame_id(fid: str) -> Optional[int]:
    try:
        return int(fid)
    except ValueError:
        return None


def read_frame_ids(path: str) -> List[str]:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def convert_pose_convention(
    poses: Sequence[np.ndarray],
    src_convention: str,
    dst_convention: str,
) -> List[np.ndarray]:
    """
    Pose convention conversion.

    - `c2w`: T_world_sensor
    - `w2c`: T_sensor_world
    """
    if src_convention == dst_convention:
        return [pose.copy() for pose in poses]
    if {src_convention, dst_convention} != {"c2w", "w2c"}:
        raise ValueError(f"Unsupported pose convention conversion: {src_convention} -> {dst_convention}")
    return [np.linalg.inv(pose) for pose in poses]


def _convert_single_sensor_pose(
    pose: np.ndarray,
    src_sensor: str,
    dst_sensor: str,
    pose_convention: str,
    T_velo_to_cam: np.ndarray,
) -> np.ndarray:
    if src_sensor == dst_sensor:
        return pose.copy()
    if {src_sensor, dst_sensor} != {"velo", "cam0"}:
        raise ValueError(f"Unsupported sensor frame conversion: {src_sensor} -> {dst_sensor}")

    T_cam_to_velo = np.linalg.inv(T_velo_to_cam)
    if src_sensor == "velo" and dst_sensor == "cam0":
        if pose_convention == "c2w":
            return pose @ T_cam_to_velo
        if pose_convention == "w2c":
            return T_velo_to_cam @ pose
    if src_sensor == "cam0" and dst_sensor == "velo":
        if pose_convention == "c2w":
            return pose @ T_velo_to_cam
        if pose_convention == "w2c":
            return T_cam_to_velo @ pose
    raise ValueError(f"Unsupported pose convention {pose_convention}")


def convert_sensor_frame(
    poses: Sequence[np.ndarray],
    src_sensor: str,
    dst_sensor: str,
    pose_convention: str,
    T_velo_to_cam: np.ndarray,
) -> List[np.ndarray]:
    return [
        _convert_single_sensor_pose(pose, src_sensor, dst_sensor, pose_convention, T_velo_to_cam)
        for pose in poses
    ]


def poses_to_xyz(
    poses: Sequence[np.ndarray],
    pose_convention: str,
) -> np.ndarray:
    c2w_poses = convert_pose_convention(poses, pose_convention, "c2w")
    if len(c2w_poses) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([pose[:3, 3] for pose in c2w_poses], dtype=np.float64)


def align_c2w_poses_to_first_frame(
    poses: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """
    Convert absolute `c2w` poses into the original pipeline style local world:
    the first frame becomes identity and all later poses are expressed relative to it.
    """
    if len(poses) == 0:
        return []
    first_w2c = np.linalg.inv(poses[0])
    return [first_w2c @ pose for pose in poses]


def align_poses_to_first_frame(
    poses: Sequence[np.ndarray],
    pose_convention: str,
) -> List[np.ndarray]:
    c2w_poses = convert_pose_convention(poses, pose_convention, "c2w")
    aligned_c2w = align_c2w_poses_to_first_frame(c2w_poses)
    return convert_pose_convention(aligned_c2w, "c2w", pose_convention)


def estimate_rotation_alignment(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    ignore_first: bool = True,
) -> np.ndarray:
    if src_xyz.shape != dst_xyz.shape:
        raise ValueError(f"Point sets must have the same shape, got {src_xyz.shape} vs {dst_xyz.shape}")
    if src_xyz.ndim != 2 or src_xyz.shape[1] != 3:
        raise ValueError(f"Point sets must have shape (N, 3), got {src_xyz.shape}")

    start_idx = 1 if ignore_first and src_xyz.shape[0] > 1 else 0
    X = src_xyz[start_idx:]
    Y = dst_xyz[start_idx:]
    if X.shape[0] == 0:
        return np.eye(3, dtype=np.float64)

    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    return R


def conjugate_c2w_poses_by_rotation(
    poses: Sequence[np.ndarray],
    rotation: np.ndarray,
) -> List[np.ndarray]:
    C = np.eye(4, dtype=np.float64)
    C[:3, :3] = rotation
    C_inv = np.eye(4, dtype=np.float64)
    C_inv[:3, :3] = rotation.T
    return [C @ pose @ C_inv for pose in poses]


def align_local_pose_axes_to_reference(
    src_poses: Sequence[np.ndarray],
    dst_poses: Sequence[np.ndarray],
    src_pose_convention: str,
    dst_pose_convention: str,
    output_pose_convention: Optional[str] = None,
    num_frames: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    if output_pose_convention is None:
        output_pose_convention = src_pose_convention

    src_local_c2w = align_c2w_poses_to_first_frame(
        convert_pose_convention(src_poses, src_pose_convention, "c2w")
    )
    dst_local_c2w = align_c2w_poses_to_first_frame(
        convert_pose_convention(dst_poses, dst_pose_convention, "c2w")
    )

    if num_frames is not None:
        use_n = max(1, min(len(src_local_c2w), len(dst_local_c2w), num_frames))
    else:
        use_n = min(len(src_local_c2w), len(dst_local_c2w))

    src_xyz = np.array([pose[:3, 3] for pose in src_local_c2w[:use_n]], dtype=np.float64)
    dst_xyz = np.array([pose[:3, 3] for pose in dst_local_c2w[:use_n]], dtype=np.float64)
    rotation = estimate_rotation_alignment(src_xyz, dst_xyz, ignore_first=True)

    aligned_local_c2w = conjugate_c2w_poses_by_rotation(src_local_c2w, rotation)
    aligned_xyz = np.array([pose[:3, 3] for pose in aligned_local_c2w[:use_n]], dtype=np.float64)
    align_err = np.linalg.norm(aligned_xyz - dst_xyz, axis=1) if use_n > 0 else np.zeros((0,), dtype=np.float64)

    info: Dict[str, object] = {
        "rotation": rotation,
        "frames_used": use_n,
        "mean_translation_error_m": float(np.mean(align_err)) if align_err.size else 0.0,
        "max_translation_error_m": float(np.max(align_err)) if align_err.size else 0.0,
    }
    return convert_pose_convention(aligned_local_c2w, "c2w", output_pose_convention), info


def summarize_axis_span(xyz: np.ndarray) -> Dict[str, float]:
    if xyz.size == 0:
        return {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        axis: float(np.max(xyz[:, idx]) - np.min(xyz[:, idx]))
        for axis, idx in AXIS_TO_INDEX.items()
    }


def describe_plot_plane_like_original() -> Tuple[Tuple[str, str], str]:
    axes = ORIGINAL_PIPELINE_PLOT_AXES
    reason = (
        "Use x-z because the original pipeline converts stored w2c poses to c2w translations "
        "and plots x against z in KITTI camera coordinates."
    )
    return axes, reason


def to_plot_trajectory(
    poses: Sequence[np.ndarray],
    pose_convention: str,
    plot_axes: Optional[Tuple[str, str]] = None,
) -> Dict[str, object]:
    axes, reason = describe_plot_plane_like_original() if plot_axes is None else (plot_axes, "custom")
    xyz = poses_to_xyz(poses, pose_convention)
    axis_0, axis_1 = axes
    return {
        "xyz": xyz,
        "plot_axes": axes,
        "plot_xy": xyz[:, [AXIS_TO_INDEX[axis_0], AXIS_TO_INDEX[axis_1]]] if xyz.size else np.zeros((0, 2)),
        "axis_span": summarize_axis_span(xyz),
        "reason": reason,
    }


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("Quaternion has zero norm.")
    r, x, y, z = q / norm
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y)],
            [2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x)],
            [2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def _load_original_pipeline_npz(path: str) -> List[np.ndarray]:
    params = dict(np.load(path, allow_pickle=True))
    if "gt_w2c_all_frames" in params:
        return [np.array(pose, dtype=np.float64) for pose in params["gt_w2c_all_frames"]]
    if "cam_unnorm_rots" not in params or "cam_trans" not in params:
        raise ValueError(
            "Original pipeline npz must contain `gt_w2c_all_frames` or (`cam_unnorm_rots`, `cam_trans`)."
        )
    cam_rots = np.array(params["cam_unnorm_rots"], dtype=np.float64)
    cam_trans = np.array(params["cam_trans"], dtype=np.float64)
    num_frames = cam_rots.shape[-1]
    poses: List[np.ndarray] = []
    for idx in range(num_frames):
        quat = cam_rots[..., idx].reshape(-1)
        trans = cam_trans[..., idx].reshape(-1)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _quat_wxyz_to_rotmat(quat)
        pose[:3, 3] = trans[:3]
        poses.append(pose)
    return poses


def load_original_pipeline_poses(
    path: str,
    pose_convention: str = ORIGINAL_PIPELINE_POSE_CONVENTION,
    sensor_frame: str = ORIGINAL_PIPELINE_SENSOR_FRAME,
) -> List[np.ndarray]:
    """
    Load original pipeline poses.

    Supported inputs:
    - `.npz`: saved pipeline params, loaded as `w2c` cam0
    - `.txt` / `.csv`: KITTI-style 12-value pose file
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        poses = _load_original_pipeline_npz(path)
        src_convention = "w2c"
    elif ext in {".txt", ".csv"}:
        poses = read_kitti_pose_file(path)
        src_convention = pose_convention
    else:
        raise ValueError(f"Unsupported original pipeline pose file: {path}")

    poses = convert_pose_convention(poses, src_convention, pose_convention)
    if sensor_frame != ORIGINAL_PIPELINE_SENSOR_FRAME:
        raise ValueError("Original pipeline poses are only defined in cam0 unless explicit calibration conversion is used.")
    return poses


def load_lidar_icp_poses(
    path: str,
    pose_convention: str,
    sensor_frame: str,
    target_pose_convention: str = ORIGINAL_PIPELINE_POSE_CONVENTION,
    target_sensor_frame: str = ORIGINAL_PIPELINE_SENSOR_FRAME,
    T_velo_to_cam: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    poses = read_kitti_pose_file(path)
    poses = convert_pose_convention(poses, pose_convention, target_pose_convention)
    if sensor_frame != target_sensor_frame:
        if T_velo_to_cam is None:
            raise ValueError("T_velo_to_cam is required when converting LiDAR ICP sensor frames.")
        poses = convert_sensor_frame(
            poses,
            src_sensor=sensor_frame,
            dst_sensor=target_sensor_frame,
            pose_convention=target_pose_convention,
            T_velo_to_cam=T_velo_to_cam,
        )
    return poses


def frame_ids_from_gt_map(frame_ids: Iterable[str], cam0_to_world: Dict[int, np.ndarray]) -> List[np.ndarray]:
    poses: List[np.ndarray] = []
    for fid in frame_ids:
        fid_int = normalize_frame_id(fid)
        if fid_int is not None and fid_int in cam0_to_world:
            poses.append(cam0_to_world[fid_int])
    return poses

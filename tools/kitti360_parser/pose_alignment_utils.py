#!/usr/bin/env python3
"""
Pose helpers shared by KITTI-360 LiDAR ICP scripts.

Design goal: follow the original project pipeline as closely as possible.

- Original pipeline pose storage: `w2c` (world -> cam0)
- Original pipeline plotting: convert to `c2w`, then plot the `x-z` plane
- Recommended fusion reference frame: `cam0`
"""

from __future__ import annotations

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


def conjugate_c2w_poses_by_rotation(
    poses: Sequence[np.ndarray],
    rotation: np.ndarray,
) -> List[np.ndarray]:
    C = np.eye(4, dtype=np.float64)
    C[:3, :3] = rotation
    C_inv = np.eye(4, dtype=np.float64)
    C_inv[:3, :3] = rotation.T
    return [C @ pose @ C_inv for pose in poses]


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

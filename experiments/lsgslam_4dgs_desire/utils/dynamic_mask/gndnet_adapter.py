import importlib.util
import collections
import collections.abc
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
import yaml


class _ConfigClass:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def _load_gndnet_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    config_dict["batch_size"] = 1
    return _ConfigClass(**config_dict)


def _install_ipdb_stub():
    if "ipdb" in sys.modules:
        return
    try:
        __import__("ipdb")
        return
    except ImportError:
        pass

    stub = types.ModuleType("ipdb")
    stub.set_trace = lambda: None
    sys.modules["ipdb"] = stub


def _install_numba_stub():
    if "numba" in sys.modules:
        return
    try:
        __import__("numba")
        return
    except ImportError:
        pass

    def jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    stub = types.ModuleType("numba")
    stub.jit = jit
    stub.types = types.SimpleNamespace()
    sys.modules["numba"] = stub


def _load_gndnet_model_class(repo_root):
    model_path = os.path.join(repo_root, "model.py")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing GndNet model.py: {model_path}")

    _install_ipdb_stub()
    _install_numba_stub()
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    spec = importlib.util.spec_from_file_location("_lsgslam_gndnet_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import GndNet model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    inserted = repo_root not in sys.path
    if inserted:
        sys.path.insert(0, repo_root)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(repo_root)
    return module.GroundEstimatorNet


def _points_to_voxel(points, voxel_size, coors_range, max_points, max_voxels, reverse_index=True):
    voxel_size = np.asarray(voxel_size, dtype=points.dtype)
    coors_range = np.asarray(coors_range, dtype=points.dtype)
    grid_size = np.round((coors_range[3:] - coors_range[:3]) / voxel_size).astype(np.int32)

    coors_xyz = np.floor((points[:, :3] - coors_range[:3]) / voxel_size).astype(np.int32)
    valid = np.all((coors_xyz >= 0) & (coors_xyz < grid_size), axis=1)
    coors_xyz = coors_xyz[valid]
    valid_points = points[valid]
    if reverse_index:
        coors_all = coors_xyz[:, [2, 1, 0]]
    else:
        coors_all = coors_xyz

    voxels = np.zeros((max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros((max_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros((max_voxels,), dtype=np.int32)
    voxel_index = {}
    voxel_num = 0

    for point, coor in zip(valid_points, coors_all):
        key = (int(coor[0]), int(coor[1]), int(coor[2]))
        idx = voxel_index.get(key)
        if idx is None:
            if voxel_num >= max_voxels:
                break
            idx = voxel_num
            voxel_index[key] = idx
            coors[idx] = coor
            voxel_num += 1

        point_idx = num_points_per_voxel[idx]
        if point_idx < max_points:
            voxels[idx, point_idx] = point
            num_points_per_voxel[idx] += 1

    return (
        voxels[:voxel_num],
        coors[:voxel_num],
        num_points_per_voxel[:voxel_num],
    )


def _segment_cloud(points, grid_range, voxel_size, elevation_map, threshold):
    grid_range = np.asarray(grid_range, dtype=np.float32)
    xy = np.floor((points[:, :2] - grid_range[:2]) / voxel_size).astype(np.int32)
    segment = np.full((points.shape[0],), -1, dtype=np.int8)
    valid = (
        (xy[:, 0] > 0)
        & (xy[:, 0] < elevation_map.shape[0])
        & (xy[:, 1] > 0)
        & (xy[:, 1] < elevation_map.shape[1])
    )
    valid_xy = xy[valid]
    valid_z = points[valid, 2]
    valid_ground = elevation_map[valid_xy[:, 0], valid_xy[:, 1]]
    segment[valid] = np.where(valid_z > valid_ground + threshold, 1, 0)
    return segment


class GndNetGroundFilter:
    def __init__(
        self,
        repo_root,
        checkpoint_path,
        config_path,
        device,
        threshold_m=0.2,
        keep_outside_as_nonground=True,
    ):
        self.repo_root = os.path.abspath(repo_root)
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.config_path = os.path.abspath(config_path)
        self.device = torch.device(device)
        self.threshold_m = float(threshold_m)
        self.keep_outside_as_nonground = bool(keep_outside_as_nonground)
        self.cfg = None
        self.model = None

    def predict_nonground_mask(self, points_xyzi):
        self._ensure_loaded()
        points = np.asarray(points_xyzi, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected point cloud with shape (N, 3+) or (N, 4), got {points.shape}")
        if points.shape[1] == 3:
            points = np.concatenate(
                [points, np.zeros((points.shape[0], 1), dtype=np.float32)],
                axis=1,
            )
        else:
            points = points[:, :4].copy()

        finite = np.isfinite(points[:, :3]).all(axis=1)
        segment = np.full((points.shape[0],), -1, dtype=np.int8)
        if not finite.any():
            nonground = np.zeros((points.shape[0],), dtype=bool)
            return nonground, self._make_info(segment, 0, nonground)

        inference_points = points[finite].copy()
        shifted_points = inference_points.copy()
        shifted_points[:, 2] += float(self.cfg.lidar_height)

        voxels, coors, num_points = _points_to_voxel(
            shifted_points,
            self.cfg.voxel_size,
            self.cfg.pc_range,
            int(self.cfg.max_points_voxel),
            int(self.cfg.max_voxels),
            reverse_index=True,
        )
        if voxels.shape[0] == 0:
            nonground = finite if self.keep_outside_as_nonground else np.zeros_like(finite)
            return nonground, self._make_info(segment, 0, nonground)

        voxels_t = torch.from_numpy(voxels).float().to(self.device)
        coors_t = torch.from_numpy(coors).to(self.device)
        coors_t = F.pad(coors_t, (1, 0), "constant", 0).float()
        num_points_t = torch.from_numpy(num_points).float().to(self.device)

        with torch.no_grad():
            elevation = self.model(voxels_t, coors_t, num_points_t)
        elevation = elevation.detach().cpu().numpy().squeeze()
        if elevation.ndim != 2:
            raise RuntimeError(f"Unexpected GndNet elevation map shape: {elevation.shape}")

        # Upstream GndNet shifts the input cloud in-place, then segments the
        # shifted coordinates against the predicted elevation map.
        segment[finite] = _segment_cloud(
            shifted_points,
            np.asarray(self.cfg.grid_range),
            float(self.cfg.voxel_size[0]),
            elevation.T,
            self.threshold_m,
        )

        nonground = segment == 1
        if self.keep_outside_as_nonground:
            nonground |= segment < 0
        nonground &= finite
        info = self._make_info(segment, voxels.shape[0], nonground)
        return nonground, info

    def _ensure_loaded(self):
        if self.model is not None:
            return
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"GndNet requested CUDA device {self.device}, but CUDA is not available")
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Missing GndNet config: {self.config_path}")
        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Missing GndNet checkpoint: {self.checkpoint_path}")

        self.cfg = _load_gndnet_config(self.config_path)
        model_cls = _load_gndnet_model_class(self.repo_root)
        model = model_cls(self.cfg).to(self.device)
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    def _make_info(self, segment, num_voxels, nonground_mask):
        return {
            "input_points": int(segment.shape[0]),
            "ground_points": int(np.count_nonzero(segment == 0)),
            "nonground_points": int(np.count_nonzero(segment == 1)),
            "outside_points": int(np.count_nonzero(segment < 0)),
            "kept_points": int(np.count_nonzero(nonground_mask)),
            "num_voxels": int(num_voxels),
            "threshold_m": float(self.threshold_m),
            "keep_outside_as_nonground": bool(self.keep_outside_as_nonground),
            "device": str(self.device),
        }

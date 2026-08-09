import math

import torch


DESIRE_4DGS_PARAM_KEYS = ("_t", "_scaling_t", "_velocity")


def cfg_enabled(cfg):
    cfg = cfg or {}
    return bool(cfg.get("enabled", cfg.get("enable_dynamic", False)))


def is_enabled(params, cfg=None):
    if cfg is not None and not cfg_enabled(cfg):
        return False
    return all(key in params for key in DESIRE_4DGS_PARAM_KEYS)


def frame_timestamp(time_idx, num_frames, cfg, device=None, dtype=None):
    cfg = cfg or {}
    time_duration = cfg.get("time_duration", [0.0, 1.0])
    start_t = float(time_duration[0])
    end_t = float(time_duration[1])
    if num_frames <= 1:
        timestamp = 0.5 * (start_t + end_t)
    else:
        timestamp = start_t + (end_t - start_t) * float(time_idx) / float(num_frames - 1)
    return torch.tensor(timestamp, device=device, dtype=dtype or torch.float32)


def initial_t(num_pts, time_idx, num_frames, cfg, device="cuda", dtype=torch.float32):
    timestamp = frame_timestamp(time_idx, num_frames, cfg, device=device, dtype=dtype)
    return torch.full((num_pts, 1), float(timestamp.item()), device=device, dtype=dtype)


def initial_scaling_t(num_pts, cfg, device="cuda", dtype=torch.float32):
    cfg = cfg or {}
    time_duration = cfg.get("time_duration", [0.0, 1.0])
    t_init = float(cfg.get("t_init", 0.1))
    dist_t = max((float(time_duration[1]) - float(time_duration[0])) * t_init, 1e-8)
    return torch.log(torch.sqrt(torch.full((num_pts, 1), dist_t, device=device, dtype=dtype)))


def initial_velocity(num_pts, device="cuda", dtype=torch.float32):
    return torch.full((num_pts, 3), 0.0, device=device, dtype=dtype)


def add_initial_params(params, num_pts, time_idx, num_frames, cfg):
    if not cfg_enabled(cfg):
        return params
    device = params["means3D"].device if isinstance(params["means3D"], torch.Tensor) else "cuda"
    dtype = params["means3D"].dtype if isinstance(params["means3D"], torch.Tensor) else torch.float32
    params["_t"] = initial_t(num_pts, time_idx, num_frames, cfg, device=device, dtype=dtype)
    params["_scaling_t"] = initial_scaling_t(num_pts, cfg, device=device, dtype=dtype)
    params["_velocity"] = initial_velocity(num_pts, device=device, dtype=dtype)
    return params


def add_missing_params(params, time_idx, num_frames, cfg):
    if not cfg_enabled(cfg) or is_enabled(params, cfg):
        return params
    num_pts = params["means3D"].shape[0]
    device = params["means3D"].device
    dtype = params["means3D"].dtype
    added = {
        "_t": initial_t(num_pts, time_idx, num_frames, cfg, device=device, dtype=dtype),
        "_scaling_t": initial_scaling_t(num_pts, cfg, device=device, dtype=dtype),
        "_velocity": initial_velocity(num_pts, device=device, dtype=dtype),
    }
    for key, value in added.items():
        params[key] = torch.nn.Parameter(value.float().contiguous().requires_grad_(True))
    return params


def get_scaling_t(params):
    return torch.exp(params["_scaling_t"])


def get_xyz_SHM(params, timestamp, cfg):
    cfg = cfg or {}
    cycle = float(cfg.get("cycle", cfg.get("T", 0.2)))
    a = 1.0 / cycle * math.pi * 2.0
    return params["means3D"] + params["_velocity"] * torch.sin((timestamp - params["_t"]) * a) / a


def get_inst_velocity(params, cfg):
    cfg = cfg or {}
    cycle = float(cfg.get("cycle", cfg.get("T", 0.2)))
    velocity_decay = float(cfg.get("velocity_decay", 1.0))
    return params["_velocity"] * torch.exp(-get_scaling_t(params) / cycle / 2.0 * velocity_decay)


def get_marginal_t(params, timestamp):
    scaling_t = get_scaling_t(params)
    return torch.exp(-0.5 * (params["_t"] - timestamp) ** 2 / scaling_t ** 2)


def get_dynamic_state(params, time_idx, num_frames, cfg, gaussians_grad=True):
    dyn_params = params
    if not gaussians_grad:
        dyn_params = dict(params)
        dyn_params["means3D"] = params["means3D"].detach()
        dyn_params["_t"] = params["_t"].detach()
        dyn_params["_scaling_t"] = params["_scaling_t"].detach()
        dyn_params["_velocity"] = params["_velocity"].detach()

    timestamp = frame_timestamp(
        time_idx,
        num_frames,
        cfg,
        device=dyn_params["means3D"].device,
        dtype=dyn_params["means3D"].dtype,
    )
    return {
        "timestamp": timestamp,
        "means3D": get_xyz_SHM(dyn_params, timestamp, cfg),
        "marginal_t": get_marginal_t(dyn_params, timestamp),
        "scaling_t": get_scaling_t(dyn_params),
        "inst_velocity": get_inst_velocity(dyn_params, cfg),
    }

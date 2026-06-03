import torch


GAUSSIAN_PARAM_KEYS = {
    "means3D",
    "rgb_colors",
    "unnorm_rotations",
    "logit_opacities",
    "log_scales",
}


def initialize_support_counts(num_gaussians, device="cuda"):
    return torch.zeros(num_gaussians, device=device, dtype=torch.float32)


def extend_support_counts(variables, num_new):
    if "support_count" not in variables:
        return variables
    if num_new <= 0:
        return variables
    zeros = torch.zeros(num_new, device=variables["support_count"].device, dtype=variables["support_count"].dtype)
    variables["support_count"] = torch.cat((variables["support_count"], zeros), dim=0)
    return variables


def _build_projected_gaussian_index_map(transformed_means3d, intrinsics, image_height, image_width):
    # transformed_means3d are already in the current camera frame.
    z = transformed_means3d[:, 2]
    valid = z > 1e-6
    if not torch.any(valid):
        num_pixels = image_height * image_width
        empty_idx = torch.full((num_pixels,), -1, dtype=torch.long, device=transformed_means3d.device)
        empty_depth = torch.full((num_pixels,), float("inf"), dtype=transformed_means3d.dtype, device=transformed_means3d.device)
        return empty_idx, empty_depth

    pts = transformed_means3d[valid]
    gaussian_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    u = fx * (pts[:, 0] / pts[:, 2]) + cx
    v = fy * (pts[:, 1] / pts[:, 2]) + cy

    x0 = torch.floor(u).long()
    y0 = torch.floor(v).long()
    x1 = x0 + 1
    y1 = y0 + 1

    xs = torch.stack((x0, x0, x1, x1), dim=1).reshape(-1)
    ys = torch.stack((y0, y1, y0, y1), dim=1).reshape(-1)
    depth = pts[:, 2].repeat_interleave(4)
    expanded_indices = gaussian_indices.repeat_interleave(4)

    inside = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    xs = xs[inside]
    ys = ys[inside]
    depth = depth[inside]
    expanded_indices = expanded_indices[inside]

    num_pixels = image_height * image_width
    idx_map = torch.full((num_pixels,), -1, dtype=torch.long, device=transformed_means3d.device)
    depth_map = torch.full((num_pixels,), float("inf"), dtype=transformed_means3d.dtype, device=transformed_means3d.device)
    if xs.numel() == 0:
        return idx_map, depth_map

    linear_idx = ys * image_width + xs
    # Sort by pixel first, then by depth, so the first occurrence per pixel is the nearest visible Gaussian.
    sort_key = linear_idx.to(depth.dtype) * (depth.max() + 1.0) + depth
    order = torch.argsort(sort_key)
    linear_idx = linear_idx[order]
    depth = depth[order]
    expanded_indices = expanded_indices[order]

    keep = torch.ones_like(linear_idx, dtype=torch.bool)
    keep[1:] = linear_idx[1:] != linear_idx[:-1]

    chosen_pixels = linear_idx[keep]
    chosen_depth = depth[keep]
    chosen_indices = expanded_indices[keep]

    idx_map[chosen_pixels] = chosen_indices
    depth_map[chosen_pixels] = chosen_depth
    return idx_map, depth_map


def update_support_counts(
    params,
    variables,
    curr_data,
    transformed_gaussians,
    color_threshold,
    depth_tolerance,
):
    if curr_data["depth"].numel() == 0 or params["means3D"].shape[0] == 0:
        return variables, 0

    image_height = curr_data["depth"].shape[1]
    image_width = curr_data["depth"].shape[2]
    idx_map, depth_map = _build_projected_gaussian_index_map(
        transformed_gaussians["means3D"].detach(),
        curr_data["intrinsics"],
        image_height,
        image_width,
    )

    gt_depth = curr_data["depth"][0].reshape(-1)
    valid_depth = gt_depth > 0
    if not torch.any(valid_depth):
        return variables, 0

    pixel_indices = torch.nonzero(valid_depth, as_tuple=False).squeeze(1)
    matched_gaussians = idx_map[pixel_indices]
    matched_depths = depth_map[pixel_indices]
    valid_gaussian = matched_gaussians >= 0
    if not torch.any(valid_gaussian):
        return variables, 0

    pixel_indices = pixel_indices[valid_gaussian]
    matched_gaussians = matched_gaussians[valid_gaussian]
    matched_depths = matched_depths[valid_gaussian]
    obs_depth = gt_depth[pixel_indices]

    depth_mask = torch.abs(matched_depths - obs_depth) < depth_tolerance
    if not torch.any(depth_mask):
        return variables, 0

    pixel_indices = pixel_indices[depth_mask]
    matched_gaussians = matched_gaussians[depth_mask]

    image_colors = curr_data["im"].permute(1, 2, 0).reshape(-1, 3)[pixel_indices]
    gaussian_colors = params["rgb_colors"].detach()[matched_gaussians]
    color_error = torch.mean(torch.abs(image_colors - gaussian_colors), dim=1)
    consistent_mask = color_error < color_threshold
    if not torch.any(consistent_mask):
        return variables, 0

    consistent_gaussians = matched_gaussians[consistent_mask]
    hits = torch.bincount(
        consistent_gaussians,
        minlength=params["means3D"].shape[0],
    ).to(variables["support_count"].dtype)
    variables["support_count"] += hits
    return variables, int(consistent_mask.sum().item())


def filter_params_by_support_count(params, support_count, min_count):
    if support_count.numel() == 0:
        keep_mask = support_count.bool()
    else:
        keep_mask = support_count >= min_count
        if not torch.any(keep_mask):
            keep_mask[torch.argmax(support_count)] = True

    filtered_params = {}
    for key, value in params.items():
        if key in GAUSSIAN_PARAM_KEYS:
            filtered_params[key] = value[keep_mask]
        else:
            filtered_params[key] = value
    return filtered_params, keep_mask

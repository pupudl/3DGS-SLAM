"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np
import torch.nn.functional as F


def _resize_exclusion_mask(mask, height, width, device):
    """Convert a stored sky/dynamic mask to a HxW boolean tensor."""
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=device)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.shape != (height, width):
        mask = F.interpolate(
            mask.float()[None, None],
            size=(height, width),
            mode="nearest",
        )[0, 0]
    return mask.bool()


@torch.no_grad()
def keyframe_selection_covisibility(
    gt_depth,
    w2c,
    intrinsics,
    keyframe_list,
    min_score=0.10,
    relative_score_ratio=0.0,
    pixels=1600,
    depth_abs_tolerance=0.20,
    depth_rel_tolerance=0.05,
    edge=20,
):
    """Return every keyframe sufficiently co-visible with the current frame.

    Co-visibility is the fraction of sampled valid current-frame depth points
    which project inside a keyframe and agree with its observed depth.  The
    effective threshold is the larger of ``min_score`` and
    ``relative_score_ratio * best_historical_score``.  The result is sorted by
    descending score and is deliberately not capped to a fixed window size.
    """
    if gt_depth.ndim != 3 or gt_depth.shape[0] != 1:
        raise ValueError("gt_depth must have shape [1, H, W]")
    if pixels <= 0:
        raise ValueError("pixels must be positive")
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be in [0, 1]")
    if not 0.0 <= relative_score_ratio <= 1.0:
        raise ValueError("relative_score_ratio must be in [0, 1]")
    if depth_abs_tolerance < 0 or depth_rel_tolerance < 0:
        raise ValueError("depth tolerances must be non-negative")
    if edge < 0:
        raise ValueError("edge must be non-negative")

    device = gt_depth.device
    dtype = gt_depth.dtype
    height, width = gt_depth.shape[-2:]
    valid_depth = torch.isfinite(gt_depth[0]) & (gt_depth[0] > 0)
    valid_indices = torch.nonzero(valid_depth, as_tuple=False)
    if valid_indices.shape[0] == 0 or not keyframe_list:
        return []

    # Deterministic subsampling keeps the adaptive window stable across runs.
    num_samples = min(int(pixels), int(valid_indices.shape[0]))
    if num_samples < valid_indices.shape[0]:
        sample_positions = torch.linspace(
            0,
            valid_indices.shape[0] - 1,
            steps=num_samples,
            device=device,
        ).round().long()
        sampled_indices = valid_indices[sample_positions]
    else:
        sampled_indices = valid_indices

    intrinsics = intrinsics[:3, :3].to(device=device, dtype=dtype)
    w2c = w2c.to(device=device, dtype=dtype)
    rows, cols = sampled_indices[:, 0], sampled_indices[:, 1]
    depth_z = gt_depth[0, rows, cols]
    x = (cols.to(dtype) - intrinsics[0, 2]) * depth_z / intrinsics[0, 0]
    y = (rows.to(dtype) - intrinsics[1, 2]) * depth_z / intrinsics[1, 1]
    points_camera = torch.stack((x, y, depth_z), dim=-1)
    points_camera_h = torch.cat(
        (points_camera, torch.ones_like(points_camera[:, :1])), dim=-1
    )
    points_world = (torch.linalg.inv(w2c) @ points_camera_h.T).T

    scored_keyframes = []
    for keyframe_idx, keyframe in enumerate(keyframe_list):
        keyframe_depth = keyframe.get("depth")
        keyframe_w2c = keyframe.get("est_w2c")
        if keyframe_depth is None or keyframe_w2c is None:
            continue

        keyframe_depth = keyframe_depth.to(device=device, dtype=dtype)
        key_height, key_width = keyframe_depth.shape[-2:]
        keyframe_intrinsics = keyframe.get("intrinsics", intrinsics)
        keyframe_intrinsics = keyframe_intrinsics[:3, :3].to(device=device, dtype=dtype)
        keyframe_w2c = keyframe_w2c.to(device=device, dtype=dtype)

        points_keyframe = (keyframe_w2c @ points_world.T).T[:, :3]
        projected = (keyframe_intrinsics @ points_keyframe.T).T
        projected_z = points_keyframe[:, 2]
        projected_xy = projected[:, :2] / projected[:, 2:3].clamp_min(1e-8)
        projected_cols = projected_xy[:, 0]
        projected_rows = projected_xy[:, 1]

        inside = (
            (projected_z > 0)
            & (projected_cols >= edge)
            & (projected_cols < key_width - edge)
            & (projected_rows >= edge)
            & (projected_rows < key_height - edge)
        )
        lookup_cols = projected_cols.round().long().clamp(0, key_width - 1)
        lookup_rows = projected_rows.round().long().clamp(0, key_height - 1)
        observed_depth = keyframe_depth[0, lookup_rows, lookup_cols]
        inside &= torch.isfinite(observed_depth) & (observed_depth > 0)

        for mask_name in ("sky_mask", "dynamic_mask"):
            exclusion_mask = _resize_exclusion_mask(
                keyframe.get(mask_name), key_height, key_width, device
            )
            if exclusion_mask is not None:
                inside &= ~exclusion_mask[lookup_rows, lookup_cols]

        tolerance = depth_abs_tolerance + depth_rel_tolerance * torch.maximum(
            observed_depth, projected_z
        )
        covisible = inside & (torch.abs(observed_depth - projected_z) <= tolerance)
        num_covisible = int(covisible.sum().item())
        score = num_covisible / float(num_samples)
        scored_keyframes.append(
            {
                "id": keyframe_idx,
                "score": score,
                "num_covisible": num_covisible,
                "num_samples": num_samples,
            }
        )

    if not scored_keyframes:
        return []
    best_historical_score = max(item["score"] for item in scored_keyframes)
    effective_threshold = max(
        float(min_score),
        float(relative_score_ratio) * best_historical_score,
    )
    scored_keyframes = [
        item for item in scored_keyframes
        if item["score"] >= effective_threshold
    ]
    scored_keyframes.sort(key=lambda item: (-item["score"], item["id"]))
    return scored_keyframes


def _dynamic_mask_confidence(keyframe):
    dynamic_mask = keyframe.get("dynamic_mask")
    if dynamic_mask is None or not bool(dynamic_mask.any()):
        return None
    dynamic_score = keyframe.get("dynamic_score")
    if dynamic_score is None:
        return 1.0

    dynamic_mask = torch.as_tensor(dynamic_mask).bool()
    dynamic_score = torch.as_tensor(dynamic_score, device=dynamic_mask.device).float()
    while dynamic_mask.ndim > 2:
        dynamic_mask = dynamic_mask[0]
    while dynamic_score.ndim > 2:
        dynamic_score = dynamic_score[0]
    if dynamic_score.shape != dynamic_mask.shape:
        dynamic_score = F.interpolate(
            dynamic_score[None, None],
            size=dynamic_mask.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return float(dynamic_score[dynamic_mask].mean().clamp(0.0, 1.0).item())


@torch.no_grad()
def dynamic_keyframe_selection_covisibility(
    dynamic_params,
    time_idx,
    keyframe_list,
    min_score=0.05,
    use_mask_confidence=True,
):
    """Select keyframes that observe at least one current dynamic object.

    Unlike static co-visibility, this score does not reproject points under a
    static-scene assumption. It uses the Dynamic 4DGS object visibility table:
    shared current objects / current visible objects, optionally multiplied by
    the historical mask confidence.
    """
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be in [0, 1]")
    visible = dynamic_params.get("dyn_obj_visible")
    if visible is None or visible.ndim != 2 or time_idx >= visible.shape[1]:
        return []

    visible = visible.bool()
    current_visible = visible[:, int(time_idx)]
    num_current_objects = int(current_visible.sum().item())
    if num_current_objects == 0:
        return []

    scored_keyframes = []
    for keyframe_idx, keyframe in enumerate(keyframe_list):
        keyframe_time_idx = int(keyframe.get("id", -1))
        if keyframe_time_idx < 0 or keyframe_time_idx >= visible.shape[1]:
            continue
        confidence = _dynamic_mask_confidence(keyframe)
        if confidence is None:
            continue

        shared_objects = current_visible & visible[:, keyframe_time_idx]
        num_shared_objects = int(shared_objects.sum().item())
        if num_shared_objects == 0:
            continue
        object_overlap = num_shared_objects / float(num_current_objects)
        score = object_overlap * confidence if use_mask_confidence else object_overlap
        if score >= min_score:
            scored_keyframes.append(
                {
                    "id": keyframe_idx,
                    "time_idx": keyframe_time_idx,
                    "score": score,
                    "object_overlap": object_overlap,
                    "mask_confidence": confidence,
                    "num_shared_objects": num_shared_objects,
                    "num_current_objects": num_current_objects,
                }
            )

    scored_keyframes.sort(
        key=lambda item: (-item["score"], -item["time_idx"], item["id"])
    )
    return scored_keyframes


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            est_w2c = keyframe['est_w2c']
            # Transform the 3D pointcloud to the keyframe's camera space
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]
            # Project the 3D pointcloud to the keyframe's image space
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]
            # Filter out the points that are outside the image
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)
            # Compute the percentage of points that are inside the image
            percent_inside = mask.sum()/projected_pts.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])

        return selected_keyframe_list

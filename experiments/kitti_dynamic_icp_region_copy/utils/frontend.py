from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def extract_feature(
    color: torch.Tensor,
    depth: torch.Tensor,
    sp_extractor,
    device: torch.device,
):
    image = torch.tensor(color[None, ...], dtype=torch.float32, device=device)
    return sp_extractor({"image": image, "depth": depth})


def extract_original_features(
    color_t: torch.Tensor,
    depth_original_t: torch.Tensor,
    sp_extractor,
    device: torch.device,
    depth_filter_far: float,
):
    color_feature = torch.clone(color_t)
    mask = (depth_original_t < 0.1) | (depth_original_t > np.min([depth_filter_far, 15.0]))
    color_feature[:, mask[0]] = 0
    feats, desc_all = extract_feature(color_feature, depth_original_t, sp_extractor, device)
    return feats, desc_all


def match_feature(
    curr_im: torch.Tensor,
    curr_feats: Dict[str, torch.Tensor],
    intrinsics: torch.Tensor,
    last_im: torch.Tensor,
    last_feats: Dict[str, torch.Tensor],
    lg_matcher,
    device: torch.device,
    topk: int = 1024,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    img = torch.tensor(curr_im[None, ...], dtype=torch.float32, device=device)
    last_img = torch.tensor(last_im[None, ...], dtype=torch.float32, device=device)
    pred = {
        **{k + "0": v for k, v in curr_feats.items()},
        **{k + "1": v for k, v in last_feats.items()},
        "image0": img,
        "image1": last_img,
    }
    pred = {**pred, **lg_matcher(pred)}
    pred = {
        k: v.to(device).detach()[0] if isinstance(v, torch.Tensor) else v
        for k, v in pred.items()
    }

    matches0, mscores0 = pred["matches0"], pred["matching_scores0"]
    valid = matches0 > -1
    matches = torch.stack([torch.where(valid)[0], matches0[valid]], -1)

    mscores, indices = mscores0[valid].sort(dim=0, descending=True)
    mscores = mscores[:topk]
    indices = indices[:topk]
    matches = matches[indices]

    kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    m_kpts0 = m_kpts0.detach().cpu().numpy()
    m_kpts1 = m_kpts1.detach().cpu().numpy()

    if m_kpts0.shape[0] < 20:
        return None, None, None

    mat_k = intrinsics.detach().cpu().numpy()
    _essential_matrix, mask = cv2.findEssentialMat(
        m_kpts0,
        m_kpts1,
        mat_k,
        cv2.RANSAC,
        0.999,
        1.0,
    )
    mask = mask.flatten().astype(bool)
    m_kpts0 = m_kpts0[mask]
    m_kpts1 = m_kpts1[mask]
    mscores = mscores[mask]
    if m_kpts0.shape[0] < 8:
        return None, None, None

    return (
        torch.tensor(np.asarray(m_kpts0), device=device),
        torch.tensor(np.asarray(m_kpts1), device=device),
        mscores,
    )


def estimate_pnp_from_matches(
    mkpts_cur: torch.Tensor,
    mkpts_last: torch.Tensor,
    last_depth_t: torch.Tensor,
    intrinsics_t: torch.Tensor,
    depth_filter_far: float,
):
    kps_curr = torch.clone(mkpts_cur).detach().cpu().numpy()
    kps_last = torch.clone(mkpts_last).detach().cpu().numpy()
    last_depth = torch.clone(last_depth_t).detach().cpu().numpy()[0, :, :]
    last_k = torch.clone(intrinsics_t).detach().cpu().numpy()

    points_in_last_cam = []
    uv_in_curr_image = []
    valid_indices = []
    for kpidx in range(kps_last.shape[0]):
        x = int(kps_last[kpidx, 0])
        y = int(kps_last[kpidx, 1])
        if y < 0 or y >= last_depth.shape[0] or x < 0 or x >= last_depth.shape[1]:
            continue
        point_depth = last_depth[y, x]
        if point_depth < 0.1 or point_depth > np.max([50.0, depth_filter_far]):
            continue
        point_in_last_cam = (
            point_depth
            * np.linalg.inv(last_k)
            @ np.array([kps_last[kpidx, 0], kps_last[kpidx, 1], 1.0]).reshape([3, 1])
        )
        points_in_last_cam.append(point_in_last_cam)
        uv_in_curr_image.append([float(kps_curr[kpidx, 0]), float(kps_curr[kpidx, 1])])
        valid_indices.append(kpidx)

    if len(points_in_last_cam) < 6:
        return None

    points_in_last_cam = np.array(points_in_last_cam).reshape([-1, 3])
    uv_in_curr_image = np.ascontiguousarray(uv_in_curr_image).reshape([-1, 1, 2])

    try:
        success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
            points_in_last_cam.astype(np.float64),
            uv_in_curr_image.astype(np.float64),
            last_k,
            np.zeros([5, 1]),
            flags=cv2.SOLVEPNP_SQPNP,
            confidence=0.90,
            reprojectionError=10.0,
            iterationsCount=200,
        )
    except cv2.error:
        return None

    if not success or inliers is None or inliers.shape[0] == 0:
        return None

    rot_curr_last = cv2.Rodrigues(rotation_vector)[0]
    trans_curr_last = translation_vector
    est_t_curr_last = np.eye(4, dtype=np.float64)
    est_t_curr_last[:3, :3] = rot_curr_last
    est_t_curr_last[:3, 3:4] = trans_curr_last
    return {
        "T_curr_last": est_t_curr_last,
        "num_inliers": int(inliers.shape[0]),
        "valid_indices": np.asarray(valid_indices, dtype=np.int32),
        "inlier_indices": np.asarray(inliers[:, 0], dtype=np.int32),
        "points_last_cam": points_in_last_cam.astype(np.float64),
        "uv_curr": uv_in_curr_image.reshape(-1, 2).astype(np.float64),
    }

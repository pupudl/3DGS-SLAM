"""Differentiable two-frame SE(3) / Gaussian ownership optimization.

The canonical Gaussian geometry is fixed throughout this module.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from diff_gaussian_rasterization import GaussianRasterizer
from utils.recon_helpers import setup_camera


def anchor_probabilities(anchors: dict, models: list[np.ndarray], sigma_m: float) -> np.ndarray:
    """HTML §6: q_jk proportional to exp(-||X'_j-H_k X_j||²/sigma_m²)."""
    a, b = anchors["x0"], anchors["x1"]
    if not len(a):
        return np.empty((0, len(models)), np.float32)
    errors = np.stack([
        np.sum((a @ H[:3, :3].T + H[:3, 3] - b)**2, axis=1)
        for H in models
    ], axis=1)
    logits = -errors / (sigma_m*sigma_m)
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(logits)
    return (prob / prob.sum(axis=1, keepdims=True)).astype(np.float32)


def hybrid_motion_costs_torch(a0, a1, uv0, uv1, H, K, cfg):
    """Differentiable counterpart of hybrid_motion.motion_costs (N x K)."""
    pred1 = torch.einsum('kij,nj->kni', H[:, :3, :3], a0) + H[:, None, :3, 3]
    pred0 = torch.einsum('kji,knj->kni', H[:, :3, :3], a1[None]-H[:, None, :3, 3])

    def project(xyz):
        z = xyz[:, :, 2].clamp_min(1e-4)
        return torch.stack((K[0, 0]*xyz[:, :, 0]/z+K[0, 2],
                            K[1, 1]*xyz[:, :, 1]/z+K[1, 2]), dim=-1)

    pixel = .5*(((project(pred1)-uv1[None])**2).sum(-1) +
                ((project(pred0)-uv0[None])**2).sum(-1))
    pixel = pixel / float(cfg.get('reprojection_threshold_px', 3.))**2
    sigma = (float(cfg.get('depth_sigma_m', .15)) +
             float(cfg.get('depth_sigma_relative', .015))*.5*(a0[:, 2]+a1[:, 2]))
    metric = (((pred1-a1[None])**2).sum(-1)/sigma[None].square()).clamp_max(
        float(cfg.get('depth_cost_cap', 4.)))
    weight = float(cfg.get('depth_weight', .2))
    cost = (1-weight)*pixel + weight*metric
    valid = (pred1[:, :, 2] > 1e-4) & (pred0[:, :, 2] > 1e-4)
    return torch.where(valid, cost, cost.new_full((), 1e6)).T


def graph_edges(gmap: dict, k: int, sigma_x: float, sigma_c: float, sigma_d: float):
    xyz, rgb = gmap["xyz"], gmap["color"]
    _, nearby = cKDTree(xyz).query(xyz, k=min(k+1, len(xyz)), workers=-1)
    if nearby.ndim == 1:
        nearby = nearby[:, None]
    source = np.repeat(np.arange(len(xyz)), nearby.shape[1]-1)
    target = nearby[:, 1:].reshape(-1)
    keep = source < target
    source, target = source[keep], target[keep]
    if not len(source):
        return np.empty((0, 2), np.int64), np.empty(0, np.float32)
    dx2 = np.sum((xyz[source]-xyz[target])**2, axis=1)
    dc2 = np.sum((rgb[source]-rgb[target])**2, axis=1)
    dz2 = (xyz[source, 2]-xyz[target, 2])**2
    weights = np.exp(-dx2/sigma_x**2-dc2/sigma_c**2-dz2/sigma_d**2)
    return np.column_stack((source, target)).astype(np.int64), weights.astype(np.float32)


def gaussian_probabilities(gmap: dict, anchors: dict, q_anchor: np.ndarray,
                           reference: int, cfg: dict, source_bgr: np.ndarray,
                           return_mass: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """HTML §7: graph-weighted sparse-to-Gaussian ownership initialization."""
    n, k = len(gmap["xyz"]), q_anchor.shape[1]
    if not len(q_anchor):
        raise ValueError("Cannot initialize ownership without sparse anchors")
    num_near = min(int(cfg.get("anchor_neighbors", 4)), len(q_anchor))
    distance, nearest = cKDTree(anchors["x0"]).query(gmap["xyz"], k=num_near, workers=-1)
    distance = np.asarray(distance).reshape(n, num_near)
    nearest = np.asarray(nearest).reshape(n, num_near)
    source_depth = anchors["x0"][:, 2]
    dz = np.abs(gmap["xyz"][:, None, 2]-source_depth[nearest])
    sx = float(cfg.get("anchor_sigma_m", 2.0))
    sd = float(cfg.get("depth_sigma_m", 1.0))
    h, w = source_bgr.shape[:2]
    anchor_uv = np.rint(anchors["matches"][:, :2]).astype(np.int32)
    anchor_uv[:, 0] = np.clip(anchor_uv[:, 0], 0, w-1)
    anchor_uv[:, 1] = np.clip(anchor_uv[:, 1], 0, h-1)
    anchor_rgb = source_bgr[anchor_uv[:, 1], anchor_uv[:, 0], ::-1].astype(np.float32)/255
    dc2 = np.sum((gmap["color"][:, None]-anchor_rgb[nearest])**2, axis=2)
    sc = float(cfg.get("color_sigma", 0.4))
    weights = np.exp(-(distance/sx)**2-(dz/sd)**2-dc2/sc**2)
    if 'confidence' in anchors:
        weights *= np.clip(anchors['confidence'][nearest], 0, 1)
    q = np.einsum("nk,nkj->nj", weights, q_anchor[nearest])
    mass = weights.sum(axis=1)
    valid = mass > float(cfg.get("min_anchor_weight", 0.02))
    q[valid] /= mass[valid, None]
    q[~valid] = 0
    q[~valid, reference] = 1
    floor = float(cfg.get("probability_floor", 0.005))
    q = q*(1-k*floor)+floor
    result = (q/q.sum(axis=1, keepdims=True)).astype(np.float32)
    return (result, mass.astype(np.float32)) if return_mass else result


def se3_from_twist(twist: torch.Tensor) -> torch.Tensor:
    """Matrix exponential maps each 6-vector to SE(3)."""
    wx, wy, wz = twist[:, 0], twist[:, 1], twist[:, 2]
    tx, ty, tz = twist[:, 3], twist[:, 4], twist[:, 5]
    zero = torch.zeros_like(wx)
    row0 = torch.stack((zero, -wz, wy, tx), dim=1)
    row1 = torch.stack((wz, zero, -wx, ty), dim=1)
    row2 = torch.stack((-wy, wx, zero, tz), dim=1)
    row3 = torch.stack((zero, zero, zero, zero), dim=1)
    return torch.matrix_exp(torch.stack((row0, row1, row2, row3), dim=1))


def simple_ssim(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Differentiable local SSIM on RGB images of shape 1x3xHxW."""
    mu1 = F.avg_pool2d(first, 7, stride=1, padding=3)
    mu2 = F.avg_pool2d(second, 7, stride=1, padding=3)
    sigma1 = F.avg_pool2d(first*first, 7, stride=1, padding=3)-mu1*mu1
    sigma2 = F.avg_pool2d(second*second, 7, stride=1, padding=3)-mu2*mu2
    cross = F.avg_pool2d(first*second, 7, stride=1, padding=3)-mu1*mu2
    value = ((2*mu1*mu2+0.01**2)*(2*cross+0.03**2) /
             ((mu1*mu1+mu2*mu2+0.01**2)*(sigma1+sigma2+0.03**2)))
    return value.clamp(-1, 1).mean()


def triangulated_depths(H: torch.Tensor, ray0: torch.Tensor,
                        ray1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-ray least-squares depths for X1 = H X0, with rays scaled to z=1."""
    a = torch.einsum('kij,nj->kni', H[:, :3, :3], ray0)
    b = ray1[None]
    t = H[:, None, :3, 3]
    aa = (a*a).sum(dim=-1)
    bb = (b*b).sum(dim=-1)
    ab = (a*b).sum(dim=-1)
    rhs0 = -(a*t).sum(dim=-1)
    rhs1 = (b*t).sum(dim=-1)
    det = (aa*bb-ab*ab).clamp_min(0)
    safe_det = det.clamp_min(1e-6)
    z0 = (bb*rhs0+ab*rhs1)/safe_det
    z1 = (ab*rhs0+aa*rhs1)/safe_det
    sin2 = det/(aa*bb).clamp_min(1e-8)
    return z0, z1, sin2


class PairwiseModel:
    def __init__(self, gmap: dict, anchors: dict, models: list[np.ndarray],
                 q_initial: np.ndarray, K: np.ndarray, target_bgr: np.ndarray,
                 target_depth: np.ndarray, config: dict,
                 target_confidence: np.ndarray | None = None):
        self.config = config
        self.shape = target_depth.shape
        self.K = K
        self.target_bgr = target_bgr
        self.target_depth = target_depth
        device = "cuda"
        self.xyz = torch.as_tensor(gmap["xyz"], dtype=torch.float32, device=device)
        self.rgb = torch.as_tensor(gmap["color"], dtype=torch.float32, device=device)
        self.gaussian_initialization = gmap.get("initialization", "surface_aligned")
        self.scale = torch.as_tensor(gmap["scale"], dtype=torch.float32, device=device)
        self.cov = (None if self.gaussian_initialization == "lsgslam_projective"
                    else self._covariances(gmap))
        self.base_H = torch.as_tensor(np.stack(models), dtype=torch.float32, device=device)
        self.q_initial = torch.as_tensor(q_initial, dtype=torch.float32, device=device)
        self.anchors0 = torch.as_tensor(anchors["x0"], dtype=torch.float32, device=device)
        self.anchors1 = torch.as_tensor(anchors["x1"], dtype=torch.float32, device=device)
        self.anchor_confidence = torch.as_tensor(
            anchors.get('confidence', np.ones(len(anchors['x0']), np.float32)),
            dtype=torch.float32, device=device).clamp(0.01, 1)
        if 'gaussian_index' in anchors:
            near = np.asarray(anchors['gaussian_index'], dtype=np.int64)
            if len(near) != len(anchors['x0']) or not np.allclose(gmap['xyz'][near], anchors['x0'], atol=1e-5):
                raise ValueError('Anchor Gaussian mapping does not match landmark geometry')
        else:
            _, near = cKDTree(gmap["uv"]).query(anchors["matches"][:, :2])
        self.anchor_g = torch.as_tensor(near, dtype=torch.long, device=device)
        matches = torch.as_tensor(anchors['matches'], dtype=torch.float32, device=device)
        self.anchor_target_uv = matches[:, 2:]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.ray0 = torch.stack(((matches[:, 0]-cx)/fx, (matches[:, 1]-cy)/fy,
                                 torch.ones_like(matches[:, 0])), dim=-1)
        self.ray1 = torch.stack(((matches[:, 2]-cx)/fx, (matches[:, 3]-cy)/fy,
                                 torch.ones_like(matches[:, 2])), dim=-1)
        min_angle = np.deg2rad(float(config.get('triangulation_min_parallax_deg', 0.5)))
        max_depth = float(config.get('triangulation_max_depth_m', 80.0))
        with torch.no_grad():
            z0, z1, sin2 = triangulated_depths(self.base_H, self.ray0, self.ray1)
            self.triangulation_valid = ((sin2 >= np.sin(min_angle)**2) &
                                        (z0 > 0) & (z1 > 0) &
                                        (z0 <= max_depth) & (z1 <= max_depth))
        graph_cfg = config.get("graph", {})
        edges, weights = graph_edges(gmap, int(graph_cfg.get("neighbors", 4)),
                                    float(graph_cfg.get("sigma_x_m", 0.8)),
                                    float(graph_cfg.get("sigma_color", 0.25)),
                                    float(graph_cfg.get("sigma_depth_m", 0.6)))
        self.edge_i = torch.as_tensor(edges[:, 0], dtype=torch.long, device=device)
        self.edge_j = torch.as_tensor(edges[:, 1], dtype=torch.long, device=device)
        self.edge_w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255
        self.rgb_full = torch.as_tensor(rgb, device=device).permute(2, 0, 1)[None]
        self.depth_full = torch.as_tensor(np.nan_to_num(target_depth, nan=0), device=device)
        self.valid_full = torch.as_tensor(np.isfinite(target_depth), device=device)
        if target_confidence is None:
            raise ValueError("A stereo-derived target depth confidence map is required")
        if target_confidence.shape != target_depth.shape:
            raise ValueError("Depth confidence and depth must have the same shape")
        confidence = np.where(np.isfinite(target_depth), target_confidence, 0)
        self.confidence_full = torch.as_tensor(confidence, device=device)
        self.cam_cache = {}
        self._target_cache = {}

    def _covariances(self, gmap):
        basis = torch.as_tensor(gmap["basis"], dtype=torch.float32, device="cuda")
        scale = torch.as_tensor(gmap["scale"], dtype=torch.float32, device="cuda")
        local = basis*scale[:, None, :]
        return local @ local.transpose(1, 2)

    def _camera(self, downsample: int):
        if downsample not in self.cam_cache:
            h, w = self.shape
            out_h, out_w = h//downsample, w//downsample
            intrinsics = self.K.copy()
            intrinsics[0, :] /= downsample
            intrinsics[1, :] /= downsample
            self.cam_cache[downsample] = GaussianRasterizer(
                raster_settings=setup_camera(out_w, out_h, intrinsics, np.eye(4), depth_threshold=float(self.config.get("render_depth_threshold_m", 1.0))))
        return self.cam_cache[downsample]

    def _targets(self, downsample: int):
        if downsample not in self._target_cache:
            if downsample == 1:
                self._target_cache[downsample] = (
                    self.rgb_full, self.depth_full, self.valid_full, self.confidence_full)
            else:
                h, w = self.shape
                out_shape = (h//downsample, w//downsample)
                rgb = F.interpolate(self.rgb_full, size=out_shape, mode="area")
                depth_np = cv2.resize(self.target_depth, out_shape[::-1], interpolation=cv2.INTER_NEAREST)
                conf_np = cv2.resize(self.confidence_full.cpu().numpy(), out_shape[::-1],
                                     interpolation=cv2.INTER_NEAREST)
                depth = torch.as_tensor(np.nan_to_num(depth_np, nan=0), device="cuda")
                valid = torch.as_tensor(np.isfinite(depth_np), device="cuda")
                conf = torch.as_tensor(conf_np, device="cuda")
                self._target_cache[downsample] = (rgb, depth, valid, conf)
        return self._target_cache[downsample]

    def render(self, H: torch.Tensor, q: torch.Tensor, downsample: int = 1):
        """HTML §8: K transformed Gaussian copies, opacity alpha_i q_ik."""
        count = self.xyz.shape[0]
        k = H.shape[0]
        rotation = H[:, :3, :3]
        translation = H[:, :3, 3]
        xyz = torch.einsum("kij,nj->kni", rotation, self.xyz)+translation[:, None, :]
        # Flatten in motion-major order, matching q.T.
        xyz = xyz.reshape(k*count, 3)
        if xyz.requires_grad:
            xyz.retain_grad()
        self._last_transformed_xyz = xyz
        if self.gaussian_initialization == "lsgslam_projective":
            # Use the legacy rasterizer inputs exactly: isotropic scales and
            # identity quaternion. A rigid transform cannot rotate a sphere.
            scales = self.scale.repeat(k, 1)
            rotations = torch.zeros((k*count, 4), dtype=xyz.dtype, device=xyz.device)
            rotations[:, 0] = 1
            cov6 = None
        else:
            cov = torch.einsum("kij,njl,kml->knim", rotation, self.cov, rotation)
            cov = cov.reshape(k*count, 3, 3)
            cov6 = torch.stack((cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                                cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]), dim=1)
            scales = None
            rotations = None
        alpha = float(self.config.get("gaussian_opacity", 0.9))
        opacity = (q.T.reshape(-1, 1)*alpha).contiguous()
        colors = self.rgb.repeat(k, 1)
        means2d = torch.zeros_like(xyz, requires_grad=True)
        common = dict(means3D=xyz, means2D=means2d, opacities=opacity,
                      cov3D_precomp=cov6, scales=scales, rotations=rotations, shs=None)
        raster = self._camera(downsample)
        rgb, _, _, _ = raster(colors_precomp=colors, **common)
        depth_color = torch.stack((xyz[:, 2], torch.ones_like(xyz[:, 2]),
                                   xyz[:, 2]**2), dim=1)
        depth_sil, _, _, _ = raster(colors_precomp=depth_color, **common)
        sil = depth_sil[1]
        depth = depth_sil[0]/sil.clamp_min(1e-5)
        return rgb[None], depth, sil

    def loss(self, H: torch.Tensor, q: torch.Tensor, downsample: int):
        rgb, depth, sil = self.render(H, q, downsample)
        target_rgb, target_depth, valid_depth, confidence = self._targets(downsample)
        # HTML §9 evaluates every observed pixel, including a map gap where
        # the current model renders little or no opacity. The map is fixed, so
        # these residuals can motivate motion discovery.
        valid = valid_depth
        valid_stereo = valid_depth & (confidence > 0)
        if not valid.any() or not valid_stereo.any():
            raise RuntimeError("No valid observed RGB/stereo depth for joint optimization")
        color_l1 = (rgb[0]-target_rgb[0]).abs()[:, valid].mean()
        ssim_loss = 1-simple_ssim(rgb*valid[None, None], target_rgb*valid[None, None])
        depth_residual = depth-target_depth
        robust_depth = torch.sqrt(depth_residual**2+0.05**2)-0.05
        depth_loss = (confidence[valid_stereo]*robust_depth[valid_stereo]).sum()/confidence[valid_stereo].sum().clamp_min(1)
        a0 = self.anchors0
        a1 = self.anchors1
        pred = torch.einsum("kij,nj->kni", H[:, :3, :3], a0)+H[:, None, :3, 3]
        err2 = ((pred-a1[None])**2).sum(dim=2).T
        sigma = float(self.config.get("anchor_sigma_m", 0.5))
        robust_err2 = 2*sigma*sigma*(torch.sqrt(1+err2/(sigma*sigma))-1)
        # Hybrid initialization uses the same geometry as candidate discovery;
        # feedback keeps its existing objective unless explicitly configured.
        hybrid = self.config.get('hybrid_geometry')
        if hybrid is not None:
            geometry = hybrid_motion_costs_torch(
                a0, a1, self.ray0[:, :2]*a0.new_tensor([self.fx, self.fy])+
                a0.new_tensor([self.cx, self.cy]), self.anchor_target_uv, H,
                a0.new_tensor(self.K), hybrid)
            temperature = float(hybrid.get('temperature', .25))
            logits = -geometry/temperature
            outlier = logits.new_full((len(a0), 1),
                                      -float(hybrid.get('outlier_cost', 1.))/temperature)
            mixture = torch.softmax(torch.cat((logits, outlier), dim=1), dim=1)
            q_anchor = torch.softmax(logits, dim=1)
            inlier_mass = mixture[:, :-1].sum(dim=1)
            anchor_each = -temperature*torch.logsumexp(torch.cat((logits, outlier), dim=1), dim=1)
        else:
            q_anchor = torch.softmax(-err2 / float(self.config.get("assignment_sigma_m", 0.75))**2, dim=1)
            inlier_mass = torch.ones_like(self.anchor_confidence)
            anchor_each = -torch.logsumexp(torch.log(q_anchor.clamp_min(1e-8))-
                                          robust_err2/(2*sigma*sigma), dim=1)
        anchor_loss = (self.anchor_confidence*anchor_each).sum()/self.anchor_confidence.sum()
        weights = self.config.get('loss_weights', {})
        if float(weights.get('anchor_reprojection', 0)) > 0:
            anchor_xyz = self.xyz[self.anchor_g]
            anchor_pred = torch.einsum('kij,nj->kni', H[:, :3, :3], anchor_xyz)+H[:, None, :3, 3]
            safe_z = anchor_pred[:, :, 2].clamp_min(0.1)
            uv = torch.stack((self.fx*anchor_pred[:, :, 0]/safe_z+self.cx,
                              self.fy*anchor_pred[:, :, 1]/safe_z+self.cy), dim=-1)
            sigma_px = float(self.config.get('reprojection_sigma_px', 2.0))
            if sigma_px <= 0:
                raise ValueError('reprojection_sigma_px must be positive')
            reproj2 = ((uv-self.anchor_target_uv[None])**2).sum(dim=-1)
            robust_reproj = torch.sqrt(1+reproj2/sigma_px**2)-1
            # Motion ownership belongs to the *rendered Gaussian*, not a
            # separate sparse point. Mixture avoids averaging pixel positions.
            log_q = torch.log(q[self.anchor_g].T.clamp_min(1e-8))
            reprojection_each = -torch.logsumexp(log_q-robust_reproj, dim=0)
            reprojection_loss = (self.anchor_confidence*inlier_mass.detach()*reprojection_each).sum()/self.anchor_confidence.sum()
        else:
            reprojection_loss = q.new_tensor(0.)
        if float(weights.get('triangulated_depth', 0)) > 0 and self.triangulation_valid.any():
            tri0, tri1, _ = triangulated_depths(H, self.ray0, self.ray1)
            depth_sigma = float(self.config.get('triangulation_depth_sigma_m', 1.0))
            if depth_sigma <= 0:
                raise ValueError('triangulation_depth_sigma_m must be positive')
            depth0 = torch.sqrt(1+((tri0-self.anchors0[None, :, 2])/depth_sigma)**2)-1
            depth1 = torch.sqrt(1+((tri1-self.anchors1[None, :, 2])/depth_sigma)**2)-1
            tri_weight = (q_anchor.T.detach()*inlier_mass[None].detach()*
                          self.triangulation_valid*self.anchor_confidence[None])
            triangulated_depth_loss = (tri_weight*0.5*(depth0+depth1)).sum()/tri_weight.sum().clamp_min(1)
        else:
            triangulated_depth_loss = q.new_tensor(0.)
        # Tie sparse evidence to the local Gaussian ownership.
        sparse_each = -(q_anchor.detach() *
                        torch.log(q[self.anchor_g].clamp_min(1e-8))).sum(dim=1)
        sparse_ownership = (self.anchor_confidence*inlier_mass.detach()*sparse_each).sum()/self.anchor_confidence.sum()
        entropy = -(q*torch.log(q.clamp_min(1e-8))).sum(dim=1).mean()
        if len(self.edge_i):
            graph = (self.edge_w*(q[self.edge_i]-q[self.edge_j]).abs().sum(dim=1)).mean()
        else:
            graph = q.new_tensor(0.)
        total = (float(weights.get("anchor", 1.0))*anchor_loss +
                 float(weights.get('anchor_reprojection', 0.0))*reprojection_loss +
                 float(weights.get('triangulated_depth', 0.0))*triangulated_depth_loss +
                 float(weights.get("sparse_ownership", 0.0))*sparse_ownership +
                 float(weights.get("color", 1.0))*color_l1 +
                 float(weights.get("ssim", 0.2))*ssim_loss +
                 float(weights.get("depth", 0.05))*depth_loss +
                 float(weights.get("entropy", 0.005))*entropy +
                 float(weights.get("graph", 0.05))*graph)
        terms = {"anchor": anchor_loss, 'anchor_reprojection': reprojection_loss,
                 'triangulated_depth': triangulated_depth_loss,
                 "sparse_ownership": sparse_ownership,
                 "color": color_l1, "ssim": ssim_loss,
                 "depth": depth_loss, "entropy": entropy, "graph": graph}
        return total, terms

    def optimize(self, iterations: int, downsample: int, pose_lr: float, q_lr: float):
        # In sequence object-window mode these local motion probabilities are
        # initialization data, not the persistent object attribute. Keep them
        # exactly unchanged, including their softness, while refining poses.
        frozen_ownership = bool(self.config.get("freeze_motion_ownership", False))
        twist = torch.nn.Parameter(torch.zeros((len(self.base_H), 6),
                                                device=self.base_H.device))
        logits = (None if frozen_ownership else
                  torch.nn.Parameter(torch.log(self.q_initial.clamp_min(1e-6))))
        groups = [{"params": [twist], "lr": pose_lr}]
        if logits is not None:
            groups.append({"params": [logits], "lr": q_lr})
        optim = torch.optim.Adam(groups)
        history = []
        for step in range(iterations):
            optim.zero_grad(set_to_none=True)
            tau = max(float(self.config.get("tau_final", 0.35)),
                      1-(1-float(self.config.get("tau_final", 0.35)))*
                      step/max(iterations-1, 1))
            q = (self.q_initial.detach() if frozen_ownership else
                 torch.softmax(logits/tau, dim=1))
            H = se3_from_twist(twist) @ self.base_H
            total, terms = self.loss(H, q, downsample)
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite joint objective at iteration {step}")
            total.backward()
            history.append({"step": step, "total": float(total.detach()),
                            **{name: float(value.detach()) for name, value in terms.items()}})
            optim.step()
        optim.zero_grad(set_to_none=True)
        # A diagnostic leaf preserves ownership gradients for motion-discovery
        # feedback without putting ownership in the optimizer or changing q.
        q = (self.q_initial.detach().clone().requires_grad_(True)
             if frozen_ownership else
             torch.softmax(logits/float(self.config.get("tau_final", 0.35)), dim=1))
        H = se3_from_twist(twist) @ self.base_H
        total, terms = self.loss(H, q, downsample)
        total.backward()
        if frozen_ownership:
            # Report the gradient of a hypothetical unit-temperature logit at
            # this fixed probability. The Jacobian removes the irrelevant
            # all-classes shift and matches the legacy feedback's logit units.
            q_grad = q.grad if q.grad is not None else torch.zeros_like(q)
            logit_gradient = q*(q_grad-(q*q_grad).sum(dim=1, keepdim=True))
            ownership_gradient = logit_gradient.detach().norm(dim=1).cpu().numpy()
        else:
            ownership_gradient = logits.grad.detach().norm(dim=1).cpu().numpy()
        # J^T r for x' = RX+t: [x' cross dL/dx', dL/dx'].
        xyz = self._last_transformed_xyz
        xyz_grad = xyz.grad.detach().reshape(len(self.base_H), len(self.xyz), 3)
        transformed = xyz.detach().reshape(len(self.base_H), len(self.xyz), 3)
        wrench = torch.cat((torch.linalg.cross(transformed, xyz_grad), xyz_grad), dim=2)
        motion_gradient = wrench.norm(dim=2).max(dim=0).values.cpu().numpy()
        prediction = torch.einsum("kij,nj->kni", H[:, :3, :3], self.anchors0) + H[:, None, :3, 3]
        anchor_error2 = ((prediction - self.anchors1[None])**2).sum(dim=2).T
        if self.config.get('hybrid_geometry') is not None:
            costs = hybrid_motion_costs_torch(
                self.anchors0, self.anchors1,
                self.ray0[:, :2]*self.anchors0.new_tensor([self.fx, self.fy])+
                self.anchors0.new_tensor([self.cx, self.cy]), self.anchor_target_uv,
                H, self.anchors0.new_tensor(self.K), self.config['hybrid_geometry'])
            q_anchor = torch.softmax(-costs/float(self.config['hybrid_geometry'].get('temperature', .25)), dim=1)
        else:
            q_anchor = torch.softmax(-anchor_error2 / float(self.config.get("assignment_sigma_m", 0.75))**2, dim=1)
        return {"H": H.detach(), "q": q.detach(), "history": history,
                "loss": float(total.detach()), "grad_q": ownership_gradient,
                "grad_motion": motion_gradient,
                "q_anchor": q_anchor.detach().cpu().numpy(),
                "terms": {name: float(value.detach()) for name, value in terms.items()}}

    def render_numpy(self, H: torch.Tensor, q: torch.Tensor):
        with torch.no_grad():
            rgb, depth, sil = self.render(H, q, 1)
        return {"rgb": rgb[0].permute(1, 2, 0).cpu().numpy(),
                "depth": depth.cpu().numpy(), "sil": sil.cpu().numpy()}


def gradient_projection(gmap: dict, H: np.ndarray, q: np.ndarray,
                        gradient: np.ndarray, K: np.ndarray,
                        shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Project gradients through every soft motion branch, without rescaling.

    The two maps give a patch its probability-weighted Gaussian gradient
    average. This preserves uncertain ownership during the soft stage.
    """
    h, w = shape
    values = np.zeros(shape, np.float32)
    counts = np.zeros(shape, np.float32)
    for k, motion in enumerate(H):
        xyz = gmap["xyz"] @ motion[:3, :3].T + motion[:3, 3]
        z = xyz[:, 2]
        x = np.rint(K[0, 0] * xyz[:, 0] / np.maximum(z, 1e-5) + K[0, 2]).astype(int)
        y = np.rint(K[1, 1] * xyz[:, 1] / np.maximum(z, 1e-5) + K[1, 2]).astype(int)
        valid = (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        branch_weight = q[valid, k].astype(np.float32)
        np.add.at(values, (y[valid], x[valid]), branch_weight * gradient[valid])
        np.add.at(counts, (y[valid], x[valid]), branch_weight)
    return values, counts

"""One-way sparse optical flow with endpoint descriptor verification."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch


def _parameters(options):
    options = {} if options is None else options
    defaults = {
        "window_size": 31, "max_level": 4, "max_iterations": 30,
        "epsilon": .01, "min_eigenvalue": 1e-4, "max_error": 30.,
        "descriptor_min_similarity": .7,
    }
    values = {}
    for name, default in defaults.items():
        raw = options.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a finite number") from error
        if isinstance(raw, (bool, np.bool_)) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        values[name] = value
    for name in ("window_size", "max_level", "max_iterations"):
        if values[name] != int(values[name]):
            raise ValueError(f"{name} must be an integer")
        values[name] = int(values[name])
    if values["window_size"] < 3 or values["window_size"] % 2 != 1:
        raise ValueError("window_size must be odd and at least 3")
    if values["max_level"] < 0 or values["max_iterations"] < 1:
        raise ValueError("max_level must be nonnegative and max_iterations positive")
    if values["epsilon"] <= 0:
        raise ValueError("epsilon must be positive")
    if values["min_eigenvalue"] < 0 or values["max_error"] < 0:
        raise ValueError("min_eigenvalue and max_error must be nonnegative")
    if not -1 <= values["descriptor_min_similarity"] <= 1:
        raise ValueError("descriptor_min_similarity must be between -1 and 1")
    return values


def _gray(image):
    image = np.asarray(image)
    if image.dtype != np.uint8:
        raise ValueError("optical flow images must be uint8")
    if not image.size:
        raise ValueError("optical flow images must be nonempty")
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim != 2:
        raise ValueError("images must have shape HxW or HxWx3 (BGR)")
    return np.ascontiguousarray(image)


def _descriptor_shape(coarse):
    if (not torch.is_tensor(coarse) or coarse.ndim != 4 or
            coarse.shape[0] != 1 or min(coarse.shape[1:]) < 1 or
            not coarse.is_floating_point()):
        raise ValueError("descriptor maps must be floating tensors with shape 1xDxHxW")
    return coarse.shape


def _covered(points, image_shape, descriptor_shape):
    # The stride-8 SuperPoint sampler maps 3.5 to -1 and coarse_size*8-1
    # to +1. Exclude padding outside that descriptor support, including
    # the bottom/right image remainder discarded by the encoder's pooling.
    height, width = image_shape
    coarse_h, coarse_w = descriptor_shape[-2:]
    return ((points[:, 0] >= 3.5) & (points[:, 1] >= 3.5) &
            (points[:, 0] < width) & (points[:, 1] < height) &
            (points[:, 0] <= coarse_w * 8 - 1) &
            (points[:, 1] <= coarse_h * 8 - 1))


def _sample(extractor, points, coarse):
    uv = torch.as_tensor(points[None], device=coarse.device, dtype=coarse.dtype)
    with torch.inference_mode():
        sampled = extractor.sample_descriptors_at(uv, coarse)
    expected = (1, len(points), coarse.shape[1])
    if tuple(sampled.shape) != expected:
        raise ValueError(f"sample_descriptors_at must return {expected}")
    return sampled[0].detach().float().cpu().numpy()


def forward_flow(image0, image1, uv0, extractor, descriptor_map0,
                 descriptor_map1, options=None):
    """Track Nx2 source points once, then verify descriptors at the endpoints.

    Returns float32 ``uv1`` (NaN for uncomputed points), an aligned Boolean
    acceptance mask, and rejection counts. Failure categories are mutually
    exclusive. Input coordinates and accepted endpoint coordinates remain
    subpixel image (x, y) positions; no target detector or reverse flow is used.
    """
    started = time.perf_counter()
    cfg = _parameters(options)
    gray0, gray1 = _gray(image0), _gray(image1)
    if gray0.shape != gray1.shape:
        raise ValueError("source and target flow images must have equal size")
    uv0 = np.asarray(uv0)
    if uv0.ndim != 2 or uv0.shape[1] != 2 or np.iscomplexobj(uv0):
        raise ValueError("uv0 must have shape Nx2 with real coordinates")
    uv0 = np.asarray(uv0, np.float32)
    uv1 = np.full(uv0.shape, np.nan, np.float32)
    valid = np.zeros(len(uv0), bool)
    report = {
        "seedcount": len(uv0), "statusfail": 0, "errorfail": 0,
        "boundsfail": 0, "finitefail": 0, "descriptorfail": 0,
        "accepted": 0, "descriptor_similarity_median": None,
    }

    def finish():
        report["accepted"] = int(valid.sum())
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        return uv1, valid, report

    if not len(uv0):
        return finish()
    shape0 = _descriptor_shape(descriptor_map0)
    shape1 = _descriptor_shape(descriptor_map1)
    if shape0[1] != shape1[1]:
        raise ValueError("source and target descriptor dimensions must agree")
    finite0 = np.isfinite(uv0).all(axis=1)
    report["finitefail"] = int((~finite0).sum())
    covered0 = _covered(uv0, gray0.shape, shape0)
    report["boundsfail"] = int((finite0 & ~covered0).sum())
    indices = np.flatnonzero(finite0 & covered0)
    if not len(indices):
        return finish()

    predicted, status, error = cv2.calcOpticalFlowPyrLK(
        gray0, gray1, np.ascontiguousarray(uv0[indices, None, :]), None,
        winSize=(cfg["window_size"], cfg["window_size"]),
        maxLevel=cfg["max_level"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                  cfg["max_iterations"], cfg["epsilon"]),
        flags=0, minEigThreshold=cfg["min_eigenvalue"])
    if predicted is None or status is None:
        report["statusfail"] = len(indices)
        return finish()
    predicted = np.asarray(predicted, np.float32).reshape(-1, 2)
    status = np.asarray(status).reshape(-1)
    if len(predicted) != len(indices) or len(status) != len(indices):
        raise ValueError("LK returned unaligned points or status")
    uv1[indices] = predicted
    status_ok = status == 1
    report["statusfail"] = int((~status_ok).sum())
    if error is None:
        error = np.full(len(indices), np.inf)
    else:
        error = np.asarray(error).reshape(-1)
        if len(error) != len(indices):
            raise ValueError("LK returned unaligned errors")
    error_ok = np.isfinite(error) & (error <= cfg["max_error"])
    report["errorfail"] = int((status_ok & ~error_ok).sum())
    finite1 = np.isfinite(predicted).all(axis=1)
    report["finitefail"] += int((status_ok & error_ok & ~finite1).sum())
    covered1 = _covered(predicted, gray1.shape, shape1)
    passed = status_ok & error_ok & finite1
    report["boundsfail"] += int((passed & ~covered1).sum())
    candidates = indices[passed & covered1]
    if not len(candidates):
        return finish()

    desc0 = _sample(extractor, uv0[candidates], descriptor_map0)
    desc1 = _sample(extractor, uv1[candidates], descriptor_map1)
    norm0, norm1 = np.linalg.norm(desc0, axis=1), np.linalg.norm(desc1, axis=1)
    descriptor_ok = (np.isfinite(desc0).all(axis=1) &
                     np.isfinite(desc1).all(axis=1) &
                     np.isfinite(norm0) & np.isfinite(norm1) &
                     (norm0 > 1e-12) & (norm1 > 1e-12))
    similarity = np.full(len(candidates), np.nan, np.float32)
    similarity[descriptor_ok] = np.clip(np.einsum(
        "ij,ij->i", desc0[descriptor_ok] / norm0[descriptor_ok, None],
        desc1[descriptor_ok] / norm1[descriptor_ok, None]), -1, 1)
    if descriptor_ok.any():
        report["descriptor_similarity_median"] = float(np.median(
            similarity[descriptor_ok]))
    accepted = descriptor_ok & (similarity >= cfg["descriptor_min_similarity"])
    report["descriptorfail"] = int((~accepted).sum())
    valid[candidates[accepted]] = True
    return finish()

"""Opaque object coverage at saved Gaussian centres, without soft blending.

Maps contain global object IDs, -1 for an uncertain surface and -2 for pixels
without a visible Gaussian centre. Rasterization changes no ownership state.
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

UNOBSERVED = -2
UNKNOWN = -1


def object_color(object_id: int) -> tuple[int, int, int]:
    """Stable saturated BGR palette shared by PNG and browser visualizations."""
    rgb = colorsys.hsv_to_rgb((.57 + int(object_id)*.61803398875) % 1., .96, 1.)
    return tuple(int(round(255*component)) for component in rgb[::-1])


def _object_columns(gmap):
    if 'object_probabilities' in gmap:
        return np.asarray(gmap['object_probabilities']), np.asarray(gmap['object_ids'], np.int64)
    mapping = gmap['local_to_object']
    if isinstance(mapping, dict):
        mapping = [mapping[str(k)] if str(k) in mapping else mapping[k] for k in range(len(mapping))]
    q = np.asarray(gmap['q'])
    local_objects = np.asarray(mapping, np.int64)
    object_ids = np.unique(local_objects[local_objects >= 0])
    probabilities = np.zeros((len(q), len(object_ids)), np.float64)
    for column, oid in enumerate(object_ids):
        probabilities[:, column] = q[:, local_objects == oid].sum(axis=1)
    return probabilities, object_ids


def ownership_maps(gmap: dict, image_shape: tuple[int, ...], terminal: bool = False) -> dict:
    """Rasterize one visible centre per pixel, using one motion per Gaussian.

    Terminal motion is the maximum-probability hypothesis, never soft copies.
    Missing/invalid winning models leave that centre unobserved. A foreground
    uncertain Gaussian still occludes accepted labels behind it. Depth ties
    deterministically prefer the smaller saved Gaussian ID (or input index).
    Legacy archives lack accepted labels, so their accepted view stays unknown.
    """
    probabilities, object_ids = _object_columns(gmap)
    xyz = np.asarray(gmap['xyz'], np.float64)
    n = len(xyz)
    h, w = image_shape[:2]
    labels = np.full(n, UNKNOWN, np.int64)
    winner_probability = np.full(n, np.nan)
    finite_probability = np.isfinite(probabilities).all(axis=1)
    has_probability = finite_probability & (probabilities.sum(axis=1) > 0)
    winners = np.zeros(n, np.int64)
    if probabilities.shape[1]:
        winners[has_probability] = probabilities[has_probability].argmax(axis=1)
        labels[has_probability] = object_ids[winners[has_probability]]
        winner_probability[has_probability] = probabilities[
            np.flatnonzero(has_probability), winners[has_probability]]
    accepted = np.asarray(gmap.get('object_labels', np.full(n, UNKNOWN)), np.int64)
    if accepted.shape != (n,):
        raise ValueError('object_labels must have one global label per Gaussian')
    if probabilities.shape[0] != n or len(object_ids) != probabilities.shape[1]:
        raise ValueError('Object probabilities and Gaussian geometry have incompatible shapes')
    if terminal:
        moved = np.full_like(xyz, np.nan)
        if 'object_probabilities' in gmap:
            # A local-motion H archive must never masquerade as global object
            # motion merely because the matrices happen to have equal count.
            models = np.asarray(gmap.get('H_object', np.empty((0, 4, 4))), float)
            model_valid = np.asarray(gmap.get('object_model_valid',
                                             np.isfinite(models).all(axis=(1, 2))), bool)
            for column, H in enumerate(models):
                if column >= len(object_ids) or not model_valid[column] or not np.isfinite(H).all():
                    continue
                selected = has_probability & (winners == column)
                moved[selected] = xyz[selected] @ H[:3, :3].T + H[:3, 3]
        else:
            # A legacy global object may combine several local motions. Its
            # most likely local hypothesis *within that object* supplies its
            # single projected centre, preserving global label/motion agreement.
            q = np.asarray(gmap['q'])
            mapping = gmap['local_to_object']
            if isinstance(mapping, dict):
                mapping = [mapping.get(str(k), mapping.get(k)) for k in range(q.shape[1])]
            local_objects = np.asarray(mapping, np.int64)
            restricted = np.where(local_objects[None] == labels[:, None], q, -np.inf)
            local_winner = restricted.argmax(axis=1)
            for column, H in enumerate(np.asarray(gmap['H'], float)):
                if not np.isfinite(H).all():
                    continue
                selected = has_probability & (local_winner == column)
                moved[selected] = xyz[selected] @ H[:3, :3].T + H[:3, 3]
        depth = moved[:, 2]
        homogeneous = moved @ np.asarray(gmap['K'], float).T
        uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-12)
    else:
        uv = np.asarray(gmap['uv'], float)
        depth = xyz[:, 2]
    finite = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 0)
    xy = np.zeros((n, 2), np.int64)
    xy[finite] = np.rint(uv[finite]).astype(np.int64)
    visible = (finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) &
               (xy[:, 1] >= 0) & (xy[:, 1] < h))
    rows = np.flatnonzero(visible)
    pixels = xy[rows, 1]*w+xy[rows, 0]
    gaussian_ids = np.asarray(gmap.get('gaussian_ids', np.arange(n)), np.int64)
    order = np.lexsort((gaussian_ids[rows], depth[rows], pixels))
    sorted_pixels = pixels[order]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]] if len(order) else np.empty(0, bool)
    rows = rows[order[first]]
    pixels = sorted_pixels[first]
    argmax_map = np.full(h*w, UNOBSERVED, np.int64)
    accepted_map = np.full(h*w, UNOBSERVED, np.int64)
    probability_map = np.full(h*w, np.nan)
    argmax_map[pixels] = labels[rows]
    accepted_map[pixels] = accepted[rows]
    probability_map[pixels] = winner_probability[rows]
    return dict(argmax=argmax_map.reshape(h, w), accepted=accepted_map.reshape(h, w),
                probability=probability_map.reshape(h, w))


def _dark_context(image):
    gray = cv2.cvtColor(np.asarray(image, np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(np.rint(gray*.18).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def isolate_object(image: np.ndarray, label_map: np.ndarray, object_id: int,
                   outline: bool = False) -> np.ndarray:
    """Opaque object color on dark context, with an optional inner outline."""
    out = _dark_context(image)
    selected = np.asarray(label_map) == int(object_id)
    out[selected] = object_color(object_id)
    if not outline:
        return out
    eroded = cv2.erode(selected.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    neighbors = cv2.filter2D(selected.astype(np.uint8), cv2.CV_16S,
                             np.ones((3, 3), np.int16), borderType=cv2.BORDER_CONSTANT)
    # Keep isolated centres in the object's colour: outlining every sparse
    # point would turn the entire mask white and hide its identity again.
    out[selected & ~eroded & (neighbors >= 4)] = (255, 255, 255)
    return out


def all_objects(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    """Full-opacity labels; unknown is gray and unobserved keeps dark context."""
    out = _dark_context(image)
    labels = np.asarray(label_map)
    out[labels == UNKNOWN] = (145, 145, 145)
    for oid in np.unique(labels[labels >= 0]):
        out[labels == oid] = object_color(int(oid))
    return out


def legend_image(object_ids, width=960) -> np.ndarray:
    columns = max(1, width//115)
    entries = [UNKNOWN, UNOBSERVED] + sorted({int(oid) for oid in object_ids if int(oid) >= 0})
    height = 34*((len(entries)+columns-1)//columns)
    canvas = np.full((height, width, 3), 246, np.uint8)
    for index, oid in enumerate(entries):
        x, y = (index % columns)*115+8, (index//columns)*34+8
        color = ((145, 145, 145) if oid == UNKNOWN else
                 (20, 20, 20) if oid == UNOBSERVED else object_color(oid))
        label = 'uncertain' if oid == UNKNOWN else 'unobserved' if oid == UNOBSERVED else f'O{oid}'
        cv2.rectangle(canvas, (x, y), (x+17, y+17), color, -1)
        cv2.putText(canvas, label, (x+23, y+14), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas

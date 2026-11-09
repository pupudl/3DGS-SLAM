#!/usr/bin/env python3
"""Inspect saved initial sparse rigid-motion hypotheses.

Only initial 3D matches, their stereo confidence, and H_sparse are used. No
Gaussian ownership, joint optimization, supplemental matches, or final motion
estimates enter the plots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pair_utils import discover, lift, load_depth, resolve, stereo_io
from strict_two_frame import stereo_confidence, with_anchor_confidence


# Same model colors as visualize_motion.py, stored here to keep this CPU-only.
PALETTE_RGB = np.array([
    [.18, .68, 1.], [1., .52, .16], [.30, .88, .42], [.92, .37, .82],
    [1., .85, .24], [.28, .86, .87], [.75, .56, 1.], [1., .47, .52],
])


def color(k: int) -> tuple[int, int, int]:
    return tuple(int(round(x * 255)) for x in PALETTE_RGB[k, ::-1])


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


def caption(image: np.ndarray, title: str, detail: str = '') -> np.ndarray:
    bar = np.full((50, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 21), cv2.FONT_HERSHEY_SIMPLEX,
                .59, (25, 25, 25), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(bar, detail, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .43, (55, 55, 55), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def dimmed(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.addWeighted(image, .43, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), .57, 0)


def draw_points(image: np.ndarray, points: np.ndarray, labels: np.ndarray,
                selected: int | None = None,
                explained: np.ndarray | None = None) -> np.ndarray:
    result = dimmed(image)
    for i, (p, label) in enumerate(zip(points, labels)):
        if not np.isfinite(p).all():
            continue
        xy = tuple(np.rint(p).astype(int))
        if selected is not None and label != selected and explained is not None and explained[i]:
            cv2.circle(result, xy, 4, (0, 225, 255), 1, cv2.LINE_AA)
        elif selected is not None and label != selected:
            cv2.circle(result, xy, 2, (150, 150, 150), -1, cv2.LINE_AA)
        elif label < 0:
            cv2.circle(result, xy, 2, (160, 160, 160), -1, cv2.LINE_AA)
        else:
            cv2.circle(result, xy, 4, color(int(label)), -1, cv2.LINE_AA)
            cv2.circle(result, xy, 5, (10, 10, 10), 1, cv2.LINE_AA)
    return result


def draw_reprojection(image: np.ndarray, measured: np.ndarray,
                      predicted: np.ndarray, valid: np.ndarray, model_color: tuple[int, int, int]) -> np.ndarray:
    result = dimmed(image)
    h, w = image.shape[:2]
    for obs, pred, okay in zip(measured, predicted, valid):
        if not np.isfinite(obs).all():
            continue
        actual = tuple(np.rint(obs).astype(int))
        # The predicted point is deliberately not clipped to the image border.
        # Clipping would make a failed hypothesis appear to have a short error.
        inside = bool(okay and np.isfinite(pred).all() and
                      0 <= pred[0] < w and 0 <= pred[1] < h)
        if inside:
            estimate = tuple(np.rint(pred).astype(int))
            cv2.line(result, actual, estimate, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(result, estimate, (255, 255, 255),
                           cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA)
            cv2.drawMarker(result, estimate, (15, 15, 15),
                           cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)
        cv2.circle(result, actual, 4, model_color, -1, cv2.LINE_AA)
        cv2.circle(result, actual, 5, (15, 15, 15), 1, cv2.LINE_AA)
    return result


def rotation_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_inputs(cfg: dict, matches: np.ndarray):
    first, second = map(int, (cfg['frames']['first'], cfg['frames']['second']))
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    K, baseline = stereo_io.read_calibration(data_root / 'calibration' / 'perspective.txt')
    images, depths, confidences = [], [], []
    for frame in (first, second):
        left = stereo_io.read_image(sequence_root / 'image_00/data_rect' / f'{frame:010d}.png')
        right = stereo_io.read_image(sequence_root / 'image_01/data_rect' / f'{frame:010d}.png')
        depth = load_depth(sequence_root, frame, left, right, K, baseline, cfg['depth'])
        sky = sequence_root / 'sky_masks' / f'{frame:010d}.png'
        if sky.is_file():
            depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
        images.append(left)
        depths.append(depth)
        confidences.append(stereo_confidence(
            left, right, depth, K, baseline,
            float(cfg['depth'].get('confidence_rgb_sigma', 0.15))))
    anchors = with_anchor_confidence(
        lift(matches, depths[0], depths[1], K), confidences[0], confidences[1], cfg)
    return images, K, anchors


def visualize(result_dir: Path, output_dir: Path | None = None) -> dict:
    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / 'sparse_motion_visualization').resolve()
    cfg = yaml.safe_load((result_dir / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    hybrid = summary.get('initialization_method') == 'hybrid'
    with np.load(result_dir / 'pairwise_state.npz') as saved:
        H = saved['H_sparse'].astype(np.float64)
        matches = (saved['anchors_initial'].copy() if 'anchors_initial' in saved.files else
                   saved['anchors_final'][:int(summary['initial_3d_anchors'])].copy())
        if hybrid:
            labels = saved['initial_motion_labels'].astype(np.int64)
            costs = saved['initial_motion_costs'].copy()
    if len(H) > len(PALETTE_RGB):
        raise ValueError('Add more colors for the saved motion count')
    images, K, anchors = load_inputs(cfg, matches)
    if len(anchors['matches']) != len(matches) or not np.allclose(anchors['matches'], matches):
        raise ValueError('Saved initial matches no longer lift to the same valid 3D anchors')
    if hybrid:
        if (labels.shape != (len(matches),) or costs.shape != (len(matches), len(H)) or
                np.any(labels < -1) or np.any(labels >= len(H))):
            raise ValueError('Saved hybrid labels/costs do not align with the initial anchors')
        supports = np.bincount(labels[labels >= 0], minlength=len(H)).tolist()
        if supports != summary['initial_model_supports']:
            raise ValueError('Saved hybrid labels do not reproduce the recorded supports')
    else:
        models, labels, supports = discover(anchors, cfg['motion'], int(cfg.get('seed', 7)))
        if (len(models) != len(H) or supports != summary['initial_model_supports'] or
                not np.allclose(np.stack(models), H, atol=1e-7, rtol=1e-7)):
            raise ValueError('Replayed RANSAC labels do not reproduce saved H_sparse and supports')
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = int(summary['reference_sparse'])
    if not 0 <= reference < len(H):
        raise ValueError('Invalid sparse reference model')
    relative = np.linalg.inv(H[reference]) @ H
    all_source = draw_points(images[0], matches[:, :2], labels)
    all_target = draw_points(images[1], matches[:, 2:], labels)
    n_unassigned = int(np.count_nonzero(labels < 0))
    assignment = 'hybrid competition' if hybrid else 'sequential RANSAC'
    rows = [np.hstack((caption(cv2.resize(all_source, None, fx=.5, fy=.5),
                               'Initial matches: frame 1', f'{len(matches)} valid 3D pairs; gray = unassigned'),
                       caption(cv2.resize(all_target, None, fx=.5, fy=.5),
                               'Observed matches: frame 2', f'{n_unassigned} gray points unassigned by {assignment}')))]
    motions = []
    threshold = float(cfg['motion'].get('inlier_threshold_m', .5))
    depth_scale = float(cfg['motion'].get('adaptive_depth_threshold_m', 0))
    hybrid_threshold = float(cfg['motion'].get('hybrid', {}).get('outlier_cost', 1.0))
    for k, transform in enumerate(H):
        selected = labels == k
        xyz_pred = anchors['x0'] @ transform[:3, :3].T + transform[:3, 3]
        residual3d = np.linalg.norm(xyz_pred - anchors['x1'], axis=1)
        positive_z = xyz_pred[:, 2] > 1e-6
        uv_pred = np.full((len(xyz_pred), 2), np.nan, np.float64)
        uv_pred[positive_z, 0] = K[0, 0] * xyz_pred[positive_z, 0] / xyz_pred[positive_z, 2] + K[0, 2]
        uv_pred[positive_z, 1] = K[1, 1] * xyz_pred[positive_z, 1] / xyz_pred[positive_z, 2] + K[1, 2]
        pixel_error = np.linalg.norm(uv_pred - matches[:, 2:], axis=1)
        h, w = images[1].shape[:2]
        inside = (positive_z & (uv_pred[:, 0] >= 0) & (uv_pred[:, 0] < w) &
                  (uv_pred[:, 1] >= 0) & (uv_pred[:, 1] < h))
        if depth_scale > 0:
            z = np.maximum(anchors['x0'][:, 2], anchors['x1'][:, 2])
            thresholds = threshold * (1 + (z / depth_scale) ** 2)
        else:
            thresholds = threshold
        explained = costs[:, k] < hybrid_threshold if hybrid else residual3d < thresholds
        median3d = float(np.median(residual3d[selected]))
        median_px = float(np.median(pixel_error[selected & inside])) if np.any(selected & inside) else None
        r = relative[k]
        motion = dict(
            model=k, reference=(k == reference), support=int(selected.sum()),
            points_explained_independently=int(explained.sum()),
            support_predictions_inside_image=int(np.count_nonzero(selected & inside)),
            support_median_3d_residual_m=median3d,
            support_median_reprojection_error_px=median_px,
            relative_rotation_deg=rotation_deg(r[:3, :3]),
            relative_translation_m=r[:3, 3].tolist(),
            H_first_camera_to_second_camera=transform.tolist(),
            H_relative_to_reference=r.tolist(),
            detail_image=f'M{k}_sparse_fit.png',
        )
        if hybrid:
            motion['support_median_hybrid_cost'] = float(np.median(costs[selected, k]))
        else:
            motion['ransac_support'] = int(selected.sum())
        motions.append(motion)
        source = draw_points(images[0], matches[:, :2], labels, k, explained)
        target = draw_reprojection(images[1], matches[selected, 2:], uv_pred[selected],
                                   inside[selected], color(k))
        role = 'REFERENCE' if k == reference else 'CANDIDATE'
        detail = (f'support={supports[k]}; explains={int(explained.sum())}; 3D median={median3d:.2f} m; '
                  f'pixel median={median_px:.1f} px' if median_px is not None else
                  f'support={supports[k]}; explains={int(explained.sum())}; '
                  f'3D median={median3d:.2f} m; no valid projections')
        full = np.hstack((caption(source, f'M{k} {role}: frame 1 support', detail),
                          caption(target, f'M{k}: frame 2 observed dots / predicted crosses',
                                  'Left: filled = assigned; yellow rings = also explained; right: yellow line = error')))
        write_image(output_dir / motion['detail_image'], full)
        small_source = cv2.resize(source, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(target, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        rows.append(np.hstack((caption(small_source, f'M{k} {role}: frame 1 support', detail),
                               caption(small_target, f'M{k}: observed dot / predicted cross',
                                       f'vs reference: rot={motion["relative_rotation_deg"]:.2f} deg; '
                                       f't={np.linalg.norm(r[:3, 3]):.2f} m'))))
    support_description = (
        'Hybrid support labels and normalized geometry costs are read from the saved '
        'initialization, without replaying sequential 3D RANSAC. Labels are exclusive '
        f'core support; independent support uses hybrid cost < {hybrid_threshold:g} '
        'and can overlap across models. ' if hybrid else
        'RANSAC support is the set selected sequentially during discovery; independently '
        'explained points can overlap across models, and the final rigid refit can move '
        'a recorded support point across the inlier threshold. ')
    report = dict(
        frames=summary['frames'], stage=('initial sparse hybrid motion competition' if hybrid
                                       else 'initial sparse RANSAC only'),
        initialization_method=summary.get('initialization_method', 'ransac3d'),
        initial_valid_3d_matches=len(matches), unassigned_matches=n_unassigned,
        reference_sparse=reference, inlier_threshold_m=threshold,
        adaptive_depth_threshold_m=depth_scale, models=motions,
        interpretation=('H_sparse maps first-frame camera 3D coordinates into the '
                        'second-frame camera coordinates. ' + support_description + 'Relative motion '
                        'is inverse(H_reference) @ H_k. Pixels and 3D residuals use '
                        'the same initial stereo matches, before Gaussian fitting '
                        'and render feedback. Sparse M indices may differ from final M indices.'))
    if hybrid:
        report['hybrid_inlier_cost_threshold'] = hybrid_threshold
    (output_dir / 'hypotheses.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    write_image(output_dir / 'overview.png', np.vstack(rows))
    print(f'{output_dir}: {len(matches)} initial 3D matches, {len(H)} sparse models, '
          f'{n_unassigned} unassigned; ' +
          ('saved hybrid support verified' if hybrid else 'RANSAC replay verified'))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True, help='Saved pair/variant directory')
    parser.add_argument('--output', type=Path, help='Defaults to RESULT/sparse_motion_visualization')
    args = parser.parse_args()
    visualize(args.result, args.output)


if __name__ == '__main__':
    main()

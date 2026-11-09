"""Visualize saved two-frame rigid motions and Gaussian motion ownership.

The first-frame Gaussian map is deterministically reconstructed from the saved
config and stereo input; no estimates are optimized or changed here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))
from pair_utils import stereo_io, resolve, load_depth, make_gaussians  # noqa: E402
from joint_pair import GaussianRasterizer, setup_camera  # noqa: E402

# RGB colors with nonzero channels so black consistently means no rendered support.
PALETTE_RGB = np.array([
    [0.18, 0.68, 1.00],  # reference blue
    [1.00, 0.52, 0.16],  # orange
    [0.30, 0.88, 0.42],  # green
    [0.92, 0.37, 0.82],  # magenta
    [1.00, 0.85, 0.24],  # yellow
    [0.28, 0.86, 0.87],  # cyan
    [0.75, 0.56, 1.00],  # purple
    [1.00, 0.47, 0.52],  # coral
], dtype=np.float32)


def source_map(cfg: dict):
    sequence = cfg['dataset']['sequence']
    first, second = int(cfg['frames']['first']), int(cfg['frames']['second'])
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / sequence
    left = sequence_root / 'image_00' / 'data_rect'
    right = sequence_root / 'image_01' / 'data_rect'
    image0 = stereo_io.read_image(left / f'{first:010d}.png')
    image1 = stereo_io.read_image(left / f'{second:010d}.png')
    right0 = stereo_io.read_image(right / f'{first:010d}.png')
    K, baseline = stereo_io.read_calibration(data_root / 'calibration' / 'perspective.txt')
    depth = load_depth(sequence_root, first, image0, right0, K, baseline, cfg['depth'])
    sky = sequence_root / 'sky_masks' / f'{first:010d}.png'
    if sky.is_file():
        depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
    gc = cfg['gaussians']
    gmap = make_gaussians(image0, depth, K, int(gc['stride']),
                          float(gc['max_depth_jump_m']),
                          float(gc.get('projected_sigma_px', 1.6)),
                          gc.get('initialization', 'surface_aligned'))
    return gmap, K, image0, image1


def rotation_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def write_image(path: Path, image: np.ndarray):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Failed to write {path}')


def to_bgr_float(rgb: torch.Tensor) -> np.ndarray:
    arr = rgb.detach().permute(1, 2, 0).cpu().numpy()
    return np.clip(arr[:, :, ::-1], 0, 1).copy()


def overlay_on_gray(image_bgr: np.ndarray, color_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)/255
    strength = np.clip(color_bgr.max(axis=2)/0.35, 0, 1)[..., None]
    result = background*(1-0.8*strength) + color_bgr*(0.8*strength)
    return np.clip(result*255, 0, 255).astype(np.uint8)


def add_caption(image: np.ndarray, label: str) -> np.ndarray:
    h, w = image.shape[:2]
    bar = np.full((36, w, 3), 255, np.uint8)
    cv2.putText(bar, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack((bar, image))


def rasterizer(shape: tuple[int, int], K: np.ndarray, threshold: float):
    h, w = shape
    return GaussianRasterizer(raster_settings=setup_camera(
        w, h, K, np.eye(4), depth_threshold=threshold))


def render(raster, xyz, colors, opacities, scales, cov6=None) -> np.ndarray:
    xyz = torch.as_tensor(np.ascontiguousarray(xyz), dtype=torch.float32, device='cuda')
    colors = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device='cuda')
    opacities = torch.as_tensor(np.ascontiguousarray(opacities), dtype=torch.float32, device='cuda')
    if cov6 is None:
        scales = torch.as_tensor(np.ascontiguousarray(scales), dtype=torch.float32, device='cuda')
        rotations = torch.zeros((len(xyz), 4), dtype=torch.float32, device='cuda')
        rotations[:, 0] = 1
        covariance = None
    else:
        scales, rotations = None, None
        covariance = torch.as_tensor(np.ascontiguousarray(cov6), dtype=torch.float32, device='cuda')
    with torch.inference_mode():
        rendered, _, _, _ = raster(
            means3D=xyz, means2D=torch.zeros_like(xyz), colors_precomp=colors,
            opacities=opacities, scales=scales, rotations=rotations,
            cov3D_precomp=covariance, shs=None)
    return to_bgr_float(rendered)


def cov6_from_basis(gmap: dict, rotation: np.ndarray) -> np.ndarray:
    local = gmap['basis']*gmap['scale'][:, None, :]
    basis = np.einsum('nij,njk->nik', rotation, local)
    cov = basis @ np.transpose(basis, (0, 2, 1))
    return np.stack((cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                     cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]), axis=1).astype(np.float32)


def write_motion_table(path: Path, items: list[dict], reference: int, width: int):
    height = 64 + len(items)*64
    canvas = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(canvas, 'Rigid motions relative to reference', (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (15, 15, 15), 2, cv2.LINE_AA)
    for k, item in enumerate(items):
        y = 70+k*64
        color = tuple(int(v*255) for v in PALETTE_RGB[k, ::-1])
        cv2.rectangle(canvas, (17, y-19), (51, y+14), color, -1)
        name = f'M{k}' + ('  CAMERA REFERENCE' if k == reference else '')
        detail = (f"hard={item['hard_gaussians']:,}  rot={item['relative_rotation_deg']:.2f} deg  "
                  f"t=({item['relative_translation_m'][0]:+.2f}, "
                  f"{item['relative_translation_m'][1]:+.2f}, "
                  f"{item['relative_translation_m'][2]:+.2f}) m")
        cv2.putText(canvas, name, (65, y-2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, detail, (65, y+22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.53, (55, 55, 55), 1, cv2.LINE_AA)
    write_image(path, canvas)


def visualize(result_dir: Path, output_dir: Path | None = None):
    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / 'motion_visualization').resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((result_dir / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    with np.load(result_dir / 'pairwise_state.npz') as archive:
        H = archive['H_final'].astype(np.float32)
        q = archive['q_gaussian_final'].astype(np.float32)
        anchor_state = ({name: archive[f'anchor_gaussian_{name}_final'].copy()
                         for name in ('indices', 'xyz', 'uv', 'color', 'scale')}
                        if 'anchor_gaussian_indices_final' in archive else None)
    gmap, K, first_image, second_image = source_map(cfg)
    if anchor_state is not None:
        ids = anchor_state['indices'].astype(np.int64)
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(gmap['xyz'])):
            raise ValueError('Invalid saved anchor Gaussian indices')
        for name in ('xyz', 'uv', 'color', 'scale'):
            gmap[name][ids] = anchor_state[name]
    n, k = q.shape
    if n != len(gmap['xyz']) or k != len(H):
        raise ValueError(f'Saved state and reconstructed map mismatch: q={q.shape}, map={len(gmap["xyz"])}, H={H.shape}')
    if k > len(PALETTE_RGB):
        raise ValueError(f'Palette has {len(PALETTE_RGB)} colors, but saved state has {k} motions')
    if not np.allclose(q.sum(axis=1), 1, atol=1e-3):
        raise ValueError('Gaussian motion probabilities do not sum to one')
    reference = int(summary['reference_final'])
    labels = q.argmax(axis=1)
    confidence = q.max(axis=1)
    h, w = second_image.shape[:2]
    source_color = first_image.astype(np.float32)
    ownership_first = source_color.copy()
    uv = np.rint(gmap['uv']).astype(np.int32)
    source_palette = PALETTE_RGB[labels, ::-1]*255
    ownership_first[uv[:, 1], uv[:, 0]] = (0.35*source_color[uv[:, 1], uv[:, 0]] +
                                             0.65*source_palette)
    write_image(output_dir / 'ownership_first_hard.png', ownership_first.astype(np.uint8))
    confidence_image = np.zeros(first_image.shape[:2], np.uint8)
    confidence_image[uv[:, 1], uv[:, 0]] = np.rint(255*confidence).astype(np.uint8)
    write_image(output_dir / 'ownership_first_confidence.png',
                cv2.applyColorMap(confidence_image, cv2.COLORMAP_VIRIDIS))

    threshold = float(cfg['joint'].get('render_depth_threshold_m', 1.0))
    alpha = float(cfg['joint'].get('gaussian_opacity', 0.9))
    raster = rasterizer((h, w), K, threshold)
    centers = gmap['xyz']
    isotropic = gmap['initialization'] == 'lsgslam_projective'
    transformed = np.stack([centers @ H[i, :3, :3].T + H[i, :3, 3]
                            for i in range(k)])
    if isotropic:
        soft_cov = None
        soft_scales = np.tile(gmap['scale'], (k, 1))
    else:
        soft_cov = np.concatenate([
            cov6_from_basis(gmap, np.broadcast_to(H[i, :3, :3], (n, 3, 3)))
            for i in range(k)])
        soft_scales = None
    soft_rgb = np.repeat(PALETTE_RGB[:k], n, axis=0)
    soft_bgr = render(raster, transformed.reshape(k*n, 3), soft_rgb,
                      (alpha*q.T.reshape(-1, 1)), soft_scales, soft_cov)
    write_image(output_dir / 'ownership_second_soft.png',
                overlay_on_gray(second_image, soft_bgr))

    chosen_R = H[labels, :3, :3]
    hard_centers = transformed[labels, np.arange(n)]
    if isotropic:
        hard_cov, hard_scales = None, gmap['scale']
    else:
        hard_cov, hard_scales = cov6_from_basis(gmap, chosen_R), None
    hard_bgr = render(raster, hard_centers, PALETTE_RGB[labels],
                      np.full((n, 1), alpha, np.float32), hard_scales, hard_cov)
    write_image(output_dir / 'ownership_second_hard.png',
                overlay_on_gray(second_image, hard_bgr))

    relative = np.linalg.inv(H[reference].astype(np.float64)) @ H.astype(np.float64)
    items = []
    component_images = []
    for i in range(k):
        selected = labels == i
        if selected.any():
            single_rgb = render(raster, hard_centers[selected], gmap['color'][selected],
                                np.full((int(selected.sum()), 1), alpha, np.float32),
                                gmap['scale'][selected] if isotropic else None,
                                hard_cov[selected] if hard_cov is not None else None)
        else:
            single_rgb = np.zeros((h, w, 3), np.float32)
        component = np.clip(single_rgb*255, 0, 255).astype(np.uint8)
        filename = f'component_{i:02d}' + ('_reference' if i == reference else '') + '.png'
        write_image(output_dir / filename, component)
        component_images.append(component)
        items.append({
            'model': i, 'reference': i == reference,
            'hard_gaussians': int(selected.sum()),
            'hard_fraction': float(selected.mean()),
            'soft_mass': float(q[:, i].sum()),
            'relative_rotation_deg': rotation_deg(relative[i, :3, :3]),
            'relative_translation_m': relative[i, :3, 3].tolist(),
            'H_first_camera_to_second_camera': H[i].tolist(),
            'H_relative_to_reference': relative[i].tolist(),
            'component_image': filename,
        })
    (output_dir / 'motions.json').write_text(json.dumps({
        'frames': summary['frames'], 'reference_model': reference,
        'gaussians': n, 'anchor_gaussians': 0 if anchor_state is None else len(ids),
        'motions': items,
        'interpretation': ('H maps first-camera coordinates to second-camera coordinates; '
                           'H_reference^-1 H_k removes the camera/reference motion. '
                           'Hard ownership is argmax(q); soft ownership uses alpha*q and '
                           'can blend colors. These are estimated motion components, not '
                           'semantic object masks or ground truth.')}, indent=2), encoding='utf-8')
    write_motion_table(output_dir / 'motion_table.png', items, reference, w)

    def small(image):
        return cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA)
    table = cv2.imread(str(output_dir / 'motion_table.png'))
    # The table text fits within 704 pixels. Keep its native text scale and
    # extend the neighboring image panel when there are several components.
    table = table[:, :704]
    table_height = max(188, table.shape[0])
    def pad_height(image):
        padded = np.full((table_height, 704, 3), 255, np.uint8)
        padded[:image.shape[0]] = image
        return padded
    table = pad_height(table)
    rows = [np.hstack((add_caption(small(ownership_first.astype(np.uint8)), 'First: hard Gaussian ownership'),
                       add_caption(small(cv2.imread(str(output_dir / 'ownership_second_soft.png'))),
                                   'Second: soft motion-weighted render'))),
            np.hstack((add_caption(pad_height(small(cv2.imread(str(output_dir / 'ownership_second_hard.png')))),
                                   'Second: hard Gaussian ownership'),
                       add_caption(table, 'Motion parameters vs reference')))]
    for i in range(0, k, 2):
        left = add_caption(small(component_images[i]), f'M{i}: hard-assigned Gaussian subset')
        right = (add_caption(small(component_images[i+1]), f'M{i+1}: hard-assigned Gaussian subset')
                 if i+1 < k else np.full_like(left, 255))
        rows.append(np.hstack((left, right)))
    write_image(output_dir / 'overview.png', np.vstack(rows))
    print(f'Wrote {output_dir}  models={k}  gaussians={n}  reference={reference}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True,
                        help='One saved pair/variant result directory')
    parser.add_argument('--output', type=Path, help='Defaults to RESULT/motion_visualization')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to render motion components')
    visualize(args.result, args.output)


if __name__ == '__main__':
    main()

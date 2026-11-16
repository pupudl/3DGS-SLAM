#!/usr/bin/env python3
"""Prepare the selected KITTI-360 frames, including frames without GT poses.

Writes only missing/invalid depth and sky masks. No pose, VPR or object masks
are required. CUDA_VISIBLE_DEVICES is honored and models run sequentially.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
from run_sequence import load_config, resolve, sequence_frames


def inspect_inputs(root, frames, cfg):
    """Check every requested image, independent of the sparse GT frame index."""
    pending = dict(depth=[], sky=[], missing_images=[])
    for frame in frames:
        stem = f'{frame:010d}'
        left, right = [root/f'image_0{c}/data_rect/{stem}.png' for c in (0, 1)]
        try:
            with Image.open(left) as image:
                shape = image.height, image.width
            with Image.open(right) as image:
                if (image.height, image.width) != shape:
                    raise ValueError('Stereo image dimensions differ')
        except (OSError, ValueError) as error:
            pending['missing_images'].append(dict(frame=frame, reason=str(error)))
            continue
        depth_path = root/cfg['depth'].get('subdir', 'depth_sceneflow')/f'{stem}.npy'
        try:
            depth = np.load(depth_path, mmap_mode='r', allow_pickle=False)
            valid = (depth.shape == shape and np.issubdtype(depth.dtype, np.floating) and
                     np.any(np.isfinite(depth) & (depth > 0)))
            del depth
        except (OSError, ValueError, EOFError):
            valid = False
        if not valid:
            pending['depth'].append(frame)
        sky_root = resolve(cfg['sky_mask']['precomputed_root']) if cfg['sky_mask'].get('precomputed_root') else root
        sky_path = sky_root/cfg['sky_mask'].get('precomputed_subdir', 'sky_masks')/f'{stem}.png'
        sky = cv2.imread(str(sky_path), cv2.IMREAD_GRAYSCALE) if sky_path.is_file() else None
        if sky is None or sky.shape != shape:
            pending['sky'].append(frame)
    return pending


def _atomic_numpy(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npy')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def disparity_to_depth(disparity, fx, baseline):
    disparity = np.asarray(disparity, np.float32)
    depth = np.zeros_like(disparity)
    valid = np.isfinite(disparity) & (disparity > 0)
    np.divide(fx*baseline, disparity, out=depth, where=valid)
    depth[~np.isfinite(depth) | (depth < .1)] = 0
    return depth


def prepare_depth(root, frames, options, checkpoint, device, iterations):
    import torch
    from tqdm import tqdm
    # This repository contains several packages called core: put IGEV first.
    sys.path.insert(0, str(ROOT/'third_party/IGEV-Stereo'))
    from core.igev_stereo import IGEVStereo
    from core.utils.utils import InputPadder
    from frontend.pair_utils import stereo_io

    K, baseline = stereo_io.read_calibration(root.parents[1]/'calibration/perspective.txt')
    args = argparse.Namespace(mixed_precision=False, hidden_dims=[128]*3,
        corr_implementation='reg', shared_backbone=False, corr_levels=2,
        corr_radius=4, n_downsample=2, slow_fast_gru=False, n_gru_layers=3, max_disp=192)
    model = IGEVStereo(args)
    weights = torch.load(checkpoint, map_location='cpu')
    weights = weights.get('state_dict', weights)
    model.load_state_dict({key.removeprefix('module.'): value for key, value in weights.items()})
    del weights
    model.to(device).eval()
    with torch.inference_mode():
        for frame in tqdm(frames, desc='IGEV depth'):
            images = []
            for camera in (0, 1):
                with Image.open(root/f'image_0{camera}/data_rect/{frame:010d}.png') as image:
                    rgb = np.array(image.convert('RGB'))
                images.append(torch.from_numpy(rgb).permute(2, 0, 1).float()[None].to(device))
            padder = InputPadder(images[0].shape, divis_by=32)
            disparity = model(*padder.pad(*images), iters=iterations, test_mode=True)
            disparity = padder.unpad(disparity).squeeze(0).squeeze(0).cpu().numpy()
            depth = disparity_to_depth(disparity, K[0, 0], baseline)
            _atomic_numpy(root/options.get('subdir', 'depth_sceneflow')/f'{frame:010d}.npy', depth)
    del model


def _signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def prepare_sky(root, frames, options, model_config, checkpoint, device):
    import torch
    from tqdm import tqdm
    from mmseg.apis import inference_model, init_model

    sky_root = resolve(options['precomputed_root']) if options.get('precomputed_root') else root
    output = sky_root/options.get('precomputed_subdir', 'sky_masks')
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(backend='mmseg_segformer', config=_signature(model_config),
        checkpoint=_signature(checkpoint), sky_class_id=10, format='binary_png_0_255')
    manifest_path = output/'_manifest.json'
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get('settings') != settings:
            raise ValueError(f'Sky model settings differ from {manifest_path}; use a separate mask directory')
    model = init_model(str(model_config), str(checkpoint), device=device)
    with torch.inference_mode():
        for frame in tqdm(frames, desc='Sky masks'):
            path = root/f'image_00/data_rect/{frame:010d}.png'
            with Image.open(path) as image:
                size = image.size
            result = inference_model(model, str(path))
            labels = result.pred_sem_seg.data.squeeze(0).cpu().numpy()
            mask = (labels == 10).astype(np.uint8)*255
            if mask.shape != size[::-1]:
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            temporary = output/f'{frame:010d}.tmp.png'
            if not cv2.imwrite(str(temporary), mask):
                raise OSError(f'Cannot write {temporary}')
            temporary.replace(output/f'{frame:010d}.png')
    temporary = manifest_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(dict(settings=settings, sequence_root=str(root.resolve()),
                                         last_prepared_frames=frames), indent=2)+'\n')
    temporary.replace(manifest_path)
    del model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=EXP/'configs/kitti360_sequence_flow.yaml')
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--end', type=int, required=True, help='Inclusive upper frame ID')
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--stage', choices=['all', 'depth', 'sky'], default='all')
    parser.add_argument('--check-only', action='store_true', help='Inspect inputs without loading models/writing files')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--igev-checkpoint', type=Path, default=ROOT/'third_party/IGEV-Stereo/pretrained_models/kitti15.pth')
    parser.add_argument('--sky-config', type=Path, default=ROOT/'checkpoints/mmseg/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.py')
    parser.add_argument('--sky-checkpoint', type=Path, default=ROOT/'checkpoints/mmseg/segformer_mit-b4_8xb1-160k_cityscapes-1024x1024.pth')
    parser.add_argument('--iterations', type=int, default=32, help='IGEV update iterations')
    args = parser.parse_args(argv)
    if not args.sequence or args.sequence in ('.', '..') or Path(args.sequence).name != args.sequence:
        parser.error('--sequence must be a sequence name without directory components')
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    cfg = load_config(args.config)
    if cfg['depth']['source'] != 'precomputed_igev' or cfg['sky_mask']['backend'] != 'precomputed_png':
        parser.error('This tool prepares precomputed IGEV depth and PNG sky masks')
    frames = sequence_frames(cfg, start=args.start, end=args.end, step=args.stride)
    root = resolve(cfg['dataset']['data_root'])/'data_2d_raw'/args.sequence
    pending = inspect_inputs(root, frames, cfg)
    pose_path = root.parents[1]/'data_poses'/args.sequence/'cam0_to_world.txt'
    truth_frames = {int(line.split()[0]) for line in pose_path.open() if line.strip()} if pose_path.is_file() else set()
    summary = dict(sequence=args.sequence, frames=len(frames), first=frames[0], last=frames[-1],
        stride=args.stride, ground_truth_frames=sum(f in truth_frames for f in frames),
        needs_depth=len(pending['depth']), needs_sky=len(pending['sky']),
        missing_images=pending['missing_images'], depth_examples=pending['depth'][:10],
        sky_examples=pending['sky'][:10])
    print(json.dumps(summary, indent=2), flush=True)
    if pending['missing_images']:
        raise FileNotFoundError('Requested stereo images are missing or have incompatible dimensions')
    if args.check_only:
        return 2 if pending['depth'] or pending['sky'] else 0
    need_depth = args.stage in ('all', 'depth') and pending['depth']
    need_sky = args.stage in ('all', 'sky') and pending['sky']
    for path in ([args.igev_checkpoint] if need_depth else [])+([args.sky_config, args.sky_checkpoint] if need_sky else []):
        if not path.is_file():
            raise FileNotFoundError(path)
    if need_depth or need_sky:
        import torch
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; activate lsgslam and check GPU access')
        if need_depth:
            prepare_depth(root, pending['depth'], cfg['depth'], args.igev_checkpoint, args.device, args.iterations)
            gc.collect()
            torch.cuda.empty_cache()
        if need_sky:
            prepare_sky(root, pending['sky'], cfg['sky_mask'], args.sky_config, args.sky_checkpoint, args.device)
    remaining = inspect_inputs(root, frames, cfg)
    requested = ['depth', 'sky'] if args.stage == 'all' else [args.stage]
    if any(remaining[key] for key in requested):
        raise RuntimeError('Preprocessing finished with missing/invalid requested outputs')
    print('Selected stages ready. Existing valid outputs were reused.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

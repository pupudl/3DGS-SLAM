#!/usr/bin/env python3
"""Browse saved learned Gaussian maps without retaining earlier frame geometry.

The RGB pass uses the mapping optimizer's native CUDA renderer. The explicitly
configured, size-limited torch_reference backend is also available for tests.
Object masks use one joint nearest-effective-splat pass with learned opacity.
These map products are separate from the original fixed-geometry object views.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import argparse
import json

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[2]

PARAMETER_KEYS = ('xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits')


def discover_map_snapshots(sequence_dir):
    """Return one disk snapshot per frame; the next submap owns shared frames.

    This keeps only paths in memory, so browsing a thousand-frame run does not
    concatenate archived geometry or upload all submaps to the renderer.
    """
    root = Path(sequence_dir).resolve()
    index = root / 'gaussian_map/submaps/manifest.json'
    if not index.is_file():
        return sorted((root / 'gaussian_map/frames').glob('frame_*.npz'))
    manifest = json.loads(index.read_text())
    if manifest.get('format') != 'fixed_gaussian_submaps':
        raise ValueError('Unrecognized Gaussian submap manifest')
    chosen = {}
    for entry in sorted(manifest['submaps'], key=lambda item: int(item['submap_id'])):
        directory = (root / entry['snapshots']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('Submap snapshot path leaves the result directory')
        for path in sorted(directory.glob('frame_*.npz')):
            chosen[path.name] = path
    return [chosen[name] for name in sorted(chosen)]


def prepare_snapshot(snapshot: dict) -> tuple[dict, dict, dict]:
    """Validate canonical parameters and prepare only owners with a known pose.

    Canonical covariance rotates with its owner into camera space. A map with
    no pose for an archived owner must not put that owner back into the scene.
    """
    batch = {key: np.asarray(snapshot[key], np.float32) for key in PARAMETER_KEYS}
    count = len(batch['xyz'])
    expected = dict(xyz=(count, 3), colors=(count, 3), log_scales=(count, 3),
                    rotations=(count, 4), opacity_logits=(count, 1))
    for key, shape in expected.items():
        if batch[key].shape != shape or not np.isfinite(batch[key]).all():
            raise ValueError(f'Map {key} must be finite with shape {shape}')
    if np.any((batch['colors'] < 0) | (batch['colors'] > 1)):
        raise ValueError('Map colors must be RGB in [0,1]')
    labels = np.asarray(snapshot['object_labels'], np.int64)
    gids = np.asarray(snapshot['map_ids'], np.int64)
    batch['map_ids'] = gids
    ids = np.asarray(snapshot['object_ids'], np.int64)
    transforms = np.asarray(snapshot['object_to_camera'], np.float32)
    K = np.asarray(snapshot['K'], np.float32)
    if labels.shape != (count,) or gids.shape != (count,):
        raise ValueError('Map labels and IDs must have shape [N]')
    if len(np.unique(gids)) != count or np.any(gids < 0) or np.any(labels < -1):
        raise ValueError('Map IDs must be unique nonnegative IDs; labels use -1 for unknown')
    if ids.ndim != 1 or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('Object IDs must be unique nonnegative IDs')
    if transforms.shape != (len(ids), 4, 4) or not np.isfinite(transforms).all():
        raise ValueError('Object-to-camera transforms must be finite [K,4,4]')
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Camera intrinsics must be finite [3,3] with positive focal lengths')
    if len(ids) and not np.allclose(transforms[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('Object-to-camera transforms must be homogeneous rigid transforms')
    if len(ids):
        rotation = transforms[:, :3, :3]
        if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3)
                or not np.allclose(np.linalg.det(rotation), 1., atol=1e-3)):
            raise ValueError('Object-to-camera transforms require proper rotation matrices')
    lengths = np.linalg.norm(batch['rotations'], axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-8):
        raise ValueError('Map rotation quaternions must have a finite nonzero length')
    order = np.argsort(ids)
    positions = np.searchsorted(ids[order], labels)
    matches = positions < len(ids)
    if len(ids):
        matches[matches] &= ids[order[positions[matches]]] == labels[matches]
    rows = np.flatnonzero(matches & (labels >= 0))
    per_point = transforms[order[positions[rows]]]
    xyz = np.einsum('nij,nj->ni', per_point[:, :3, :3], batch['xyz'][rows]) + per_point[:, :3, 3]
    q = batch['rotations'][rows] / lengths[rows]
    w, x, y, z = q.T
    basis = np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
                      2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
                      2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), 1).reshape(-1, 3, 3)
    with np.errstate(over='ignore', under='ignore'):
        scales = np.exp(batch['log_scales'][rows])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Learned log scales must produce finite positive scales')
    basis = per_point[:, :3, :3] @ basis
    scaled = basis * scales[:, None, :]
    covariance = scaled @ scaled.transpose(0, 2, 1)
    if not np.isfinite(xyz).all() or not np.isfinite(covariance).all():
        raise ValueError('Learned camera geometry and covariance must be finite')
    logits = batch['opacity_logits'][rows, 0]
    # Stable sigmoid also handles large, finite saved logits.
    opacity = np.exp(-np.logaddexp(0., -logits))
    view = dict(rows=rows, transforms=per_point, K=K)
    geometry = dict(xyz=xyz, covariance=covariance, opacity=opacity,
                    labels=labels[rows], map_ids=gids[rows],
                    input_gaussians=count, skipped_without_pose=int(count - len(rows)))
    return batch, view, geometry


def _array(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _render(batch, view, cfg, shape, renderer):
    from mapping.gaussian_map_optimizer import render_map_view
    # The public helper stages its batch on the rendering device. Compacting
    # first avoids uploading unrelated owners again for each object RGB pass.
    rows = view['rows']
    selected = {key: value[rows] for key, value in batch.items()}
    view = dict(view, rows=np.arange(len(rows), dtype=np.int64), image_shape=tuple(shape[:2]))
    result = render_map_view(selected, view, cfg, renderer=renderer)
    rgb = _array(result['rgb'])
    if rgb.shape == (3, *shape[:2]):
        rgb = rgb.transpose(1, 2, 0)
    if rgb.shape != (*shape[:2], 3) or not np.isfinite(rgb).all():
        raise ValueError('Map renderer must return finite image-sized RGB')
    bgr = np.rint(np.clip(rgb[:, :, ::-1], 0, 1) * 255).astype(np.uint8)
    return bgr, result


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not save map visualization: {path}')


def _write_index(destination, manifest):
    payload = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>联合优化高斯地图</title>
<style>body{font:16px system-ui,sans-serif;margin:24px;background:#f4f5f7;color:#20242b}
main{max-width:1600px;margin:auto}select,button{font:inherit;padding:8px;margin:4px}
img{display:block;max-width:100%;height:auto;margin-top:16px}p{line-height:1.6}</style>
<main><h1>联合优化高斯地图</h1><p>场景视图对比原图与优化后的高斯 RGB。
Object 视图上排为该物体的高斯 RGB，下排为原图上的可见 mask。
位置、尺度、旋转、颜色和不透明度均来自保存的地图参数；遮挡由所有当前物体共同计算。</p>
<label>视图 <select id="object"><option value="scene">完整场景</option></select></label>
<label>帧 <select id="frame"></select></label>
<button id="prev">上一帧</button><button id="next">下一帧</button>
<a id="download">打开图片</a><p id="status"></p><img id="image" alt="Learned Gaussian map">
<script>const data=__DATA__,o=document.querySelector('#object'),f=document.querySelector('#frame');
for(const item of data.objects)o.add(new Option('Object '+item.object_id,item.object_id));
function record(){return o.value==='scene'?data.scene:data.objects.find(x=>x.object_id==o.value)}
function show(){const item=record(),v=item?.views[f.selectedIndex];if(!v)return;
document.querySelector('#image').src=v.image;document.querySelector('#download').href=v.image;
document.querySelector('#status').textContent='帧 '+v.frame+' · '+data.renderer+
(v.silhouette_pixels===undefined?'':' · 可见像素 '+v.silhouette_pixels);
history.replaceState(null,'','#'+new URLSearchParams({object:o.value,frame:v.frame}));}
function choose(frame){f.replaceChildren();const item=record();if(!item)return;
for(const v of item.views)f.add(new Option(v.frame,v.frame));
if(frame!==undefined&&[...f.options].some(x=>x.value==frame))f.value=frame;show()}
o.onchange=()=>choose(f.value);f.onchange=show;
document.querySelector('#prev').onclick=()=>{f.selectedIndex=Math.max(0,f.selectedIndex-1);show()};
document.querySelector('#next').onclick=()=>{f.selectedIndex=Math.min(f.options.length-1,f.selectedIndex+1);show()};
const initial=new URLSearchParams(location.hash.slice(1));if(initial.has('object')){
const value=initial.get('object');if([...o.options].some(x=>x.value===value))o.value=value;}
choose(initial.get('frame')??undefined);</script></main>'''.replace('__DATA__', payload)
    (destination / 'index.html').write_text(page, encoding='utf-8')


def visualize_map(sequence_dir, state=None, width=640, object_id=None, write_html=True):
    """Render one saved learned map at a time with full-scene occlusion."""
    from ownership.joint_gaussian_visibility import nearest_gaussian_owners
    from mapping.gaussian_map_optimizer import CudaMapRenderer, TorchReferenceRenderer
    from tools.visualization.visualize_objects import (MASK_OVERLAY_ALPHA, caption, make_render_overview,
                                   object_color, read_image, read_sky_mask, mask_sky_visibility,
                                   mask_sky_image, resized)

    sequence_dir = Path(sequence_dir).resolve()
    if width <= 0 or (object_id is not None and int(object_id) < 0):
        raise ValueError('Width must be positive and the requested object ID nonnegative')
    if state is None:
        state = json.loads((sequence_dir / 'sequence_state.json').read_text())
    cfg = state['config']
    map_cfg = cfg.get('gaussian_map', {})
    optimization = dict(map_cfg)
    optimization.update(map_cfg.get('optimization', {}))
    backend = optimization.get('backend', 'cuda')
    if backend not in ('cuda', 'torch_reference'):
        raise ValueError('Map backend must be cuda or explicitly selected torch_reference')
    device = optimization.get('device', 'cpu' if backend == 'torch_reference' else 'cuda')
    snapshots = discover_map_snapshots(sequence_dir)
    if not snapshots:
        raise ValueError('No learned Gaussian map frame snapshots found')
    renderer = (TorchReferenceRenderer() if backend == 'torch_reference' else
                CudaMapRenderer(device, optimization.get('render_depth_threshold_m', 1.),
                                optimization.get('far', 1000.)))
    destination = sequence_dir / 'gaussian_map' / 'visualization'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'visibility').mkdir(exist_ok=True)
    previous = None
    manifest_path = destination / 'manifest.json'
    if object_id is not None and manifest_path.is_file():
        saved_manifest = json.loads(manifest_path.read_text())
        if saved_manifest.get('format') == 'learned_gaussian_map_frames_v1':
            previous = saved_manifest
    partial_refresh = previous is not None
    records = {}
    scene_views = previous['scene']['views'] if partial_refresh else []
    options = cfg.get('object_render', {})
    visibility_reports = previous['visibility']['frames'] if partial_refresh else []
    for snapshot_path in snapshots:
        with np.load(snapshot_path, allow_pickle=False) as archive:
            snapshot = {key: archive[key] for key in archive.files}
        frame = int(np.asarray(snapshot['frame']).item())
        source_submap = snapshot_path.parent.parent.name
        submap_id = (int(source_submap.removeprefix('submap_'))
                     if source_submap.startswith('submap_') else None)
        image = read_image(cfg, frame)
        batch, view, geometry = prepare_snapshot(snapshot)
        visibility = nearest_gaussian_owners(
            geometry['xyz'], geometry['covariance'], geometry['labels'], geometry['map_ids'],
            view['K'], image.shape, opacity=geometry['opacity'],
            threshold=float(options.get('visibility_alpha_threshold', .05)), device=device,
            max_candidate_pixels=int(options.get('visibility_candidate_pixel_budget', 1_000_000)))
        sky = read_sky_mask(cfg, frame, image.shape[:2])
        mask_sky_visibility(visibility, sky)
        if not partial_refresh:
            rgb, render_result = _render(batch, view, map_cfg, image.shape, renderer)
            rgb = mask_sky_image(rgb, sky)
            relative = f'frames/frame_{frame:010d}_scene.png'
            comparison = np.hstack((caption(resized(image, width), f'Frame {frame}: original RGB'),
                                    caption(resized(rgb, width), 'Learned Gaussian map RGB')))
            _write_image(destination / relative, comparison)
            scene_views.append(dict(frame=frame, image=relative, gaussians=len(view['rows']),
                                    snapshot=snapshot_path.relative_to(sequence_dir).as_posix(),
                                    snapshot_scope=str(np.asarray(snapshot.get('snapshot_scope',
                                                       'visible_map_within_active_budget')).item())))
            visibility_path = f'visibility/frame_{frame:010d}.npz'
            saved = dict(owner=visibility['owner'], depth=visibility['depth'],
                         map_ids=visibility['gaussian_ids'])
            if submap_id is not None:
                saved['submap_id'] = np.asarray(submap_id, np.int64)
            for key in ('alpha', 'depth'):
                if key in render_result:
                    saved[f'render_{key}'] = _array(render_result[key])
            np.savez_compressed(destination / visibility_path, **saved)
            visibility_reports.append(dict(frame=frame, owner_map=visibility_path,
                                           **visibility['metadata'],
                                           source_snapshot=snapshot_path.relative_to(sequence_dir).as_posix(),
                                           submap_id=submap_id,
                                           skipped_without_pose=geometry['skipped_without_pose']))
            del rgb, render_result
        ids = sorted(map(int, np.unique(geometry['labels'])))
        requested = ids if object_id is None else [int(object_id)]
        for oid in requested:
            selected = geometry['labels'] == oid
            if not selected.any():
                continue
            silhouette = visibility['owner'] == oid
            object_view = dict(view, rows=view['rows'][selected], transforms=view['transforms'][selected])
            object_rgb, _ = _render(batch, object_view, map_cfg, image.shape, renderer)
            object_rgb[~silhouette] = 0
            overlay = image.copy()
            overlay[silhouette] = np.rint(
                (1 - MASK_OVERLAY_ALPHA) * image[silhouette] +
                MASK_OVERLAY_ALPHA * np.asarray(object_color(oid))).astype(np.uint8)
            panel, _ = make_render_overview([object_rgb], [overlay], [frame], oid, width)
            relative = f'object{oid}/frame_{frame:010d}.png'
            _write_image(destination / relative, panel)
            record = records.setdefault(oid, dict(object_id=oid, views=[]))
            record['views'].append(dict(frame=frame, image=relative,
                                        silhouette_pixels=int(silhouette.sum())))
        # Only manifest metadata persists across frames; release all geometry.
        del snapshot, batch, view, geometry, image, visibility
    if partial_refresh:
        for item in previous['objects']:
            if int(item['object_id']) != int(object_id):
                records[int(item['object_id'])] = item
    manifest = dict(format='learned_gaussian_map_frames_v1', renderer=backend,
                    optimized_attributes=list(PARAMETER_KEYS), opacity='learned_per_gaussian',
                    scene=dict(views=scene_views), objects=[records[k] for k in sorted(records)],
                    visibility=dict(method='nearest_effective_gaussian_camera_z',
                                    all_scene_objects_considered=True, frames=visibility_reports))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if write_html:
        _write_index(destination, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path, help='Sequence result directory')
    parser.add_argument('--object-id', type=int)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--no-html', action='store_true', help='Save ordinary PNGs and manifests without an HTML page')
    args = parser.parse_args()
    manifest = visualize_map(args.result, width=args.width, object_id=args.object_id, write_html=not args.no_html)
    print(json.dumps(dict(frames=len(manifest['scene']['views']),
                          objects=len(manifest['objects']),
                          images=str(args.result / 'gaussian_map/visualization/frames')),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()

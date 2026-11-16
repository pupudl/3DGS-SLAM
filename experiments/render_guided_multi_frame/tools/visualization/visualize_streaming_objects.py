"""Render long runs one frame at a time and browse individual object images."""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import json
from contextlib import nullcontext

import cv2
import numpy as np


def visualize_streaming(sequence_dir, state, object_id=None, width=640, renderer=None):
    from ownership.object_gaussian_renderer import GaussianObjectRenderer
    from tools.visualization.visualize_objects import (MASK_OVERLAY_ALPHA, make_render_overview,
                                   object_color, read_image, read_sky_mask, mask_sky_visibility,
                                   mask_sky_image, resolved_path)

    sequence_dir = Path(sequence_dir)
    if object_id is not None and int(object_id) < 0:
        raise ValueError('Unknown labels are not objects to visualize')
    cfg = state['config']
    renderer = GaussianObjectRenderer() if renderer is None else renderer
    destination = sequence_dir / 'objects'
    destination.mkdir(exist_ok=True)
    visibility_dir = destination / 'visibility'
    visibility_dir.mkdir(exist_ok=True)
    geometry_keys = {'xyz', 'uv', 'color', 'scale', 'K', 'object_ids', 'object_labels',
                     'H_object', 'object_model_valid', 'basis', 'cov', 'covariance',
                     'cov6', 'cov3D_precomp', 'gaussian_ids'}
    records = {}
    pairs = state['pairs']
    visibility_path = destination / 'visibility.jsonl'
    with (visibility_path.open('w') if object_id is None else nullcontext(None)) as journal:
        for index, pair in enumerate(pairs):
            directory = resolved_path(pair['result_path'], sequence_dir)
            # Source geometry comes from this frame's real observations. Only
            # the final image needs projection from its incoming pair.
            endpoints = [(int(pair['first']), False)]
            if index == len(pairs) - 1:
                endpoints.append((int(pair['second']), True))
            for frame, terminal in endpoints:
                with np.load(directory / 'object_state.npz') as archive:
                    gmap = {key: archive[key] for key in geometry_keys if key in archive.files}
                ids = sorted(map(int, np.unique(gmap['object_labels'][gmap['object_labels'] >= 0])))
                if terminal:
                    temporal = directory / 'temporal_state.json'
                    if temporal.is_file():
                        evidence = json.loads(temporal.read_text()).get('object_window', {}).get('lifecycle', {})
                        observed = evidence.get('target_observed_object_ids')
                        if observed is not None:
                            ids = sorted(set(ids) & set(map(int, observed)))
                image = read_image(cfg, frame)
                sources = [dict(gmap=gmap, object_ids=ids, terminal=terminal)]
                requested = ids if object_id is None else [int(object_id)]
                rendered, visibility = renderer.render_scene(sources, image.shape, cfg, requested)
                sky = read_sky_mask(cfg, frame, image.shape[:2])
                mask_sky_visibility(visibility, sky)
                if journal is not None:
                    np.savez_compressed(visibility_dir / f'frame_{frame:010d}.npz',
                                        owner=visibility['owner'], depth=visibility['depth'],
                                        gaussian_ids=visibility['gaussian_ids'])
                    journal.write(json.dumps(dict(frame=frame, **visibility['metadata'])) + '\n')
                for oid, (rgb, silhouette, metadata) in rendered.items():
                    rgb, silhouette = np.asarray(rgb), np.asarray(silhouette, bool)
                    if rgb.shape != image.shape or rgb.dtype != np.uint8 or silhouette.shape != image.shape[:2]:
                        raise ValueError('Renderer must return image-sized uint8 BGR and a silhouette')
                    if np.any(rgb[~silhouette]):
                        raise ValueError('Object RGB must use the same joint visible region as its mask')
                    rgb = mask_sky_image(rgb, sky)
                    silhouette = silhouette & ~sky
                    metadata = dict(metadata, silhouette_pixels=int(silhouette.sum()), sky_mask_pixels=int(sky.sum()))
                    if not np.any(silhouette):
                        continue
                    overlay = image.copy()
                    overlay[silhouette] = np.rint(
                        (1 - MASK_OVERLAY_ALPHA) * image[silhouette] +
                        MASK_OVERLAY_ALPHA * np.asarray(object_color(oid))).astype(np.uint8)
                    panel, _ = make_render_overview([rgb], [overlay], [frame], oid, width)
                    folder = destination / f'object{oid}'
                    folder.mkdir(exist_ok=True)
                    relative = f'object{oid}/frame_{frame:010d}.png'
                    if not cv2.imwrite(str(destination / relative), panel):
                        raise OSError(destination / relative)
                    record = records.setdefault(int(oid), dict(object_id=int(oid), views=[]))
                    record['views'].append(dict(frame=frame, image=relative,
                                                silhouette_pixels=int(silhouette.sum()),
                                                projected=terminal))
                # No cloud or rendered image from an earlier frame is cached.
                del gmap, sources, image, rendered, visibility
    for oid, record in records.items():
        life = state.get('objects', {}).get(str(oid), {})
        record.update(status=life.get('status'), first_observed_frame=life.get('first_observed_frame'),
                      last_observed_frame=life.get('last_observed_frame'),
                      frames=[view['frame'] for view in record['views']])
    manifest = dict(format='streaming_object_frames_v1', objects=[records[k] for k in sorted(records)],
                    visibility=dict(method='nearest_effective_gaussian_camera_z',
                                    journal='visibility.jsonl' if visibility_path.exists() else None),
                    description='One frame in memory; each image has Gaussian RGB above its visible mask.')
    manifest_path = destination / 'manifest.json'
    if object_id is not None and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get('format') == manifest['format']:
            manifest['objects'] += [item for item in previous['objects'] if item['object_id'] != int(object_id)]
            manifest['objects'].sort(key=lambda item: item['object_id'])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not cfg.get('sequence', {}).get('visualize_html', False):
        return manifest
    data = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>序列 object</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f5f7;color:#20242b}main{max-width:1100px;margin:auto}
select,button{font:inherit;padding:8px;margin:4px}img{display:block;max-width:100%;margin-top:16px}p{line-height:1.6}</style>
<main><h1>序列 object</h1><p>每张图上排为高斯 RGB，下排为原图上的可见区域。逐帧浏览已保存的归属结果。</p>
<label>Object <select id="object"></select></label><label>帧 <select id="frame"></select></label>
<button id="prev">上一帧</button><button id="next">下一帧</button><a id="download">打开图片</a>
<p id="status"></p><img id="image" alt="object Gaussian RGB and mask"></main>
<script>const data=__DATA__,o=document.querySelector('#object'),f=document.querySelector('#frame');
for(const item of data.objects)o.add(new Option('Object '+item.object_id,item.object_id));
function show(){const item=data.objects.find(x=>x.object_id==o.value);if(!item)return;
const view=item.views[f.selectedIndex];if(!view)return;document.querySelector('#image').src=view.image;
document.querySelector('#download').href=view.image;document.querySelector('#status').textContent=
'Object '+item.object_id+' · '+(item.status||'')+' · 首次观测 '+(item.first_observed_frame??item.frames[0])+
' · 最后观测 '+(item.last_observed_frame??item.frames.at(-1));}
function choose(){f.replaceChildren();const item=data.objects.find(x=>x.object_id==o.value);if(!item)return;
for(const view of item.views)f.add(new Option(view.frame,view.frame));show();}
o.onchange=choose;f.onchange=show;document.querySelector('#prev').onclick=()=>{f.selectedIndex=Math.max(0,f.selectedIndex-1);show()};
document.querySelector('#next').onclick=()=>{f.selectedIndex=Math.min(f.options.length-1,f.selectedIndex+1);show()};choose();</script>'''.replace('__DATA__', data)
    (destination / 'index.html').write_text(page, encoding='utf-8')
    return manifest

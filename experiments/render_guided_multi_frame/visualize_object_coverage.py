#!/usr/bin/env python3
"""Draw opaque object coverage from saved states, without running inference."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from object_masks import all_objects, legend_image, object_color, ownership_maps
from visualize_objects import caption, pair_frames, read_image, resized, resolved_path


def encoded_image(image, extension, parameters=()):
    ok, encoded = cv2.imencode(extension, image, list(parameters))
    if not ok:
        raise ValueError(f'Could not encode {extension} image')
    return ('data:image/'+('jpeg' if extension == '.jpg' else 'png')+';base64,'+
            base64.b64encode(encoded).decode('ascii'))


def encode_label_map(labels, object_ids):
    """Lossless RGB24 indices readable through a browser canvas, no 16-bit PNG loss."""
    codes = np.zeros(labels.shape, np.uint32)
    codes[labels == -1] = 1
    for index, oid in enumerate(object_ids, start=2):
        codes[labels == oid] = index
    bgr = np.stack(((codes >> 16) & 255, (codes >> 8) & 255, codes & 255), axis=-1).astype(np.uint8)
    return encoded_image(bgr, '.png', (cv2.IMWRITE_PNG_COMPRESSION, 9))


def coverage_html(payload):
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object 覆盖区域</title><style>
body{font:16px system-ui,sans-serif;margin:24px;background:#151a22;color:#f3f4f6}
main{max-width:1900px;margin:auto}p{line-height:1.65;color:#d3d8df}label{margin-right:18px;display:inline-block;margin-bottom:12px}
select{font:inherit;padding:7px;background:#fff;color:#151a22;border:0;border-radius:4px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}img,canvas{width:100%;height:auto;display:block;background:#101318;image-rendering:pixelated}
figure{margin:0}figcaption{padding:10px 0;color:#e8edf5}.legend{display:flex;flex-wrap:wrap;gap:14px;margin:18px 0}
.swatch{width:16px;height:16px;display:inline-block;vertical-align:middle;margin-right:6px;border:1px solid #aaa}
#stats{font-variant-numeric:tabular-nums}a{color:#8bd2ff}@media(max-width:850px){.views{grid-template-columns:1fr}}
</style><main><h1>Object 覆盖区域</h1>
<p>颜色完全不透明，直接显示保存的高斯中心覆盖范围。单独选择 object 时，其区域使用鲜明颜色，其余图像变暗，可选白色内边界。
“最大概率”包括尚未确定的高斯；“已接受标签”严格使用保存的标签，灰色只表示未确定，不表示原分配已被证明错误。
末帧按每个高斯最大概率对应的一种运动投影，并处理深度遮挡；这些图不是完整高斯渲染或语义物体轮廓。</p>
<label>帧 <select id="frame"></select></label>
<label>Object <select id="object"><option value="all">全部 object</option></select></label>
<label>显示 <select id="mode"><option value="argmax">最大概率（包含未确定）</option><option value="accepted">已接受标签（灰色为未确定）</option></select></label>
<label><input id="outline" type="checkbox"> 显示白色内边界</label>
<div class="legend" id="legend"></div><p id="stats"></p><div class="views">
<figure><figcaption>源 RGB（原始分辨率）</figcaption><img id="source" alt="源 RGB"></figure>
<figure><figcaption id="title"></figcaption><canvas id="coverage"></canvas></figure></div>
<p><a href="../object_coverage_overview.png">打开全部帧总览</a> · <a href="index.html">逐 object 多帧图</a></p>
<script>const data=__DATA__;const frames=document.querySelector('#frame'),objects=document.querySelector('#object'),mode=document.querySelector('#mode'),outline=document.querySelector('#outline');
for(let i=0;i<data.frames.length;i++){let f=data.frames[i];frames.add(new Option(f.frame+(f.terminal?'（末帧投影）':''),i))}
for(let oid of data.object_ids)objects.add(new Option('Object '+oid,oid));
const rgb={};for(let oid of data.object_ids){rgb[oid]=data.colors[oid];let e=document.createElement('span');let s=document.createElement('i');s.className='swatch';s.style.background='rgb('+rgb[oid].join(',')+')';e.append(s,document.createTextNode('O'+oid));document.querySelector('#legend').append(e)}
for(let [name,c] of [['未确定','#919191'],['无高斯中心','#161616']]){let e=document.createElement('span'),s=document.createElement('i');s.className='swatch';s.style.background=c;e.append(s,document.createTextNode(name));document.querySelector('#legend').append(e)}
function loadImage(url){return new Promise((resolve,reject)=>{let image=new Image();image.onload=()=>resolve(image);image.onerror=reject;image.src=url})}
const cache=new Map();async function frameData(i){if(cache.has(i))return cache.get(i);let f=data.frames[i];let images=await Promise.all([loadImage(f.source),loadImage(f.argmax),loadImage(f.accepted)]);
let scratch=document.createElement('canvas');scratch.width=f.width;scratch.height=f.height;let ctx=scratch.getContext('2d',{willReadFrequently:true});ctx.drawImage(images[0],0,0);let source=ctx.getImageData(0,0,f.width,f.height).data;
let maps=[];for(let image of images.slice(1)){ctx.clearRect(0,0,f.width,f.height);ctx.drawImage(image,0,0);let pixels=ctx.getImageData(0,0,f.width,f.height).data;let labels=new Int32Array(f.width*f.height);for(let j=0;j<labels.length;j++){let code=pixels[4*j]+256*pixels[4*j+1]+65536*pixels[4*j+2];labels[j]=code===0?-2:code===1?-1:data.object_ids[code-2]}maps.push(labels)}let result={source,maps};cache.set(i,result);return result}
let revision=0;async function update(){let version=++revision,i=Number(frames.value),f=data.frames[i],loaded=await frameData(i);if(version!==revision)return;
document.querySelector('#source').src=f.source;let labels=loaded.maps[mode.value==='argmax'?0:1],choice=objects.value,w=f.width,h=f.height;let canvas=document.querySelector('#coverage');canvas.width=w;canvas.height=h;let ctx=canvas.getContext('2d'),out=ctx.createImageData(w,h);let assigned=0,unknown=0,unobserved=0,selected=0;
for(let j=0;j<labels.length;j++){let label=labels[j],p=4*j,gray=Math.round((.299*loaded.source[p]+.587*loaded.source[p+1]+.114*loaded.source[p+2])*.18),color=[gray,gray,gray];if(label>=0)assigned++;else if(label===-1)unknown++;else unobserved++;
if(choice==='all'){if(label>=0)color=rgb[label];else if(label===-1)color=[145,145,145]}else if(label===Number(choice)){color=rgb[label];selected++;if(outline.checked){let x=j%w,y=Math.floor(j/w),neighbors=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){let nx=x+dx,ny=y+dy;if(nx>=0&&ny>=0&&nx<w&&ny<h&&labels[ny*w+nx]===label)neighbors++}if(neighbors>=4&&neighbors<9)color=[255,255,255]}}
out.data[p]=color[0];out.data[p+1]=color[1];out.data[p+2]=color[2];out.data[p+3]=255}ctx.putImageData(out,0,0);
document.querySelector('#title').textContent=(choice==='all'?'全部 object':'Object '+choice)+' · '+(mode.value==='argmax'?'最大概率，包含未确定':'已接受标签；灰色为未确定')+(f.terminal?' · 末帧单运动深度投影':' · 源帧高斯中心');
document.querySelector('#stats').textContent='分辨率 '+w+' × '+h+'；'+(choice==='all'?'object 标签像素 '+assigned:'所选 object 像素 '+selected)+'；未确定 '+unknown+'；无高斯中心 '+unobserved+'。不透明度 100%。'}
frames.onchange=objects.onchange=mode.onchange=outline.onchange=update;update();</script></main>'''.replace('__DATA__', data)


def visualize_coverage(result: Path, width=960):
    result = Path(result).resolve()
    state = json.loads((result / 'sequence_state.json').read_text())
    cfg = state.get('config')
    if cfg is None:
        cfg = yaml.safe_load((result / 'config.yaml').read_text())
    views = {}
    for pair in state['pairs']:
        first, second = pair_frames(pair)
        directory = resolved_path(pair['result_path'], result)
        views[first] = (directory, False, pair)
    if state['pairs']:
        pair = state['pairs'][-1]
        _, second = pair_frames(pair)
        views.setdefault(second, (resolved_path(pair['result_path'], result), True, pair))
    object_ids = set()
    entries, rows = [], []
    for frame, (directory, terminal, pair) in sorted(views.items()):
        path = directory / 'object_state.npz'
        if not path.is_file():
            path = directory / 'gaussian_state.npz'
        with np.load(path) as archive:
            gmap = {key: archive[key] for key in archive.files}
        if 'object_probabilities' not in gmap and 'local_to_object' not in gmap:
            gmap['local_to_object'] = pair['local_to_object']
        if 'object_ids' in gmap:
            object_ids.update(int(oid) for oid in np.asarray(gmap['object_ids']).reshape(-1) if oid >= 0)
        else:
            mapping = gmap['local_to_object']
            object_ids.update(int(oid) for oid in (mapping.values() if isinstance(mapping, dict) else mapping)
                              if int(oid) >= 0)
        image = read_image(cfg, frame)
        maps = ownership_maps(gmap, image.shape, terminal=terminal)
        for name in ('argmax', 'accepted'):
            object_ids.update(map(int, np.unique(maps[name][maps[name] >= 0])))
        entries.append((frame, terminal, image, maps))
        heading = f'Frame {frame}'+(' (terminal projection)' if terminal else '')
        rgb = caption(resized(image, width), heading+': source RGB', 'Original image')
        size = (width, max(1, round(image.shape[0]*width/image.shape[1])))
        provisional = caption(cv2.resize(all_objects(image, maps['argmax']), size,
                                        interpolation=cv2.INTER_NEAREST),
                              'Maximum probability: includes uncertain',
                              '100% opacity; one visible Gaussian centre per pixel')
        accepted = caption(cv2.resize(all_objects(image, maps['accepted']), size,
                                     interpolation=cv2.INTER_NEAREST),
                           'Accepted labels only: gray = uncertain',
                           'Gray is not evidence that an initial label was wrong')
        rows.append(np.hstack((rgb, provisional, accepted)))
    object_ids = sorted(object_ids)
    frames = []
    for frame, terminal, image, maps in entries:
        frames.append(dict(frame=frame, terminal=terminal, width=image.shape[1], height=image.shape[0],
                           source=encoded_image(image, '.jpg', (cv2.IMWRITE_JPEG_QUALITY, 90)),
                           argmax=encode_label_map(maps['argmax'], object_ids),
                           accepted=encode_label_map(maps['accepted'], object_ids)))
    payload = dict(object_ids=object_ids,
                   colors={str(oid): list(object_color(oid)[::-1]) for oid in object_ids}, frames=frames)
    if rows:
        overview = np.vstack([legend_image(object_ids, 3*width), *rows])
        if not cv2.imwrite(str(result / 'object_coverage_overview.png'), overview):
            raise OSError(result / 'object_coverage_overview.png')
    output = result / 'objects' / 'coverage.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(coverage_html(payload), encoding='utf-8')
    return dict(frames=len(frames), object_ids=object_ids, interactive=str(output),
                overview=str(result / 'object_coverage_overview.png'))


# Keep the short CLI/helper entry point available for existing callers.
visualize = visualize_coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--width', type=int, default=960)
    args = parser.parse_args()
    if args.width < 160:
        parser.error('--width must be at least 160')
    print(json.dumps(visualize_coverage(args.result, args.width)))


if __name__ == '__main__':
    main()

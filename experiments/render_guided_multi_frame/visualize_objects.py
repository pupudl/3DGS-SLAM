#!/usr/bin/env python3
"""One two-row overview per object: RGB renders above masks on original images.

Every frame has one column. Final labels select Gaussian subsets, while one
joint Gaussian Z buffer makes displayed object regions mutually exclusive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from object_masks import object_color
from object_gaussian_renderer import GaussianObjectRenderer

PROJECT_ROOT = HERE.parents[1]
MASK_OVERLAY_ALPHA = .65


def resolved_path(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    candidate = base / value
    return candidate if candidate.exists() else PROJECT_ROOT / value


def pair_frames(pair: dict) -> tuple[int, int]:
    if 'first' in pair:
        return int(pair['first']), int(pair['second'])
    return tuple(map(int, pair['frames']))


def pair_mapping(pair: dict, saved: dict) -> np.ndarray:
    if 'local_to_object' in saved:
        return np.asarray(saved['local_to_object'], np.int64).reshape(-1)
    values = pair['local_to_object']
    return np.asarray([values[str(k)] for k in range(len(values))], np.int64)


def caption(image: np.ndarray, title: str, detail: str = '') -> np.ndarray:
    bar = np.full((49 if detail else 30, image.shape[1], 3), 250, np.uint8)
    cv2.putText(bar, title, (9, 20), cv2.FONT_HERSHEY_SIMPLEX,
                .51, (25, 25, 25), 1, cv2.LINE_AA)
    if detail:
        cv2.putText(bar, detail, (9, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, (70, 70, 70), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def resized(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def frame_sources(state: dict, sequence_dir: Path) -> list:
    """Select outgoing estimates per object, retaining its last incoming view."""
    result = {}
    for pair in state['pairs']:
        first, second = pair_frames(pair)
        path = resolved_path(pair['result_path'], sequence_dir)
        object_path = path / 'object_state.npz'
        if object_path.is_file():
            with np.load(object_path) as archive:
                object_ids = archive['object_ids'].reshape(-1)
        else:
            object_ids = pair['local_to_object'].values()
        for oid in set(map(int, object_ids)):
            result.setdefault(first, {})[oid] = (pair, path, False)
            result.setdefault(second, {}).setdefault(oid, (pair, path, True))
    # Group objects that share a source so each saved map is read once per view.
    grouped = []
    for frame, objects in sorted(result.items()):
        views = {}
        for oid, (pair, path, terminal) in objects.items():
            key = (path, terminal)
            if key not in views:
                views[key] = (pair, path, terminal, [])
            views[key][3].append(oid)
        grouped.extend((frame, *view) for view in views.values())
    return grouped


def read_image(cfg: dict, frame: int) -> np.ndarray:
    root = resolved_path(cfg['dataset']['data_root'], PROJECT_ROOT)
    path = (root / 'data_2d_raw' / cfg['dataset']['sequence'] /
            'image_00/data_rect' / f'{frame:010d}.png')
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def make_render_overview(renders: list[np.ndarray], masks: list[np.ndarray],
                         frames: list[int], object_id: int, width: int) -> tuple[np.ndarray, dict]:
    """Keep all frames on two rows, including original images for absent objects."""
    if not frames or len(renders) != len(frames) or len(masks) != len(frames):
        raise ValueError('Each frame needs exactly one Gaussian render and one mask')
    if width <= 0:
        raise ValueError('Tile width must be positive')
    # Nearest sampling preserves exclusive ownership at overview resolution;
    # area interpolation would mix adjacent owners into the same small pixel.
    render_tiles = []
    for frame, image in zip(frames, renders):
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        render_tiles.append(caption(tile, f'Frame {frame}: Gaussian render'))
    mask_tiles = []
    for frame, mask in zip(frames, masks):
        height = max(1, round(mask.shape[0] * width / mask.shape[1]))
        tile = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_tiles.append(caption(tile, f'Frame {frame}: object {object_id} overlay'))
    tile_height = max(tile.shape[0] for tile in render_tiles + mask_tiles)
    header_height = 40
    canvas = np.zeros((header_height + 2 * tile_height, len(frames) * width, 3), np.uint8)
    canvas[:header_height] = 250
    cv2.putText(canvas, f'OBJECT {object_id}', (12, 27), cv2.FONT_HERSHEY_SIMPLEX,
                .7, object_color(object_id), 2, cv2.LINE_AA)
    for row, tiles in enumerate((render_tiles, mask_tiles)):
        for column, tile in enumerate(tiles):
            y, x = header_height + row * tile_height, column * width
            canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return canvas, dict(rows=2, columns=len(frames), tile_width=width,
                        tile_height=tile_height, header_height=header_height)


def write_index(output_dir: Path, manifest: dict):
    payload = json.dumps(manifest, ensure_ascii=False).replace('</', '<\\/')
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object 高斯渲染与 mask</title><style>
body{font:16px system-ui,sans-serif;margin:28px;background:#f4f5f7;color:#20242b}
main{max-width:2000px;margin:auto}select{font:inherit;padding:7px;min-width:160px}
img{width:100%;height:auto;margin-top:18px;background:black}
p{line-height:1.6}a{color:#1769aa}
</style><main><h1>Object 高斯渲染与 mask</h1>
<p>每列一帧：上排是最终 object 属性对应的高斯 RGB 渲染，下排在原图上叠加同一批高斯的彩色 mask。
所有 object 共同判断遮挡，每个像素只显示最近的有效高斯所属 object。
RGB 与 mask 使用同一可见区域；不同 object 使用不同颜色，未覆盖或未知区域保留原图。</p>
<label>Object <select id="objects"></select></label> <a id="download">打开原图</a>
<img id="overview" alt="object 多帧高斯渲染与 mask">
<script>const data=__DATA__;const select=document.querySelector('#objects');
for(const item of data.objects){select.add(new Option('Object '+item.object_id,item.object_id))}
function update(){const item=data.objects.find(x=>x.object_id==select.value);if(!item)return;
document.querySelector('#overview').src=item.overview;document.querySelector('#download').href=item.overview;
history.replaceState(null,'','#object='+item.object_id)}
let chosen=new URLSearchParams(location.hash.slice(1)).get('object');if(chosen)select.value=chosen;
select.onchange=update;update();</script></main>""".replace('__DATA__', payload)
    (output_dir / 'index.html').write_text(page, encoding='utf-8')


def visualize(sequence_dir: Path, object_id: int | None = None,
              max_columns: int | None = None, width: int = 640, renderer=None) -> dict:
    """Render saved final objects; max_columns is retained only for old callers."""
    sequence_dir = Path(sequence_dir).resolve()
    state = json.loads((sequence_dir / 'sequence_state.json').read_text())
    sources = frame_sources(state, sequence_dir)
    frames = list(map(int, state.get('frames') or sorted({source[0] for source in sources})))
    if not frames:
        raise ValueError('No frames found in the sequence result')
    object_ids = ([int(object_id)] if object_id is not None else
                  sorted(set(map(int, state.get('objects', []))) |
                         {oid for source in sources for oid in source[-1]}))
    if any(oid < 0 for oid in object_ids):
        raise ValueError('Unknown labels are not objects to visualize')
    renderer = GaussianObjectRenderer() if renderer is None else renderer
    views = {oid: {} for oid in object_ids}
    frame_images = {}
    config_cache = {}
    geometry_keys = {'xyz', 'uv', 'color', 'scale', 'K', 'object_ids', 'object_labels',
                     'H_object', 'object_model_valid', 'basis', 'cov', 'covariance',
                     'cov6', 'cov3D_precomp', 'gaussian_ids'}
    scenes = {}
    for frame, pair, result_dir, terminal, source_objects in sources:
        if frame not in frames:
            continue
        if result_dir not in config_cache:
            config_path = result_dir / 'config.yaml'
            config_cache[result_dir] = (yaml.safe_load(config_path.read_text())
                                       if config_path.is_file() else state.get('config'))
        cfg = config_cache[result_dir]
        if cfg is None:
            raise ValueError(f'Missing dataset/render configuration for {result_dir}')
        if frame not in frame_images:
            frame_images[frame] = read_image(cfg, frame)
        path = result_dir / 'object_state.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Final optimized object state is required: {path}')
        with np.load(path) as archive:
            if 'object_labels' not in archive.files:
                raise ValueError(f'Missing final object_labels in {path}')
            gmap = {key: archive[key] for key in geometry_keys if key in archive.files}
        scenes.setdefault(frame, []).append(dict(gmap=gmap, terminal=terminal,
                                                  object_ids=source_objects, config=cfg,
                                                  result_path=str(result_dir)))
    output_dir = sequence_dir / 'objects'
    output_dir.mkdir(parents=True, exist_ok=True)
    visibility_dir = output_dir / 'visibility'
    visibility_dir.mkdir(exist_ok=True)
    visibility_reports = []
    for frame, scene in sorted(scenes.items()):
        image = frame_images[frame]
        shape = image.shape
        # Prefer the outgoing source for unknown occluders in mixed-source views.
        scene.sort(key=lambda source: source['terminal'])
        rendered, visibility = renderer.render_scene(scene, shape, scene[0]['config'], object_ids)
        mask_count = np.zeros(shape[:2], np.uint16)
        for oid, (rgb, silhouette, metadata) in rendered.items():
            rgb = np.asarray(rgb)
            silhouette = np.asarray(silhouette, bool)
            if rgb.shape != shape or rgb.dtype != np.uint8 or silhouette.shape != shape[:2]:
                raise ValueError('Renderer must return image-sized uint8 BGR and a silhouette')
            if np.any(rgb[~silhouette]):
                raise ValueError('Object RGB must use the same joint visible region as its mask')
            mask_count += silhouette
            mask = image.copy()
            mask[silhouette] = np.rint(
                (1 - MASK_OVERLAY_ALPHA) * image[silhouette] +
                MASK_OVERLAY_ALPHA * np.asarray(object_color(oid))).astype(np.uint8)
            views[oid][frame] = (rgb, mask, dict(metadata, frame=frame,
                                                projected=bool(metadata.get('terminal', False))))
        if np.any(mask_count > 1):
            raise ValueError('Joint visibility returned overlapping object masks')
        relative = f'visibility/frame_{frame:010d}.npz'
        np.savez_compressed(output_dir / relative, owner=visibility['owner'],
                            depth=visibility['depth'], gaussian_ids=visibility['gaussian_ids'])
        visibility_reports.append(dict(visibility['metadata'], frame=int(frame), owner_map=relative,
                                       source_paths=[source['result_path'] for source in scene]))
    for frame in frames:
        if frame not in frame_images:
            cfg = state.get('config') or next(iter(config_cache.values()), None)
            if cfg is None:
                raise ValueError(f'Missing image dimensions for frame {frame}')
            frame_images[frame] = read_image(cfg, frame)
    manifest = dict(objects=[], description=(
        'Two rows, one column per frame: final object Gaussian RGB renders and '
        'colored masks overlaid on original images, clipped to a joint nearest '
        'effective Gaussian owner map. Saved final labels only.'),
        visibility=dict(method='nearest_effective_gaussian_camera_z', frames=visibility_reports,
                        unknown_source_policy='occlude_without_display',
                        unknown_terminal_policy='no_motion_available_not_projected',
                        rgb='native_object_rgb_clipped_to_joint_owner_mask'))
    for oid in object_ids:
        renders, masks, details = [], [], []
        for frame in frames:
            view = views[oid].get(frame)
            if view is None:
                image = frame_images[frame]
                view = (np.zeros_like(image), image.copy(),
                        dict(frame=frame, object_id=oid, selected_gaussians=0,
                             silhouette_pixels=0, renderer='empty',
                             reason='object_absent_in_saved_frame'))
            rgb, mask, metadata = view
            renders.append(rgb)
            masks.append(mask)
            details.append(metadata)
        overview, layout = make_render_overview(renders, masks, frames, oid, width)
        directory = output_dir / f'object{oid}'
        old_directory = output_dir / f'object_{oid:04d}'
        if old_directory.is_dir() and not directory.exists():
            old_directory.rename(directory)
        directory.mkdir(exist_ok=True)
        if not cv2.imwrite(str(directory / 'overview.png'), overview):
            raise OSError(directory / 'overview.png')
        item = dict(object_id=oid, overview=f'object{oid}/overview.png', frames=frames,
                    panel_count=2 * len(frames), layout=layout,
                    mask_overlay_alpha=MASK_OVERLAY_ALPHA,
                    color_bgr=list(object_color(oid)), frame_details=details,
                    selection='saved_final_object_labels_equal_to_object_id',
                    visibility='joint_nearest_effective_gaussian_camera_z')
        (directory / 'summary.json').write_text(json.dumps(item, indent=2))
        manifest['objects'].append(item)
    if object_id is not None and (output_dir / 'manifest.json').is_file():
        previous = json.loads((output_dir / 'manifest.json').read_text())
        manifest['objects'] += [item for item in previous['objects']
                                if int(item['object_id']) != int(object_id)]
        manifest['objects'].sort(key=lambda item: int(item['object_id']))
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output_dir / 'visibility.json').write_text(json.dumps(manifest['visibility'], indent=2))
    write_index(output_dir, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path, help='Sequence output directory')
    parser.add_argument('--object-id', type=int, help='Only regenerate this object overview')
    parser.add_argument('--columns', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--width', type=int, default=640)
    args = parser.parse_args()
    manifest = visualize(args.result, args.object_id, args.columns, args.width)
    print(f'{args.result}/objects/index.html: {len(manifest["objects"])} objects; '
          'two rows, one column per frame')


if __name__ == '__main__':
    main()

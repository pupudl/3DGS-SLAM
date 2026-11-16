#!/usr/bin/env python3
"""Index global submaps, optionally streaming a stitched background PLY.

Background centers/covariances already share the world frame. Moving owners
remain canonical geometry plus time-indexed poses; their different positions
must not be concatenated into a static background cloud. Exact boundary seed
copies are removed by provenance, rather than approximate geometric overlap.
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import argparse
import json

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def _path(root, relative):
    value = (root / relative).resolve()
    if not value.is_relative_to(root):
        raise ValueError('Submap artifact path leaves the result directory')
    return value


def _read(root):
    manifest = json.loads((root / 'gaussian_map/submaps/manifest.json').read_text())
    if manifest.get('format') != 'fixed_gaussian_submaps':
        raise ValueError('Expected a fixed Gaussian submap manifest')
    entries = sorted(manifest['submaps'], key=lambda row: int(row['submap_id']))
    return manifest, entries


def _blocks(root, entries, object_id=0):
    """Yield one object block, without retaining earlier geometry."""
    for entry in entries:
        path = _path(root, entry['checkpoint'])
        manifest = json.loads(path.read_text())
        for key, metadata in manifest.get('objects', {}).items():
            if object_id is not None and int(key) != int(object_id):
                continue
            filename = metadata['filename']
            if Path(filename).name != filename:
                raise ValueError('Invalid checkpoint object filename')
            with np.load(path.parent / filename, allow_pickle=False) as archive:
                yield entry, int(key), {name: archive[name] for name in archive.files}


def _superseded_seeds(root, entries):
    """Skip older copies only when a live successor exists in the next map.

    The persistent metadata is bounded by boundary anchors, not dense map size.
    A successor that was pruned must not erase a still valid earlier surface.
    """
    skipped = {}
    for entry in entries:
        seed_path = _path(root, entry['directory']) / 'boundary_seed.json'
        if not seed_path.is_file():
            continue
        seed = json.loads(seed_path.read_text())
        source = np.asarray(seed['source_map_ids'], np.int64)
        destination = np.asarray(seed['destination_map_ids'], np.int64)
        if source.shape != destination.shape or source.ndim != 1:
            raise ValueError('Boundary provenance requires parallel map-ID arrays')
        live = set()
        wanted = set(map(int, destination))
        checkpoint_path = _path(root, entry['checkpoint'])
        checkpoint = json.loads(checkpoint_path.read_text())
        for metadata in checkpoint.get('objects', {}).values():
            filename = metadata['filename']
            if Path(filename).name != filename:
                raise ValueError('Invalid checkpoint object filename')
            with np.load(checkpoint_path.parent / filename, allow_pickle=False) as archive:
                ids = archive['map_ids']
                live.update(int(mid) for mid in ids if int(mid) in wanted)
                del ids
        previous = skipped.setdefault(int(seed['source_submap_id']), set())
        previous.update(int(old) for old, new in zip(source, destination) if int(new) in live)
    return skipped


def export_submaps(result, background_ply=None):
    """Write a lightweight global index and an optional streaming binary PLY.

    No full-scene array is constructed. The global index remains the primary
    representation, preserving canonical dynamic objects and their trajectories.
    """
    root = Path(result).resolve()
    manifest, entries = _read(root)
    chunks, owners = [], {}
    for entry in entries:
        checkpoint = json.loads(_path(root, entry['checkpoint']).read_text())
        chunk = dict(submap_id=entry['submap_id'], checkpoint=entry['checkpoint'],
                     frame_ids=entry['frame_ids'], world_from_submap=entry['world_from_submap'],
                     map_id_namespace=entry['map_id_namespace'])
        seed_path = _path(root, entry['directory']) / 'boundary_seed.json'
        if seed_path.is_file():
            chunk['boundary_seed_provenance'] = seed_path.relative_to(root).as_posix()
        chunks.append(chunk)
        for key, row in checkpoint.get('objects', {}).items():
            if int(key) != 0 and row['gaussian_count']:
                owners.setdefault(key, []).append(int(entry['submap_id']))
    index = dict(format='global_gaussian_submap_scene_v1',
                 source_manifest='submaps/manifest.json',
                 coordinate_system=manifest['coordinate_system'], chunks=chunks,
                 background_object_id=0, dynamic_objects=[dict(object_id=int(key), submap_ids=value)
                   for key, value in sorted(owners.items(), key=lambda item: int(item[0]))],
                 dynamic_placement='canonical geometry + global object trajectory at requested frame',
                 trajectory='trajectory.jsonl', map_id='(submap_id, local_map_id)',
                 geometry_memory='one object checkpoint at a time',
                 seam_policy='shared camera/world frame; exact copied seeds deduplicated for PLY; no approximate surface fusion')
    if background_ply is not None:
        destination = Path(background_ply)
        destination.parent.mkdir(parents=True, exist_ok=True)
        skipped = _superseded_seeds(root, entries)
        count = 0
        for entry, _, block in _blocks(root, entries):
            count += int((~np.isin(block['map_ids'], list(skipped.get(int(entry['submap_id']), ())))).sum())
            del block
        dtype = np.dtype([(name, '<f4') for name in ('x', 'y', 'z')] +
                         [(name, 'u1') for name in ('red', 'green', 'blue')] +
                         [(name, '<f4') for name in ('opacity', 'scale_0', 'scale_1', 'scale_2',
                                                   'rot_0', 'rot_1', 'rot_2', 'rot_3')] +
                         [('submap_id', '<u4'), ('local_map_id', '<u4')])
        header = ['ply', 'format binary_little_endian 1.0',
                  'comment Background only; canonical moving objects are in global_scene.json',
                  f'element vertex {count}']
        for name in dtype.names:
            header.append(f'property {"uchar" if dtype[name].kind == "u" and dtype[name].itemsize == 1 else "uint" if dtype[name].kind == "u" else "float"} {name}')
        header.append('end_header\n')
        temporary = destination.with_name(destination.name + '.tmp')
        try:
            with temporary.open('wb') as stream:
                stream.write('\n'.join(header).encode('ascii'))
                for entry, _, block in _blocks(root, entries):
                    keep = ~np.isin(block['map_ids'], list(skipped.get(int(entry['submap_id']), ())))
                    ids = block['map_ids'][keep]
                    if len(ids) and ids.max() > np.iinfo(np.uint32).max:
                        raise ValueError('PLY local map IDs exceed uint32')
                    rows = np.empty(len(ids), dtype=dtype)
                    for i, name in enumerate(('x', 'y', 'z')):
                        rows[name] = block['xyz'][keep, i]
                    rgb = np.rint(np.clip(block['colors'][keep], 0, 1)*255).astype(np.uint8)
                    for i, name in enumerate(('red', 'green', 'blue')):
                        rows[name] = rgb[:, i]
                    rows['opacity'] = np.exp(-np.logaddexp(0., -block['opacity_logits'][keep, 0]))
                    for i in range(3):
                        rows[f'scale_{i}'] = np.exp(block['log_scales'][keep, i])
                    for i in range(4):
                        rows[f'rot_{i}'] = block['rotations'][keep, i]
                    rows['submap_id'] = int(entry['submap_id'])
                    rows['local_map_id'] = ids
                    stream.write(rows.tobytes())
                    del block, rows, ids, rgb
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        index['background_ply'] = dict(path=str(destination), gaussians=count,
                                       exact_duplicate_seeds_omitted=sum(map(len, skipped.values())))
    destination = root / 'gaussian_map/global_scene.json'
    temporary = destination.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(index, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(destination)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--background-ply', type=Path,
                        help='Optional binary PLY; excludes moving owners and exact boundary seed copies')
    args = parser.parse_args()
    print(json.dumps(export_submaps(args.result, args.background_ply), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

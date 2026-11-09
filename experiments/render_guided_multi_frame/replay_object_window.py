#!/usr/bin/env python3
"""Replay saved geometry through the Gaussian object window, without pair estimation.

Use --reuse-saved-objects to preserve Gaussian/object IDs, initial probabilities,
motions and measured costs, changing only the object solver in a new directory.
--render-optimize uses differentiable CUDA Gaussian rendering; explicitly pass
--no-render-optimize for the centre/track-only CPU baseline. Neither mode reruns
two-frame estimation or treats source final labels/posteriors as supervision.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in (HERE, HERE / 'render_guided_two_frame'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from gaussian_object_backend import SequenceObjectBackend
from hybrid_motion import _core_labels, motion_probabilities
from object_state import ObjectRegistry
from object_window import ObjectWindow
from pair_utils import lift, load_depth, resolve, stereo_io
from run_sequence import configure_object_window, load_config


def stereo_confidence(left, right, depth, K, baseline, sigma_rgb):
    """CPU equivalent of strict_two_frame.stereo_confidence, without renderer imports."""
    h, w = depth.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    finite = np.isfinite(depth) & (depth > 0)
    disparity = K[0, 0]*baseline/np.maximum(np.nan_to_num(depth, nan=0), 1e-5)
    right_x = xx-disparity.astype(np.float32)
    warped = cv2.remap(right, right_x, yy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT).astype(np.float32)/255
    difference = np.abs(left.astype(np.float32)/255-warped).mean(axis=2)
    confidence = np.exp(-difference/sigma_rgb)
    confidence[~finite | (right_x < 1) | (right_x >= w-2)] = 0
    return confidence.astype(np.float32)


def _anchor_confidence(anchors, source_confidence, target_confidence, cfg):
    """Reconstruct saved anchors' measured stereo/depth reliability.

    Saved matches already passed the original sparse depth-range filter; rows
    stay aligned with their saved track IDs and bound Gaussian indices.
    """
    if not cfg.get('ownership', {}).get('stereo_weighted_anchors', False):
        return np.ones(len(anchors['matches']), np.float32)
    uv = np.rint(anchors['matches']).astype(np.int32)
    h, w = source_confidence.shape
    uv[:, (0, 2)] = np.clip(uv[:, (0, 2)], 0, w-1)
    uv[:, (1, 3)] = np.clip(uv[:, (1, 3)], 0, h-1)
    c0 = source_confidence[uv[:, 1], uv[:, 0]]
    c1 = target_confidence[uv[:, 3], uv[:, 2]]
    z = np.maximum(anchors['x0'][:, 2], anchors['x1'][:, 2])
    depth_weight = 1/(1+(z/float(cfg['ownership'].get('reliable_depth_m', 35)))**2)
    confidence = np.clip(np.sqrt(c0*c1)*depth_weight, .05, 1)
    features = cfg.get('features', {})
    near = features.get('match_depth_near_reliable_m')
    far = features.get('match_depth_far_reliable_m')
    if near is not None or far is not None:
        if near is None or far is None:
            raise ValueError('Set both near and far reliable sparse-match depths')
        minimum = float(features.get('match_depth_min_m', cfg['depth'].get('min_m', 0)))
        maximum = float(features.get('match_depth_max_m', cfg['depth'].get('max_m', float('inf'))))
        near, far = float(near), float(far)
        edge = float(features.get('match_depth_edge_weight', .05))
        if not minimum < near <= far < maximum:
            raise ValueError('Require min < near_reliable <= far_reliable < max')
        z_near = np.minimum(anchors['x0'][:, 2], anchors['x1'][:, 2])
        near_factor = np.clip((z_near-minimum)/(near-minimum), 0, 1)
        far_factor = np.clip((maximum-z)/(maximum-far), 0, 1)
        confidence = np.clip(confidence*(edge+(1-edge)*near_factor*far_factor), .01, 1)
    return confidence.astype(np.float32)


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _lift_rows(matches, depth0, depth1, K):
    """Use the real lift helper and recover its row selection without relabeling tracks."""
    anchors = lift(matches, depth0, depth1, K)
    if len(anchors['matches']) == len(matches):
        return anchors, np.arange(len(matches))
    # lift preserves order. This supports repeated coordinates without a
    # dictionary silently collapsing distinct saved observations.
    selected = []
    cursor = 0
    for match in anchors['matches']:
        while cursor < len(matches) and not np.array_equal(matches[cursor], match):
            cursor += 1
        if cursor == len(matches):
            raise ValueError('Cannot align lifted anchors to saved track rows')
        selected.append(cursor)
        cursor += 1
    return anchors, np.asarray(selected, np.int64)


def run_replay(source, output, window_frames=5, visualize_results=True, *,
               render_optimize=None, reuse_saved_objects=False,
               no_height_constraint=False, no_connectivity_constraint=False):
    """Replay a saved sequence into a new directory; source files remain untouched."""
    source, output = resolve(source).resolve(), resolve(output).resolve()
    if source == output or source in output.parents:
        raise ValueError('Replay output must be outside the source sequence directory')
    if (output / 'sequence_state.json').exists():
        raise FileExistsError(f'Refusing to overwrite saved sequence: {output}')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Replay output must be a new or empty directory: {output}')
    if int(window_frames) < 2:
        raise ValueError('window_frames must count at least two images')
    source_state = json.loads((source / 'sequence_state.json').read_text())
    cfg = load_config(source / 'config.yaml')
    defaults = load_config(HERE / 'configs' / 'kitti360_short_sequence.yaml')['object_window']
    cfg['object_window'] = {**defaults, **cfg.get('object_window', {}),
                            'enabled': True, 'window_frames': int(window_frames)}
    for section, disabled in (('height', no_height_constraint),
                              ('connectivity', no_connectivity_constraint)):
        settings = {**defaults.get(section, {}), **cfg['object_window'].get(section, {})}
        if disabled:
            settings['enabled'] = False
        cfg['object_window'][section] = settings
    render_cfg = {**defaults.get('render', {}), **cfg['object_window'].get('render', {})}
    # Replay's default is the full render solver. The old CPU solver requires
    # an explicit selection, even when replaying an old pre-render config.
    render_cfg['enabled'] = render_optimize is not False
    cfg['object_window']['render'] = render_cfg
    render_enabled = bool(render_cfg.get('enabled', False))
    if render_enabled:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('object_window.render.enabled=true requires CUDA. '
                               'Use --no-render-optimize explicitly for the CPU centre/track baseline; '
                               '--no-visualize only disables output images.')
    cfg = configure_object_window(cfg)
    cfg.setdefault('sequence', {})['output_dir'] = str(output)
    cfg['output_dir'] = str(output / 'pairs')
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    K, baseline = stereo_io.read_calibration(data_root / 'calibration' / 'perspective.txt')
    pairs = source_state['pairs']
    for pair in pairs:
        source_pair = (source / pair['result_path']).resolve()
        if not source_pair.is_relative_to(source):
            raise ValueError('Saved result paths must remain within the source sequence')
        needed = ('gaussian_state.npz', 'pair_tracks.npz',
                  'object_state.npz' if reuse_saved_objects else 'pairwise_state.npz')
        for filename in needed:
            if not (source_pair / filename).is_file():
                raise FileNotFoundError(source_pair / filename)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    tracking_dir = source / source_state.get('tracking_dir', 'tracks')
    if tracking_dir.is_dir():
        shutil.copytree(tracking_dir, output / 'tracks')
    provenance = dict(
        mode=('saved_object_window_solver_replay' if reuse_saved_objects else 'saved_pair_geometry_replay'),
        source=str(source), render_optimize=render_enabled,
        local_motion_initialization=('object_state.npz:q_motion_initial (immutable)'
                                     if reuse_saved_objects else 'pairwise_state.npz:q_gaussian_sparse (immutable)'),
        motion_proposals=('object_state.npz:H_object (unchanged)'
                          if reuse_saved_objects else 'pairwise_state.npz:H_sparse (saved sparse initialization)'),
        source_pose_refinement_and_feedback_replayed=False,
        source_final_gaussian_geometry_used=True,
        cuda_pair_optimizer_rerun=False,
        source_object_labels_used_as_ground_truth=False,
        source_object_ids_reused=bool(reuse_saved_objects),
        source_gaussian_ids_reused=bool(reuse_saved_objects),
        source_final_posteriors_used_as_supervision=False,
        height_constraint_enabled=bool(cfg['object_window']['height'].get('enabled', False)),
        connectivity_constraint_enabled=bool(cfg['object_window']['connectivity'].get('enabled', False)),
        object_motion_fitting_rerun=not reuse_saved_objects,
        feature_height_model_refit_applied=False,
        feature_height_model_fitting_enabled=(not reuse_saved_objects and
                                             bool(cfg['object_window']['height'].get('enabled', False))),
        feature_height_model_fitting_note=(
            'Not applied: saved object motions are reused unchanged. Only the Gaussian '
            'object-attribute height/connectivity terms are evaluated in this same-ID replay.'
            if reuse_saved_objects else
            'Height gating is evaluated for nonbackground object motion support; invalid '
            'ground estimates skip it. Actual application is recorded per pair.'),
        raw_observations='original left/right images, depth, sky masks, calibration, tracks')
    (output / 'replay_provenance.json').write_text(json.dumps(provenance, indent=2))
    registry = ObjectRegistry(cfg)
    backend = None if reuse_saved_objects else SequenceObjectBackend(cfg)
    window = ObjectWindow(cfg) if reuse_saved_objects else backend.window
    track_snapshots = {}
    if reuse_saved_objects:
        # These are identity/history metadata only. No old posterior or label
        # is passed into the solver, and no association/model fitting is rerun.
        registry.objects = {int(key): deepcopy(value)
                            for key, value in source_state.get('objects', {}).items()}
        registry._next_object = max(registry.objects, default=-1)+1
    state = dict(frames=source_state['frames'], sequence=cfg['dataset']['sequence'],
                 dataset=cfg['dataset'], config=cfg, tracking_dir='tracks',
                 pairs=[], objects={}, replay=provenance)
    frame_cache = {}

    def observations(frame):
        frame = int(frame)
        if frame not in frame_cache:
            name = f'{frame:010d}.png'
            image = stereo_io.read_image(sequence_root / 'image_00' / 'data_rect' / name)
            right = stereo_io.read_image(sequence_root / 'image_01' / 'data_rect' / name)
            if image.shape != right.shape:
                raise ValueError('Stereo image dimensions differ')
            depth = load_depth(sequence_root, frame, image, right, K, baseline, cfg['depth'])
            sky = sequence_root / 'sky_masks' / name
            if sky.is_file():
                depth[stereo_io.read_mask(sky, depth.shape)] = np.nan
            confidence = stereo_confidence(image, right, depth, K, baseline,
                                            float(cfg['depth'].get('confidence_rgb_sigma', .15)))
            frame_cache[frame] = (image, depth, confidence)
        return frame_cache[frame]

    for pair in pairs:
        first, second = int(pair['first']), int(pair['second'])
        mode = 'CUDA render' if render_enabled else 'CPU centre/track'
        print(f'[{first}→{second}] {mode} object-window replay...', flush=True)
        source_pair = (source / pair['result_path']).resolve()
        destination = output / pair['result_path']
        destination.mkdir(parents=True, exist_ok=True)
        image0, depth0, confidence0 = observations(first)
        image1, depth1, confidence1 = observations(second)
        raw_observations = dict(source=image0, target=image1,
                                source_depth=depth0, target_depth=depth1,
                                source_confidence=confidence0, target_confidence=confidence1)
        if reuse_saved_objects:
            archive = _npz(source_pair / 'object_state.npz')
            input_ids = np.asarray(archive.get('evidence_object_ids', archive['object_ids']), np.int64)
            columns = {int(oid): i for i, oid in enumerate(archive['object_ids'])}
            selected = np.asarray([columns[int(oid)] for oid in input_ids], np.int64)
            payload = {name: archive[name] for name in ('xyz', 'uv', 'color', 'scale', 'K')}
            payload['H_object'] = archive['H_object'][selected]
            payload['object_model_valid'] = archive.get(
                'object_model_valid', np.isfinite(archive['H_object']).all(axis=(1, 2)))[selected]
            for name in ('basis', 'matches', 'anchor_track_ids', 'ground_plane_normal',
                         'ground_plane_offset', 'ground_plane_confidence', 'ground_plane_valid',
                         'ground_plane_report_json'):
                if name in archive:
                    payload[name] = archive[name]
            if not np.allclose(payload['K'], K):
                raise ValueError('Saved object intrinsics differ from raw dataset calibration')
            # Preserve legacy archives byte-for-byte. They remain diagnostics;
            # the new optimized attributes live separately in object_state.npz.
            for filename in ('gaussian_state.npz', 'pair_tracks.npz', 'pairwise_state.npz', 'summary.json'):
                if (source_pair / filename).is_file():
                    shutil.copy2(source_pair / filename, destination / filename)
            pair_cfg = deepcopy(cfg)
            pair_cfg['frames'] = dict(first=first, second=second)
            (destination / 'config.yaml').write_text(yaml.safe_dump(pair_cfg, sort_keys=False))
            render_arguments = {'render_observations': raw_observations} if render_enabled else {}
            posterior = window.update(
                first, second, archive['gaussian_ids'], input_ids,
                archive['q_object_initial'], archive['observation_costs'],
                weights=archive['observation_weights'], output_dir=destination,
                track_ids=archive.get('track_ids'), payload=payload,
                q_motion_initial=archive['q_motion_initial'], **render_arguments)
            report = dict(**posterior['report'], current_gaussians=len(archive['gaussian_ids']),
                          geometry_optimized=False, full_gaussian_raster_loss=render_enabled,
                          reused_saved_object_ids=True, reused_saved_gaussian_ids=True,
                          old_posteriors_used_as_supervision=False,
                          feature_height_gate=dict(
                              enabled=bool(cfg['object_window']['height'].get('enabled', False)),
                              applied=False, reason='saved_object_motions_reused_without_refitting'))
            row = deepcopy(pair)
            row['source_object_window'] = row.pop('object_window', None)
            row['object_window'] = report
            registry.pairs.append(dict(first=first, second=second,
                                       result_path=pair['result_path'], reference=pair.get('reference', 0),
                                       local_to_object=pair.get('local_to_object', {})))
            if 'track_ids' in archive:
                track_snapshots[(first, second)] = (archive['gaussian_ids'], archive['track_ids'])
                active = {tuple(p) for p in report['window_pairs']}
                track_snapshots = {p: snapshot for p, snapshot in track_snapshots.items() if p in active}
                associations = sorted({(int(track), int(gid))
                                       for ids, tracks in track_snapshots.values()
                                       for gid, track in zip(ids, tracks) if track >= 0})
                if associations:
                    unique = np.asarray(sorted({gid for _, gid in associations}), np.int64)
                    shared = window.get_state(unique)
                    lookup = {int(gid): i for i, gid in enumerate(unique)}
                    indices = np.asarray([lookup[gid] for _, gid in associations], np.int64)
                    registry.commit_window_tracks(
                        np.asarray([track for track, _ in associations], np.int64), shared['object_ids'],
                        shared['object_probabilities'][indices], shared['object_labels'][indices])
            (destination / 'temporal_state.json').write_text(json.dumps(row, indent=2))
            state['pairs'].append(row)
            state.update({key: value for key, value in registry.to_dict().items() if key != 'pairs'})
            state['object_window'] = window.to_dict()
            (output / 'sequence_state.json').write_text(json.dumps(state, indent=2))
            frame_cache = {second: frame_cache[second]}
            print(f'[{first}→{second}] {report["current_gaussians"]} Gaussians; '
                  f'{len(report["object_ids"])} preserved objects; solver updated', flush=True)
            continue
        saved = _npz(source_pair / 'gaussian_state.npz')
        saved_tracks = _npz(source_pair / 'pair_tracks.npz')
        # Read only the needed arrays from the large pairwise archive.
        with np.load(source_pair / 'pairwise_state.npz', allow_pickle=False) as initial:
            q = initial['q_gaussian_sparse'].copy()
            H = initial['H_sparse'].copy()
            bound = (initial['anchor_gaussian_indices_final'].copy()
                     if 'anchor_gaussian_indices_final' in initial else None)
        if q.shape != (len(saved['xyz']), len(H)):
            raise ValueError(f'{source_pair}: sparse q dimensions do not match saved Gaussian geometry '
                             'and sparse motion proposals')
        if not np.allclose(saved['K'], K):
            raise ValueError('Saved intrinsics differ from raw dataset calibration')
        anchors, rows = _lift_rows(saved_tracks['matches'], depth0, depth1, K)
        anchors['confidence'] = _anchor_confidence(anchors, confidence0, confidence1, cfg)
        track_ids = saved_tracks['track_ids'][rows]
        costs, labels = _core_labels(anchors, list(H), K, cfg['motion']['hybrid'])
        q_anchor, _ = motion_probabilities(costs, cfg['motion']['hybrid'])
        summary_path = source_pair / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        reference = int(summary.get('reference_sparse', 0))
        inherited = sum(int(track) in registry.track_objects for track in track_ids if track >= 0)
        mapping = registry.finish_pair(first, second, H, q_anchor, track_ids, reference,
                                       pair['result_path'], labels=labels)
        columns = np.asarray([mapping[str(k)] for k in range(len(H))], np.int64)
        gmap = {key: saved[key] for key in ('xyz', 'uv', 'color', 'scale')}
        if bound is not None:
            if len(bound) != len(saved_tracks['track_ids']):
                raise ValueError('Saved Gaussian bindings do not align with saved anchor rows')
            gmap['anchor_gaussian_indices'] = bound[rows]
        pair_state = dict(H=H, q=q, q_anchor=q_anchor, reference=reference)
        np.savez_compressed(destination / 'gaussian_state.npz', **{
            **gmap, 'q': q, 'H': H, 'K': K, 'local_to_object': columns})
        np.savez_compressed(destination / 'pair_tracks.npz', matches=anchors['matches'],
                            track_ids=track_ids, labels=labels, q_anchor=q_anchor,
                            local_to_object=columns)
        pair_cfg = deepcopy(cfg)
        pair_cfg['frames'] = dict(first=first, second=second)
        (destination / 'config.yaml').write_text(yaml.safe_dump(pair_cfg, sort_keys=False))
        report = backend.process(first, second, gmap, anchors, track_ids, mapping,
                                 pair_state, K, image0, image1, depth0, depth1,
                                 confidence0, confidence1, destination, registry)
        if report.get('feature_height_gate', {}).get('applied', False) and any(
                row.get('height_gated_refit', False) and row.get('valid', False)
                for row in report.get('models', [])):
            provenance['feature_height_model_refit_applied'] = True
        row = dict(first=first, second=second, result_path=pair['result_path'],
                   reference=reference, local_to_object=mapping,
                   source_local_to_object=pair.get('local_to_object'),
                   inherited_3d_tracks=inherited,
                   retained_3d_tracks=int(np.count_nonzero(track_ids >= 0)),
                   core_3d_tracks=int(np.count_nonzero(labels >= 0)),
                   initialization=dict(mode='saved_sparse_motion_probabilities',
                                       filtered_unliftable_anchors=len(saved_tracks['track_ids'])-len(rows)),
                   object_window=report)
        (destination / 'temporal_state.json').write_text(json.dumps(row, indent=2))
        state['pairs'].append(row)
        state.update({key: value for key, value in registry.to_dict().items() if key != 'pairs'})
        state['object_window'] = backend.window.to_dict()
        (output / 'sequence_state.json').write_text(json.dumps(state, indent=2))
        frame_cache = {second: frame_cache[second]}
        print(f'[{first}→{second}] {report["current_gaussians"]} Gaussians; '
              f'{report["links"]}; {len(report["object_ids"])} persistent objects', flush=True)
    if visualize_results:
        from visualize_objects import visualize
        visualize(output)
    (output / 'replay_provenance.json').write_text(json.dumps(provenance, indent=2))
    identity_note = (
        'Saved Gaussian IDs, object IDs, first initializers, object motions and measured costs '
        'are reused unchanged. Source final posteriors and hard labels are not supervision. '
        'This isolates changes in the object solver.\n\n' if reuse_saved_objects else
        'Object IDs are reassociated from the observations; source object IDs are diagnostics.\n\n')
    objective_note = (
        'The object solver jointly differentiates Gaussian RGB-D raster losses across the active '
        'window, together with the measured centre/track terms. Geometry and object motions '
        'remain fixed; only persistent object logits are optimized. CUDA is required.\n\n'
        if render_enabled else
        'This explicitly selected CPU baseline uses measured track and RGB-D centre errors. '
        'It does not optimize a Gaussian raster loss.\n\n')
    proposal_note = (
        'Saved object motions are reused; neither object association nor motion fitting is rerun. '
        if reuse_saved_objects else 'Saved sparse poses serve as proposals; source object association is rebuilt. ')
    spatial_note = (
        f"Height constraint: {provenance['height_constraint_enabled']}; "
        f"connectivity constraint: {provenance['connectivity_constraint_enabled']}. "
        + provenance['feature_height_model_fitting_note'] + '\n\n')
    (output / 'README.md').write_text(
        '# Gaussian object-window solver replay\n\n'
        f'Source: `{source}`. Window: {window_frames} image frames.\n\n'
        'This run preserves the source files and re-evaluates the shared object attributes '
        'against the original RGB-D, stereo confidence and measured tracks. '
        'Sparse local motion probabilities are immutable initialization. '
        + identity_note + objective_note + spatial_note +
        '**The CUDA two-frame estimator was not rerun.** ' + proposal_note +
        'this replay does not validate frozen local-motion optimization in a new CUDA run. '
        'Source CUDA pose refinement and feedback are not replayed. Source final Gaussian '
        'positions and final bound track observations are reused; therefore this is a backend '
        'replay, not an independent end-to-end reconstruction of the sparse initialization '
        'or a fused geometry optimization.\n\n'
        '[Object histories](objects/index.html) (when visualization was enabled). '
        'See `replay_provenance.json`, `sequence_state.json`, and each pair’s '
        '`object_state.npz` for the saved window evidence and results.\n', encoding='utf-8')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Saved sequence directory')
    parser.add_argument('--output', required=True, type=Path, help='New replay output directory')
    parser.add_argument('--window-frames', type=int, default=5, help='Window size in image frames')
    parser.add_argument('--no-visualize', action='store_true')
    render = parser.add_mutually_exclusive_group()
    render.add_argument('--render-optimize', dest='render_optimize', action='store_true',
                        help='Optimize window Gaussian RGB-D render losses on CUDA (default)')
    render.add_argument('--no-render-optimize', dest='render_optimize', action='store_false',
                        help='Explicitly select the CPU centre/track-only baseline')
    parser.set_defaults(render_optimize=None)
    parser.add_argument('--reuse-saved-objects', action='store_true',
                        help='Preserve saved Gaussian/object IDs, initializers, motions and raw costs; '
                             'change only the object solver')
    parser.add_argument('--no-height-constraint', action='store_true',
                        help='Disable height gating/regularization independently of connectivity')
    parser.add_argument('--no-connectivity-constraint', action='store_true',
                        help='Disable core connectivity independently of height')
    args = parser.parse_args()
    state = run_replay(args.source, args.output, args.window_frames, not args.no_visualize,
                       render_optimize=args.render_optimize, reuse_saved_objects=args.reuse_saved_objects,
                       no_height_constraint=args.no_height_constraint,
                       no_connectivity_constraint=args.no_connectivity_constraint)
    print(f'{args.output}: replayed {len(state["pairs"])} pairs', flush=True)


if __name__ == '__main__':
    main()

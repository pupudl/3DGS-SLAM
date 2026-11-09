#!/usr/bin/env python3
"""Compare ownership before and after the complete two-frame feedback loop.

Color tracks motion lineage across structural edits, so removing a component
does not make a surviving component look newly assigned merely due to renumbering.
No model is re-estimated by this visualization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pair_utils import resolve, stereo_io
from visualize_ownership_stages import (
    PALETTE_RGB, caption, changed_image, confidence_image, load_first_map,
    ownership_image, pixel_coordinates, save,
)


def motion_lineage(initial_count: int, cycles: list[dict]) -> tuple[list[int], list[dict]]:
    """Map each final column back to an initial or newly created component."""
    identities = list(range(initial_count))
    next_identity = initial_count
    accepted = []
    for cycle in cycles:
        for record in cycle['proposals']:
            if not record.get('accepted', False):
                continue
            kind = record['kind']
            event = {'cycle': int(cycle['cycle']), 'kind': kind}
            if kind == 'birth':
                identities.append(next_identity)
                event['created_identity'] = next_identity
                next_identity += 1
            elif kind == 'prune':
                index = int(record['component'])
                event['removed_identity'] = identities.pop(index)
            elif kind == 'split':
                index = int(record['component'])
                event['removed_identity'] = identities.pop(index)
                event['created_identities'] = [next_identity, next_identity + 1]
                identities.extend(event['created_identities'])
                next_identity += 2
            elif kind == 'merge':
                a, b = (int(v) for v in record['components'])
                event['merged_identities'] = [identities[a], identities[b]]
                identities = [v for j, v in enumerate(identities) if j not in (a, b)]
                identities.append(next_identity)
                event['created_identity'] = next_identity
                next_identity += 1
            elif kind == 'repair':
                # A replacement changes H_k, not the component's identity.
                event['repaired_identity'] = identities[int(record['component'])]
            else:
                raise ValueError(f'Unknown accepted structural edit: {kind}')
            accepted.append(event)
    if next_identity > len(PALETTE_RGB):
        raise ValueError('More component identities than available colors')
    return identities, accepted


def apply_anchor_state(gmap: dict, state: dict[str, np.ndarray], count: int) -> dict:
    result = dict(gmap)
    for key in ('xyz', 'uv', 'color', 'scale'):
        result[key] = gmap[key].copy()
    ids = state['indices'][:count].astype(np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(gmap['xyz'])):
        raise ValueError('Invalid saved anchor Gaussian indices')
    for key in ('xyz', 'uv', 'color', 'scale'):
        result[key][ids] = state[key][:count]
    return result


def legend_image(width: int, lineage: list[int], accepted: list[dict],
                 initial_count: int, before: np.ndarray, after: np.ndarray,
                 changed: np.ndarray) -> np.ndarray:
    height = 96 if len(lineage) <= 4 else 142
    board = np.full((height, width, 3), 255, np.uint8)
    n = len(after)
    operations = ', '.join(f'{e["kind"]} (round {e["cycle"] + 1})' for e in accepted) or 'none'
    cv2.putText(board, f'Hard ownership changed: {changed.sum():,}/{n:,} ({changed.mean():.1%}); '
                f'accepted structure: {operations}', (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, .58, (25, 25, 25), 2, cv2.LINE_AA)
    for final_index, identity in enumerate(lineage):
        row, column = divmod(final_index, 4)
        x, y = 17 + 524 * column, 59 + 42 * row
        color = tuple(int(round(v * 255)) for v in PALETTE_RGB[identity, ::-1])
        cv2.rectangle(board, (x, y - 16), (x + 25, y + 9), color, -1)
        origin = f'initial M{identity}' if identity < initial_count else 'new motion'
        count = int((after.argmax(axis=1) == final_index).sum())
        cv2.putText(board, f'Final M{final_index} <- {origin}; {count:,} Gaussians',
                    (x + 35, y + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (40, 40, 40), 1, cv2.LINE_AA)
    return board


def visualize(result_dir: Path, output_dir: Path | None = None) -> dict:
    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / 'feedback_final_visualization').resolve()
    cfg = yaml.safe_load((result_dir / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    cycles = json.loads((result_dir / 'cycles.json').read_text(encoding='utf-8'))
    with np.load(result_dir / 'pairwise_state.npz') as saved:
        q_joint = saved['q_gaussian_joint'].copy()
        q_final = saved['q_gaussian_final'].copy()
        anchor_state = ({key: saved[f'anchor_gaussian_{key}_final'].copy()
                         for key in ('indices', 'xyz', 'uv', 'color', 'scale')}
                        if 'anchor_gaussian_indices_final' in saved.files else None)
    gmap, first_image = load_first_map(cfg)
    n, initial_k = q_joint.shape
    lineage, accepted = motion_lineage(initial_k, cycles)
    if (n != len(gmap['xyz']) or len(q_final) != n or
            len(lineage) != q_final.shape[1] or
            len(lineage) != int(summary['final_models'])):
        raise ValueError('Saved states and accepted structural edits disagree')
    if not (np.allclose(q_joint.sum(axis=1), 1, atol=1e-3) and
            np.allclose(q_final.sum(axis=1), 1, atol=1e-3)):
        raise ValueError('Invalid Gaussian ownership probabilities')
    if anchor_state is not None:
        if len(anchor_state['indices']) != int(summary['final_3d_anchors']):
            raise ValueError('Final anchor count disagrees with summary')
        joint_map = apply_anchor_state(gmap, anchor_state,
                                       int(summary['initial_3d_anchors']))
        final_map = apply_anchor_state(gmap, anchor_state, len(anchor_state['indices']))
    else:
        joint_map = final_map = gmap
    joint_rows, joint_cols = pixel_coordinates(joint_map, first_image.shape[:2])
    final_rows, final_cols = pixel_coordinates(final_map, first_image.shape[:2])
    before_labels = q_joint.argmax(axis=1)
    after_indices = q_final.argmax(axis=1)
    after_identities = np.asarray(lineage, np.int32)[after_indices]
    changed = before_labels != after_identities
    output_dir.mkdir(parents=True, exist_ok=True)
    images = {
        'before_feedback_hard.png': ownership_image(first_image, joint_rows, joint_cols,
                                                      before_labels),
        'final_hard.png': ownership_image(first_image, final_rows, final_cols,
                                          after_identities),
        'changed_hard.png': changed_image(first_image, final_rows, final_cols, changed),
        'final_confidence.png': confidence_image(first_image, final_rows, final_cols,
                                                  q_final.max(axis=1)),
    }
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    second_frame = int(summary['frames'][1])
    observed = stereo_io.read_image(sequence_root / 'image_00/data_rect' /
                                    f'{second_frame:010d}.png')
    rendered = cv2.imread(str(result_dir / 'render_final.png'), cv2.IMREAD_COLOR)
    if rendered is None and summary.get('compact_output', False):
        with np.load(result_dir / 'score_maps.npz') as saved:
            rendered = cv2.cvtColor(saved['rendered_rgb_final'], cv2.COLOR_RGB2BGR)
    if rendered is None or rendered.shape != observed.shape:
        raise ValueError('Missing or incompatible saved final RGB rendering')
    for name, value in images.items():
        save(output_dir / name, value)
    def small(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA)
    top = np.hstack((
        caption(small(images['before_feedback_hard.png']), 'Before feedback',
                'After initial joint optimization; first-frame Gaussians'),
        caption(small(images['final_hard.png']), 'Final after all feedback',
                'Colors track motion identity, including any renumbering'),
        caption(small(images['changed_hard.png']), 'Changed hard ownership',
                f'Red = changed motion identity; {changed.sum():,}/{n:,}'),
    ))
    bottom = np.hstack((
        caption(small(images['final_confidence.png']), 'Final ownership confidence',
                'Yellow = high max probability, not ground-truth correctness'),
        caption(small(observed), 'Observed frame 2', 'Target RGB image'),
        caption(small(rendered), 'Final rendered frame 2',
                'Rendering after supplementation and structural decisions'),
    ))
    legend = legend_image(top.shape[1], lineage, accepted, initial_k,
                          q_joint, q_final, changed)
    save(output_dir / 'overview.png', np.vstack((top, bottom, legend)))
    report = {
        'frames': summary['frames'],
        'initial_joint_models': initial_k,
        'final_models': len(lineage),
        'final_index_to_identity': lineage,
        'accepted_structure': accepted,
        'gaussians': n,
        'hard_changed_gaussians': int(changed.sum()),
        'hard_changed_fraction': float(changed.mean()),
        'joint_hard_counts_by_initial_index': np.bincount(before_labels, minlength=initial_k).tolist(),
        'final_hard_counts_by_final_index': np.bincount(after_indices, minlength=len(lineage)).tolist(),
        'interpretation': ('Both ownership maps show first-frame Gaussian labels. '
                           'Colors are lineage-aligned across renumbering; final M indices '
                           'are given by final_index_to_identity. Red indicates an '
                           'argmax identity change, not necessarily corrected motion. '
                           'Observed and rendered RGB use the second frame.'),
    }
    (output_dir / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{output_dir}: {initial_k}->{len(lineage)} motions; '
          f'{changed.sum():,}/{n:,} Gaussian labels changed')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--result', type=Path, help='One saved pair/full_feedback directory')
    choice.add_argument('--group', type=Path, help='Four-pair output directory')
    parser.add_argument('--output', type=Path, help='Override output for one result')
    args = parser.parse_args()
    if args.group:
        if args.output:
            parser.error('--output is only supported with --result')
        results = sorted(p.parent for p in args.group.glob('*/full_feedback/pairwise_state.npz'))
        if not results:
            parser.error(f'No saved pair results found under {args.group}')
        for result in results:
            visualize(result)
    else:
        visualize(args.result, args.output)


if __name__ == '__main__':
    main()

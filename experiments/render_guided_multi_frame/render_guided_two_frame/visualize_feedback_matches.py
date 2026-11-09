#!/usr/bin/env python3
"""Visualize saved feedback masks, matching evidence and final retained matches.

Pixel-mask runs include per-round evidence; historical patch runs have fewer
saved fields. This script never re-runs feature extraction or motion estimation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from pair_utils import resolve, stereo_io


NEW_COLOR = (0, 0, 255)  # Red means retained new match, never a motion label.
OLD_COLOR = (155, 155, 155)
BOX_COLOR = (0, 255, 255)


def save(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot write {path}')


def caption(image: np.ndarray, title: str, detail: str = '') -> np.ndarray:
    bar = np.full((53, image.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .61, (22, 22, 22), 2, cv2.LINE_AA)
    if detail:
        cv2.putText(bar, detail, (10, 43), cv2.FONT_HERSHEY_SIMPLEX,
                    .45, (55, 55, 55), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def mark_points(image: np.ndarray, old: np.ndarray, new: np.ndarray,
                offset: int) -> np.ndarray:
    result = image.copy()
    for point in old:
        cv2.circle(result, tuple(np.rint(point[offset:offset + 2]).astype(int)),
                   2, OLD_COLOR, -1, cv2.LINE_AA)
    for point in new:
        center = tuple(np.rint(point[offset:offset + 2]).astype(int))
        cv2.circle(result, center, 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(result, center, 3, NEW_COLOR, -1, cv2.LINE_AA)
    return result


def selected_regions(image: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    result = image.copy()
    for x0, y0, x1, y1 in regions:
        cv2.rectangle(result, (x0, y0), (x1, y1), BOX_COLOR, 2, cv2.LINE_AA)
    return result


def search_window_example(first: np.ndarray, second: np.ndarray,
                          box: list[int], cfg: dict) -> np.ndarray:
    x0, y0, x1, y1 = box
    h, w = first.shape[:2]
    target_margin = int(cfg['discovery']['target_margin_px'])
    source_margin = int(cfg['discovery']['source_margin_px'])
    tx0, ty0 = max(0, x0 - target_margin), max(0, y0 - target_margin)
    tx1, ty1 = min(w, x1 + target_margin), min(h, y1 + target_margin)
    sx0, sy0 = max(0, tx0 - source_margin), max(0, ty0 - source_margin)
    sx1, sy1 = min(w, tx1 + source_margin), min(h, ty1 + source_margin)
    left, right = first.copy(), second.copy()
    cv2.rectangle(left, (sx0, sy0), (sx1, sy1), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.rectangle(right, (tx0, ty0), (tx1, ty1), (255, 0, 0), 3, cv2.LINE_AA)
    cv2.rectangle(right, (x0, y0), (x1, y1), BOX_COLOR, 2, cv2.LINE_AA)
    return np.hstack((left, right))


def correspondence_image(first: np.ndarray, second: np.ndarray,
                         matches: np.ndarray, max_lines: int = 180) -> tuple[np.ndarray, int]:
    h, w = first.shape[:2]
    base = np.hstack((first, second))
    tint = base.copy()
    count = min(len(matches), max_lines)
    indices = np.linspace(0, len(matches) - 1, count, dtype=int) if count else []
    for i in indices:
        x0, y0, x1, y1 = np.rint(matches[i]).astype(int)
        cv2.line(tint, (x0, y0), (x1 + w, y1), NEW_COLOR, 1, cv2.LINE_AA)
    result = cv2.addWeighted(base, .45, tint, .55, 0)
    for i in indices:
        x0, y0, x1, y1 = np.rint(matches[i]).astype(int)
        cv2.circle(result, (x0, y0), 3, NEW_COLOR, -1, cv2.LINE_AA)
        cv2.circle(result, (x1 + w, y1), 3, NEW_COLOR, -1, cv2.LINE_AA)
    return result, count


def mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[mask] = (.65*result[mask]+.35*np.array([0, 255, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
    return result


def visualize_mask_feedback(result_dir, output_dir, summary, cycles, initial, retained,
                            first, second):
    """Display the actual saved pixel masks and their evidence, without boxes."""
    masks, cycle_reports = [], []
    first_signals = None
    for cycle in cycles:
        index = int(cycle['cycle'])
        with np.load(result_dir / f'feedback_evidence_cycle_{index:02d}.npz') as data:
            mask = data['problem_mask'].astype(bool)
            novel = data['novel_matches']
            raw = data['raw_matches'] if 'raw_matches' in data else novel
            if index == 0 and 'residual_trigger' in data:
                first_signals = (data['residual_trigger'].copy(), data['gradient_trigger'].copy())
        masks.append(mask)
        uv = np.rint(raw[:, 2:]).astype(int)
        outside = int((~mask[uv[:, 1], uv[:, 0]]).sum()) if len(uv) else 0
        if outside:
            raise ValueError(f'{outside} raw matches lie outside cycle {index} mask')
        selected = mask_overlay(second, mask)
        save(output_dir / f'problem_mask_cycle_{index:02d}.png', mask.astype(np.uint8)*255)
        save(output_dir / f'mask_overlay_cycle_{index:02d}.png', selected)
        save(output_dir / f'raw_matches_cycle_{index:02d}.png',
             mark_points(selected, np.empty((0, 4)), raw, 2))
        save(output_dir / f'novel_matches_cycle_{index:02d}.png',
             mark_points(selected, np.empty((0, 4)), novel, 2))
        cycle_reports.append(dict(cycle=index, mask_pixels=int(mask.sum()),
                                  raw_matches=int(len(raw)), novel_3d_matches=int(len(novel)),
                                  matches_outside_mask=outside))
    union = np.logical_or.reduce(masks)
    uv = np.rint(retained[:, 2:]).astype(int)
    outside_union = int((~union[uv[:, 1], uv[:, 0]]).sum()) if len(uv) else 0
    if outside_union:
        raise ValueError('Retained new matches lie outside the union of saved masks')
    selected = mask_overlay(second, masks[0])
    target_matches = mark_points(mask_overlay(second, union), initial, retained, 2)
    source_matches = mark_points(first, initial, retained, 0)
    save(output_dir / 'first_cycle_selected.png', selected)
    save(output_dir / 'matches_first.png', source_matches)
    save(output_dir / 'matches_second.png', target_matches)
    save(output_dir / 'all_cycles_mask.png', union.astype(np.uint8)*255)
    signals = mask_overlay(second, union)
    signal_title = '2  Mask union across feedback rounds'
    signal_detail = 'Yellow = pixels allowed in at least one round'
    if first_signals is not None:
        residual, gradient = first_signals
        colors = second.copy()
        colors[residual] = (40, 40, 255)
        colors[gradient] = (255, 40, 190)
        colors[residual & gradient] = (0, 225, 255)
        signals = cv2.addWeighted(second, .5, colors, .5, 0)
        signal_title = '2  Pixel signals before mask cleanup'
        signal_detail = 'Red = residual; purple = ownership gradient; gold = both'
    save(output_dir / 'mask_signals.png', signals)
    def panel(image, title, detail):
        return caption(cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA), title, detail)
    score_mode = summary.get('feedback_mode') == 'score_mask_repair'
    global_mode = score_mode or summary.get('feedback_mode') == 'global_mask_repair'
    top = np.hstack((
        panel(selected, '1  Actual frame-2 problem mask',
              ('Full-image pixel selection; ' if global_mode else 'Historical patch-limited mask; ') +
              f'{masks[0].sum():,} selected pixels'),
        panel(signals, signal_title, signal_detail)))
    middle = np.hstack((
        panel(source_matches, '3  Corresponding points in frame 1',
              'Full-image matching; red = retained new matches' if global_mode else
              'Red = retained new matches'),
        panel(target_matches, '4  Retained matches and mask union in frame 2',
              f'{len(retained)} red matches; yellow = union of actual masks')))
    lines, drawn = correspondence_image(first, second, retained)
    save(output_dir / 'correspondences.png', lines)
    bottom = caption(cv2.resize(lines, (1408, 188), interpolation=cv2.INTER_AREA),
                     '5  Cross-frame correspondences',
                     f'{drawn} lines sampled from {len(retained)} retained new 3D matches')
    save(output_dir / 'overview.png', np.vstack((top, middle, bottom)))
    report = dict(frames=summary['frames'], selection_type=(
        'composite_score_mask' if score_mode else
        'global_pixel_mask' if global_mode else 'historical_patch_limited_mask'),
        initial_valid_3d_matches=int(len(initial)), retained_new_3d_matches_all_cycles=int(len(retained)),
        retained_matches_outside_all_masks=outside_union, cycles=cycle_reports,
        interpretation='Masks are the exact saved target pixel masks. Full-image pixel mode '
                       'uses no rectangular selection or crop. Red matches aggregate all rounds; '
                       'the final target panel shows their mask union. No ground truth is used.')
    (output_dir / 'visualization.json').write_text(json.dumps(report, indent=2))
    print(f'{output_dir}: {report["selection_type"]}; {len(retained)} retained new matches')
    return report


def visualize(result_dir: Path, output_dir: Path | None = None) -> dict:
    result_dir = result_dir.resolve()
    output_dir = (output_dir or result_dir / 'feedback_match_visualization').resolve()
    cfg = yaml.safe_load((result_dir / 'config.yaml').read_text(encoding='utf-8'))
    summary = json.loads((result_dir / 'summary.json').read_text(encoding='utf-8'))
    cycles = json.loads((result_dir / 'cycles.json').read_text(encoding='utf-8'))
    if not cycles:
        raise ValueError('Saved run contains no feedback cycle')
    n_initial = int(summary['initial_3d_anchors'])
    with np.load(result_dir / 'pairwise_state.npz') as saved:
        matches = saved['anchors_final'].copy()
    if len(matches) != int(summary['final_3d_anchors']) or n_initial > len(matches):
        raise ValueError('Saved match counts disagree with summary.json')
    initial, retained = matches[:n_initial], matches[n_initial:]
    data_root = resolve(cfg['dataset']['data_root'])
    sequence_root = data_root / 'data_2d_raw' / cfg['dataset']['sequence']
    first_frame, second_frame = (int(v) for v in summary['frames'])
    first = stereo_io.read_image(sequence_root / 'image_00/data_rect' /
                                 f'{first_frame:010d}.png')
    second = stereo_io.read_image(sequence_root / 'image_00/data_rect' /
                                  f'{second_frame:010d}.png')
    if first.shape != second.shape:
        raise ValueError('Frame dimensions differ')
    output_dir.mkdir(parents=True, exist_ok=True)
    if summary.get('feedback_mode') in ('global_mask_repair', 'masked_repair', 'score_mask_repair'):
        return visualize_mask_feedback(result_dir, output_dir, summary, cycles, initial, retained,
                                       first, second)
    for index, cycle in enumerate(cycles):
        save(output_dir / f'selected_regions_cycle_{index:02d}.png',
             selected_regions(second, cycle['regions']))
    first_new = mark_points(first, initial, retained, 0)
    second_new = mark_points(second, initial, retained, 2)
    save(output_dir / 'matches_first.png', first_new)
    save(output_dir / 'matches_second.png', second_new)
    selected = selected_regions(second, cycles[0]['regions'])
    save(output_dir / 'first_cycle_selected.png', selected)
    example = search_window_example(first, second, cycles[0]['regions'][0], cfg)
    save(output_dir / 'search_window_example.png', example)
    lines, drawn = correspondence_image(first, second, retained)
    save(output_dir / 'correspondences.png', lines)

    def small(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (704, 188), interpolation=cv2.INTER_AREA)

    # The example contains two full-width frames; preserve its aspect ratio.
    example_preview = np.full((188, 704, 3), 255, np.uint8)
    example_height = round(example.shape[0] * 704 / example.shape[1])
    example_preview[(188 - example_height) // 2:
                    (188 - example_height) // 2 + example_height] = cv2.resize(
                        example, (704, example_height), interpolation=cv2.INTER_AREA)

    top = np.hstack((
        caption(small(selected), '1  Choose regions in frame 2',
                'Yellow boxes = first feedback cycle search targets'),
        caption(small(second_new), '2  New matches in frame 2',
                'Red = retained new 3D match; gray = initial match'),
    ))
    middle = np.hstack((
        caption(small(first_new), '3  Their partners in frame 1',
                'Red points correspond to red points in frame 2'),
        caption(example_preview, '4  Search window example',
                'Left source search is wider than right target search'),
    ))
    # Keep the full two-frame width so the correspondence lines remain visible.
    line_panel = cv2.resize(lines, (1408, 188), interpolation=cv2.INTER_AREA)
    bottom = caption(line_panel, '5  Cross-frame correspondences',
                     f'{drawn} lines sampled from {len(retained)} retained new 3D matches')
    save(output_dir / 'overview.png', np.vstack((top, middle, bottom)))
    report = {
        'frames': summary['frames'],
        'initial_valid_3d_matches': n_initial,
        'retained_new_3d_matches_all_cycles': int(len(retained)),
        'first_cycle_selected_regions': len(cycles[0]['regions']),
        'first_cycle_raw_2d_matches_count': cycles[0]['supplemental_matches'],
        'first_cycle_novel_valid_3d_count': cycles[0]['novel_3d_anchors'],
        'correspondence_lines_drawn': drawn,
        'interpretation': ('Selected boxes show the first cycle on frame 2. Red points '
                           'are the exact new 3D matches retained in the final saved '
                           'state across all feedback cycles. Rejected, depth-invalid '
                           'and duplicate candidate coordinates were not saved, so '
                           'they cannot be plotted from this run. Red is not a motion label.'),
    }
    (output_dir / 'visualization.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'{output_dir}: {len(cycles[0]["regions"])} first-cycle regions; '
          f'{len(retained)} retained new 3D matches')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--result', type=Path, help='One saved pair/full_feedback directory')
    choice.add_argument('--group', type=Path, help='Four-pair output directory')
    parser.add_argument('--output', type=Path, help='Override output for a single result')
    args = parser.parse_args()
    if args.group:
        if args.output:
            parser.error('--output is only supported with --result')
        results = sorted(p.parent for p in args.group.glob('*/*/pairwise_state.npz'))
        if not results:
            parser.error(f'No saved pair results found under {args.group}')
        for result in results:
            visualize(result)
    else:
        visualize(args.result, args.output)


if __name__ == '__main__':
    main()

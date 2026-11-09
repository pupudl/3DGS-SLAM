#!/usr/bin/env python3
"""Audit a full frontend run with feature height gating and object refitting.

Reads saved artifacts on CPU. A passing report establishes execution and state
consistency, not semantic segmentation accuracy. Pose deltas are diagnostic:
rerunning a fit is allowed to return the same transform.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _archive(path, names):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in names}


def _pose_delta(estimate, reference):
    rotation = estimate[:3, :3] @ reference[:3, :3].T
    angle = np.arccos(np.clip((np.trace(rotation)-1)/2, -1., 1.))
    return dict(rotation_degrees=float(np.degrees(angle)),
                translation_m=float(np.linalg.norm(estimate[:3, 3]-reference[:3, 3])),
                matrix_max_absolute=float(np.max(np.abs(estimate-reference))))


def verify(result):
    result = Path(result).resolve()
    state = json.loads((result / 'sequence_state.json').read_text())
    frames = list(map(int, state['frames']))
    expected = list(zip(frames[:-1], frames[1:]))
    observed = [(int(row['first']), int(row['second'])) for row in state['pairs']]
    cfg = state.get('config', {})
    options = cfg.get('object_window', {})
    checks = dict(
        complete_ordered_sequence=len(frames) >= 2 and observed == expected,
        full_frontend_not_replay=not state.get('replay', False) and
            not (result / 'replay_provenance.json').exists(),
        height_and_connectivity_enabled=bool(options.get('height', {}).get('enabled')) and
            bool(options.get('connectivity', {}).get('enabled')),
        gaussian_rendering_enabled=bool(options.get('render', {}).get('enabled')),
        configured_motion_ownership_frozen=bool(cfg.get('joint', {}).get('freeze_motion_ownership')))
    records = []
    for row in state['pairs']:
        first, second = int(row['first']), int(row['second'])
        directory = Path(row['result_path'])
        directory = directory if directory.is_absolute() else result / directory
        temporal = json.loads((directory / 'temporal_state.json').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        report = temporal.get('object_window', {})
        gate = report.get('feature_height_gate', {})
        plane = gate.get('plane', {})
        models = report.get('models', [])
        render = report.get('render_optimization', {})
        spatial = render.get('spatial_constraints', {})
        motion = _archive(directory / 'pairwise_state.npz', (
            'H_final', 'q_gaussian_sparse', 'q_gaussian_joint', 'q_gaussian_final'))
        gaussian = _archive(directory / 'gaussian_state.npz', ('H', 'q', 'local_to_object'))
        saved = _archive(directory / 'object_state.npz', (
            'gaussian_ids', 'object_ids', 'H_object', 'object_model_valid',
            'q_motion_initial', 'object_probabilities', 'ground_plane_valid',
            'ground_plane_normal', 'ground_plane_offset', 'ground_plane_confidence'))
        nonbackground = [model for model in models if int(model['object_id']) != 0]
        supported = [model for model in nonbackground if model.get('valid')]
        background = [model for model in models if int(model['object_id']) == 0]
        q = motion['q_gaussian_sparse']
        probability = saved['object_probabilities']
        log_path = result / 'logs' / f'{first}_{second}_estimate.log'
        pair_checks = dict(
            frontend_artifacts_present=log_path.is_file() and log_path.stat().st_size > 0 and
                summary.get('initial_matches', 0) > 0 and summary.get('initial_3d_anchors', 0) > 0,
            no_saved_identity_replay=not report.get('reused_saved_object_ids', False) and
                not report.get('reused_saved_gaussian_ids', False),
            sequence_and_temporal_gate_agree=row.get('object_window', {}).get('feature_height_gate') == gate,
            measured_ground_and_feature_gate_active=bool(gate.get('enabled')) and
                bool(gate.get('applied')) and bool(plane.get('valid')),
            background_features_preserved=gate.get('background_filtered') is False and bool(background) and
                all(not model.get('height_gating_applied', False) and
                    model.get('height_excluded_anchors', 0) == 0 and
                    model.get('height_downweighted_anchors', 0) == 0 for model in background),
            nonbackground_candidates_height_gated=all(model.get('height_gating_applied') for model in nonbackground),
            supported_nonbackground_models_refitted=all(model.get('height_gated_refit') for model in supported),
            saved_gate_plane_matches_estimation=bool(plane.get('valid')) and
                np.asarray(plane.get('normal', [])).shape == (3,) and bool(saved['ground_plane_valid']) and
                np.allclose(saved['ground_plane_normal'], plane.get('normal', [])) and
                np.isclose(float(saved['ground_plane_offset']), float(plane.get('offset', np.nan))) and
                np.isclose(float(saved['ground_plane_confidence']), float(plane.get('confidence', np.nan))),
            original_motion_probabilities_frozen=all(np.array_equal(q, value) for value in (
                motion['q_gaussian_joint'], motion['q_gaussian_final'], gaussian['q'], saved['q_motion_initial'])),
            raw_pair_motion_preserved=np.array_equal(gaussian['H'], motion['H_final']),
            posterior_finite_normalized=probability.shape == (len(saved['gaussian_ids']), len(saved['object_ids'])) and
                bool(np.isfinite(probability).all()) and bool((probability >= 0).all()) and
                bool(np.allclose(probability.sum(axis=1), 1., atol=1e-5)),
            all_valid_object_motions_finite=bool(np.isfinite(saved['H_object'][saved['object_model_valid']]).all()),
            actual_object_logits_gradients=render.get('optimized_parameter') == 'object_logits' and
                render.get('motion_probabilities_optimized') is False and
                render.get('iterations_completed', 0) > 0 and render.get('nonzero_gradient_steps', 0) > 0 and
                render.get('render_passes', 0) > 0,
            spatial_losses_active=spatial.get('height', {}).get('enabled') is True and
                spatial.get('connectivity', {}).get('enabled') is True and
                render.get('weights', {}).get('height', 0) > 0 and
                render.get('weights', {}).get('connectivity', 0) > 0,
            render_losses_finite=bool(render.get('initial_losses')) and bool(render.get('final_losses')) and
                all(np.isfinite(value) for stage in ('initial_losses', 'final_losses')
                    for value in render.get(stage, {}).values()))
        pose_deltas = []
        columns = {int(oid): column for column, oid in enumerate(saved['object_ids'])}
        for model in models:
            if not model.get('valid'):
                continue
            oid = int(model['object_id'])
            local = np.flatnonzero(gaussian['local_to_object'] == oid)
            estimate = saved['H_object'][columns[oid]]
            candidates = [dict(local_model=int(index), **_pose_delta(estimate, gaussian['H'][index]))
                          for index in local]
            pose_deltas.append(dict(object_id=oid, source=model.get('source'),
                                    height_gated_refit=bool(model.get('height_gated_refit')),
                                    mapped_raw_motion_deltas=candidates))
        records.append(dict(first=first, second=second, checks={key: bool(value) for key, value in pair_checks.items()},
                            feature_height_gate=gate, models=models, pose_comparisons=pose_deltas,
                            valid_nonbackground_models=len(supported),
                            refitted_nonbackground_models=sum(bool(model.get('height_gated_refit')) for model in supported),
                            iterations_completed=render.get('iterations_completed', 0),
                            nonzero_gradient_steps=render.get('nonzero_gradient_steps', 0),
                            render_passes=render.get('render_passes', 0),
                            initial_losses=render.get('initial_losses', {}), final_losses=render.get('final_losses', {})))
    checks.update(
        every_pair_execution_and_state_checks=bool(records) and
            all(all(record['checks'].values()) for record in records),
        actual_feature_filtering_observed=sum(record['feature_height_gate'].get('excluded_anchors', 0) +
            record['feature_height_gate'].get('downweighted_anchors', 0) for record in records) > 0,
        actual_nonbackground_refitting_observed=sum(record['refitted_nonbackground_models'] for record in records) > 0,
        final_render_window_covers_requested_frames=bool(state['pairs']) and
            set(state['pairs'][-1].get('object_window', {}).get('render_optimization', {}).get('frames', [])) == set(frames))
    checks = {key: bool(value) for key, value in checks.items()}
    report = dict(result=str(result), frames=frames, checks=checks, passed=all(checks.values()), pairs=records,
                  refitted_nonbackground_models=sum(record['refitted_nonbackground_models'] for record in records),
                  excluded_anchor_observations=sum(record['feature_height_gate'].get('excluded_anchors', 0) for record in records),
                  downweighted_anchor_observations=sum(record['feature_height_gate'].get('downweighted_anchors', 0) for record in records),
                  ground_truth_used_for_this_audit=False, semantic_accuracy_evaluated=False,
                  notes=['Height screening and refitting apply to the object backend after fresh pairwise estimation.',
                         'Raw pair H remains archived; H_object is the new motion consumed by the object window.',
                         'Unchanged H is allowed after an actual refit. Invalid objects may have no supported motion.',
                         'Filtering counts sum anchor observations; they do not count unique physical features.'])
    (result / 'full_spatial_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True, type=Path)
    args = parser.parse_args()
    report = verify(args.result)
    print(json.dumps(dict(passed=report['passed'], checks=report['checks'],
                          refitted_nonbackground_models=report['refitted_nonbackground_models'],
                          excluded_anchor_observations=report['excluded_anchor_observations'],
                          downweighted_anchor_observations=report['downweighted_anchor_observations'],
                          report=str(args.result / 'full_spatial_verification.json')), indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

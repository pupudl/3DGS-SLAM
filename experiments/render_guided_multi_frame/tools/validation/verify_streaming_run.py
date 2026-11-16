#!/usr/bin/env python3
"""Audit a long object-window run without loading its whole Gaussian history."""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import argparse
import json
import sqlite3

import numpy as np


def verify(result):
    result = Path(result).resolve()
    state = json.loads((result / 'sequence_state.json').read_text())
    if not state.get('config', {}).get('sequence', {}).get('streaming', False):
        raise ValueError('This auditor requires a streaming sequence')
    frames = state['frames']
    size = int(state['config']['object_window']['window_frames'])
    step = int(state['config']['object_window'].get('window_step', 1))
    if not 1 <= step < size:
        raise ValueError('window_step must be positive and smaller than window_frames')
    final_start = ((max(0, len(frames)-size)+step-1)//step)*step
    pairs = state['pairs']
    checks = dict(complete=state.get('status') == 'complete',
                  pair_order=[(r['first'], r['second']) for r in pairs] == list(zip(frames[:-1], frames[1:])),
                  lifecycle_finalized=bool(state.get('lifecycle', {}).get('finalized')),
                  immutable_motion=True, finite_probabilities=True,
                  frozen_window_boundaries=True, bounded_parameter_cache=True,
                  shared_active_state=True, lifecycle_intervals=True)
    active = {tuple(pair) for pair in state['object_window']['window_pairs']}
    active_states = []
    observed_intervals = {}
    supported_objects = set()
    for index, pair in enumerate(pairs):
        directory = Path(pair['result_path'])
        if not directory.is_absolute():
            directory = result / directory
        with np.load(directory / 'object_state.npz') as archive:
            probability = archive['object_probabilities']
            ids, gids = archive['object_ids'], archive['gaussian_ids']
            labels = archive['object_labels']
            supported_objects.update(map(int, np.unique(labels[labels >= 0])))
            checks['finite_probabilities'] &= bool(
                probability.shape == (len(gids), len(ids)) and
                np.isfinite(probability).all() and (probability >= 0).all() and
                np.allclose(probability.sum(axis=1), 1., atol=1e-5) and
                np.isin(labels, np.r_[ids, -1]).all() and len(np.unique(gids)) == len(gids))
            with np.load(directory / 'gaussian_state.npz') as original:
                checks['immutable_motion'] &= np.array_equal(archive['q_motion_initial'], original['q'])
            # This pair's last legal rewrite is immediately before its source
            # frame leaves the lag, or the end of the sequence.
            start = min((index//step)*step, final_start)
            expected = frames[start:min(start+size, len(frames))]
            checks['frozen_window_boundaries'] &= np.array_equal(archive['window_frames'], expected)
            report = json.loads(str(archive['report_json'].item()))
            checks['bounded_parameter_cache'] &= bool(
                report['retained_gaussians'] == report['active_gaussians'] and
                report['retained_gaussians'] + report['forgotten_gaussians'] == report['total_gaussians'] and
                report['retained_pairs'] <= size - 1 and report['retained_frames'] <= size)
            if (pair['first'], pair['second']) in active:
                order = np.argsort(gids)
                current = (gids[order], ids.copy(), probability[order].copy(), labels[order].copy())
                for previous_active in active_states:
                    _, ia, ib = np.intersect1d(previous_active[0], current[0], return_indices=True)
                    checks['shared_active_state'] &= bool(
                        np.array_equal(previous_active[1], current[1]) and
                        np.allclose(previous_active[2][ia], current[2][ib], atol=1e-8) and
                        np.array_equal(previous_active[3][ia], current[3][ib]))
                active_states.append(current)
        temporal = json.loads((directory / 'temporal_state.json').read_text())
        life = temporal['object_window']['lifecycle']
        for endpoint in ('source', 'target'):
            row = life[endpoint]
            for oid in row['observed_object_ids']:
                interval = observed_intervals.setdefault(str(oid), [row['frame'], row['frame']])
                interval[0], interval[1] = min(interval[0], row['frame']), max(interval[1], row['frame'])
    # Empty final identities are excluded from primary objects but their past
    # lifecycle events must still form a complete, auditable journal.
    lifecycle_records = {**state.get('discarded_objects', {}), **state['objects']}
    for oid, record in lifecycle_records.items():
        if record['confirmed']:
            checks['lifecycle_intervals'] &= observed_intervals.get(oid) == [
                record['first_observed_frame'], record['last_observed_frame']]
        if record['status'] == 'retired':
            checks['lifecycle_intervals'] &= record['end_frame'] == record['last_observed_frame']
    event_count = 0
    born = set()
    checks['unique_birth_ids'] = True
    checks['complete_event_journal'] = True
    with (result / 'object_lifecycle.jsonl').open() as journal:
        for line in journal:
            event_count += 1
            row = json.loads(line)
            checks['complete_event_journal'] &= row['event_index'] == event_count
            if row['type'] == 'birth':
                checks['unique_birth_ids'] &= row['object_id'] not in born
                born.add(row['object_id'])
    checks['complete_event_journal'] &= event_count == state['lifecycle']['event_count']
    checks['unique_birth_ids'] &= born == {int(oid) for oid, r in lifecycle_records.items() if r['confirmed']}
    if state['config']['object_window'].get('prune_empty_objects', False):
        manifests = [result / 'gaussian_map/map/manifest.json',
                     *result.glob('gaussian_map/submaps/submap_*/map/manifest.json')]
        for path in manifests:
            if path.is_file():
                records = json.loads(path.read_text())['objects']
                supported_objects.update(int(oid) for oid, record in records.items() if record['gaussian_count'] > 0)
        checks['no_empty_objects'] = set(map(int, state['objects'])) == supported_objects
    database_uri = (result / 'tracks/track_index.sqlite3').as_uri() + '?mode=ro'
    with sqlite3.connect(database_uri, uri=True) as database:
        statistics = dict(database.execute('SELECT name, value FROM stats'))
        checks['track_archive'] = bool(
            statistics['observations'] == state['track_summary']['observations'] and
            statistics['tracks'] == state['track_summary']['tracks'] and
            statistics['frame_count'] == len(frames) and
            database.execute('SELECT COUNT(*) FROM frames WHERE sealed = 0').fetchone()[0] == 0 and
            (result / 'tracks/observations.npz').is_file())
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(passed=all(checks.values()), checks=checks, frames=len(frames), pairs=len(pairs),
                confirmed_objects=len(born), lifecycle_events=event_count,
                final_window=state['object_window'],
                scope='saved-state consistency; no segmentation accuracy or CUDA equivalence claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.result)
    path = args.result / 'streaming_verification.json'
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(passed=report['passed'], checks=report['checks']), indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

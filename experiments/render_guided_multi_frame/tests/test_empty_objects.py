"""Solved windows discard empty objects without turning uncertainty into labels."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ownership.object_window import ObjectWindow
from ownership.object_render_optimizer import build_render_views
from ownership.object_connectivity_constraints import build_connectivity_state
from tracking.object_state import ObjectRegistry
from run_sequence import prune_empty_object_records


class EmptyObjectTests(unittest.TestCase):
    def config(self):
        return dict(sequence=dict(streaming=True, frames=[0, 1, 2, 3, 4]),
                    object_lifecycle=dict(enabled=True), object_window=dict(
                        window_frames=4, window_step=2, prune_empty_objects=True,
                        temperature=.03, prior_weight=.01))

    def update(self, window, first, output=None, costs=None, oids=(0, 1, 2)):
        k = len(oids)
        depth = np.full((2, 2), 5.)
        observations = dict(source=np.zeros((2, 2, 3), np.uint8),
                            target=np.zeros((2, 2, 3), np.uint8), source_depth=depth,
                            target_depth=depth, source_confidence=np.ones((2, 2)),
                            target_confidence=np.ones((2, 2)))
        payload = dict(xyz=np.array([[0., 0., 5.], [1., 0., 5.]]),
                       uv=np.array([[0., 0.], [1., 0.]]), color=np.zeros((2, 3)),
                       scale=np.full((2, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], k, axis=0))
        return window.update(first, first+1, np.array([40, 41]), np.array(oids),
                             np.full((2, k), 1./k),
                             np.array([[.01, 2., 2.], [2., .01, 2.]]) if costs is None else costs,
                             output_dir=output, payload=payload, render_observations=observations)

    def test_waits_for_solved_window_preserves_observations_and_does_not_revive_empty_id(self):
        cfg = self.config()
        window = ObjectWindow(cfg)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for first in (0, 1):
                result = self.update(window, first, root / str(first))
                np.testing.assert_array_equal(result['object_ids'], [0, 1, 2])
                self.assertNotIn('empty_object_pruning', result['report'])
            result = self.update(window, 2, root / '2')
            np.testing.assert_array_equal(result['object_ids'], [0, 1])
            np.testing.assert_array_equal(result['object_labels'], [0, 1])
            self.assertEqual(result['report']['empty_object_pruning']['removed_object_ids'], [2])
            for first in (0, 1, 2):
                with np.load(root / str(first) / 'object_state.npz') as saved:
                    np.testing.assert_array_equal(saved['object_ids'], [0, 1])
                    np.testing.assert_array_equal(saved['evidence_object_ids'], [0, 1, 2])
                    self.assertEqual(saved['q_motion_initial'].shape, (2, 3))
                    self.assertEqual(saved['q_object_initial'].shape, (2, 3))
                    self.assertEqual(saved['H_object'].shape, (2, 4, 4))
                    np.testing.assert_allclose(saved['object_probabilities'].sum(1), 1.)
            frozen = (root / '0/object_state.npz').read_bytes()
            # Old immutable snapshots still contain O2, while the new interval
            # discovers O3. Neither optimizer planning nor boundary remapping
            # may reintroduce the discarded identity from those snapshots.
            self.update(window, 3, root / '3', oids=(0, 1, 3))
            self.assertEqual(window.object_ids, [0, 1])
            self.assertEqual((root / '0/object_state.npz').read_bytes(), frozen)
            graph = build_connectivity_state(window)
            np.testing.assert_array_equal(graph['object_ids'], [0, 1])

    def test_pruning_preserves_unknown_labels_even_when_background_probability_becomes_one(self):
        window = ObjectWindow(self.config())
        for first in (0, 1, 2):
            result = self.update(window, first, costs=np.full((2, 3), np.nan))
        np.testing.assert_array_equal(result['object_ids'], [0])
        np.testing.assert_array_equal(result['object_labels'], [-1, -1])
        np.testing.assert_array_equal(result['object_probabilities'], [[1.], [1.]])

    def test_render_solver_repeats_with_pruned_original_candidate_columns(self):
        cfg = self.config()
        cfg['object_window']['render'] = dict(enabled=True)
        window = ObjectWindow(cfg)
        planned = []

        def solve(window, options, *, priors):
            planned.append([view['object_ids'].tolist() for view in build_render_views(window)])
            probabilities = np.zeros((2, len(window.object_ids)))
            probabilities[0, 0] = probabilities[1, 1] = 1.
            return dict(probabilities=probabilities, render_support=probabilities > 0, report={})

        with patch('ownership.object_render_optimizer.optimize_render_window', side_effect=solve):
            for first in (0, 1, 2, 3):
                self.update(window, first, oids=(0, 1, 2) if first < 3 else (0, 1, 3))
        self.assertEqual(len(planned), 2)
        self.assertTrue(planned[-1])
        self.assertTrue(all(2 not in ids for ids in planned[-1]))
        self.assertEqual(window.object_ids, [0, 1])

    def test_registry_immediately_retires_empty_identity_and_drops_track_priors(self):
        registry = ObjectRegistry(self.config())
        registry.finish_pair(0, 1, [np.eye(4)]*2, np.eye(2), np.array([10, 11]),
                             0, 'pair', labels=np.array([0, 1]))
        registry.observe_frame(0, [0, 1])
        registry.observe_frame(1, [0, 1])
        registry.commit_window_tracks([11], [0, 1], [[.01, .99]], [1])
        registry.discard_empty_objects(1, [1])
        self.assertEqual(registry.candidate_object_ids(), [0])
        self.assertEqual(registry.objects[1]['retirement_reason'], 'no_gaussians')
        self.assertNotIn(11, registry.track_objects)
        self.assertFalse(any(1 in prior for prior in registry.track_probabilities.values()))
        registry.observe_frame(1, [0, 1])
        self.assertEqual(registry.objects[1]['status'], 'retired')
        self.assertGreater(registry._create_object(2), 1)

    def test_final_records_keep_historical_objects_with_gaussians_and_exclude_empty_ones(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pair, labels in [('early', [0, 1, -1]), ('late', [0, 0, -1])]:
                (root / pair).mkdir()
                np.savez_compressed(root / pair / 'object_state.npz', object_labels=np.array(labels))
            state = dict(pairs=[dict(result_path='early'), dict(result_path='late')],
                         objects={str(oid): dict(object_id=oid) for oid in (0, 1, 2)})
            prune_empty_object_records(root, state)
            self.assertEqual(set(state['objects']), {'0', '1'})
            self.assertEqual(set(state['discarded_objects']), {'2'})

    def test_final_records_keep_objects_with_persistent_map_gaussians(self):
        for relative in ('gaussian_map/map/manifest.json',
                         'gaussian_map/submaps/submap_000000/map/manifest.json'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = root / relative
                manifest.parent.mkdir(parents=True)
                manifest.write_text(json.dumps(dict(objects={
                    '1': dict(gaussian_count=5), '2': dict(gaussian_count=0)})))
                state = dict(pairs=[], objects={'1': {}, '2': {}})
                prune_empty_object_records(root, state)
                self.assertEqual(set(state['objects']), {'1'})
                self.assertEqual(set(state['discarded_objects']), {'2'})


if __name__ == '__main__':
    unittest.main()

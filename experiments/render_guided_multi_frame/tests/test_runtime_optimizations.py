"""Behavioral regression checks for equivalent CPU work and archive changes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copy import deepcopy
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial import cKDTree

from tracking.feature_tracks import FeatureTrackStore
from ownership.object_window import ObjectWindow
from frontend.masked_feedback import feedback_cycle
from run_sequence import SequencePairContext, snapshot_sources


def original_keypoints(existing, points, radius):
    """Pre-optimization sequential rule, including float32 norm rounding."""
    points = np.asarray(points, np.float32)
    ids = np.full(len(points), -1, np.int64)
    if len(existing) and len(points):
        distance, nearest = cKDTree(existing).query(points)
        close = distance <= radius
        ids[close] = nearest[close]
    additions = []
    for row in np.flatnonzero(ids < 0):
        if additions:
            distances = np.linalg.norm(np.asarray(additions)-points[row], axis=1)
            best = int(np.argmin(distances))
            if distances[best] <= radius:
                ids[row] = len(existing)+best
                continue
        ids[row] = len(existing)+len(additions)
        additions.append(points[row])
    return ids, np.concatenate((existing, np.asarray(additions, np.float32).reshape(-1, 2)))


class RuntimeOptimizationTests(unittest.TestCase):
    def test_source_snapshot_is_frozen_before_later_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output = Path(directory)/'source', Path(directory)/'output'
            (root/'frontend').mkdir(parents=True)
            output.mkdir()
            (root/'run_sequence.py').write_text('runner before')
            (root/'frontend/example.py').write_text('frontend before')
            (output/'config.yaml').write_text('seed: 7\n')
            snapshot_sources(output, root)
            (root/'frontend/example.py').write_text('frontend after')
            self.assertEqual((output/'source_snapshot/frontend/example.py').read_text(), 'frontend before')
            self.assertEqual((output/'source_snapshot/config.yaml').read_text(), 'seed: 7\n')

    def test_spatial_index_preserves_original_nearest_ties_and_float32_boundaries(self):
        rng = np.random.default_rng(71)
        cases = [
            (np.empty((0, 2), np.float32), rng.uniform(-50, 50, (4096, 2)), 1.5),
            (np.array([[0., 0.]], np.float32), [[1., 0.], [-1., 0.], [0., 0.]], 1.),
            (np.empty((0, 2), np.float32), [[-1., 0.], [1., 0.], [0., 0.]], 1.),
            (np.empty((0, 2), np.float32),
             [[np.nextafter(np.float32(1), np.float32(0)), 0.], [2., 0.]], 1.),
            (np.empty((0, 2), np.float32), [[0., 0.], [1e-23, 0.], [-0., 0.]], 0.),
        ]
        for existing, points, radius in cases:
            with self.subTest(radius=radius, size=len(points)):
                store = FeatureTrackStore({'tracking': {'join_radius_px': radius}})
                record = store._frame(1)
                record['uv'] = existing.copy()
                record['track_ids'] = np.full(len(existing), -1, np.int64)
                expected_ids, expected_points = original_keypoints(existing, points, radius)
                actual = store._keypoint_ids(1, points, append=True)
                np.testing.assert_array_equal(actual, expected_ids)
                np.testing.assert_array_equal(record['uv'], expected_points)

    def test_batch_priors_preserve_original_columns_after_pruning_and_append(self):
        for streaming in (False, True):
            window = ObjectWindow({'sequence': {'streaming': streaming},
                                   'object_window': {'window_frames': 4}})
            window.update(0, 1, np.array([8, 3]), np.array([9, 0]),
                          np.array([[.9, .1], [0., 0.]]), np.full((2, 2), np.nan))
            window.update(1, 2, np.array([3, 12]), np.array([0, 5, 9]),
                          np.array([[.2, .5, .3], [.1, .7, .2]]), np.full((2, 3), np.nan))
            if streaming:
                window.object_ids = [5, 0]  # Discard original ID 9 and reorder.
            gids = np.array([12, 8, 3])
            expected = np.stack([window._prior(int(gid)) for gid in gids])
            np.testing.assert_array_equal(window._priors(gids), expected)
            self.assertEqual(window._priors(np.empty(0, np.int64)).shape,
                             (0, len(window.object_ids)))

    def test_archive_levels_preserve_all_fields_and_failed_write_preserves_previous(self):
        with tempfile.TemporaryDirectory() as directory:
            archives = []
            for level in (0, 1, 6):
                output = Path(directory)/str(level)
                window = ObjectWindow({'archive_compression': level})
                window.update(0, 1, np.array([8, 3]), np.array([0, 9]),
                              np.array([[.9, .1], [.1, .9]]), np.array([[0., 2.], [2., 0.]]),
                              payload={'xyz': np.arange(6).reshape(2, 3)}, output_dir=output)
                path = output/'object_state.npz'
                with np.load(path, allow_pickle=False) as saved:
                    archives.append({name: saved[name] for name in saved.files})
                previous = path.read_bytes()
                with patch('numpy.lib.format.write_array', side_effect=OSError('disk failure')):
                    with self.assertRaisesRegex(OSError, 'disk failure'):
                        window._write_active()
                self.assertEqual(path.read_bytes(), previous)
                self.assertFalse(path.with_suffix('.tmp.npz').exists())
            for other in archives[1:]:
                self.assertEqual(set(archives[0]), set(other))
                for name in other:
                    np.testing.assert_array_equal(archives[0][name], other[name])

    def test_network_models_load_once_across_pair_contexts(self):
        tracks, objects = Mock(), Mock()
        objects.to_dict.return_value = {}
        shared = {}
        first = SequencePairContext(0, 1, tracks, objects, Path('/tmp'), feature_models=shared)
        second = SequencePairContext(1, 2, tracks, objects, Path('/tmp'), feature_models=shared)
        factory = Mock(return_value=(object(), object()))
        self.assertIs(first.feature_models(factory), second.feature_models(factory))
        factory.assert_called_once()

    def test_feedback_attach_removes_only_duplicate_solves(self):
        anchors = {'matches': np.zeros((1, 4)), 'x0': np.zeros((1, 3)), 'x1': np.zeros((1, 3))}
        active = {name: values+1 for name, values in anchors.items()}
        state = dict(H=np.eye(4)[None], q=np.array([[.6, .4]]), loss=1.)
        guard = dict(accepted=True, render_gain=.1, mask_gain=.1, outside_gain=0.)
        for frozen, expected_calls in ((True, 2), (False, 3)):
            calls = []
            def prepare(matches):
                return {}, dict(x0=np.zeros((len(matches), 3)), matches=matches)
            def optimize(gmap, evidence, models, q):
                calls.append((deepcopy(models), q.copy()))
                return dict(H=np.asarray(models), q=q.copy(), loss=.5)
            gate = Mock(side_effect=AssertionError('Attach must not use a structural gate'))
            with patch('frontend.masked_feedback.fit_proposals', return_value=[]), \
                 patch('frontend.masked_feedback.warm_ownership', return_value=state['q'].copy()), \
                 patch('frontend.masked_feedback.region_gate', return_value=guard), \
                 patch('frontend.masked_feedback.minimum_errors', return_value=np.array([0.])):
                result = feedback_cycle(state, anchors, {}, active, None,
                    dict(joint=dict(freeze_motion_ownership=frozen), discovery={}),
                    0, None, prepare, optimize, gate)
            self.assertEqual(len(calls), expected_calls)
            self.assertEqual(result[-1]['retained_3d_anchors'], 1)
            self.assertEqual(result[-1]['selected_update'], 'attach')
            np.testing.assert_array_equal(state['q'], [[.6, .4]])


if __name__ == '__main__':
    unittest.main()

"""A bound foreground motion must survive background growth and ID changes."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from ownership.gaussian_object_backend import SequenceObjectBackend
from test_object_state import anchors_for, pose


class BoundObjectModelsTests(unittest.TestCase):
    def config(self):
        return dict(sequence=dict(streaming=True), tracking={},
            object_window=dict(enabled=True, window_frames=7, window_step=5,
                               render=dict(enabled=False)),
            motion=dict(hybrid=dict(min_support=6, refit_max_nfev=35, depth_weight=.05)))

    def test_extra_background_inliers_do_not_refit_a_bound_mover_into_background(self):
        for background_count in (12, 120, 1200):
            with self.subTest(background_count=background_count):
                anchors, K, _, _ = anchors_for([0.]*background_count+[.08]*12, group_size=1)
                backend = SequenceObjectBackend(self.config())
                ids, models, valid, reports = backend._models(
                    dict(H=np.stack([pose(0), pose(.08)])), anchors,
                    np.arange(len(anchors['x0'])), {'0': 0, '1': 1},
                    SimpleNamespace(candidate_object_ids=lambda: [0, 1]), K,
                    height_gate=dict(allowed=np.ones(len(anchors['x0']), bool),
                                     weights=np.ones(len(anchors['x0']))))
                column = list(ids).index(1)
                self.assertTrue(valid[column])
                np.testing.assert_allclose(models[column], pose(.08), atol=1e-7)
                self.assertEqual(reports[column]['refit_bound_support'], 12)
                self.assertEqual(reports[column]['independent_support'], 12)

    def test_foreground_id_permutation_does_not_change_selected_motion(self):
        anchors, K, _, _ = anchors_for([0.]*6+[.08]*6+[.12]*12, group_size=1)
        anchors['x0'][:6, 2] = anchors['x1'][:6, 2] = 80.
        project = lambda x: x[:, :2]/x[:, 2:]*[K[0, 0], K[1, 1]]+K[:2, 2]
        anchors['matches'] = np.column_stack((project(anchors['x0']), project(anchors['x1'])))
        state = dict(H=np.stack([pose(0), pose(.08), pose(.11), pose(.12)]))
        results = []
        for first_id, second_id in ((1, 2), (2, 1)):
            ids, models, valid, reports = SequenceObjectBackend(self.config())._models(
                state, anchors, np.arange(len(anchors['x0'])),
                {'0': 0, '1': first_id, '2': first_id, '3': second_id},
                SimpleNamespace(candidate_object_ids=lambda: [0, 1, 2]), K)
            column = list(ids).index(first_id)
            self.assertTrue(valid[column])
            results.append(models[column])
            self.assertEqual(reports[column]['independent_support'], 6)
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_allclose(results[0], pose(.08), atol=1e-8)


if __name__ == '__main__':
    unittest.main()

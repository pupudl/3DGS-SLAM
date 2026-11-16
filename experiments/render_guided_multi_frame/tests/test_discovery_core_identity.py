"""Real regression: unrelated inliers diluted a valid cyclist core into O0."""
import sys
from pathlib import Path
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from run_sequence import load_config
from tracking.object_state import ObjectRegistry
from frontend.hybrid_motion import _core_labels, motion_costs, motion_probabilities


class DiscoveryCoreIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(HERE / 'configs/kitti360_sequence.yaml')
        with np.load(HERE / 'tests/fixtures/cyclist_discovery_core.npz') as saved:
            self.anchors = {k: saved[k] for k in ('x0', 'x1', 'matches', 'confidence')}
            self.models, self.final_models = saved['models'], saved['final_models']
            self.labels, self.K = saved['labels'], saved['K']
        self.tracks = np.arange(len(self.labels))
        self.registry = ObjectRegistry(self.cfg)
        self.registry._create_object(0)
        self.registry.observe_frame(0, [0])
        self.registry.track_objects = {int(t): 0 for t in self.tracks}

    def prepare(self):
        return self.registry.prepare_pair(1, 2, self.anchors, self.K, self.cfg['motion'],
                                           list(self.models), self.labels, self.tracks)

    def gate(self, models, anchors=None, tracks=None):
        anchors = self.anchors if anchors is None else anchors
        tracks = self.tracks if tracks is None else tracks
        costs, labels = _core_labels(anchors, models, self.K, self.cfg['motion']['hybrid'])
        q, _ = motion_probabilities(costs, self.cfg['motion']['hybrid'])
        return self.registry._motion_identity_gate(models, labels, q, tracks, anchors,
                                                   self.K, self.cfg['motion'])

    def test_no_warm_update_preserves_discovery_labels_and_background_rejection(self):
        models, labels, _, report = self.prepare()
        self.assertFalse(report['core_labels_recomputed'])
        np.testing.assert_array_equal(models, self.models)
        np.testing.assert_array_equal(labels, self.labels)
        self.assertEqual(int(np.sum(labels == 8)), 9)
        self.assertEqual(report['initial_local_to_object'][8], -1)
        self.assertTrue(any(row['local_model'] == 8 and row['object_id'] == 0
                            for row in report['motion_identity']['rejected']))

    def test_added_background_inliers_do_not_dilute_discovery_evidence(self):
        self.prepare()
        _, expanded = _core_labels(self.anchors, self.models, self.K, self.cfg['motion']['hybrid'])
        self.assertEqual(int(np.sum(expanded == 8)), 16)
        forbidden, report = self.gate(self.models)
        self.assertIn((8, 0), forbidden)
        self.assertEqual(report['core_evidence']['8']['reliable_unique_support'], 9)

    def test_final_pose_revalidates_same_core_after_row_reordering(self):
        self.prepare()
        order = np.arange(len(self.labels))[::-1]
        anchors = {key: value[order] for key, value in self.anchors.items()}
        forbidden, report = self.gate(self.final_models, anchors, self.tracks[order])
        self.assertIn((8, 0), forbidden)
        # One original match no longer fits the final pose and is excluded.
        self.assertEqual(report['core_evidence']['8']['reliable_unique_support'], 8)
        costs, labels = _core_labels(anchors, self.final_models, self.K, self.cfg['motion']['hybrid'])
        q, _ = motion_probabilities(costs, self.cfg['motion']['hybrid'])
        mapping = self.registry.finish_pair(1, 2, list(self.final_models), q, self.tracks[order],
            0, 'current', labels=labels, anchors=anchors, K=self.K, motion_cfg=self.cfg['motion'])
        self.assertNotEqual(mapping['8'], 0)

    def test_invalidated_core_does_not_force_background_split(self):
        self.prepare()
        models = self.models.copy()
        models[8] = models[0]
        forbidden, report = self.gate(models)
        self.assertNotIn((8, 0), forbidden)
        self.assertLess(report['core_evidence']['8']['reliable_unique_support'], 6)

    def test_repeated_track_is_not_independent_core_support(self):
        self.prepare()
        tracks = self.tracks.copy()
        tracks[self.labels == 8] = 10_000
        forbidden, report = self.gate(self.models, tracks=tracks)
        self.assertNotIn((8, 0), forbidden)
        self.assertEqual(report['core_evidence']['8']['reliable_unique_support'], 1)

    def test_removed_core_and_added_matches_use_current_observations(self):
        self.prepare()
        costs = motion_costs(self.anchors, self.models, self.K, self.cfg['motion']['hybrid'])
        removed = np.flatnonzero((self.labels == 8) & (costs[:, 0] >= 4))[0]
        keep = np.arange(len(self.labels)) != removed
        anchors = {key: value[keep].copy() for key, value in self.anchors.items()}
        tracks = self.tracks[keep]
        # Add new observations near the extra background inliers. They must
        # not become members of the original independent evidence set.
        _, expanded = _core_labels(self.anchors, self.models, self.K, self.cfg['motion']['hybrid'])
        extra = np.flatnonzero((expanded == 8) & (self.labels != 8))
        for key in anchors:
            additions = self.anchors[key][extra].copy()
            if key == 'matches':
                additions += .001
            anchors[key] = np.concatenate((anchors[key], additions))
        tracks = np.r_[tracks, np.arange(10_000, 10_000+len(extra))]
        forbidden, report = self.gate(self.models, anchors, tracks)
        self.assertIn((8, 0), forbidden)
        self.assertEqual(report['core_evidence']['8']['reliable_unique_support'], 8)


if __name__ == '__main__':
    unittest.main()

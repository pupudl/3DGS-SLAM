"""Small moving regions get a new discovery opportunity without target masks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import unittest

import numpy as np

from frontend.hybrid_motion import _core_labels, _discover_residual_motions
from test_hybrid_motion import K, application_config, scene
from test_object_state import anchors_for, pose
from tracking.object_state import ObjectRegistry
from frontend.hybrid_motion import motion_costs, motion_probabilities


class ResidualMotionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = application_config()['hybrid']
        self.cfg['residual_discovery'] = dict(local_seeds=16, max_new_components=2,
                                             extra_component_budget=1)

    def test_recovers_unproposed_small_motion_even_when_original_slots_are_full(self):
        anchors, truth, true_models = scene((200, 12), outliers=12)
        models = [true_models[0]]
        _, labels = _core_labels(anchors, models, K, self.cfg)
        result, final, sources, report = _discover_residual_motions(
            anchors, models, labels, ['background'], K, self.cfg, 7, 6, 1)
        self.assertEqual(len(result), 2)
        self.assertGreaterEqual(np.count_nonzero(final[truth == 1] == 1), 10)
        np.testing.assert_array_equal(result[0], models[0])
        np.testing.assert_array_equal(final[labels >= 0], labels[labels >= 0])
        self.assertEqual(report['extra_component_budget'], 1)

    def test_does_not_create_a_motion_in_a_static_scene(self):
        anchors, truth, models = scene((200,), outliers=20)
        models = models[:1]
        _, labels = _core_labels(anchors, models, K, self.cfg)
        result, final, _, report = _discover_residual_motions(
            anchors, models, labels, ['background'], K, self.cfg, 7, 6, 1)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(final, labels)
        self.assertEqual(report['added_models'], 0)

    def test_no_overflow_reports_capacity_instead_of_replacing_an_old_core(self):
        anchors, truth, models = scene((200, 12))
        models = models[:1]
        _, labels = _core_labels(anchors, models, K, self.cfg)
        self.cfg['residual_discovery']['extra_component_budget'] = 0
        result, final, _, report = _discover_residual_motions(
            anchors, models, labels, ['background'], K, self.cfg, 7, 6, 1)
        self.assertEqual(len(result), 1)
        self.assertEqual(report['skip_reason'], 'component_limit')
        np.testing.assert_array_equal(final, labels)



class IndependentBackgroundCoreTests(unittest.TestCase):
    def setUp(self):
        self.motion = dict(hybrid=dict(min_support=6, max_components=8,
                                       depth_weight=.05, core_margin=.1))
        self.registry = ObjectRegistry(dict(sequence=dict(streaming=True),
            object_window=dict(enabled=True), motion=self.motion,
            tracking=dict(motion_identity=dict(enabled=True))))
        self.tracks = np.arange(24)
        self.registry.finish_pair(0, 1, [pose()], np.ones((24, 1)), self.tracks,
                                  0, 'seed', labels=np.zeros(24, np.int32))
        self.registry.observe_frame(0, [0])
        self.registry.observe_frame(1, [0])
    def test_small_but_independent_motion_can_override_old_background(self):
        anchors, intrinsics, models, labels = anchors_for([.15, .15, .28])
        costs = motion_costs(anchors, models, intrinsics, self.motion['hybrid'])
        q, _ = motion_probabilities(costs, self.motion['hybrid'])
        old_cost = costs[16:, 0]
        self.assertTrue(np.all((old_cost > 1.) & (old_cost < 4.)))
        self.registry.identity_cfg['split_background_on_independent_core'] = True
        mapping = self.registry.finish_pair(1, 2, models, q, self.tracks, 0, 'current',
            labels=labels, anchors=anchors, K=intrinsics, motion_cfg=self.motion)
        self.assertEqual(mapping['0'], 0)
        self.assertEqual(mapping['1'], 0)
        self.assertNotEqual(mapping['2'], 0)
        rejection = self.registry.pairs[-1]['motion_identity']['rejected']
        self.assertTrue(any(row['reason'] == 'independent_current_core_not_explained_by_background'
                            for row in rejection))


if __name__ == '__main__':
    unittest.main()

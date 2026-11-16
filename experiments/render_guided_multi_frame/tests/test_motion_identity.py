"""Current metric motion must be able to overrule old background identity."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import unittest

import numpy as np

from test_object_state import ObjectRegistry, anchors_for, pose
from frontend.hybrid_motion import _core_labels, motion_costs, motion_probabilities


class MotionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.motion = dict(hybrid=dict(min_support=6, max_components=8,
                                       depth_weight=.05, core_margin=.1,
                                       refit_max_nfev=20))
        self.cfg = dict(sequence=dict(streaming=True),
                        object_window=dict(enabled=True), motion=self.motion,
                        tracking=dict(prior_weight=.25))
        self.registry = ObjectRegistry(self.cfg)
        # All 24 tracks used to belong to one stationary object (background).
        self.tracks = np.arange(24)
        self.registry.finish_pair(0, 1, [pose()], np.ones((24, 1)), self.tracks,
                                  0, "seed", labels=np.zeros(24, np.int32))
        self.registry.observe_frame(0, [0])
        self.registry.observe_frame(1, [0])

    def scene(self):
        anchors, K, models, labels = anchors_for([.15, .15, 1.5])
        # Two compatible background fragments and one independently moving
        # group carrying historical background tracks.
        costs = motion_costs(anchors, models, K, self.motion['hybrid'])
        q, _ = motion_probabilities(costs, self.motion['hybrid'])
        return anchors, K, models, labels, q

    def finish_scene(self, anchors, K, models, labels, q, **kwargs):
        return self.registry.finish_pair(1, 2, models, q, self.tracks, 0, "current",
            labels=labels, anchors=anchors, K=K, motion_cfg=self.motion, **kwargs)

    def test_new_motion_splits_from_historical_background_but_aliases_do_not(self):
        anchors, K, models, labels, q = self.scene()
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping, {'0': 0, '1': 0, '2': 1})
        self.assertEqual(self.registry.objects[1]['status'], 'provisional')
        self.assertEqual(self.registry.objects[1]['parent_id'], 0)
        rejection = next(e for e in self.registry.events
                         if e['type'] == 'identity_motion_rejected')
        self.assertEqual(rejection['local_model'], 2)
        self.assertGreater(rejection['inherited_cost_median'], 100.)
        self.assertLess(rejection['candidate_cost_median'], 1e-8)
        # The newest pair must still wait for the joint ownership posterior
        # before revising inherited hard track assignments.
        self.assertEqual(self.registry.track_objects[23], 0)

    def test_grey_soft_identity_cannot_bypass_the_motion_gate(self):
        anchors, K, models, labels, q = self.scene()
        labels[16:] = -1
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping['2'], 1)
        self.assertFalse(any(e['type'] == 'identity_from_soft_tracks' and
                             e['local_model'] == 2 for e in self.registry.events))

    def test_prepare_retains_candidate_without_warm_fit_or_old_prior_bias(self):
        anchors, K, models, labels = anchors_for([.15, 1.5, .15])
        # Collapse the duplicate background fragment for an exclusive core.
        models = models[:2]
        _, labels = _core_labels(anchors, models, K, self.motion['hybrid'])
        result, labels, q, report = self.registry.prepare_pair(
            1, 2, anchors, K, self.motion, models, labels, self.tracks)
        self.assertEqual(report['initial_local_to_object'], [0, -1])
        np.testing.assert_allclose(result[1], models[1])
        np.testing.assert_allclose(q[8:16, 1], 1., atol=1e-8)
        self.assertEqual(report['prior_mixed_tracks'], 16)
        self.assertEqual(report['warm_fits'][0]['motion_compatible_tracks'], 16)

    def test_current_object_can_change_speed_without_previous_pose_gate(self):
        # A current-pair rigid motion can differ arbitrarily from last pair.
        anchors, K, models, labels = anchors_for([1.5, 1.5, 1.5])
        labels[:] = 0
        models = models[:1]
        mapping = self.finish_scene(anchors, K, models, labels, np.ones((24, 1)))
        self.assertEqual(mapping, {'0': 0})

    def test_background_reference_with_fresh_tracks_can_reject_old_background(self):
        anchors, K, models, labels, q = self.scene()
        # Only the independent mover retains old background tracks; the
        # broad actual background now consists of freshly detected tracks.
        self.tracks[:16] += 100
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping['0'], 0)
        self.assertNotEqual(mapping['2'], 0)

    def test_insufficient_independent_support_does_not_force_split(self):
        anchors, K, models, labels, q = self.scene()
        labels[19:] = -1
        q[19:] = [1., 0., 0.]
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping['2'], 0)

    def test_repeated_observations_of_one_track_are_not_independent_birth_support(self):
        anchors, K, models, labels, q = self.scene()
        self.tracks[16:] = 16
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping['2'], 0)

    def test_explicit_disable_and_old_api_preserve_historical_behavior(self):
        anchors, K, models, labels, q = self.scene()
        self.registry.identity_cfg['enabled'] = False
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(set(mapping.values()), {0})
        self.registry.identity_cfg['enabled'] = True
        mapping = self.registry.finish_pair(2, 3, models, q, self.tracks, 0,
                                             'legacy', labels=labels)
        self.assertEqual(set(mapping.values()), {0})

    def test_one_residual_vehicle_track_cannot_steal_rejected_background_motion(self):
        anchors, K, models, labels, q = self.scene()
        old_vehicle = self.registry._create_object(0)
        self.registry.track_objects[23] = old_vehicle
        self.registry.track_probabilities[23] = {old_vehicle: 1.}
        mapping = self.finish_scene(anchors, K, models, labels, q)
        self.assertEqual(mapping['2'], 2)
        self.assertEqual(self.registry.objects[2]['parent_id'], 0)
        rejection = next(e for e in self.registry.pairs[-1]['motion_identity']['rejected']
                         if e['object_id'] == old_vehicle and e['local_model'] == 2)
        self.assertEqual(rejection['reason'], 'insufficient_compatible_identity_support')


if __name__ == '__main__':
    unittest.main()

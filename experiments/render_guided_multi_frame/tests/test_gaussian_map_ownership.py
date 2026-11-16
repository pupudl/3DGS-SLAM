"""Measured ownership corrections preserve a persistent map's geometry and IDs."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
from mapping.gaussian_map_backend import GaussianMapBackend, transform_points
from mapping.gaussian_map_ownership import MapOwnershipLinks
from mapping.persistent_gaussian_map import PersistentGaussianMap


def pose(translation=(0., 0., 0.), angles=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', angles).as_matrix()
    result[:3, 3] = translation
    return result


def covariance(batch):
    rotation = Rotation.from_quat(batch['rotations'][:, [1, 2, 3, 0]]).as_matrix()
    scaled = rotation * np.exp(batch['log_scales'])[:, None, :]
    return scaled @ scaled.transpose(0, 2, 1)


class MapOwnershipLinksTests(unittest.TestCase):
    def setUp(self):
        self.store = PersistentGaussianMap(dict(association_distance=.02, color_distance=.2))
        self.ids = self.store.integrate(0, np.asarray([[1., .3, 8.]]),
            np.asarray([[.7, .2, .3]]), scales=np.asarray([[.2, .07, .03]]),
            rotations=Rotation.from_euler('xyz', [[.3, -.2, .1]]).as_quat()[:, [3, 0, 1, 2]],
            frame_id=0)['map_ids']
        self.links = MapOwnershipLinks(dict(ownership_correction=dict(
            enabled=True, confirmation_updates=2, max_observation_links=3,
            max_descendants_per_observation=4)))
        self.links.remember(np.asarray([100]), self.ids, 0)
        self.camera = pose((.2, -.3, .4), (.1, -.2, .05))
        self.poses = {0: np.eye(4), 1: pose((1.4, -.2, 4.), (.2, .4, -.3))}
        self.registry = SimpleNamespace(objects={0: dict(status='active'), 1: dict(status='active')})

    def correct(self, update, labels=(1,), probabilities=((.02, .98),),
                observation_ids=(100,), poses=None, views=None, window=None):
        observations = np.asarray(observation_ids, np.int64)
        hints = self.links.hints(observations, np.full(len(labels), -1), {}, self.store)
        return self.links.correct(self.store, 10+update, update, observations,
            dict(object_labels=np.asarray(labels), object_probabilities=np.asarray(probabilities)),
            hints, self.poses if poses is None else poses, self.camera,
            {} if views is None else views, self.registry, window)

    def test_two_confirmations_transfer_stable_id_preserving_world_centre_covariance(self):
        before = self.store.select(self.ids)
        first = self.correct(1)
        self.assertEqual((first['transferred'], first['pending']), (0, 1))
        np.testing.assert_array_equal(self.store.ids_for_object(0), self.ids)
        final = self.correct(2)
        self.assertEqual((final['transferred'], final['pending']), (1, 0))
        after = self.store.select(self.ids)
        self.assertEqual(len(self.store), 1)
        self.assertEqual(self.store.next_id, 1)
        np.testing.assert_array_equal(after['object_ids'], [1])
        np.testing.assert_allclose(transform_points(after['xyz'], self.poses[1]), before['xyz'], atol=1e-6)
        R = self.poses[1][:3, :3]
        np.testing.assert_allclose(R @ covariance(after) @ R.T, covariance(before), atol=1e-8)
        self.assertEqual(self.store.ownership_events[-1]['map_ids'], self.ids.tolist())
        self.assertEqual(self.correct(3)['transferred'], 0)

    def test_confirmation_wait_uses_first_reliable_frame_then_follows_new_owner_motion(self):
        before = self.store.select(self.ids)
        reference_poses = {oid: T.copy() for oid, T in self.poses.items()}
        historical = {11: dict(object_to_camera={oid: np.linalg.inv(self.camera) @ T
                                               for oid, T in reference_poses.items()})}
        self.assertEqual(self.correct(1)['pending'], 1)
        # Both the object and camera move before the second confirmation. The
        # canonical change must preserve frame 11 geometry, then carry that
        # geometry along the object's measured 11->12 motion.
        motion = pose((1., -.3, .2), (.03, -.06, .17))
        current_poses = {0: np.eye(4), 1: motion @ reference_poses[1]}
        self.camera = pose((-.4, .3, .1), (-.2, .1, -.15))
        result = self.correct(2, poses=current_poses, views=historical)
        self.assertEqual(result['transferred'], 1)
        self.assertEqual(result['events'][0]['reference_frame'], 11)
        after = self.store.select(self.ids)
        np.testing.assert_allclose(transform_points(after['xyz'], reference_poses[1]),
                                   before['xyz'], atol=1e-6)
        np.testing.assert_allclose(transform_points(after['xyz'], current_poses[1]),
                                   transform_points(before['xyz'], motion), atol=1e-6)
        rotation = current_poses[1][:3, :3]
        np.testing.assert_allclose(rotation @ covariance(after) @ rotation.T,
            motion[:3, :3] @ covariance(before) @ motion[:3, :3].T, atol=1e-8)

    def test_conflicting_observations_for_one_map_point_cancel_pending(self):
        self.correct(1)
        self.links.remember(np.asarray([101]), self.ids, 1)
        report = self.correct(2, labels=(0, 1),
            probabilities=((.98, .02), (.02, .98)), observation_ids=(100, 101))
        self.assertEqual((report['transferred'], report['conflicts'], report['pending']), (0, 1, 0))
        np.testing.assert_array_equal(self.store.ids_for_object(0), self.ids)

    def test_missing_same_instant_owner_poses_never_combine_different_times(self):
        self.correct(1, poses={0: self.poses[0]})
        report = self.correct(2, poses={0: self.poses[0]}, views={
            7: dict(object_to_camera={1: np.linalg.inv(self.camera) @ self.poses[1]})})
        self.assertEqual(report['transferred'], 0)
        self.assertEqual(report['skipped_without_pose'], 1)
        np.testing.assert_array_equal(self.store.ids_for_object(0), self.ids)

    def test_shared_historical_pose_pair_can_support_correction(self):
        self.correct(1)
        historical = {7: dict(object_to_camera={oid: np.linalg.inv(self.camera) @ T
                                              for oid, T in self.poses.items()})}
        result = self.correct(2, poses={0: self.poses[0]}, views=historical)
        self.assertEqual(result['transferred'], 1)
        self.assertEqual(result['events'][0]['reference_frame'], 7)

    def test_low_probability_unknown_or_unconfirmed_owner_does_not_transfer(self):
        for labels, probabilities in [((1,), ((.2, .8),)), ((-1,), ((.02, .98),)),
                                      ((1,), ((.49, .51),))]:
            self.assertEqual(self.correct(1, labels, probabilities)['pending'], 0)
            self.assertEqual(self.correct(2, labels, probabilities)['transferred'], 0)
        for state in ('provisional', 'retired', 'archived'):
            self.registry.objects[1]['status'] = state
            self.assertEqual(self.correct(3)['transferred'], 0)
            self.assertEqual(self.correct(4)['pending'], 0)

    def test_gap_or_contradiction_requires_fresh_consecutive_confirmation(self):
        self.correct(1)
        self.assertEqual(self.correct(3)['transferred'], 0)
        self.assertEqual(self.correct(4, labels=(0,), probabilities=((.98, .02),))['pending'], 0)
        self.assertEqual(self.correct(5)['transferred'], 0)
        self.assertEqual(self.correct(6)['transferred'], 1)

    def test_links_pending_and_descendants_are_bounded_and_window_pruned(self):
        self.correct(1)
        for observation in range(101, 111):
            self.links.remember(np.asarray([observation]), self.ids, observation)
        self.assertEqual(len(self.links.links), 3)
        self.links.retain(self.store, first_frame=1000)
        self.assertEqual(self.links.links, {})
        self.assertEqual(self.links.pending, {})
        self.links.remember(np.asarray([200, 201]), np.repeat(self.ids, 2), 2)
        window = SimpleNamespace(_initial={201: None})
        self.links.retain(self.store, first_frame=0, window=window)
        self.assertEqual(set(self.links.links), {201})
        children = self.store.append_from_parent(np.repeat(self.ids, 10), reason='clone')
        self.links.topology(np.repeat(self.ids, 10), children, np.asarray([], np.int64))
        self.assertLessEqual(len(self.links.links[201]), 4)

    def test_clone_and_split_descendants_inherit_links_and_transfer_together(self):
        clones = self.store.append_from_parent(self.ids, reason='clone')
        self.links.topology(self.ids, clones, np.asarray([], np.int64))
        self.assertEqual(set(self.links.links[100]), {int(self.ids[0]), int(clones[0])})
        split = self.store.append_from_parent(np.repeat(clones, 2), reason='split')
        self.store.remove(clones, reason='split_parent')
        self.links.topology(np.repeat(clones, 2), split, clones)
        self.links.remember(np.asarray([100]), self.ids, 2)
        self.assertEqual(set(self.links.links[100]), {int(self.ids[0]), *map(int, split)})
        self.correct(1)
        self.assertEqual(self.correct(2)['transferred'], 3)
        np.testing.assert_array_equal(self.store.ids_for_object(1), np.r_[self.ids, split])

    def test_track_hints_support_confirmation_without_observation_ids(self):
        self.links.links.clear()
        self.links.last_frames.clear()
        tracks = np.asarray([40])
        hints = self.links.hints(None, tracks, {40: (0, int(self.ids[0]))}, self.store)
        posterior = dict(object_labels=np.asarray([1]), object_probabilities=np.asarray([[.02, .98]]))
        first = self.links.correct(self.store, 10, 1, None, posterior, hints,
            self.poses, self.camera, {}, self.registry)
        self.assertEqual(first['pending'], 1)
        self.links.retain(self.store, 100, extra_map_ids=hints)
        second = self.links.correct(self.store, 11, 2, None, posterior, hints,
            self.poses, self.camera, {}, self.registry)
        self.assertEqual(second['transferred'], 1)


class MapOwnershipBackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cfg = dict(gaussian_map=dict(enabled=True, window_frames=3,
            backend='torch_reference', iterations=0, downsample=1,
            observation_stride=1, initial_scale_multiplier=1.,
            max_active_gaussians=100, max_samples_per_frame=100,
            max_new_gaussians_per_frame=100, checkpoint_interval=10,
            association_distance=.02, color_distance=.2,
            ownership_correction=dict(enabled=True, confirmation_updates=2)))
        self.backend = GaussianMapBackend(self.cfg, Path(temporary.name))
        self.K = np.asarray([[20., 0., 16.], [0., 20., 12.], [0., 0., 1.]])
        self.xyz = np.asarray([[-2., -1., 8.], [-1., -.5, 8.], [-2., 1., 8.],
                               [1., -.2, 7.], [1.5, .3, 7.], [1.8, -.3, 7.]])
        self.tracks = np.arange(6)
        self.observations = dict(source=np.zeros((25, 33, 3), np.uint8),
            target=np.zeros((25, 33, 3), np.uint8), source_depth=np.full((25, 33), 8.),
            target_depth=np.full((25, 33), 8.), source_confidence=np.ones((25, 33)),
            target_confidence=np.ones((25, 33)))

    def process(self, frame, corrected, observation_ids=True, quality=1., probability=1.):
        labels = np.asarray([0, 0, 0, 1, 1, 1] if corrected else [0]*6)
        object_ids = np.asarray([0, 1] if corrected else [0])
        probabilities = np.asarray([[(probability if oid == label else 1-probability)
            if len(object_ids) > 1 else 1. for oid in object_ids] for label in labels])
        gmap = dict(xyz=self.xyz.copy(), uv=(self.xyz @ self.K.T)[:, :2]/self.xyz[:, 2:],
                    color=np.tile([.8, .2, .3], (6, 1)), scale=np.tile([.1, .06, .02], (6, 1)))
        raw = dict(self.observations)
        raw['source_confidence'] = np.full((25, 33), quality)
        registry = SimpleNamespace(objects={oid: dict(status='active', last_observed_frame=frame)
                                            for oid in object_ids})
        return self.backend.process(frame, frame+1, gmap,
            dict(object_labels=labels, object_probabilities=probabilities),
            object_ids, np.repeat(np.eye(4)[None], len(object_ids), axis=0),
            np.ones(len(object_ids), bool), self.K, raw, registry, self.tracks,
            presence=dict(source=dict(observed_object_ids=object_ids.tolist()),
                          target=dict(observed_object_ids=object_ids.tolist())),
            observation_ids=np.arange(100, 106) if observation_ids else None)

    def test_shared_observation_ids_wait_then_transfer_without_duplicate_insertion(self):
        self.process(0, False)
        originals = self.backend.store.ids_for_object(0).copy()
        first = self.process(1, True)
        self.assertEqual(first['ownership_correction']['pending'], 3)
        self.assertEqual(first['ownership_deferred_observations'], 3)
        self.assertEqual(first['inserted_gaussians'], 0)
        self.assertEqual(len(self.backend.store), 6)
        second = self.process(2, True)
        self.assertEqual(second['ownership_correction']['transferred'], 3)
        self.assertEqual(second['inserted_gaussians'], 0)
        np.testing.assert_array_equal(self.backend.store.ids_for_object(1), originals[3:])
        for track in self.tracks[3:]:
            self.assertEqual(self.backend._track_landmarks[int(track)][0], 1)
        learned = self.backend.store.select(originals[3:])
        np.testing.assert_allclose(transform_points(learned['xyz'], self.backend._object_to_world[1]),
                                   self.xyz[3:], atol=1e-6)

    def test_track_only_compatibility_waits_then_transfers_without_duplicate_map(self):
        self.process(0, False, observation_ids=False)
        first = self.process(1, True, observation_ids=False)
        second = self.process(2, True, observation_ids=False)
        self.assertEqual(first['ownership_deferred_observations'], 3)
        self.assertEqual(second['ownership_correction']['transferred'], 3)
        self.assertEqual(second['inserted_gaussians'], 0)
        self.assertEqual(len(self.backend.store), 6)

    def test_low_measured_confidence_does_not_supply_migration_evidence(self):
        self.process(0, False)
        for frame in (1, 2):
            report = self.process(frame, True, quality=.01)
            self.assertEqual(report['ownership_correction']['transferred'], 0)
            self.assertEqual(report['ownership_correction']['pending'], 0)
        self.assertEqual(len(self.backend.store.ids_for_object(0)), 6)

    def test_unconfirmed_changed_label_waits_without_duplicate_map_point(self):
        self.process(0, False)
        for frame in (1, 2):
            report = self.process(frame, True, probability=.8)
            self.assertEqual(report['ownership_correction']['transferred'], 0)
            self.assertEqual(report['inserted_gaussians'], 0)
            self.assertEqual(len(self.backend.store), 6)


if __name__ == '__main__':
    unittest.main()

"""CPU coverage of measured evidence and cross-frame Gaussian identities."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


MODULE = Path(__file__).resolve().parents[1] / 'gaussian_object_backend.py'
SPEC = importlib.util.spec_from_file_location('gaussian_object_backend', MODULE)
backend_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backend_module)


class Registry:
    def __init__(self):
        self.objects = {0: {}, 1: {}}
        self.track_objects = {tid: int(tid > 3) for tid in range(1, 7)}
        self.track_probabilities = {}
        self.commits = []

    def commit_window_tracks(self, tracks, object_ids, probabilities, labels):
        self.commits.append((tracks.copy(), probabilities.copy(), labels.copy()))
        for tid, p, label in zip(tracks, probabilities, labels):
            self.track_probabilities[int(tid)] = dict(zip(map(int, object_ids), map(float, p)))
            if label >= 0:
                self.track_objects[int(tid)] = int(label)


class GaussianObjectBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.K = np.array([[10., 0., 24.], [0., 10., 16.], [0., 0., 1.]])
        self.uv = np.array([[8., 8.], [10., 12.], [14., 8.],
                            [28., 18.], [30., 22.], [34., 18.], [18., 24.]])
        self.colors = np.array([[1., .1, .1], [.1, 1., .1], [.1, .1, 1.],
                                [.8, .8, .1], [.1, .8, .8], [.8, .1, .8], [1., 1., 1.]])
        self.depth = np.full((32, 48), 5., np.float32)
        self.confidence = np.ones_like(self.depth)
        self.models = np.repeat(np.eye(4)[None], 2, axis=0)
        self.models[1, 0, 3] = 1.
        self.cfg = dict(motion={'hybrid': {'min_support': 3, 'reprojection_threshold_px': .5,
                                         'depth_sigma_m': .1, 'depth_weight': .2}},
                        object_window={'enabled': True, 'window_frames': 5,
                                       'max_cost': 3., 'temperature': .05,
                                       'prior_weight': .05, 'color_sigma': .05,
                                       'link_radius_px': .75, 'model_min_support': 3})

    def frame(self, step, decoy=False):
        uv = self.uv.copy()
        uv[3:6, 0] += 2*step
        xyz = np.column_stack(((uv-self.K[:2, 2])*5./10., np.full(len(uv), 5.)))
        image = np.zeros((32, 48, 3), np.uint8)
        for point, color in zip(uv, self.colors):
            cv2.circle(image, tuple(point.astype(int)), 1,
                       tuple(map(int, np.rint(color[::-1]*255))), -1)
        if decoy:
            cv2.circle(image, (20, 24), 1, (255, 255, 255), -1)
        return dict(xyz=xyz, uv=uv, color=self.colors.copy(),
                    scale=np.ones((len(uv), 3))*.1,
                    anchor_gaussian_indices=np.arange(6)), image

    def process(self, backend, registry, step, *, missing_car=False, decoy=False,
                raw_motion_initial=False):
        gmap, source = self.frame(step, decoy=step == 1)
        following, target = self.frame(step+1, decoy=decoy)
        anchors = dict(x0=gmap['xyz'][:6], x1=following['xyz'][:6],
                       matches=np.column_stack((gmap['uv'][:6], following['uv'][:6])),
                       confidence=np.ones(6))
        q = np.zeros((7, 1 if missing_car else 2))
        if missing_car:
            q[:] = 1.
        else:
            q[:3, 0] = 1.
            q[3:6, 1] = 1.
            q[6] = [.01, .99]  # Static surface mistakenly initialized as car.
        frozen_q = q.copy()
        state = dict(H=self.models[:1] if missing_car else self.models, q=q)
        raw_q = np.column_stack((q, np.full(len(q), .25))) if raw_motion_initial else frozen_q
        if raw_motion_initial:
            state['q_motion_initial'] = raw_q.copy()
        output = self.root / f'{step}_{step+1}'
        report = backend.process(
            step, step+1, gmap, anchors, np.arange(1, 7),
            {'0': 0} if missing_car else {'0': 0, '1': 1}, state, self.K,
            source, target, self.depth, self.depth, self.confidence, self.confidence,
            output, registry)
        np.testing.assert_array_equal(state['q'], frozen_q)
        with np.load(output / 'object_state.npz') as archive:
            saved = {key: archive[key] for key in archive.files}
        np.testing.assert_array_equal(saved['q_motion_initial'], raw_q)
        return report, saved

    def test_all_motion_links_preserve_wrongly_initialized_surface_and_correct_history(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        registry = Registry()
        _, first = self.process(backend, registry, 0, decoy=True)
        static_gid = int(first['gaussian_ids'][6])
        # The first image pair cannot disambiguate this surface; its initial
        # car probability is still higher. Linking must test background too.
        self.assertGreater(first['object_probabilities'][6, 1], .5)
        report, second = self.process(backend, registry, 1)
        self.assertEqual(int(second['gaussian_ids'][6]), static_gid)
        self.assertEqual(report['links']['track_links'], 6)
        self.assertGreaterEqual(report['links']['geometry_links'], 1)
        self.assertEqual(len(set(second['gaussian_ids'])), len(second['gaussian_ids']))
        self.assertEqual(int(second['object_labels'][6]), 0)
        with np.load(self.root / '0_1' / 'object_state.npz') as rewritten:
            self.assertEqual(int(rewritten['object_labels'][6]), 0)
            np.testing.assert_array_equal(rewritten['window_frames'], [0, 1, 2])
        self.assertEqual(len(registry.commits), 2)

    def test_missing_object_refits_current_tracks_instead_of_reusing_previous_pose(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        registry = Registry()
        self.process(backend, registry, 0)
        # Current actual car displacement changes; the absent candidate must
        # be recovered from the current interval's anchor geometry.
        self.models[1, 0, 3] = 1.5
        original_frame = self.frame
        def frame(step, decoy=False):
            cloud, image = original_frame(step, decoy)
            if step >= 2:
                cloud['xyz'][3:6, 0] += .5
                cloud['uv'][3:6, 0] += 1.
            return cloud, image
        self.frame = frame
        report, saved = self.process(backend, registry, 1, missing_car=True)
        model = next(row for row in report['models'] if row['object_id'] == 1)
        self.assertTrue(model['valid'])
        self.assertEqual(model['source'], 'current_historical_track_refit')
        car_column = int(np.flatnonzero(saved['object_ids'] == 1)[0])
        self.assertAlmostEqual(float(saved['H_object'][car_column, 0, 3]), 1.5, places=5)

    def test_dense_occlusion_and_out_of_frame_are_missing_not_zero_error(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        gmap, image = self.frame(0)
        models = np.repeat(np.eye(4)[None], 3, axis=0)
        models[1, 0, 3] = 100.
        models[2, 2, 3] = 2.
        costs, weights = backend._raw_costs(gmap, models, np.ones(3, bool), self.K,
                                           image, self.depth, self.confidence, self.confidence)
        self.assertTrue(np.isfinite(costs[:, 0]).all())
        self.assertTrue(np.isnan(costs[:, 1:]).all())
        self.assertTrue((weights[:, 1:] == 0).all())

    def test_no_current_support_keeps_missing_model_invalid(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        registry = Registry()
        report, saved = self.process(backend, registry, 0, missing_car=True)
        car_column = int(np.flatnonzero(saved['object_ids'] == 1)[0])
        self.assertFalse(saved['object_model_valid'][car_column])
        self.assertTrue(np.isnan(saved['H_object'][car_column]).all())
        self.assertFalse(next(row for row in report['models'] if row['object_id'] == 1)['valid'])

    def test_raw_motion_initializer_can_have_different_columns_from_current_models(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        _, saved = self.process(backend, Registry(), 0, raw_motion_initial=True)
        self.assertEqual(saved['q_motion_initial'].shape, (7, 3))
        self.assertEqual(saved['q_object_initial'].shape, (7, 2))
        np.testing.assert_array_equal(saved['q_object_initial'][:3], [[1., 0.]]*3)

    def test_mixed_object_selects_independent_motion_despite_larger_inherited_background(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        registry = Registry()
        gmap, _ = self.frame(0)
        following, _ = self.frame(1)
        # Twelve inherited background tracks and three true moving tracks all
        # claim O1. O0 already explains the background, so O1 must keep its
        # independently supported motion, rather than its larger alias.
        indices = np.r_[np.tile(np.arange(3), 4), np.arange(3, 6)]
        anchors = dict(x0=gmap['xyz'][indices], x1=following['xyz'][indices],
                       matches=np.column_stack((gmap['uv'][indices], following['uv'][indices])),
                       confidence=np.ones(len(indices)))
        track_ids = np.arange(100, 100+len(indices))
        backend._history = {int(track): 1 for track in track_ids}
        state = dict(H=np.stack((self.models[0], self.models[0], self.models[1])))
        object_ids, models, valid, reports = backend._models(
            state, anchors, track_ids, {'0': 0, '1': 1, '2': 1}, registry, self.K)
        column = int(np.flatnonzero(object_ids == 1)[0])
        self.assertTrue(valid[column])
        np.testing.assert_allclose(models[column], self.models[1], atol=1e-6)
        chosen = next(row for row in reports if row['object_id'] == 1)
        self.assertEqual(chosen['core_support'], 3)
        self.assertEqual(chosen['independent_support'], 3)
        # The same true object remains recoverable when local discovery
        # entirely misses it, despite a larger contaminated historical set.
        object_ids, models, valid, reports = backend._models(
            {'H': self.models[:1]}, anchors, track_ids, {'0': 0}, registry, self.K)
        column = int(np.flatnonzero(object_ids == 1)[0])
        self.assertTrue(valid[column])
        np.testing.assert_allclose(models[column], self.models[1], atol=1e-6)

    def test_ambiguous_duplicate_samples_do_not_share_a_gaussian_id(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        gmap, _ = self.frame(0)
        point = {key: value[6:7] for key, value in gmap.items()
                 if key in ('xyz', 'uv', 'color', 'scale')}
        backend._previous = dict(second=1, gaussian_ids=np.array([10]),
                                 xyz=point['xyz'], color=point['color'],
                                 models=self.models[:1], model_valid=np.array([True]), K=self.K)
        duplicated = {key: np.repeat(value, 2, axis=0) for key, value in point.items()}
        assigned = np.full(2, -1, np.int64)
        count = backend._geometry_links(duplicated, 1, self.depth, self.confidence,
                                        assigned, set())
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(assigned, [-1, -1])

    def test_sampling_large_gaussian_cloud_avoids_opencv_map_dimension_limit(self):
        uv = np.tile([[12., 10.], [-1., 5.]], (40000, 1))
        scalar, valid = backend_module._sample(self.depth, uv)
        self.assertEqual(scalar.shape, (80000,))
        np.testing.assert_array_equal(scalar[::2], np.full(40000, 5.))
        self.assertTrue(np.isnan(scalar[1::2]).all())
        image = np.full((32, 48, 3), 7., np.float32)
        rgb, _ = backend_module._sample(image, uv, linear=True)
        self.assertEqual(rgb.shape, (80000, 3))
        np.testing.assert_array_equal(rgb[::2], np.full((40000, 3), 7.))
        self.assertTrue(valid[::2].all())

    def test_height_gate_only_removes_nonbackground_motion_support(self):
        backend = backend_module.SequenceObjectBackend(self.cfg)
        gmap, _ = self.frame(0)
        following, _ = self.frame(1)
        anchors = dict(x0=gmap['xyz'][:6], x1=following['xyz'][:6],
                       matches=np.column_stack((gmap['uv'][:6], following['uv'][:6])),
                       confidence=np.ones(6))
        original_confidence = anchors['confidence'].copy()
        state = dict(H=self.models.copy(), q=np.tile([.1, .9], (7, 1)))
        original_q = state['q'].copy()
        # The same high features remain observations for O0; only their
        # ability to support a foreground-object motion is removed.
        allowed = np.array([True]*3+[False]*3)
        objects, _, valid, reports = backend._models(
            state, anchors, np.arange(1, 7), {'0': 0, '1': 1}, Registry(), self.K,
            height_gate=dict(allowed=allowed, weights=allowed.astype(float)))
        self.assertTrue(valid[np.flatnonzero(objects == 0)[0]])
        self.assertFalse(valid[np.flatnonzero(objects == 1)[0]])
        background = next(row for row in reports if row['object_id'] == 0)
        foreground = next(row for row in reports if row['object_id'] == 1)
        self.assertFalse(background['height_gating_applied'])
        self.assertEqual(background['height_excluded_anchors'], 0)
        self.assertTrue(foreground['height_gating_applied'])
        self.assertEqual(foreground['height_excluded_anchors'], 3)
        self.assertEqual(len(anchors['x0']), 6)
        np.testing.assert_array_equal(anchors['confidence'], original_confidence)
        np.testing.assert_array_equal(state['q'], original_q)
        np.testing.assert_array_equal(state['H'], self.models)

    def test_invalid_ground_estimate_skips_feature_height_gating(self):
        cfg = {**self.cfg, 'object_window': {**self.cfg['object_window'],
                                            'height': dict(enabled=True)}}
        backend = backend_module.SequenceObjectBackend(cfg)
        gmap, image = self.frame(0)
        height = types.ModuleType('object_height_constraints')
        height.estimate_ground_plane = Mock(return_value=dict(
            valid=False, confidence=0., reason='insufficient_ground_support'))
        height.height_penalty = Mock(side_effect=AssertionError('Invalid planes must not be used'))
        with patch.dict('sys.modules', {'object_height_constraints': height}):
            gate, report = backend._feature_height_gate(
                gmap, dict(x0=gmap['xyz'][:6]), image, self.confidence)
        self.assertIsNone(gate)
        self.assertTrue(report['enabled'])
        self.assertFalse(report['applied'])
        self.assertEqual(report['reason'], 'insufficient_ground_support')
        height.height_penalty.assert_not_called()

    def test_measured_height_band_excludes_high_features_only_for_nonzero_objects(self):
        import object_height_constraints as height
        cfg = {**self.cfg, 'object_window': {**self.cfg['object_window'], 'height': dict(
            enabled=True, above_camera_m=2., ground_min_m=-.2,
            soft_margin_m=.3, feature_min_reliability=.5)}}
        backend = backend_module.SequenceObjectBackend(cfg)
        gmap, image = self.frame(0)
        following, _ = self.frame(1)
        x0, x1 = gmap['xyz'][:6].copy(), following['xyz'][:6].copy()
        x0[3:6, 1] = x1[3:6, 1] = -4.  # 5.5m above a measured 1.5m ground offset.
        project = lambda xyz: xyz[:, :2]/xyz[:, 2:]*self.K.diagonal()[:2]+self.K[:2, 2]
        anchors = dict(x0=x0, x1=x1, matches=np.column_stack((project(x0), project(x1))),
                       confidence=np.ones(6))
        originals = {key: value.copy() for key, value in anchors.items()}
        plane = dict(valid=True, reason='synthetic_measured_ground', normal=[0., -1., 0.],
                     offset=1.5, camera_height=1.5, confidence=1., fit_rmse_m=0.)
        with patch.object(height, 'estimate_ground_plane', return_value=plane):
            gate, report = backend._feature_height_gate(gmap, anchors, image, self.confidence)
        self.assertTrue(report['applied'])
        self.assertTrue((~gate['allowed'][3:6]).all())
        self.assertTrue((gate['weights'][3:6] == 0).all())
        state = dict(H=self.models.copy(), q=np.tile([.1, .9], (7, 1)))
        q_before = state['q'].copy()
        objects, _, valid, _ = backend._models(
            state, anchors, np.arange(1, 7), {'0': 0, '1': 1}, Registry(), self.K, height_gate=gate)
        self.assertTrue(valid[np.flatnonzero(objects == 0)[0]])
        self.assertFalse(valid[np.flatnonzero(objects == 1)[0]])
        for key, value in originals.items():
            np.testing.assert_array_equal(anchors[key], value)
        np.testing.assert_array_equal(state['q'], q_before)

    def test_local_candidate_height_refit_uses_only_eligible_core_and_leaves_background_untouched(self):
        for foreground_count in (3, 4):
            with self.subTest(foreground_count=foreground_count):
                backend = backend_module.SequenceObjectBackend(self.cfg)
                gmap, _ = self.frame(0)
                foreground = gmap['xyz'][3:6].copy()
                if foreground_count == 4:
                    foreground = np.vstack((foreground, [[6., 2., 5.]]))
                x0 = np.vstack((gmap['xyz'][:3], foreground, [[0., -6., 5.]]))
                x1 = x0.copy()
                x1[3:3+foreground_count, 0] += 1.
                x1[-1, 0] += 1.1
                project = lambda xyz: xyz[:, :2]/xyz[:, 2:]*self.K.diagonal()[:2]+self.K[:2, 2]
                anchors = dict(x0=x0, x1=x1,
                               matches=np.column_stack((project(x0), project(x1))),
                               confidence=np.ones(len(x0)))
                originals = {key: value.copy() for key, value in anchors.items()}
                H = self.models.copy()
                H[1, 0, 3] = 1.03  # An otherwise usable but contaminated local proposal.
                state = dict(H=H, q=np.tile([.2, .8], (len(x0), 1)))
                original_H, original_q = state['H'].copy(), state['q'].copy()
                allowed = np.ones(len(x0), bool)
                allowed[-1] = False
                with patch.object(backend_module, '_refit', wraps=backend_module._refit) as refit:
                    objects, models, valid, reports = backend._models(
                        state, anchors, np.arange(len(x0))+10, {'0': 0, '1': 1}, Registry(), self.K,
                        height_gate=dict(allowed=allowed, weights=allowed.astype(float)))
                self.assertEqual(refit.call_count, 1)  # No added background refit.
                fit_anchors, _, fit_ids, _, _ = refit.call_args.args
                np.testing.assert_array_equal(fit_ids, np.arange(3, 3+foreground_count))
                self.assertEqual(fit_anchors['confidence'][-1], 0.)
                self.assertTrue(valid.all())
                np.testing.assert_allclose(models[np.flatnonzero(objects == 0)[0]], self.models[0], atol=0.)
                np.testing.assert_allclose(models[np.flatnonzero(objects == 1)[0]], self.models[1], atol=1e-5)
                background = next(row for row in reports if row['object_id'] == 0)
                foreground_report = next(row for row in reports if row['object_id'] == 1)
                self.assertFalse(background['height_gated_refit'])
                self.assertTrue(foreground_report['height_gated_refit'])
                self.assertEqual(foreground_report['source'], 'local_1')
                for key, value in originals.items():
                    np.testing.assert_array_equal(anchors[key], value)
                np.testing.assert_array_equal(state['H'], original_H)
                np.testing.assert_array_equal(state['q'], original_q)


if __name__ == '__main__':
    unittest.main()

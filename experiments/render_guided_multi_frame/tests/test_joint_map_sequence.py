"""Full sequence entry with persistent geometry and the exact synthetic frontend."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import json
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import test_streaming_sequence as synthetic


class JointMapSequenceTests(unittest.TestCase):
    def test_stopped_vehicle_starts_moving_and_corrects_existing_background_map(self):
        fixture = synthetic.StreamingSequenceTests()
        fixture.setUp()
        self.addCleanup(fixture.temporary.cleanup)
        fixture.cfg['gaussian_map'] = dict(
            enabled=True, observation_stride=1, initial_scale_multiplier=1.,
            # Motion is one metre per frame, much larger than the association
            # gate. Shared observation identity must preserve existing map IDs
            # while confirmed ownership changes their canonical frame.
            association_distance=.05, color_distance=.2, voxel_size=.05,
            min_probability=.6, max_active_gaussians=24, max_samples_per_frame=16,
            max_new_gaussians_per_frame=8, checkpoint_interval=3,
            ownership_correction=dict(enabled=True, confirmation_updates=2),
            optimization=dict(backend='torch_reference', device='cpu', iterations=1,
                              downsample=1, verbose=False))

        def stopped_frame(frame):
            uv = [[8., 8.], [14., 8.], [8., 14.], [14., 14.]]
            if 10 <= frame <= 16:
                offset = 2*max(0, frame-11)
                uv += [[28.+offset, 28.], [34.+offset, 28.],
                       [28.+offset, 34.], [34.+offset, 34.]]
            uv = np.asarray(uv)
            xyz = np.column_stack(((uv-fixture.K[:2, 2])*5./
                                   np.asarray([fixture.K[0, 0], fixture.K[1, 1]]),
                                   np.full(len(uv), 5.)))
            colors = fixture.colors[:len(uv)]
            image = np.zeros((48, 64, 3), np.uint8)
            for point, rgb in zip(uv, colors):
                cv2.circle(image, tuple(point.astype(int)), 1,
                           tuple(map(int, rgb[::-1]*255)), -1)
            return dict(uv=uv, xyz=xyz, color=colors,
                        scale=np.full((len(uv), 3), .1)), image

        def frontend(cfg, pair_context):
            from frontend.hybrid_motion import _core_labels, motion_probabilities
            first, second = cfg['frames']['first'], cfg['frames']['second']
            gmap, source = stopped_frame(first)
            following, target = stopped_frame(second)
            car = len(gmap['xyz']) == 8 and len(following['xyz']) == 8
            moving = car and not np.array_equal(gmap['uv'][4:], following['uv'][4:])
            n = 8 if car else 4
            anchors = dict(x0=gmap['xyz'][:n], x1=following['xyz'][:n],
                matches=np.column_stack((gmap['uv'][:n], following['uv'][:n])),
                confidence=np.ones(n))
            models = np.repeat(np.eye(4)[None], 2 if moving else 1, axis=0)
            if moving:
                models[1, :3, 3] = following['xyz'][4]-gmap['xyz'][4]
            _, labels = _core_labels(anchors, models, fixture.K, cfg['motion']['hybrid'])
            models, labels, q_anchor, _ = pair_context.initialize(
                anchors, fixture.K, models, labels, cfg)
            # Initialize the observation Gaussians from measured current motion
            # probabilities; never override registry/window/global ownership.
            q = np.zeros((len(gmap['xyz']), len(models)))
            q[:, 0] = 1.
            q[:n] = motion_probabilities(_core_labels(anchors, models, fixture.K,
                cfg['motion']['hybrid'])[0], cfg['motion']['hybrid'])[0]
            gmap['anchor_gaussian_indices'] = np.arange(n)
            pair_output = Path(cfg['output_dir']) / f'{first}_{second}'
            pair_output.mkdir(parents=True)
            depth = np.full((48, 64), 5., np.float32)
            pair_context.finish(anchors, gmap, dict(H=np.asarray(models), q=q,
                q_anchor=q_anchor, reference=0), fixture.K, pair_output, cfg,
                dict(source=source, target=target, source_depth=depth, target_depth=depth,
                     source_confidence=np.ones_like(depth), target_confidence=np.ones_like(depth)))
            return dict(final_models=len(models))

        output = fixture.root / 'stopped_then_moving_map'
        with patch('frontend.strict_two_frame.run', side_effect=frontend):
            state = synthetic.runner.run_sequence(fixture.cfg, list(range(23)), output)
        from mapping.persistent_gaussian_map import PersistentGaussianMap
        store = PersistentGaussianMap.load(output / 'gaussian_map/map')
        dynamic = [row for oid, row in state['objects'].items()
                   if oid != '0' and row['confirmed']]
        self.assertEqual(len(dynamic), 1, state['objects'])
        vehicle = dynamic[0]
        self.assertEqual(vehicle['first_observed_frame'], 11)
        self.assertEqual(vehicle['last_observed_frame'], 16)
        self.assertEqual(vehicle['status'], 'retired')
        self.assertEqual(vehicle['retired_at_frame'], 18)
        self.assertEqual(len(store), 8)
        self.assertEqual(store.next_id, 8)
        events = store.ownership_events
        moved = [mid for event in events if event['new_object_id'] == vehicle['object_id']
                 for mid, owner in zip(event['map_ids'], event['old_object_ids']) if owner == 0]
        self.assertEqual(len(moved), 4, events)
        np.testing.assert_array_equal(sorted(moved), store.ids_for_object(vehicle['object_id']))
        pairs = [json.loads(line) for line in (output / 'pairs.jsonl').read_text().splitlines()]
        stopped = next(row for row in pairs if row['first'] == 10)
        self.assertEqual(set(stopped['local_to_object'].values()), {0})
        self.assertTrue(any(row['gaussian_map']['ownership_deferred_observations'] > 0
                            for row in pairs))
        self.assertTrue(any(row['gaussian_map']['ownership_correction']['transferred'] == 4
                            for row in pairs))
        # Confirmation completes at frame 12. The transferred surface should
        # already have followed the vehicle's one-metre 11->12 displacement,
        # rather than remaining at its old background position for one frame.
        from mapping.gaussian_map_backend import transform_points
        with np.load(output / 'gaussian_map/frames/frame_0000000012.npz') as snapshot:
            selected = snapshot['object_labels'] == vehicle['object_id']
            pose_row = int(np.flatnonzero(snapshot['object_ids'] == vehicle['object_id'])[0])
            actual = transform_points(snapshot['xyz'][selected], snapshot['object_to_camera'][pose_row])
            np.testing.assert_allclose(actual, stopped_frame(12)[0]['xyz'][4:], atol=.01)
        for row in pairs:
            with np.load(output / row['result_path'] / 'gaussian_state.npz') as saved:
                np.testing.assert_array_equal(saved['xyz'], stopped_frame(row['first'])[0]['xyz'])
        from tools.validation.verify_streaming_run import verify
        from tools.validation.verify_gaussian_map_run import verify as verify_map
        self.assertTrue(verify(output)['passed'])
        audit = verify_map(output)
        self.assertTrue(audit['passed'], audit)

    def test_36_frame_map_keeps_geometry_adam_and_object_intervals(self):
        fixture = synthetic.StreamingSequenceTests()
        fixture.setUp()
        self.addCleanup(fixture.temporary.cleanup)
        fixture.cfg['gaussian_map'] = dict(
            enabled=True, observation_stride=1, initial_scale_multiplier=1.,
            association_distance=.2, voxel_size=.08, min_probability=.6,
            max_active_gaussians=24, max_samples_per_frame=16,
            max_new_gaussians_per_frame=8, checkpoint_interval=3,
            optimization=dict(backend='torch_reference', device='cpu', iterations=2,
                              downsample=1, verbose=False))
        output = fixture.root / 'joint_map'
        with patch('frontend.strict_two_frame.run', side_effect=fixture.frontend):
            state = synthetic.runner.run_sequence(fixture.cfg, list(range(36)), output)
        report = state['gaussian_map']
        self.assertTrue(report['enabled'])
        self.assertTrue(report['finalized'])
        self.assertTrue(report['geometry_optimized'])
        self.assertEqual(report['window_frames'], [31, 32, 33, 34, 35])
        self.assertLessEqual(report['active_map_gaussians'], 24)
        self.assertEqual(report['total_map_gaussians'], 12)
        self.assertEqual(report['optimization']['optimized_parameters'],
                         ['xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits'])
        from mapping.persistent_gaussian_map import PersistentGaussianMap
        store = PersistentGaussianMap.load(output / 'gaussian_map/map')
        self.assertEqual(len(store), 12)
        self.assertEqual(store.get_status(1), 'retired')
        self.assertEqual(store.get_status(2), 'retired')
        self.assertTrue(set(store.ids_for_object(1)).isdisjoint(store.ids_for_object(2)))
        static = store.export_object(0, include_optimizer=True)
        self.assertGreater(static['optimizer_state']['params']['xyz']['row_steps'].max(), 2)
        self.assertGreater(np.max(np.abs(static['xyz']-static['initial_xyz'])), 1e-6)
        self.assertTrue(np.isfinite(static['colors']).all())
        self.assertTrue(np.allclose(np.linalg.norm(static['rotations'], axis=1), 1, atol=1e-5))
        frames = output / 'gaussian_map/frames'
        self.assertEqual(len(list(frames.glob('frame_*.npz'))), 36)
        with np.load(frames / 'frame_0000000016.npz') as present:
            self.assertIn(1, present['object_ids'])
        with np.load(frames / 'frame_0000000017.npz') as absent:
            self.assertNotIn(1, absent['object_ids'])
        with np.load(frames / 'frame_0000000035.npz') as final:
            np.testing.assert_array_equal(final['object_ids'], [0])
        self.assertEqual(state['objects']['1']['last_observed_frame'], 16)
        self.assertEqual(state['objects']['2']['first_observed_frame'], 22)
        records = [json.loads(line) for line in (output / 'gaussian_map/trajectory.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 35)
        np.testing.assert_allclose(records[0]['camera_to_world_first'], np.eye(4))
        self.assertNotIn('1', records[16]['object_to_world_second'])
        # The ownership archives still describe immutable observation geometry.
        with np.load(output / 'pairs/0_1/gaussian_state.npz') as original:
            np.testing.assert_array_equal(original['xyz'], fixture.frame(0)[0]['xyz'])
        from tools.validation.verify_streaming_run import verify
        self.assertTrue(verify(output)['passed'])
        from tools.validation.verify_gaussian_map_run import verify as verify_map
        audit = verify_map(output)
        self.assertTrue(audit['passed'], audit)
        # Exercise the ordinary visualization entry, which must choose learned
        # map geometry rather than the fixed ownership Gaussian snapshots.
        images = fixture.root / 'dataset/data_2d_raw/synthetic/image_00/data_rect'
        images.mkdir(parents=True)
        for frame in range(36):
            cv2.imwrite(str(images / f'{frame:010d}.png'), fixture.frame(frame)[1])
        from tools.visualization.visualize_objects import visualize
        manifest = visualize(output, width=64)
        self.assertEqual(manifest['format'], 'learned_gaussian_map_frames_v1')
        self.assertEqual(len(manifest['scene']['views']), 36)
        self.assertTrue((output / 'gaussian_map/visualization/index.html').is_file())

    def test_map_config_requires_streaming_and_remains_opt_in(self):
        cfg = synthetic.runner.load_config(synthetic.HERE / 'configs/kitti360_joint_gaussian_map.yaml')
        self.assertTrue(cfg['gaussian_map']['enabled'])
        self.assertTrue(cfg['sequence']['streaming'])
        self.assertEqual(synthetic.runner.sequence_frames(cfg), list(range(2433, 2464, 2)))
        baseline = synthetic.runner.load_config(synthetic.HERE / 'configs/kitti360_sequence.yaml')
        self.assertFalse(baseline.get('gaussian_map', {}).get('enabled', False))


if __name__ == '__main__':
    unittest.main()

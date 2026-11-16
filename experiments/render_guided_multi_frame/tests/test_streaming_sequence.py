"""Long rolling CPU integration with real tracks, registry and Gaussian backend.

The frontend supplies exact synthetic motions, matching the lifecycle contract:
the ownership observations are reliable. CUDA estimation is tested separately.
"""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('streaming_sequence_runner', HERE / 'run_sequence.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class StreamingSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.K = np.array([[10., 0., 32.], [0., 10., 24.], [0., 0., 1.]])
        self.colors = np.array([[1., .1, .1], [.1, 1., .1], [.1, .1, 1.], [.8, .8, .1],
                                [.1, .8, .8], [.8, .1, .8], [.9, .9, .9], [.5, .3, .9]])
        self.cfg = dict(
            dataset=dict(data_root=str(self.root / 'dataset'), sequence='synthetic'),
            sequence=dict(streaming=True, stage_overviews=False, visualize_objects=False),
            output={}, tracking=dict(join_radius_px=.25, cache_frames=2),
            motion=dict(hybrid=dict(min_support=3, max_components=4,
                                   reprojection_threshold_px=.5, depth_sigma_m=.1,
                                   depth_weight=.2, refit_max_nfev=10)),
            object_lifecycle=dict(enabled=True, missing_grace_frames=2),
            object_window=dict(enabled=True, window_frames=5, temperature=.03,
                               prior_weight=.03, max_cost=3., max_fit_cost=1.,
                               color_sigma=.05, link_radius_px=.5,
                               render=dict(enabled=False)))

    def frame(self, frame):
        uv = [[8., 8.], [14., 8.], [8., 14.], [14., 14.]]
        if 10 <= frame <= 16:
            offset = frame - 10
        elif 22 <= frame <= 26:
            offset = frame - 22
        else:
            offset = None
        if offset is not None:
            uv += [[28.+offset, 28.], [34.+offset, 28.],
                   [28.+offset, 34.], [34.+offset, 34.]]
        uv = np.asarray(uv)
        xyz = np.column_stack(((uv-self.K[:2, 2])*.5, np.full(len(uv), 5.)))
        color = self.colors[:len(uv)]
        image = np.zeros((48, 64, 3), np.uint8)
        for point, rgb in zip(uv, color):
            cv2.circle(image, tuple(point.astype(int)), 1, tuple(map(int, rgb[::-1]*255)), -1)
        return dict(uv=uv, xyz=xyz, color=color, scale=np.full((len(uv), 3), .1)), image

    def frontend(self, cfg, pair_context):
        first, second = cfg['frames']['first'], cfg['frames']['second']
        gmap, source = self.frame(first)
        following, target = self.frame(second)
        car = len(gmap['xyz']) == 8 and len(following['xyz']) == 8
        n = 8 if car else 4
        anchors = dict(x0=gmap['xyz'][:n], x1=following['xyz'][:n],
                       matches=np.column_stack((gmap['uv'][:n], following['uv'][:n])),
                       confidence=np.ones(n))
        models = np.repeat(np.eye(4)[None], 2 if car else 1, axis=0)
        if car:
            models[1, 0, 3] = .5
        labels = np.r_[np.zeros(4, int), np.ones(4, int)] if car else np.zeros(4, int)
        models, labels, q_anchor, _ = pair_context.initialize(anchors, self.K, models, labels, cfg)
        q = np.zeros((len(gmap['xyz']), len(models)))
        q[:, 0] = 1.
        if car:
            q[4:, 0], q[4:, 1] = 0., 1.
        gmap['anchor_gaussian_indices'] = np.arange(n)
        output = Path(cfg['output_dir']) / f'{first}_{second}'
        output.mkdir(parents=True)
        depth = np.full((48, 64), 5., np.float32)
        observations = dict(source=source, target=target, source_depth=depth,
                            target_depth=depth, source_confidence=np.ones_like(depth),
                            target_confidence=np.ones_like(depth))
        pair_context.finish(anchors, gmap, dict(H=np.asarray(models), q=q,
                            q_anchor=q_anchor, reference=0), self.K, output, cfg, observations)
        return dict(final_models=len(models))

    def test_localization_visualization_runs_automatically_when_enabled(self):
        self.cfg['sequence']['visualize_localization'] = True
        with patch('frontend.strict_two_frame.run', side_effect=self.frontend):
            state = runner.run_sequence(self.cfg, [0, 1, 2], self.root/'auto_localization')
        self.assertEqual(state['localization']['valid_intervals'], 2)
        self.assertTrue((self.root/'auto_localization/localization/trajectory.png').is_file())
        self.assertIsNone(state['localization']['ate_rmse_m'])

    def test_vehicle_interval_new_birth_and_real_window_rollover(self):
        output = self.root / 'run'
        with patch('frontend.strict_two_frame.run', side_effect=self.frontend), \
                patch.object(runner, 'stage_overviews') as stages:
            state = runner.run_sequence(self.cfg, list(range(36)), output)
        stages.assert_not_called()
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(len(state['pairs']), 35)
        confirmed = [record for oid, record in state['objects'].items()
                     if oid != '0' and record['confirmed']]
        self.assertEqual(len(confirmed), 2)
        confirmed.sort(key=lambda row: row['first_observed_frame'])
        old, new = confirmed
        self.assertEqual((old['first_observed_frame'], old['last_observed_frame'],
                          old['retired_at_frame']), (10, 16, 18))
        self.assertEqual((new['first_observed_frame'], new['last_observed_frame'],
                          new['retired_at_frame']), (22, 26, 28))
        self.assertGreater(new['object_id'], old['object_id'])
        self.assertEqual(state['lifecycle']['candidate_object_ids'], [0])
        window = state['object_window']
        self.assertEqual(window['window_frames'], [31, 32, 33, 34, 35])
        self.assertEqual(window['retained_objects'], 1)
        self.assertLessEqual(window['retained_pairs'], 4)
        self.assertLessEqual(window['retained_gaussians'], 16)
        self.assertGreater(window['forgotten_gaussians'], 0)
        self.assertEqual(window['finalized_pairs'], 31)
        with np.load(output / 'tracks/observations.npz') as archive:
            self.assertEqual(set(archive['frames']), set(range(36)))
            self.assertEqual(archive['uv'].shape[1], 2)
        events = [json.loads(line) for line in (output / 'object_lifecycle.jsonl').read_text().splitlines()]
        births = [row for row in events if row['type'] == 'birth' and row['object_id'] != 0]
        self.assertEqual([row['frame'] for row in births], [10, 22])
        self.assertEqual([row['event_index'] for row in events], list(range(1, len(events)+1)))
        self.assertEqual(len((output / 'pairs.jsonl').read_text().splitlines()), 35)
        self.assertTrue((output / 'object_intervals.json').is_file())
        # Early files retain their original window and candidate columns.
        with np.load(output / 'pairs/0_1/object_state.npz') as early:
            np.testing.assert_array_equal(early['window_frames'], [0, 1, 2, 3, 4])
            np.testing.assert_array_equal(early['object_ids'], [0])
        from tools.validation.verify_streaming_run import verify
        audit = verify(output)
        self.assertTrue(audit['passed'], audit['checks'])
        # A plausible looking interval must still be rejected if it includes
        # unobserved frames after the object left the scene.
        state_path = output / 'sequence_state.json'
        corrupt = json.loads(state_path.read_text())
        corrupt['objects'][str(old['object_id'])]['last_observed_frame'] = 19
        state_path.write_text(json.dumps(corrupt))
        self.assertFalse(verify(output)['checks']['lifecycle_intervals'])
        with patch('frontend.strict_two_frame.run') as frontend:
            with self.assertRaises(FileExistsError):
                runner.run_sequence(self.cfg, [0, 1], output)
        frontend.assert_not_called()

    def test_empty_candidate_is_pruned_from_live_window_and_final_result_objects(self):
        self.cfg['object_window'].update(prune_empty_objects=True, window_step=3)
        self.cfg['object_lifecycle']['missing_grace_frames'] = 100
        output = self.root / 'empty_candidate'

        def frontend(cfg, pair_context):
            result = self.frontend(cfg, pair_context)
            if cfg['frames']['first'] == 0:
                # A proposed identity has no actual Gaussian support in any
                # subsequent observation. It must not survive until the much
                # longer missing-grace limit.
                pair_context.objects._create_object(0)
            return result

        with patch('frontend.strict_two_frame.run', side_effect=frontend):
            state = runner.run_sequence(self.cfg, list(range(8)), output)
        self.assertEqual(set(state['objects']), {'0'})
        self.assertEqual(set(state['discarded_objects']), {'1'})
        self.assertEqual(state['discarded_objects']['1']['retirement_reason'], 'no_gaussians')
        self.assertEqual(state['discarded_objects']['1']['retired_at_frame'], 4)
        self.assertEqual(state['object_window']['object_ids'], [0])
        self.assertEqual(set(json.loads((output / 'object_intervals.json').read_text())), {'0'})
        with np.load(output / 'pairs/1_2/object_state.npz') as saved:
            np.testing.assert_array_equal(saved['object_ids'], [0])
            np.testing.assert_array_equal(saved['evidence_object_ids'], [0, 1])
        from tools.validation.verify_streaming_run import verify
        audit = verify(output)
        self.assertTrue(audit['passed'], audit['checks'])
        self.assertTrue(audit['checks']['no_empty_objects'])

    def test_sequence_frame_selection_and_invalid_ranges(self):
        cfg = runner.load_config(HERE / 'configs/kitti360_drive0002_4914_4934.yaml')
        self.assertEqual(runner.sequence_frames(cfg), list(range(4914, 4935, 3)))
        self.assertEqual(runner.sequence_frames(cfg, start=10, end=16, step=1), list(range(10, 17)))
        self.assertEqual(runner.sequence_frames(cfg, frames=[10, 13, 16]), [10, 13, 16])
        for options in (dict(start=1), dict(end=4), dict(start=4, end=1),
                        dict(start=1, end=4, step=0), dict(frames=[2, 1])):
            with self.assertRaises(ValueError):
                runner.sequence_frames(cfg, **options)

    def test_seven_frame_windows_with_five_frame_advance_and_three_frame_sampling(self):
        self.cfg['object_window'].update(window_frames=7, window_step=5)
        frames = list(range(0, 40, 3))
        output = self.root / 'batch_run'
        with patch('frontend.strict_two_frame.run', side_effect=self.frontend):
            state = runner.run_sequence(self.cfg, frames, output)
        self.assertEqual(state['object_window']['window_frames'], frames[10:])
        self.assertEqual(state['object_window']['window_schedule']['reason'], 'final_partial_window')
        from tools.validation.verify_streaming_run import verify
        report = verify(output)
        self.assertTrue(report['passed'], report['checks'])
        for index, expected in ((0, frames[:7]), (5, frames[5:12]), (10, frames[10:])):
            with np.load(output / f'pairs/{frames[index]}_{frames[index+1]}/object_state.npz') as saved:
                np.testing.assert_array_equal(saved['window_frames'], expected)


if __name__ == '__main__':
    unittest.main()

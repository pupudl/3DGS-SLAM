"""Batch-window boundaries, posterior handoff, and final partial-window solves."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tempfile
import json
import unittest
from unittest.mock import patch

import numpy as np

import run_sequence as runner
from ownership.object_window import ObjectWindow
from run_sequence import load_config, sequence_frames


class WindowScheduleTests(unittest.TestCase):
    def config(self, frames=None, render=True):
        return dict(sequence=dict(streaming=True, frames=frames), object_window=dict(
            window_frames=7, window_step=5, temperature=.2, prior_weight=.03,
            render=dict(enabled=render)))

    def inputs(self):
        payload = dict(xyz=np.array([[0., 0., 5.]]), uv=np.array([[0., 0.]]),
                       color=np.ones((1, 3)), scale=np.full((1, 3), .1), K=np.eye(3),
                       H_object=np.repeat(np.eye(4)[None], 2, axis=0))
        depth = np.full((2, 2), 5.)
        observations = dict(source=np.zeros((2, 2, 3), np.uint8),
                            target=np.zeros((2, 2, 3), np.uint8),
                            source_depth=depth, target_depth=depth,
                            source_confidence=np.ones((2, 2)), target_confidence=np.ones((2, 2)))
        return payload, observations

    def update(self, window, first, second, **kwargs):
        payload, observations = self.inputs()
        return window.update(first, second, np.array([42]), np.array([0, 1]),
                             np.array([[.8, .2]]), np.array([[.01, .9]]),
                             payload=payload, render_observations=observations, **kwargs)

    def test_default_configs_use_seven_five_and_frame_interval_three(self):
        for name in ('kitti360_drive0002_4914_4934.yaml', 'kitti360_drive0004_0776_0796.yaml'):
            cfg = load_config(HERE / 'configs' / name)
            self.assertEqual(cfg['object_window']['window_frames'], 7)
            self.assertEqual(cfg['object_window']['window_step'], 5)
            self.assertTrue(cfg['object_window']['connectivity']['no_core_to_background'])
            self.assertEqual(cfg['sequence']['step'], 3)
            self.assertEqual(cfg['gaussians']['stride'], 1)
            frames = sequence_frames(cfg)
            np.testing.assert_array_equal(np.diff(frames), 3)

    def test_sequence_presets_share_method_and_allow_schedule_overrides(self):
        configs = [load_config(HERE / 'configs' / name) for name in (
            'kitti360_drive0002_4914_4934.yaml', 'kitti360_drive0004_0776_0796.yaml')]
        for cfg in configs:
            self.assertTrue(cfg['sequence']['streaming'])
            self.assertTrue(cfg['object_lifecycle']['enabled'])
            self.assertTrue(cfg['tracking']['motion_identity']['enabled'])
            self.assertTrue(cfg['object_window']['prune_empty_objects'])
        for cfg in configs:
            cfg['sequence'] = {key: value for key, value in cfg['sequence'].items()
                               if key not in ('frames', 'start', 'end', 'output_dir')}
            cfg['dataset'].pop('sequence')
        self.assertEqual(configs[0], configs[1])
        configs[0]['object_window'].update(window_frames=5, window_step=2)
        configs[0]['sequence']['step'] = 1
        window = ObjectWindow(configs[0])
        self.assertEqual((window.window_size, window.window_step), (5, 2))
        self.assertTrue(configs[0]['tracking']['motion_identity']['enabled'])

    def test_cli_schedule_overrides_preserve_the_shared_method_and_saved_stride(self):
        for name in ('kitti360_drive0002_4914_4934.yaml', 'kitti360_drive0004_0776_0796.yaml'):
            argv = ['run_sequence.py', '--config', str(HERE / 'configs' / name),
                    '--start', '2439', '--end', '2460', '--stride', '1',
                    '--window-frames', '5', '--window-step', '2']
            with patch.object(sys, 'argv', argv), patch.object(runner, 'run_sequence') as run:
                runner.main()
            cfg, frames, _ = run.call_args.args
            self.assertEqual(frames, list(range(2439, 2461)))
            self.assertEqual(cfg['sequence']['step'], 1)
            self.assertEqual((cfg['object_window']['window_frames'], cfg['object_window']['window_step']), (5, 2))
            self.assertTrue(cfg['tracking']['motion_identity']['enabled'])
            self.assertTrue(cfg['sequence']['streaming'])

    def test_full_windows_overlap_by_two_and_keep_posteriors_until_next_solve(self):
        frames = list(range(2433, 2473, 3))  # 14 images: two full windows and a tail.
        window = ObjectWindow(self.config(frames))
        captured = []

        def solve(current, cfg, *, priors):
            captured.append((sorted(current._frames), current._render_initial_probabilities.copy()))
            p = np.array([[.1+.1*len(captured), .9-.1*len(captured)]])
            return dict(probabilities=p, render_support=np.ones_like(p, bool),
                        report=dict(render_passes=1, iterations_completed=1))

        with tempfile.TemporaryDirectory() as temporary, \
                patch('ownership.object_render_optimizer.optimize_render_window', side_effect=solve):
            root = Path(temporary)
            frozen = None
            last_solved = None
            for index, (first, second) in enumerate(zip(frames[:-1], frames[1:])):
                result = self.update(window, first, second, output_dir=root / str(index))
                schedule = result['report']['window_schedule']
                self.assertEqual(schedule['render_optimization_performed'], index in (5, 10, 12))
                if schedule['render_optimization_performed']:
                    last_solved = result['object_probabilities'].copy()
                elif last_solved is not None:
                    np.testing.assert_array_equal(result['object_probabilities'], last_solved)
                    self.assertEqual(result['report']['render_optimization']['render_passes'], 0)
                self.assertLessEqual(len(window._frames), 7)
                self.assertLessEqual(len(window._active), 6)
                np.testing.assert_allclose(window._prior(42), [.8, .2])
                if index == 5:
                    frozen = (root / '0/object_state.npz').read_bytes()
                if index > 5:
                    self.assertEqual((root / '0/object_state.npz').read_bytes(), frozen)
            self.assertEqual([row[0] for row in captured], [frames[:7], frames[5:12], frames[10:]])
            np.testing.assert_allclose(captured[1][1], [[.2, .8]])
            np.testing.assert_allclose(captured[2][1], [[.3, .7]])
            self.assertEqual(window.to_dict()['window_schedule']['reason'], 'final_partial_window')
            for index, expected in ((0, frames[:7]), (5, frames[5:12]), (10, frames[10:])):
                with np.load(root / str(index) / 'object_state.npz') as saved:
                    np.testing.assert_array_equal(saved['window_frames'], expected)
                    np.testing.assert_array_equal(saved['q_object_initial'], [[.8, .2]])

    def test_centre_boundary_evidence_is_not_duplicated_at_five_frame_advance(self):
        streaming = ObjectWindow(self.config(render=False))
        full = ObjectWindow(dict(window_frames=100, temperature=.2, prior_weight=.03))
        frames = list(range(2433, 2490, 3))
        for i, (first, second) in enumerate(zip(frames[:-1], frames[1:])):
            costs = np.array([[.05, .9] if i % 2 else [.8, .1], [.1, .9]])
            args = (first, second, np.array([42, 100+i]), np.array([0, 1]),
                    np.array([[.8, .2], [.5, .5]]), costs)
            actual = streaming.update(*args)
            expected = full.update(*args)
            np.testing.assert_allclose(actual['object_probabilities'], expected['object_probabilities'], atol=1e-14)
            self.assertLessEqual(len(streaming._initial), 7)
        self.assertEqual(streaming.to_dict()['finalized_pairs'], 15)

    def test_explicit_flush_solves_an_incomplete_live_window(self):
        window = ObjectWindow(self.config())
        with patch('ownership.object_render_optimizer.optimize_render_window', return_value=dict(
                probabilities=np.array([[.2, .8]]), render_support=np.ones((1, 2), bool),
                report=dict(render_passes=1))) as solve:
            self.update(window, 0, 1)
            self.update(window, 1, 2)
            solve.assert_not_called()
            result = self.update(window, 2, 3, force_optimize=True)
        solve.assert_called_once()
        self.assertEqual(result['report']['window_schedule']['reason'], 'final_partial_window')

    def test_invalid_advance_is_rejected(self):
        for step in (0, -1, 7, 8):
            cfg = self.config()
            cfg['object_window']['window_step'] = step
            with self.assertRaisesRegex(ValueError, 'window_step'):
                ObjectWindow(cfg)

    def test_pruned_motion_columns_remain_available_for_exact_solver_replay(self):
        cfg = self.config(frames=[0, 1], render=False)
        cfg['object_window']['prune_empty_objects'] = True
        window = ObjectWindow(cfg)
        payload, _ = self.inputs()
        payload['H_object'][1, 0, 3] = 1.
        with tempfile.TemporaryDirectory() as directory:
            original = window.update(0, 1, [42], [0, 1], [[1., 0.]], [[.01, 2.]],
                                     payload=payload, output_dir=directory)
            with np.load(Path(directory) / 'object_state.npz') as saved:
                np.testing.assert_array_equal(saved['object_ids'], [0])
                np.testing.assert_array_equal(saved['evidence_object_ids'], [0, 1])
                self.assertEqual(saved['H_object'].shape, (1, 4, 4))
                replayed = ObjectWindow(cfg).update(0, 1, saved['gaussian_ids'],
                    saved['evidence_object_ids'], saved['q_object_initial'], saved['observation_costs'],
                    weights=saved['observation_weights'], payload=dict(payload,
                        H_object=saved['evidence_H_object'],
                        object_model_valid=saved['evidence_object_model_valid']))
            np.testing.assert_array_equal(replayed['object_ids'], original['object_ids'])
            np.testing.assert_array_equal(replayed['object_labels'], original['object_labels'])
            np.testing.assert_array_equal(replayed['object_probabilities'], original['object_probabilities'])

    def test_hard_height_applies_before_the_window_is_complete(self):
        cfg = self.config()
        cfg['object_window']['height'] = dict(enabled=True, above_camera_m=.5, hard_upper_bound=True)
        window = ObjectWindow(cfg)
        payload, observations = self.inputs()
        payload['xyz'][0, 1] = -.6
        payload['ground_plane_report_json'] = np.asarray(json.dumps(dict(
            valid=True, normal=[0., -1., 0.], offset=1.5, camera_height=1.5,
            confidence=1., fit_rmse_m=0.)))
        with patch('ownership.object_render_optimizer.optimize_render_window') as solve:
            result = window.update(0, 1, np.array([42]), np.array([0, 1]),
                                   np.array([[.01, .99]]), np.array([[.9, .01]]),
                                   payload=payload, render_observations=observations)
        solve.assert_not_called()
        np.testing.assert_array_equal(result['object_probabilities'], [[1., 0.]])
        np.testing.assert_array_equal(result['object_labels'], [0])
        self.assertEqual(result['report']['window_schedule']['reason'], 'collecting_window')

    def test_missing_core_is_decided_at_the_solve_instead_of_during_collection(self):
        cfg = self.config()
        cfg['object_window']['connectivity'] = dict(enabled=True, no_core_to_background=True)
        window = ObjectWindow(cfg)
        payload, observations = self.inputs()
        state = window.update(0, 1, np.array([42]), np.array([0, 1]), [[.01, .99]], [[.9, .01]],
                              payload=payload, render_observations=observations)
        self.assertEqual(state['object_labels'][0], 1)
        self.assertEqual(state['report']['window_schedule']['reason'], 'collecting_window')
        self.assertEqual(len(window._active_no_core_disabled), 2)
        self.assertFalse(window._active_no_core_disabled.any())


if __name__ == '__main__':
    unittest.main()

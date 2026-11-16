"""Select datasets and sampled frames without starting an expensive GPU run."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_sequence as runner


class SequenceCliTests(unittest.TestCase):
    def invoke(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, 'run_sequence') as run:
            runner.main(list(args))
        return run.call_args.args

    def test_one_default_config_selects_all_three_experiments(self):
        for drive, start, end in ((0, 2439, 2466), (2, 4914, 4934), (4, 776, 796)):
            sequence = f'2013_05_28_drive_{drive:04d}_sync'
            with self.subTest(sequence=sequence):
                cfg, frames, output = self.invoke(
                    '--sequence', sequence, '--start', str(start), '--end', str(end))
                self.assertEqual(cfg['dataset']['sequence'], sequence)
                self.assertEqual(frames, list(range(start, end+1, 2)))
                self.assertEqual(Path(output).name, f'{sequence}_{start:04d}_{frames[-1]:04d}_s2')
                self.assertEqual(cfg['object_window']['window_frames'], 7)
                self.assertEqual(cfg['object_window']['window_step'], 5)
                self.assertTrue(cfg['tracking']['motion_identity']['enabled'])

    def test_unmodified_legacy_preset_keeps_its_output(self):
        preset = HERE / 'configs/kitti360_drive0002_4914_4934.yaml'
        cfg, frames, output = self.invoke('--config', str(preset))
        self.assertEqual(frames, list(range(4914, 4935, 3)))
        self.assertEqual(output, cfg['sequence']['output_dir'])

    def test_overridden_legacy_input_gets_a_matching_output(self):
        preset = HERE / 'configs/kitti360_drive0002_4914_4934.yaml'
        for extra in (['--sequence', '2013_05_28_drive_0004_sync'],
                      ['--stride', '2'], ['--start', '10', '--end', '20'],
                      ['--frames', '10', '11', '20']):
            with self.subTest(extra=extra):
                cfg, frames, output = self.invoke('--config', str(preset), *extra)
                self.assertNotEqual(str(output), cfg['sequence']['output_dir'])
                self.assertTrue(Path(output).name.startswith(cfg['dataset']['sequence'] + '_'))
                self.assertIn(f'_{frames[0]:04d}_{frames[-1]:04d}_', Path(output).name)

    def test_explicit_output_has_priority(self):
        cfg, frames, output = self.invoke('--sequence', '2013_05_28_drive_0000_sync',
                                          '--frames', '10', '12', '19', '--output', '/tmp/my_run')
        self.assertEqual(output, Path('/tmp/my_run'))
        self.assertEqual(frames, [10, 12, 19])

    def test_irregular_frame_selections_get_distinct_stable_names(self):
        first = runner.default_sequence_output('example', [10, 12, 19])
        second = runner.default_sequence_output('example', [10, 13, 19])
        self.assertNotEqual(first, second)
        self.assertEqual(first, runner.default_sequence_output('example', [10, 12, 19]))

    def test_cli_range_replaces_config_explicit_frames(self):
        cfg = dict(sequence=dict(frames=[100, 103, 106], step=3))
        self.assertEqual(runner.sequence_frames(cfg, start=10, end=16, step=2), [10, 12, 14, 16])
        with self.assertRaisesRegex(ValueError, 'explicit frames'):
            runner.sequence_frames(cfg, step=2)

    def test_invalid_inputs_do_not_start_estimation(self):
        for args in ([], ['--start', '1'], ['--start', '1', '--end', '5', '--stride', '0'],
                     ['--frames', '1', '2', '--stride', '3'],
                     ['--frames', '1', '2', '--end', '5'], ['--frames', '-1', '2'],
                     ['--frames', '3', '2'], ['--frames', '1', '2', '--sequence', '../other'],
                     ['--frames', '1', '2', '--window-frames', '1'],
                     ['--frames', '1', '2', '--window-step', '7']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), \
                    patch.object(runner, 'run_sequence') as run, self.assertRaises(SystemExit) as error:
                runner.main(args)
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_dry_run_reports_inputs_without_starting_estimation(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), patch.object(runner, 'run_sequence') as run:
            runner.main(['--sequence', '2013_05_28_drive_0004_sync',
                         '--start', '776', '--end', '796', '--dry-run'])
        run.assert_not_called()
        self.assertIn('Frames (11): [776, 778, 780, 782, 784, 786, 788, 790, 792, 794, 796]', stdout.getvalue())
        self.assertIn('2013_05_28_drive_0004_sync_0776_0796_s2', stdout.getvalue())


if __name__ == '__main__':
    unittest.main()

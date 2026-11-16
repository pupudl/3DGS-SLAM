"""Requested image ranges must not be restricted to sparse GT pose frames."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from tools.prepare_sequence import disparity_to_depth, inspect_inputs, main


class PrepareSequenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sequence = self.root/'data_2d_raw/fixture'
        self.cfg = dict(dataset=dict(data_root=str(self.root)),
            depth=dict(source='precomputed_igev', subdir='depth_sceneflow'),
            sky_mask=dict(enabled=True, backend='precomputed_png', precomputed_subdir='sky_masks'))
        for frame in (0, 2, 4):
            for camera in (0, 1):
                path = self.sequence/f'image_0{camera}/data_rect/{frame:010d}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.ones((8, 16, 3), np.uint8))

    def test_frames_without_any_ground_truth_still_require_both_outputs(self):
        pending = inspect_inputs(self.sequence, [0, 2, 4], self.cfg)
        self.assertEqual(pending, dict(depth=[0, 2, 4], sky=[0, 2, 4], missing_images=[]))

    def test_valid_outputs_are_reused_and_wrong_sized_or_broken_outputs_are_rebuilt(self):
        depth = self.sequence/'depth_sceneflow'
        sky = self.sequence/'sky_masks'
        depth.mkdir()
        sky.mkdir()
        np.save(depth/'0000000000.npy', np.ones((8, 16), np.float32))
        np.save(depth/'0000000002.npy', np.ones((4, 8), np.float32))
        (depth/'0000000004.npy').write_bytes(b'interrupted write')
        cv2.imwrite(str(sky/'0000000000.png'), np.zeros((8, 16), np.uint8))
        cv2.imwrite(str(sky/'0000000002.png'), np.zeros((4, 8), np.uint8))
        before = (depth/'0000000000.npy').stat().st_mtime_ns
        pending = inspect_inputs(self.sequence, [0, 2, 4], self.cfg)
        self.assertEqual(pending['depth'], [2, 4])
        self.assertEqual(pending['sky'], [2, 4])
        self.assertEqual(before, (depth/'0000000000.npy').stat().st_mtime_ns)

    def test_missing_right_image_is_reported_instead_of_silently_skipping_frame(self):
        (self.sequence/'image_01/data_rect/0000000002.png').unlink()
        pending = inspect_inputs(self.sequence, [0, 2, 4], self.cfg)
        self.assertEqual([r['frame'] for r in pending['missing_images']], [2])

    def test_bad_disparities_do_not_generate_infinite_depth(self):
        depth = disparity_to_depth(np.array([[5., 0., -1., np.nan, np.inf]]), 100., .5)
        np.testing.assert_array_equal(depth, [[10., 0., 0., 0., 0.]])

    def test_check_only_neither_loads_models_nor_creates_output_directories(self):
        config = self.root/'config.yaml'
        config.write_text(yaml.safe_dump(self.cfg))
        with patch('tools.prepare_sequence.prepare_depth') as depth, patch('tools.prepare_sequence.prepare_sky') as sky:
            with contextlib.redirect_stdout(io.StringIO()):
                status = main(['--config', str(config), '--sequence', 'fixture',
                               '--start', '0', '--end', '5', '--stride', '2', '--check-only'])
        self.assertEqual(status, 2)
        depth.assert_not_called()
        sky.assert_not_called()
        self.assertFalse((self.sequence/'depth_sceneflow').exists())
        self.assertFalse((self.sequence/'sky_masks').exists())


if __name__ == '__main__':
    unittest.main()

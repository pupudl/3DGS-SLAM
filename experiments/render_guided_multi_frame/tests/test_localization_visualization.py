"""Pose direction, metric scale, gaps and optional GT in the saved-run viewer."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

EXP = Path(__file__).resolve().parents[1]
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))
from tools.visualization.visualize_localization import (
    align_camera_centers, background_motion, compose_trajectory, compare_ground_truth, visualize_localization)


def pose(x=0., z=0., angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = [[c, 0., s], [0., 1., 0.], [-s, 0., c]]
    result[:3, 3] = [x, 0., z]
    return result


class LocalizationVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.poses = [pose(), pose(1, 2, .2), pose(3, 4, .4)]
        self.pairs = [dict(first=10, second=12), dict(first=12, second=14)]
        self.motions = [(np.linalg.inv(b) @ a, 'test_background')
                        for a, b in zip(self.poses[:-1], self.poses[1:])]

    def test_static_scene_transform_accumulates_inverse_in_correct_order(self):
        segments, _ = compose_trajectory(self.pairs, self.motions)
        self.assertEqual(len(segments), 1)
        np.testing.assert_allclose(segments[0]['poses'], self.poses, atol=1e-12)

    def test_invalid_interval_splits_instead_of_connecting_or_filling_identity(self):
        pairs = self.pairs+[dict(first=14, second=16)]
        segments, rows = compose_trajectory(pairs, [self.motions[0], (None, 'invalid'), self.motions[1]])
        self.assertEqual([s['frames'] for s in segments], [[10, 12], [14, 16]])
        np.testing.assert_array_equal(segments[1]['poses'][0], np.eye(4))
        self.assertFalse(rows[1]['valid'])

    def test_arbitrary_ground_truth_world_pose_cancels_by_se3_alignment(self):
        segments, intervals = compose_trajectory(self.pairs, self.motions)
        world = pose(100, -30, .8)
        truth = {frame: world @ p for frame, p in zip([10, 12, 14], self.poses)}
        rows = compare_ground_truth(segments, intervals, truth)
        self.assertLess(max(r['position_error_m'] for r in rows), 1e-10)
        self.assertLess(max(p['translation_error_m'] for p in intervals), 1e-10)
        self.assertLess(max(p['rotation_error_deg'] for p in intervals), 1e-5)

    def test_alignment_recovers_a_global_rigid_transform_without_scaling(self):
        points = np.asarray([[0, 0, 0], [2, 0, 0], [0, 3, 0], [.4, .2, 4.]])
        transform = pose(100, -20, .9)
        reference = points @ transform[:3, :3].T+transform[:3, 3]
        np.testing.assert_allclose(align_camera_centers(points, reference), transform, atol=1e-12)

    def test_alignment_forbids_reflections(self):
        points = np.asarray([[0, 0, 0], [2, 0, 0], [0, 3, 0], [.4, .2, 4.]])
        reference = points * [-1, 1, 1]
        transform = align_camera_centers(points, reference)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.)
        residual = points @ transform[:3, :3].T+transform[:3, 3]-reference
        self.assertGreater(np.linalg.norm(residual), .1)

    def test_metric_scale_error_is_not_removed_by_alignment(self):
        segments, intervals = compose_trajectory(self.pairs, self.motions)
        truth = {}
        for frame, p in zip([10, 12, 14], self.poses):
            truth[frame] = p.copy()
            truth[frame][:3, 3] *= 2
        rows = compare_ground_truth(segments, intervals, truth)
        # Optimal translation centers the errors rather than fixing frame 10.
        points = np.asarray([p[:3, 3] for p in self.poses])
        expected_rmse = np.sqrt(np.mean(np.sum((points-points.mean(axis=0))**2, axis=1)))
        actual_rmse = np.sqrt(np.mean([r['position_error_m']**2 for r in rows]))
        self.assertAlmostEqual(actual_rmse, expected_rmse)
        self.assertGreater(rows[0]['position_error_m'], 1)
        self.assertGreater(intervals[0]['translation_error_m'], 1)

    def test_partial_ground_truth_aligns_only_common_centers_without_interpolation(self):
        segments, intervals = compose_trajectory(self.pairs, self.motions)
        rows = compare_ground_truth(segments, intervals, {12: self.poses[1], 14: self.poses[2]})
        self.assertTrue(all(r['coordinate_reference_frame'] == 12 for r in rows))
        self.assertIsNone(rows[0]['position_error_m'])
        self.assertIsNone(intervals[0]['translation_error_m'])
        self.assertLess(rows[-1]['position_error_m'], 1e-12)

    def test_explicit_object0_evidence_is_used_independent_of_final_column_order(self):
        np.savez(self.root/'object_state.npz',
            evidence_object_ids=[7, 0], evidence_H_object=[pose(50), self.motions[0][0]],
            evidence_object_model_valid=[True, True],
            object_ids=[0, 7], H_object=[pose(80), pose(90)])
        motion, source = background_motion(self.root, dict(reference=0))
        np.testing.assert_allclose(motion, self.motions[0][0])
        self.assertIn('evidence_H_object', source)

    def test_invalid_object0_does_not_fall_back_to_a_moving_reference(self):
        np.savez(self.root/'object_state.npz', evidence_object_ids=[0, 1],
                 evidence_H_object=[pose(), pose(50)], evidence_object_model_valid=[False, True])
        np.savez(self.root/'gaussian_state.npz', H=[pose(100)], local_to_object=[0])
        motion, reason = background_motion(self.root, dict(reference=1))
        self.assertIsNone(motion)
        self.assertEqual(reason, 'object0_model_invalid')

    def export(self, ground_truth=True, invalid=False):
        pairs = []
        for i, (pair, (motion, _)) in enumerate(zip(self.pairs, self.motions)):
            folder = self.root/f'pair{i}'
            folder.mkdir()
            np.savez(folder/'object_state.npz', evidence_object_ids=[0],
                     evidence_H_object=[motion], evidence_object_model_valid=[not (invalid and i == 1)])
            pairs.append(dict(pair, result_path=folder.name))
        state = dict(frames=[10, 12, 14], pairs=pairs, status='complete',
                     config=dict(dataset=dict(data_root=str(self.root/'data'), sequence='test')))
        if ground_truth:
            target = self.root/'data/data_poses/test/cam0_to_world.txt'
            target.parent.mkdir(parents=True)
            target.write_text('\n'.join(str(frame)+' '+' '.join(map(str, p.reshape(-1)))
                                        for frame, p in zip([10, 12, 14], self.poses)))
        (self.root/'README.md').write_text('# Original report\n')
        old_plot = self.root/'localization/errors.png'
        old_plot.parent.mkdir(exist_ok=True)
        old_plot.write_bytes(b'old version chart')
        result = visualize_localization(self.root, state=state)
        with Image.open(self.root/'localization/trajectory.png') as image:
            image.verify()
        self.assertEqual([p.name for p in old_plot.parent.glob('*.png')], ['trajectory.png'])
        return result

    def test_complete_export_has_zero_errors_and_png_csv_pose_artifacts(self):
        result = self.export()
        self.assertTrue(result['full_trajectory_available'])
        self.assertLess(result['ate_rmse_m'], 1e-10)
        self.assertEqual(result['ate_translation_dimensions'], 3)
        self.assertFalse(result['ground_truth_used_for_estimation'])
        with np.load(self.root/'localization/trajectory.npz') as saved:
            np.testing.assert_allclose(saved['camera_to_segment_start'], self.poses, atol=1e-12)
        self.assertEqual(len((self.root/'localization/errors.csv').read_text().splitlines()), 4)
        self.assertIn('localization/trajectory.png', (self.root/'README.md').read_text())

    def test_missing_gt_and_invalid_motion_do_not_produce_a_fake_full_score(self):
        result = self.export(ground_truth=False, invalid=True)
        self.assertFalse(result['full_trajectory_available'])
        self.assertIsNone(result['ate_rmse_m'])
        self.assertIsNone(result['relative_translation_rmse_m'])
        self.assertEqual(len(result['invalid_intervals']), 1)
        json.loads((self.root/'localization/metrics.json').read_text())

    def test_gaps_with_ground_truth_only_get_partial_segment_ate(self):
        result = self.export(invalid=True)
        self.assertIsNone(result['ate_rmse_m'])
        self.assertLess(result['ate_rmse_on_matched_segments_m'], 1e-10)
        self.assertEqual(result['matched_ground_truth_frames'], 2)


if __name__ == '__main__':
    unittest.main()

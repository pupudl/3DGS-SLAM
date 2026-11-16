"""Learned geometry/opacity rendering and streaming visualization regressions."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

from copy import deepcopy
import hashlib
import json
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ownership.joint_gaussian_visibility import nearest_gaussian_owners, projected_footprints
from tools.visualization.visualize_gaussian_map import prepare_snapshot, visualize_map


def snapshot(frame=10):
    return dict(frame=np.int64(frame), xyz=np.array([[0., 0., 3.], [0., 0., 2.]], np.float32),
                colors=np.array([[0., 1., 0.], [1., 0., 0.]], np.float32),
                log_scales=np.full((2, 3), np.log(.1), np.float32),
                rotations=np.array([[1., 0., 0., 0.]] * 2, np.float32),
                opacity_logits=np.array([[np.log(9)], [-9.]], np.float32),
                map_ids=np.array([10, 11]), object_labels=np.array([0, 1]),
                object_ids=np.array([0, 1]),
                object_to_camera=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
                K=np.array([[16., 0., 8.5], [0., 16., 8.5], [0., 0., 1.]], np.float32))


class LearnedOpacityVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.state = snapshot()
        self.xyz = self.state['xyz'][::-1]
        self.cov = np.repeat((np.eye(3) * .01)[None], 2, axis=0)

    def render(self, opacity):
        return nearest_gaussian_owners(self.xyz, self.cov, [1, 0], [11, 10],
                                       self.state['K'], (17, 17), opacity=opacity, device='cpu')

    def test_almost_transparent_near_point_does_not_hide_opaque_rear(self):
        out = self.render(np.array([.001, .9]))
        self.assertEqual(out['owner'][8, 8], 0)
        self.assertEqual(out['gaussian_ids'][8, 8], 10)
        self.assertEqual(out['metadata']['projected_gaussians'], 1)
        opaque = self.render(np.array([.9, .9]))
        self.assertEqual(opaque['owner'][8, 8], 1)

    def test_scalar_vector_and_column_opacity_have_same_footprints(self):
        shared = self.render(.9)
        for opacity in (np.full(2, .9), np.full((2, 1), .9)):
            out = self.render(opacity)
            np.testing.assert_array_equal(out['owner'], shared['owner'])
            np.testing.assert_array_equal(out['gaussian_ids'], shared['gaussian_ids'])

    def test_invalid_opacity_shapes_range_and_nonfinite_values_are_rejected(self):
        for opacity in ([.9], np.ones((2, 2)), np.ones((1, 2)),
                        [.9, float('nan')], [-.1, .9], [.9, 1.1], float('inf')):
            with self.subTest(opacity=opacity), self.assertRaises(ValueError):
                self.render(opacity)
        with self.assertRaises(ValueError):
            projected_footprints(np.empty((0, 3)), np.empty((0, 3, 3)),
                                 self.state['K'], (17, 17), opacity=-1.)


class MapSnapshotGeometryTests(unittest.TestCase):
    def test_owner_pose_rotates_learned_anisotropy_and_uses_wxyz_rotation(self):
        state = snapshot()
        state['xyz'][0] = [1., 0., 0.]
        state['rotations'][0] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
        state['log_scales'][0] = np.log([.1, .2, .3])
        state['object_to_camera'][0, :3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
        state['object_to_camera'][0, :3, 3] = [0., 0., 4.]
        original = deepcopy(state)
        batch, view, geometry = prepare_snapshot(state)
        np.testing.assert_allclose(geometry['xyz'][0], [0., 0., 3.], atol=1e-6)
        np.testing.assert_allclose(geometry['covariance'][0], np.diag([.09, .01, .04]), atol=1e-6)
        np.testing.assert_allclose(geometry['opacity'][0], .9, atol=1e-6)
        np.testing.assert_array_equal(view['rows'], [0, 1])
        np.testing.assert_array_equal(batch['map_ids'], [10, 11])
        for key in state:
            np.testing.assert_array_equal(state[key], original[key])

    def test_missing_owner_pose_excludes_archived_geometry_and_id_order_does_not_matter(self):
        state = snapshot()
        state['object_ids'] = np.array([1])
        state['object_to_camera'] = state['object_to_camera'][1:]
        _, view, geometry = prepare_snapshot(state)
        np.testing.assert_array_equal(view['rows'], [1])
        np.testing.assert_array_equal(geometry['labels'], [1])
        self.assertEqual(geometry['skipped_without_pose'], 1)
        state = snapshot()
        state['object_ids'] = state['object_ids'][::-1]
        state['object_to_camera'] = state['object_to_camera'][::-1]
        state['object_to_camera'][0, 0, 3] = 2.
        _, _, geometry = prepare_snapshot(state)
        np.testing.assert_allclose(geometry['xyz'], [[0., 0., 3.], [2., 0., 2.]])

    def test_empty_and_unknown_only_snapshots_do_not_create_object_geometry(self):
        state = snapshot()
        state['object_ids'] = np.empty(0, np.int64)
        state['object_to_camera'] = np.empty((0, 4, 4), np.float32)
        _, view, geometry = prepare_snapshot(state)
        self.assertEqual(geometry['xyz'].shape, (0, 3))
        self.assertEqual(geometry['covariance'].shape, (0, 3, 3))
        self.assertEqual(len(view['rows']), 0)
        state = snapshot()
        state['object_labels'][:] = -1
        _, view, _ = prepare_snapshot(state)
        self.assertEqual(len(view['rows']), 0)

    def test_invalid_learned_quaternion_or_nonrigid_pose_rejected(self):
        state = snapshot()
        state['rotations'][0] = 0.
        with self.assertRaises(ValueError):
            prepare_snapshot(state)
        state = snapshot()
        state['object_to_camera'][0, 0, 0] = 2.
        with self.assertRaises(ValueError):
            prepare_snapshot(state)


class MapVisualizationIntegrationTests(unittest.TestCase):
    def write_run(self, root, backend='torch_reference'):
        cfg = dict(dataset=dict(data_root=str(root / 'dataset'), sequence='fixture_sync'),
                   gaussian_map=dict(enabled=True, optimization=dict(backend=backend)))
        frames = root / 'gaussian_map' / 'frames'
        frames.mkdir(parents=True)
        images = root / 'dataset/data_2d_raw/fixture_sync/image_00/data_rect'
        images.mkdir(parents=True)
        for frame in (10, 12):
            cv2.imwrite(str(images / f'{frame:010d}.png'), np.full((17, 17, 3), 32, np.uint8))
            state = snapshot(frame)
            if frame == 12:
                state['opacity_logits'][1] = np.log(9.)
            np.savez_compressed(frames / f'frame_{frame:010d}.npz', **state)
        state = dict(config=cfg, frames=[10, 12])
        (root / 'sequence_state.json').write_text(json.dumps(state))
        return state

    def test_saved_learned_map_rgb_and_joint_masks_change_with_opacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root)
            manifest = visualize_map(root, width=32)
            output = root / 'gaussian_map/visualization'
            self.assertEqual(manifest['renderer'], 'torch_reference')
            self.assertEqual(manifest['optimized_attributes'],
                             ['xyz', 'colors', 'log_scales', 'rotations', 'opacity_logits'])
            self.assertTrue((output / 'index.html').is_file())
            self.assertEqual([item['frame'] for item in manifest['scene']['views']], [10, 12])
            with np.load(output / 'visibility/frame_0000000010.npz') as visibility:
                self.assertEqual(visibility['owner'][8, 8], 0)
                self.assertEqual(visibility['map_ids'][8, 8], 10)
                self.assertIn('render_alpha', visibility.files)
                self.assertIn('render_depth', visibility.files)
            with np.load(output / 'visibility/frame_0000000012.npz') as visibility:
                self.assertEqual(visibility['owner'][8, 8], 1)
                self.assertEqual(visibility['map_ids'][8, 8], 11)
            for item in manifest['scene']['views']:
                panel = cv2.imread(str(output / item['image']))
                self.assertIsNotNone(panel)
                self.assertTrue(np.any(panel[30:, 32:]))
            self.assertFalse((root / 'objects').exists())

    def test_requested_object_still_observes_other_objects_for_occlusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = self.write_run(root)
            manifest = visualize_map(root, state=state, width=32, object_id=0)
            self.assertEqual([item['object_id'] for item in manifest['objects']], [0])
            with np.load(root / 'gaussian_map/visualization/visibility/frame_0000000012.npz') as visibility:
                self.assertEqual(visibility['owner'][8, 8], 1)

    def test_partial_refresh_preserves_other_objects_and_global_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root)
            original = visualize_map(root, width=32)
            output = root / 'gaussian_map/visualization'
            protected = ([output / item['image'] for item in original['scene']['views']] +
                         [output / item['owner_map'] for item in original['visibility']['frames']] +
                         list((output / 'object1').glob('*.png')))
            hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
            source = root / 'gaussian_map/frames/frame_0000000010.npz'
            with np.load(source) as saved:
                changed = {key: saved[key] for key in saved.files}
            changed['colors'][0] = [0., 0., 1.]
            np.savez_compressed(source, **changed)
            updated = visualize_map(root, width=40, object_id=0)
            self.assertEqual([item['object_id'] for item in updated['objects']], [0, 1])
            self.assertEqual(updated['objects'][1], original['objects'][1])
            self.assertEqual(updated['scene'], original['scene'])
            self.assertEqual(updated['visibility'], original['visibility'])
            self.assertEqual(hashes, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected})

    def test_reference_renderer_rejects_real_image_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root)
            path = root / 'dataset/data_2d_raw/fixture_sync/image_00/data_rect/0000000010.png'
            cv2.imwrite(str(path), np.zeros((200, 200, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, 'tiny fixtures'):
                visualize_map(root)


if __name__ == '__main__':
    unittest.main()

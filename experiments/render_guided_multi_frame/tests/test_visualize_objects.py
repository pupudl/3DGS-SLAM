"""CPU integration checks for final-object RGB renders and matching masks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np


MODULE = Path(__file__).resolve().parents[1] / 'visualize_objects.py'
SPEC = importlib.util.spec_from_file_location('visualize_objects', MODULE)
visuals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(visuals)
from object_gaussian_renderer import prepare_object_gaussians


class RecordingRenderer:
    """Use real final-label/motion preparation with deterministic CPU images."""

    def __init__(self):
        self.calls = []
        self.scene_calls = []

    def render(self, gmap, object_id, image_shape, cfg, terminal=False):
        prepared = prepare_object_gaussians(gmap, object_id, terminal)
        self.calls.append(dict(object_id=object_id, terminal=terminal,
                               indices=prepared['indices'].copy(),
                               xyz=prepared['xyz'].copy(), keys=set(gmap),
                               models=gmap['H_object'].copy(),
                               object_ids=gmap['object_ids'].copy()))
        h, w = image_shape[:2]
        rgb = np.zeros((h, w, 3), np.uint8)
        silhouette = np.zeros((h, w), bool)
        if len(prepared['indices']):
            left = 12 if object_id == 0 else 34
            silhouette[8:18, left:left+12] = True
            # Object zero is deliberately black: masks must use rendered
            # opacity rather than thresholding nonblack RGB.
            rgb[silhouette] = np.rint(prepared['color'][0, ::-1]*255).astype(np.uint8)
        metadata = dict(object_id=int(object_id), terminal=bool(terminal),
                        selected_gaussians=len(prepared['indices']),
                        model_valid=bool(prepared['model_valid']), reason=prepared['reason'],
                        silhouette_pixels=int(silhouette.sum()), renderer='cpu_fixture')
        return rgb, silhouette, metadata

    def render_scene(self, sources, image_shape, cfg, object_ids=None):
        self.scene_calls.append([dict(object_ids=source['object_ids'],
                                     terminal=source['terminal']) for source in sources])
        h, w = image_shape[:2]
        owner = np.full((h, w), -2, np.int32)
        result = {}
        for source in sources:
            for oid in source['object_ids']:
                if object_ids is not None and oid not in object_ids:
                    continue
                render = self.render(source['gmap'], oid, image_shape, cfg, source['terminal'])
                owner[render[1]] = oid
                result[oid] = render
        visibility = dict(owner=owner, depth=np.full((h, w), np.inf, np.float32),
                          gaussian_ids=np.full((h, w), -1, np.int64),
                          metadata=dict(visible_object_overlap_pixels=0,
                                        all_scene_objects_considered=True))
        return result, visibility


class ObjectVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.height, self.width = 40, 64
        dataset = self.root / 'dataset'
        images = dataset / 'data_2d_raw' / 'test_sequence' / 'image_00' / 'data_rect'
        images.mkdir(parents=True)
        for frame in (1, 2, 3):
            cv2.imwrite(str(images / f'{frame:010d}.png'),
                        np.full((self.height, self.width, 3), 100 + frame, np.uint8))
        self.cfg = dict(dataset=dict(data_root=str(dataset), sequence='test_sequence'))
        K = np.array([[20., 0., 32.], [0., 20., 20.], [0., 0., 1.]])
        self.xyz = np.array([[0., 0., 2.], [.5, .2, 3.], [-.3, .1, 2.]])
        uv = self.xyz[:, :2]/self.xyz[:, 2:] * 20 + K[:2, 2]
        self.models = np.repeat(np.eye(4)[None], 3, axis=0)
        self.models[0, 1, 3] = .25  # O7, not O0, occupies model column zero.
        self.models[1, 0, 3] = .5
        self.pairs = []
        for first, second in ((1, 2), (2, 3)):
            relative = f'pairs/{first}_{second}'
            directory = self.root / relative
            directory.mkdir(parents=True)
            np.savez_compressed(
                directory / 'object_state.npz', xyz=self.xyz, uv=uv, K=K,
                color=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
                scale=np.full((3, 3), .02), H_object=self.models,
                object_ids=np.array([7, 0, 11]), object_labels=np.array([0, 7, -1]),
                object_model_valid=np.ones(3, bool),
                # Every probability argmax contradicts the accepted labels.
                object_probabilities=np.array([[.99, .01, 0.], [.01, .99, 0.], [.99, .01, 0.]]),
                q=np.array([[1., 0.]]*3), q_motion_initial=np.array([[1., 0.]]*3),
                window_frames=np.array([1, 2, 3]))
            # An unrelated legacy archive must not supply Gaussian selection.
            np.savez_compressed(directory / 'gaussian_state.npz', q=np.ones((3, 1)),
                                xyz=self.xyz, local_to_object=np.array([99]))
            self.pairs.append(dict(first=first, second=second, result_path=relative,
                                   local_to_object={'0': 99}, reference=0))
        self.state = dict(frames=[1, 2, 3], pairs=self.pairs,
                          objects={'0': {}, '7': {}, '11': {}}, config=self.cfg)
        self.state_path = self.root / 'sequence_state.json'
        self.state_path.write_text(json.dumps(self.state))

    def render(self, **kwargs):
        renderer = RecordingRenderer()
        manifest = visuals.visualize(self.root, renderer=renderer, width=self.width, **kwargs)
        return manifest, renderer

    def contents(self, overview, layout, row, column):
        caption_height = layout['tile_height'] - self.height
        y = layout['header_height'] + row*layout['tile_height'] + caption_height
        x = column*layout['tile_width']
        return overview[y:y+self.height, x:x+self.width]

    def test_exactly_two_rows_and_all_frame_columns_without_wrapping(self):
        manifest, _ = self.render(max_columns=1)
        self.assertEqual([item['object_id'] for item in manifest['objects']], [0, 7, 11])
        for item in manifest['objects']:
            self.assertEqual(item['frames'], [1, 2, 3])
            self.assertEqual(item['panel_count'], 6)
            layout = item['layout']
            self.assertEqual((layout['rows'], layout['columns']), (2, 3))
            self.assertEqual(item['overview'], f"object{item['object_id']}/overview.png")
            image = cv2.imread(str(self.root / 'objects' / item['overview']))
            self.assertEqual(image.shape,
                             (layout['header_height']+2*layout['tile_height'], 3*self.width, 3))
        self.assertEqual(len(list((self.root / 'objects').rglob('*.png'))), 3)
        self.assertFalse(list((self.root / 'objects').glob('object_*')))

    def test_black_gaussian_rgb_keeps_colored_silhouette_overlay_on_original(self):
        manifest, _ = self.render()
        item = next(item for item in manifest['objects'] if item['object_id'] == 0)
        image = cv2.imread(str(self.root / 'objects' / item['overview']))
        for column in range(3):
            source_value = 101 + column
            expected = np.full((self.height, self.width, 3), source_value, np.uint8)
            expected[8:18, 12:24] = np.rint(
                .65*np.asarray(visuals.object_color(0)) + .35*source_value).astype(np.uint8)
            np.testing.assert_array_equal(self.contents(image, item['layout'], 0, column), 0)
            np.testing.assert_array_equal(self.contents(image, item['layout'], 1, column), expected)

    def test_final_labels_select_gaussians_and_probabilities_are_not_loaded(self):
        before = {path: path.read_bytes() for path in self.root.rglob('*.npz')}
        manifest, renderer = self.render()
        for call in renderer.calls:
            expected = {0: [0], 7: [1], 11: []}[call['object_id']]
            np.testing.assert_array_equal(call['indices'], expected)
            self.assertFalse({'q', 'q_motion_initial', 'object_probabilities'} & call['keys'])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(len(list((self.root / 'objects' / 'visibility').glob('*.npz'))), 3)
        item = next(item for item in manifest['objects'] if item['object_id'] == 7)
        image = cv2.imread(str(self.root / 'objects' / item['overview']))
        expected_rgb = np.zeros((self.height, self.width, 3), np.uint8)
        expected_rgb[8:18, 34:46] = (0, 0, 255)
        for column in range(3):
            np.testing.assert_array_equal(self.contents(image, item['layout'], 0, column), expected_rgb)

    def test_terminal_frame_uses_matching_global_object_model(self):
        _, renderer = self.render()
        for oid, selected in ((0, 0), (7, 1)):
            calls = [call for call in renderer.calls if call['object_id'] == oid]
            self.assertEqual([call['terminal'] for call in calls], [False, False, True])
            column = int(np.flatnonzero(calls[-1]['object_ids'] == oid)[0])
            H = self.models[column]
            expected = self.xyz[[selected]] @ H[:3, :3].T + H[:3, 3]
            np.testing.assert_allclose(calls[-1]['xyz'], expected, atol=1e-7)

    def test_absent_object_keeps_empty_rgb_and_original_mask_for_every_frame(self):
        manifest, _ = self.render()
        item = next(item for item in manifest['objects'] if item['object_id'] == 11)
        self.assertEqual(item['frames'], [1, 2, 3])
        self.assertTrue(all(row['selected_gaussians'] == 0 for row in item['frame_details']))
        image = cv2.imread(str(self.root / 'objects' / item['overview']))
        for column in range(3):
            np.testing.assert_array_equal(self.contents(image, item['layout'], 0, column), 0)
            np.testing.assert_array_equal(self.contents(image, item['layout'], 1, column), 101+column)

    def test_disappearing_object_retains_incoming_view_then_empty_last_column(self):
        path = self.root / self.pairs[1]['result_path'] / 'object_state.npz'
        with np.load(path) as archive:
            saved = {key: archive[key] for key in archive.files}
        saved['object_ids'] = np.array([0, 11])
        saved['H_object'] = self.models[[1, 2]]
        saved['object_model_valid'] = np.ones(2, bool)
        saved['object_labels'] = np.array([0, -1, -1])
        np.savez_compressed(path, **saved)
        manifest, renderer = self.render()
        calls = [call for call in renderer.calls if call['object_id'] == 7]
        self.assertEqual([call['terminal'] for call in calls], [False, True])
        item = next(item for item in manifest['objects'] if item['object_id'] == 7)
        self.assertEqual(item['frames'], [1, 2, 3])
        image = cv2.imread(str(self.root / 'objects' / item['overview']))
        expected_overlay = np.full((self.height, self.width, 3), 102, np.uint8)
        expected_overlay[8:18, 34:46] = np.rint(
            .65*np.asarray(visuals.object_color(7)) + .35*102).astype(np.uint8)
        np.testing.assert_array_equal(self.contents(image, item['layout'], 1, 1), expected_overlay)
        np.testing.assert_array_equal(self.contents(image, item['layout'], 0, 2), 0)
        np.testing.assert_array_equal(self.contents(image, item['layout'], 1, 2), 103)

    def test_missing_terminal_motion_leaves_empty_rgb_and_original_mask(self):
        path = self.root / self.pairs[1]['result_path'] / 'object_state.npz'
        with np.load(path) as archive:
            saved = {key: archive[key] for key in archive.files}
        saved['object_model_valid'][1] = False
        np.savez_compressed(path, **saved)
        manifest, _ = self.render()
        item = next(item for item in manifest['objects'] if item['object_id'] == 0)
        self.assertFalse(item['frame_details'][-1]['model_valid'])
        image = cv2.imread(str(self.root / 'objects' / item['overview']))
        np.testing.assert_array_equal(self.contents(image, item['layout'], 0, 2), 0)
        np.testing.assert_array_equal(self.contents(image, item['layout'], 1, 2), 103)

    def test_default_output_has_simple_index_and_no_coverage_page(self):
        self.render()
        index = (self.root / 'objects' / 'index.html').read_text()
        self.assertIn('Object 高斯渲染与 mask', index)
        self.assertNotIn('coverage.html', index)
        self.assertNotIn('支持 track', index)
        self.assertFalse((self.root / 'objects' / 'coverage.html').exists())
        self.assertFalse((self.root / 'objects' / 'coverage_manifest.json').exists())

    def test_single_object_refresh_preserves_other_index_entries(self):
        self.render()
        manifest, renderer = self.render(object_id=7)
        self.assertEqual([item['object_id'] for item in manifest['objects']], [0, 7, 11])
        self.assertEqual({call['object_id'] for call in renderer.calls}, {7})
        for scene in renderer.scene_calls:
            self.assertEqual({oid for source in scene for oid in source['object_ids']}, {0, 7, 11})
        self.assertEqual(len(list((self.root / 'objects').rglob('*.png'))), 3)

    def test_visibility_owner_maps_and_reports_are_written_for_every_frame(self):
        manifest, _ = self.render()
        reports = json.loads((self.root / 'objects' / 'visibility.json').read_text())
        self.assertEqual(reports, manifest['visibility'])
        self.assertEqual([item['frame'] for item in reports['frames']], [1, 2, 3])
        for item in reports['frames']:
            self.assertEqual(item['visible_object_overlap_pixels'], 0)
            with np.load(self.root / 'objects' / item['owner_map']) as archive:
                self.assertEqual(archive['owner'].shape, (self.height, self.width))
                self.assertEqual(archive['owner'][10, 15], 0)
                self.assertEqual(archive['owner'][10, 35], 7)

    def test_overview_downsample_keeps_rgb_and_overlay_owners_exclusive(self):
        owner = np.tile(np.arange(8) % 2, (4, 1))
        displayed_rgb_masks = []
        for oid in (0, 1):
            rgb = np.zeros((4, 8, 3), np.uint8)
            rgb[owner == oid] = (50, 100, 200)
            overlay = np.full((4, 8, 3), 100, np.uint8)
            overlay[owner == oid] = (70, 130, 190)
            image, layout = visuals.make_render_overview([rgb], [overlay], [1], oid, 4)
            content_height = 2
            rgb_y = layout['header_height'] + 30
            mask_y = layout['header_height'] + layout['tile_height'] + 30
            rgb_mask = np.any(image[rgb_y:rgb_y+content_height] != 0, axis=2)
            mask_mask = np.any(image[mask_y:mask_y+content_height] != 100, axis=2)
            np.testing.assert_array_equal(rgb_mask, mask_mask)
            displayed_rgb_masks.append(rgb_mask)
        self.assertFalse(np.any(displayed_rgb_masks[0] & displayed_rgb_masks[1]))


if __name__ == '__main__':
    unittest.main()

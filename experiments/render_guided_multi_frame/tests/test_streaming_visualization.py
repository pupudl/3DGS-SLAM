"""Small CPU scenes verify bounded-frame browsing and shared visibility."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import importlib.util
import json
import tempfile
import unittest

import cv2
import numpy as np


HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('streaming_visualization_entry', HERE / 'tools/visualization/visualize_objects.py')
visuals = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(visuals)
from ownership.object_gaussian_renderer import GaussianObjectRenderer, prepare_object_gaussians


class CpuSceneRenderer(GaussianObjectRenderer):
    """Use real scene-wide CPU Z buffers with deterministic object RGB."""

    def __init__(self):
        super().__init__(device='cpu')
        self.scenes = []
        self.calls = []
        self.visibility_reports = []
        self.owner_values = []

    def render_scene(self, sources, image_shape, cfg, object_ids=None):
        self.scenes.append(dict(
            object_ids=[list(source['object_ids']) for source in sources],
            unknown=[int((source['gmap']['object_labels'] < 0).sum()) for source in sources],
            terminal=[source['terminal'] for source in sources],
            requested=object_ids))
        rendered, visibility = super().render_scene(sources, image_shape, cfg, object_ids)
        self.visibility_reports.append(dict(visibility['metadata']))
        self.owner_values.append(set(np.unique(visibility['owner'])))
        return rendered, visibility

    def render(self, gmap, object_id, image_shape, cfg, terminal=False):
        selected = prepare_object_gaussians(gmap, object_id, terminal)
        self.calls.append(dict(object_id=object_id, keys=set(gmap), terminal=terminal,
                               selected=selected['indices'].copy()))
        rgb = np.full(image_shape, 40 + object_id * 20, np.uint8)
        mask = np.ones(image_shape[:2], bool)
        return rgb, mask, dict(terminal=terminal, silhouette_pixels=int(mask.sum()))


class StreamingVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        dataset = self.root / 'dataset'
        images = dataset / 'data_2d_raw/test_sequence/image_00/data_rect'
        images.mkdir(parents=True)
        self.shape = (40, 64, 3)
        for frame in (1, 2, 3):
            cv2.imwrite(str(images / f'{frame:010d}.png'), np.full(self.shape, 100 + frame, np.uint8))
        self.cfg = dict(dataset=dict(data_root=str(dataset), sequence='test_sequence'),
                        sequence=dict(streaming=True))
        self.pairs = []
        self.xyz = np.asarray([[-2., 0., 4.], [0., 0., 4.], [.1, 0., 1.]], np.float32)
        for first, second in ((1, 2), (2, 3)):
            relative = f'pairs/{first}_{second}'
            directory = self.root / relative
            directory.mkdir(parents=True)
            np.savez_compressed(
                directory / 'object_state.npz', xyz=self.xyz,
                uv=np.zeros((3, 2)), K=np.asarray([[20., 0., 32.], [0., 20., 20.], [0., 0., 1.]]),
                color=np.ones((3, 3)) * .5, scale=np.ones((3, 3)) * .03,
                object_ids=np.asarray([0, 7]), object_labels=np.asarray([0, 7, -1]),
                gaussian_ids=np.asarray([100, 101, 102]),
                H_object=np.repeat(np.eye(4)[None], 2, axis=0), object_model_valid=np.ones(2, bool),
                object_probabilities=np.ones((3, 2)), q=np.ones((3, 2)), q_motion_initial=np.ones((3, 2)))
            (directory / 'temporal_state.json').write_text(json.dumps(dict(
                object_window=dict(lifecycle=dict(target_observed_object_ids=[0])))))
            self.pairs.append(dict(first=first, second=second, result_path=relative))
        self.state = dict(config=self.cfg, frames=[1, 2, 3], pairs=self.pairs,
                          objects={'0': dict(status='active', first_observed_frame=1, last_observed_frame=3),
                                   '7': dict(status='archived', first_observed_frame=1, last_observed_frame=2)})
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))

    def render(self, **kwargs):
        renderer = CpuSceneRenderer()
        result = visuals.visualize(self.root, renderer=renderer, width=self.shape[1], **kwargs)
        return result, renderer

    def test_each_frame_has_own_image_and_terminal_presence_is_filtered(self):
        manifest, renderer = self.render(max_columns=1)
        objects = {item['object_id']: item for item in manifest['objects']}
        self.assertEqual(manifest['format'], 'streaming_object_frames_v1')
        self.assertEqual(objects[0]['frames'], [1, 2, 3])
        self.assertEqual(objects[7]['frames'], [1, 2])
        self.assertEqual(objects[7]['status'], 'archived')
        self.assertEqual(objects[7]['last_observed_frame'], 2)
        self.assertEqual([scene['terminal'] for scene in renderer.scenes], [[False], [False], [True]])
        self.assertEqual(renderer.scenes[-1]['object_ids'], [[0]])
        self.assertFalse(list((self.root / 'objects').rglob('overview.png')))
        images = list((self.root / 'objects').rglob('*.png'))
        self.assertEqual(len(images), 5)
        for image in images:
            rendered = cv2.imread(str(image))
            self.assertEqual(rendered.shape[1], self.shape[1])
            self.assertLess(rendered.shape[0], self.shape[0] * 2 + 150)
        self.assertFalse((self.root / 'objects/index.html').exists())
        self.assertTrue(all(not {'q', 'q_motion_initial', 'object_probabilities'} & call['keys']
                            for call in renderer.calls))
        reports = [json.loads(line) for line in (self.root / 'objects/visibility.jsonl').read_text().splitlines()]
        self.assertEqual([item['frame'] for item in reports], [1, 2, 3])
        for frame in (1, 2, 3):
            with np.load(self.root / 'objects/visibility' / f'frame_{frame:010d}.npz') as archive:
                self.assertEqual(archive['owner'].shape, self.shape[:2])

    def test_single_object_keeps_other_object_and_unknown_occluders_in_scene(self):
        manifest, renderer = self.render(object_id=7)
        self.assertEqual([item['object_id'] for item in manifest['objects']], [7])
        self.assertEqual(renderer.scenes[0]['object_ids'], [[0, 7]])
        self.assertEqual(renderer.scenes[0]['unknown'], [1])
        self.assertEqual(renderer.scenes[0]['requested'], [7])
        reports = renderer.visibility_reports
        self.assertGreater(reports[0]['unknown_occluder_pixels'], 0)
        self.assertTrue(reports[0]['object_pixel_counts']['0'])
        self.assertTrue(reports[0]['object_pixel_counts']['7'])
        self.assertTrue({-2, -1, 0, 7}.issubset(renderer.owner_values[0]))
        self.assertNotIn(7, renderer.owner_values[-1])
        self.assertNotIn(-1, renderer.owner_values[-1])
        self.assertIsNone(manifest['visibility']['journal'])
        self.assertFalse((self.root / 'objects/visibility.jsonl').exists())

    def test_partial_refresh_preserves_other_entries_and_global_visibility_artifacts(self):
        self.render()
        destination = self.root / 'objects'
        # These represent an existing complete visualization from a previous
        # renderer/version. A targeted image refresh must preserve its journal.
        journal = destination / 'visibility.jsonl'
        journal.write_text(journal.read_text() + '{"complete_visibility_marker": true}\n')
        owner_path = destination / 'visibility/frame_0000000001.npz'
        with np.load(owner_path) as archive:
            fields = {key: archive[key] for key in archive.files}
        fields['complete_visibility_marker'] = np.asarray([1])
        np.savez_compressed(owner_path, **fields)
        before = {path: path.read_bytes() for path in [journal, *sorted((destination / 'visibility').glob('*.npz'))]}
        unrequested = {path: path.read_bytes() for path in (destination / 'object0').glob('*.png')}
        manifest, _ = self.render(object_id=7)
        self.assertEqual([item['object_id'] for item in manifest['objects']], [0, 7])
        for path, content in before.items():
            self.assertTrue(content == path.read_bytes(), f'Partial refresh changed {path.name}')
        self.assertEqual(unrequested, {path: path.read_bytes() for path in unrequested})

    def test_unknown_id_is_rejected_without_creating_output(self):
        with self.assertRaises(ValueError):
            self.render(object_id=-1)
        self.assertFalse((self.root / 'objects').exists())

    def test_legacy_mode_keeps_multiframe_overviews(self):
        self.state['config']['sequence']['streaming'] = False
        (self.root / 'sequence_state.json').write_text(json.dumps(self.state))
        manifest, _ = self.render()
        self.assertNotIn('format', manifest)
        for item in manifest['objects']:
            self.assertEqual(item['frames'], [1, 2, 3])
            self.assertEqual(item['layout']['columns'], 3)
            self.assertTrue((self.root / 'objects' / item['overview']).is_file())


if __name__ == '__main__':
    unittest.main()

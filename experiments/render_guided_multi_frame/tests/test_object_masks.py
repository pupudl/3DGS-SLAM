"""CPU checks for opaque global-object coverage and visibility semantics."""
from pathlib import Path
import base64
import json
import sys
import tempfile
import unittest

import cv2
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from object_masks import all_objects, isolate_object, object_color, ownership_maps
from visualize_object_coverage import encode_label_map, visualize


class ObjectCoverageTests(unittest.TestCase):
    def gaussian(self):
        return dict(xyz=np.array([[0., 0., 2.]]), uv=np.array([[2., 2.]]),
                    object_ids=np.array([0, 7]), object_probabilities=np.array([[.8, .2]]),
                    object_labels=np.array([-1]), gaussian_ids=np.array([10]),
                    H_object=np.repeat(np.eye(4)[None], 2, axis=0),
                    object_model_valid=np.array([True, True]), K=np.eye(3))

    def test_source_nearest_depth_unknown_occludes_farther_accepted_surface(self):
        g = self.gaussian()
        g.update(xyz=np.array([[0., 0., 1.123456789], [0., 0., 2.]]),
                 uv=np.array([[2., 2.], [2., 2.]]),
                 object_probabilities=np.array([[.8, .2], [.1, .9]]),
                 object_labels=np.array([-1, 7]), gaussian_ids=np.array([10, 11]))
        maps = ownership_maps(g, (5, 5, 3))
        self.assertEqual(maps['argmax'][2, 2], 0)
        self.assertEqual(maps['accepted'][2, 2], -1)
        self.assertEqual(maps['accepted'][0, 0], -2)
        self.assertAlmostEqual(maps['probability'][2, 2], .8)

    def test_terminal_projects_only_winning_motion_and_retains_unknown(self):
        g = self.gaussian()
        g['K'] = np.array([[1., 0., 2.], [0., 1., 2.], [0., 0., 1.]])
        g['H_object'][1, 0, 3] = 2.
        maps = ownership_maps(g, (5, 5, 3), terminal=True)
        self.assertEqual(maps['argmax'][2, 2], 0)
        self.assertEqual(maps['accepted'][2, 2], -1)
        self.assertEqual(maps['argmax'][2, 3], -2)  # No soft copy of object 7.
        g['object_model_valid'][0] = False
        invalid = ownership_maps(g, (5, 5, 3), terminal=True)
        self.assertTrue((invalid['argmax'] == -2).all())
        del g['H_object']
        g['H'] = np.repeat(np.eye(4)[None], 2, axis=0)
        missing = ownership_maps(g, (5, 5, 3), terminal=True)
        self.assertTrue((missing['argmax'] == -2).all())

    def test_depth_ties_are_deterministic_without_float32_rounding_loss(self):
        g = self.gaussian()
        g.update(xyz=np.array([[0., 0., 1.123456789], [0., 0., 1.123456789]]),
                 uv=np.array([[2., 2.], [2., 2.]]),
                 object_probabilities=np.array([[.8, .2], [.1, .9]]),
                 object_labels=np.array([0, 7]), gaussian_ids=np.array([20, 10]))
        g['K'] = np.array([[1., 0., 2.], [0., 1., 2.], [0., 0., 1.]])
        maps = ownership_maps(g, (5, 5, 3), terminal=True)
        self.assertEqual(maps['argmax'][2, 2], 7)
        self.assertEqual(np.count_nonzero(maps['argmax'] >= 0), 1)

    def test_legacy_many_local_motions_use_motion_of_winning_global_object(self):
        models = np.repeat(np.eye(4)[None], 3, axis=0)
        models[1, 0, 3] = 2.
        models[2, 0, 3] = -2.
        g = dict(xyz=np.array([[0., 0., 2.]]), uv=np.array([[2., 2.]]),
                 q=np.array([[.35, .40, .25]]), local_to_object=np.array([0, 7, 0]), H=models,
                 K=np.array([[1., 0., 2.], [0., 1., 2.], [0., 0., 1.]]))
        maps = ownership_maps(g, (5, 5, 3), terminal=True)
        self.assertEqual(maps['argmax'][2, 2], 0)
        self.assertEqual(maps['accepted'][2, 2], -1)
        self.assertEqual(maps['argmax'][2, 3], -2)
        self.assertAlmostEqual(maps['probability'][2, 2], .6)

    def test_palette_is_opaque_and_isolation_outline_does_not_expand_coverage(self):
        image = np.full((9, 9, 3), 255, np.uint8)
        labels = np.full((9, 9), -2)
        labels[2:7, 2:7] = 7
        labels[0, 0] = -1
        result = all_objects(image, labels)
        np.testing.assert_array_equal(result[4, 4], object_color(7))
        self.assertGreater(max(object_color(7))-min(object_color(7)), 200)
        np.testing.assert_array_equal(result[0, 0], [145, 145, 145])
        selected = isolate_object(image, labels, 7, outline=True)
        np.testing.assert_array_equal(selected[4, 4], object_color(7))
        np.testing.assert_array_equal(selected[2, 2], [255, 255, 255])
        self.assertTrue((selected[labels != 7] < 60).all())

    def test_lossless_browser_label_encoding_keeps_large_global_ids(self):
        objects = list(range(500)) + [100000]
        labels = np.array([[-2, -1, 0, 499, 100000]])
        uri = encode_label_map(labels, objects)
        png = cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        codes = png[..., 2].astype(int) + 256*png[..., 1] + 65536*png[..., 0]
        np.testing.assert_array_equal(codes, [[0, 1, 2, 501, 502]])

    def test_coverage_export_keeps_existing_index_and_writes_full_resolution_embedded_maps(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            images = root/'dataset/data_2d_raw/test/image_00/data_rect'
            images.mkdir(parents=True)
            for frame in (1, 2):
                cv2.imwrite(str(images/f'{frame:010d}.png'), np.full((5, 5, 3), 80, np.uint8))
            directory = root/'pairs/1_2'
            directory.mkdir(parents=True)
            np.savez_compressed(directory/'object_state.npz', **self.gaussian())
            state = dict(config={'dataset':{'data_root':str(root/'dataset'), 'sequence':'test'}},
                         pairs=[dict(first=1,second=2,result_path='pairs/1_2',local_to_object={'0':0,'1':7})])
            (root/'sequence_state.json').write_text(json.dumps(state))
            (root/'objects').mkdir()
            (root/'objects/index.html').write_text('original index')
            report = visualize(root, width=160)
            self.assertEqual(report['frames'], 2)
            self.assertEqual(report['object_ids'], [0, 7])
            self.assertEqual((root/'objects/index.html').read_text(), 'original index')
            html = (root/'objects/coverage.html').read_text()
            self.assertIn('"width": 5, "height": 5', html)
            self.assertIn('data:image/png;base64,', html)
            self.assertTrue((root/'object_coverage_overview.png').is_file())


if __name__ == '__main__':
    unittest.main()

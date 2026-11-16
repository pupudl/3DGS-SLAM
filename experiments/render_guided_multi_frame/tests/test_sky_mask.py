"""Sky exclusion across PNG loading, geometry, losses and displayed owners."""
import sys
from pathlib import Path

_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import tempfile
import unittest

import cv2
import numpy as np
import torch

from frontend.pair_utils import load_sky_mask, mask_sky_image, make_gaussians, lift
from frontend.joint_pair import simple_ssim
from mapping.gaussian_map_optimizer import structural_dissimilarity
from tools.visualization.visualize_objects import mask_sky_visibility


class SkyMaskTests(unittest.TestCase):
    def test_windows_png_semantics_and_nearest_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sky_masks').mkdir()
            cv2.imwrite(str(root / 'sky_masks/0000002433.png'), np.array([[0, 1], [255, 0]], np.uint8))
            mask = load_sky_mask(root, 2433, (4, 4), {'sky_mask': {'enabled': True}})
            expected = np.repeat(np.repeat([[False, True], [True, False]], 2, axis=0), 2, axis=1)
            np.testing.assert_array_equal(mask, expected)

    def test_missing_mask_is_explicit_and_disabled_mode_needs_no_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, 'Sky mask not found'):
                load_sky_mask(directory, 1, (4, 4), {'sky_mask': {'enabled': True}})
            self.assertFalse(load_sky_mask(directory, 1, (4, 4),
                {'sky_mask': {'enabled': False}}).any())
            # Old saved runs previously allowed absent masks.
            self.assertFalse(load_sky_mask(directory, 1, (4, 4), {}).any())

    def test_sky_has_no_gaussians_or_lifted_anchors_and_is_black(self):
        image = np.full((8, 8, 3), 120, np.uint8)
        sky = np.zeros((8, 8), bool)
        sky[:3] = True
        depth = np.full((8, 8), 4., np.float32)
        depth[sky] = np.nan
        masked = mask_sky_image(image, sky)
        self.assertFalse(masked[sky].any())
        self.assertTrue((image == 120).all())
        K = np.array([[10., 0., 4.], [0., 10., 4.], [0., 0., 1.]])
        gmap = make_gaussians(masked, depth, K, 1, 1., initialization='lsgslam_projective')
        uv = np.rint(gmap['uv']).astype(int)
        self.assertFalse(sky[uv[:, 1], uv[:, 0]].any())
        matches = np.array([[2, 1, 2, 4], [2, 4, 2, 1], [2, 4, 2, 4]], np.float32)
        anchors = lift(matches, depth, depth, K)
        np.testing.assert_array_equal(anchors['matches'], matches[2:])

    def test_masked_ssim_ignores_sky_values_and_sends_no_gradient_to_sky(self):
        generator = torch.Generator().manual_seed(19)
        image = torch.rand((1, 3, 12, 14), generator=generator, requires_grad=True)
        target = torch.rand((1, 3, 12, 14), generator=generator)
        valid = torch.ones((12, 14), dtype=torch.bool)
        valid[:5] = False
        poisoned = image.detach().clone()
        poisoned[:, :, ~valid] = 123.
        target_poisoned = target.clone()
        target_poisoned[:, :, ~valid] = -31.
        score = simple_ssim(image, target, valid)
        torch.testing.assert_close(score, simple_ssim(poisoned, target_poisoned, valid))
        score.backward()
        self.assertEqual(float(image.grad[:, :, ~valid].abs().sum()), 0.)
        self.assertGreater(float(image.grad[:, :, valid].abs().sum()), 0.)
        image.grad.zero_()
        value = (structural_dissimilarity(image[0], target[0], mask=valid)*valid).sum()
        other = (structural_dissimilarity(poisoned[0], target_poisoned[0], mask=valid)*valid).sum()
        torch.testing.assert_close(value, other)
        value.backward()
        self.assertEqual(float(image.grad[:, :, ~valid].abs().sum()), 0.)

    def test_projected_splats_cannot_claim_a_masked_sky_pixel_in_visualization(self):
        visibility = dict(owner=np.zeros((4, 4), np.int32), depth=np.ones((4, 4)),
                          gaussian_ids=np.ones((4, 4), np.int64),
                          metadata={'object_pixel_counts': {'0': 16}})
        sky = np.zeros((4, 4), bool)
        sky[:2] = True
        mask_sky_visibility(visibility, sky)
        self.assertTrue((visibility['owner'][sky] == -2).all())
        self.assertTrue(np.isinf(visibility['depth'][sky]).all())
        self.assertTrue((visibility['gaussian_ids'][sky] == -1).all())
        self.assertEqual(visibility['metadata']['object_pixel_counts']['0'], 8)


if __name__ == '__main__':
    unittest.main()

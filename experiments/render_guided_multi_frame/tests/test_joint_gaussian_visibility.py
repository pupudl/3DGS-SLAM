"""Nearest-effective-splat visibility and native CUDA footprint checks."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from joint_gaussian_visibility import nearest_gaussian_owners, projected_footprints
from object_gaussian_renderer import GaussianObjectRenderer, prepare_object_gaussians


class JointVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[32., 0., 16.5], [0., 32., 16.5], [0., 0., 1.]])

    def owners(self, xyz, labels, scales=None, gids=None, **options):
        xyz = np.asarray(xyz)
        scales = np.full(len(xyz), .1) if scales is None else np.asarray(scales)
        cov = np.eye(3)[None] * scales[:, None, None] ** 2
        return nearest_gaussian_owners(
            xyz, cov, labels, np.arange(len(xyz)) if gids is None else gids,
            self.K, (33, 33), device='cpu', **options)

    def test_joint_camera_z_selects_nearest_gaussian_not_object_average(self):
        # O1 has a nearer centre plus a far centre, so its mean depth exceeds O0.
        out = self.owners([[0, 0, 2], [0, 0, 20], [0, 0, 3]], [1, 1, 0])
        self.assertEqual(out['owner'][16, 16], 1)
        self.assertEqual(out['depth'][16, 16], 2)
        self.assertEqual(out['metadata']['overlap_pixels'], 0)

    def test_weak_near_gaussian_tail_does_not_hide_far_surface(self):
        out = self.owners([[.15, 0, 2], [0, 0, 3]], [1, 0], scales=[.015, .1])
        self.assertEqual(out['owner'][16, 16], 0)
        self.assertEqual(out['owner'][16, 18], 1)

    def test_unknown_nearest_splat_occludes_without_inventing_background(self):
        out = self.owners([[0, 0, 2], [0, 0, 3]], [-1, 0])
        self.assertEqual(out['owner'][16, 16], -1)
        self.assertGreater(out['metadata']['unknown_occluder_pixels'], 0)
        self.assertEqual(out['owner'][0, 0], -2)

    def test_equal_depth_tie_is_stable_gaussian_id_not_input_order(self):
        out = self.owners([[0, 0, 2], [0, 0, 2]], [7, 3], gids=[100, 8])
        reversed_out = self.owners([[0, 0, 2], [0, 0, 2]], [3, 7], gids=[8, 100])
        np.testing.assert_array_equal(out['owner'], reversed_out['owner'])
        self.assertEqual(out['owner'][16, 16], 3)

    def test_chunk_budget_including_single_large_splat_preserves_result(self):
        args = ([[0, 0, 2], [.3, .1, 3]], [1, 0])
        small = self.owners(*args, scales=[1., .5], max_candidate_pixels=17)
        large = self.owners(*args, scales=[1., .5], max_candidate_pixels=10000)
        np.testing.assert_array_equal(small['owner'], large['owner'])
        np.testing.assert_array_equal(small['gaussian_ids'], large['gaussian_ids'])

    def test_native_near_clip_and_empty_scene(self):
        out = self.owners([[0, 0, .1]], [1])
        self.assertTrue(np.all(out['owner'] == -2))
        empty = nearest_gaussian_owners(np.empty((0, 3)), np.empty((0, 3, 3)), [], [],
                                       self.K, (33, 33), device='cpu')
        self.assertTrue(np.all(empty['owner'] == -2))

    def test_anisotropic_footprint_rotates_on_image(self):
        cov = np.diag([.2**2, .02**2, .02**2])[None]
        out = nearest_gaussian_owners([[0, 0, 2]], cov, [1], [10], self.K,
                                      (33, 33), device='cpu')
        mask = out['owner'] == 1
        self.assertGreater(mask[16].sum(), mask[:, 16].sum())
        cov = cov[:, [1, 0, 2]][:, :, [1, 0, 2]]
        rotated = nearest_gaussian_owners([[0, 0, 2]], cov, [1], [10], self.K,
                                          (33, 33), device='cpu')
        np.testing.assert_array_equal(mask.T, rotated['owner'] == 1)

    def test_scene_subset_still_observes_other_objects_and_unknown_occluders(self):
        class CpuFixtureRenderer(GaussianObjectRenderer):
            def render(inner, gmap, oid, image_shape, cfg, terminal=False):
                prepared = prepare_object_gaussians(gmap, oid, terminal)
                cov = np.eye(3)[None] * prepared['scale'][:, :, None] ** 2
                visibility = nearest_gaussian_owners(
                    prepared['xyz'], cov, np.full(len(prepared['xyz']), oid),
                    np.arange(len(prepared['xyz'])), gmap['K'], image_shape, device='cpu')
                mask = visibility['owner'] == oid
                rgb = np.zeros((*image_shape[:2], 3), np.uint8)
                rgb[mask] = 255
                return rgb, mask, dict(terminal=terminal)

        renderer = CpuFixtureRenderer(device='cpu')
        gmap = dict(xyz=np.array([[0., 0., 3.], [0., 0., 2.], [.3, 0., 1.5]]),
                    color=np.ones((3, 3)), scale=np.full(3, .1), K=self.K,
                    object_ids=np.array([0, 1]), object_labels=np.array([0, 1, -1]),
                    gaussian_ids=np.array([10, 11, 12]))
        original = {key: value.copy() for key, value in gmap.items()}
        source = dict(gmap=gmap, terminal=False, object_ids=[0, 1])
        outputs, visibility = renderer.render_scene([source], (33, 33, 3), {}, [0])
        self.assertEqual(set(outputs), {0})
        self.assertEqual(visibility['owner'][16, 16], 1)
        self.assertFalse(outputs[0][1][16, 16])
        self.assertTrue(np.any(visibility['owner'] == -1))
        self.assertGreater(visibility['metadata']['independent_object_overlap_pixels'], 0)
        self.assertEqual(visibility['metadata']['visible_object_overlap_pixels'], 0)
        for key, value in original.items():
            np.testing.assert_array_equal(gmap[key], value)


@unittest.skipUnless(torch.cuda.is_available(), 'Requires native CUDA Gaussian rasterizer')
class JointVisibilityCudaTests(unittest.TestCase):
    def scene(self):
        return dict(xyz=np.array([[0., 0., 2.], [0., 0., 3.], [.5, 0., 2.5]]),
                    color=np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                    scale=np.array([[.1]*3, [.15]*3, [.04]*3]),
                    K=np.array([[32., 0., 16.5], [0., 32., 16.5], [0., 0., 1.]]),
                    gaussian_ids=np.array([20, 30, 40]), object_ids=np.array([0, 1]),
                    object_labels=np.array([1, 0, -1]),
                    H_object=np.repeat(np.eye(4)[None], 2, 0),
                    object_model_valid=np.ones(2, bool))

    def test_single_native_cuda_silhouette_matches_analytic_footprint(self):
        gmap = self.scene()
        gmap['object_labels'] = np.array([1, -1, -1])
        renderer = GaussianObjectRenderer()
        _, mask, _ = renderer.render(gmap, 1, (33, 33, 3), {})
        expected = nearest_gaussian_owners(gmap['xyz'][:1], np.eye(3)[None] * .01,
                                          [1], [20], gmap['K'], (33, 33), device='cuda')
        np.testing.assert_array_equal(mask, expected['owner'] == 1)

    def test_joint_native_rgb_masks_and_subset_request_share_same_visibility(self):
        renderer, gmap = GaussianObjectRenderer(), self.scene()
        source = dict(gmap=gmap, terminal=False, object_ids=[0, 1])
        full, visibility = renderer.render_scene([source], (33, 33, 3), {})
        subset, subset_visibility = renderer.render_scene([source], (33, 33, 3), {}, [0])
        self.assertFalse(np.any(full[0][1] & full[1][1]))
        self.assertTrue(full[1][1][16, 16])
        self.assertFalse(full[0][1][16, 16])
        np.testing.assert_array_equal(full[0][1], subset[0][1])
        np.testing.assert_array_equal(visibility['owner'], subset_visibility['owner'])
        for rgb, mask, _ in full.values():
            self.assertFalse(np.any(rgb[~mask]))
        self.assertGreater(visibility['metadata']['independent_object_overlap_pixels'], 0)
        self.assertEqual(visibility['metadata']['visible_object_overlap_pixels'], 0)
        self.assertGreater(visibility['metadata']['unknown_source_gaussians'], 0)

    def test_native_offaxis_rotated_anisotropic_covariance_footprint(self):
        renderer, gmap = GaussianObjectRenderer(), self.scene()
        angle = .6
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        covariance = rotation @ np.diag([.08**2, .025**2, .03**2]) @ rotation.T
        for position in ([.55, .3, 2.], [1.6, 0., 2.]):
            with self.subTest(position=position):
                local_covariance = covariance if position[0] < 1 else covariance * 100
                gmap['xyz'][0] = position
                gmap['object_labels'] = np.array([1, -1, -1])
                gmap['cov'] = np.repeat(local_covariance[None], 3, 0)
                _, mask, _ = renderer.render(gmap, 1, (33, 33, 3), {})
                expected = nearest_gaussian_owners(
                    gmap['xyz'][:1], local_covariance[None], [1], [20],
                    gmap['K'], (33, 33), device='cuda')
                self.assertTrue(mask.any())
                np.testing.assert_array_equal(mask, expected['owner'] == 1)

    def test_terminal_unknown_motion_is_not_fabricated(self):
        renderer, gmap = GaussianObjectRenderer(), self.scene()
        source = dict(gmap=gmap, terminal=True, object_ids=[0, 1])
        _, visibility = renderer.render_scene([source], (33, 33, 3), {})
        self.assertEqual(visibility['metadata']['unknown_source_gaussians'], 0)
        self.assertEqual(visibility['metadata']['terminal_unknown_gaussians_without_motion'], 1)


if __name__ == '__main__':
    unittest.main()

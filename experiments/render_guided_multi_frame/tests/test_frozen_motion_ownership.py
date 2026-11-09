"""Exercise the actual pose/ownership optimizer on CPU without rasterization."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

# Load native dependencies before restoring sys.modules around renderer stubs;
# importing OpenCV twice in one process is unsupported by some installations.
import cv2
import numpy as np
import scipy.spatial
import torch


def load_pairwise_model():
    raster = types.ModuleType("diff_gaussian_rasterization")
    raster.GaussianRasterizer = None
    helpers = types.ModuleType("utils.recon_helpers")
    helpers.setup_camera = None
    path = (Path(__file__).resolve().parents[1] /
            "render_guided_two_frame" / "joint_pair.py")
    spec = importlib.util.spec_from_file_location("frozen_ownership_joint_pair", path)
    module = importlib.util.module_from_spec(spec)
    # These dependencies are used only by the CUDA renderer, not optimize().
    with patch.dict("sys.modules", {"diff_gaussian_rasterization": raster,
                                   "utils.recon_helpers": helpers}):
        spec.loader.exec_module(module)
    return module.PairwiseModel


PairwiseModel = load_pairwise_model()


def load_warm_ownership():
    pair_utils = types.ModuleType("pair_utils")
    for name in ("add_unique", "discover", "errors", "unexplained_clusters"):
        setattr(pair_utils, name, None)
    path = (Path(__file__).resolve().parents[1] /
            "render_guided_two_frame" / "masked_feedback.py")
    spec = importlib.util.spec_from_file_location("frozen_ownership_feedback", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", {"pair_utils": pair_utils}):
        spec.loader.exec_module(module)
    return module.warm_ownership


warm_ownership = load_warm_ownership()


def cpu_model(config):
    model = PairwiseModel.__new__(PairwiseModel)
    model.config = dict(config)
    model.base_H = torch.eye(4).repeat(2, 1, 1)
    model.xyz = torch.tensor([[0., 0., 2.], [.2, 0., 3.], [0., .3, 4.]])
    model.anchors0 = model.xyz.clone()
    model.anchors1 = model.xyz + torch.tensor([.4, 0., 0.])
    model.q_initial = torch.tensor([[.9, .1], [.25, .75], [1., 0.]])
    observed_q = []

    def loss(H, q, downsample):
        observed_q.append(q.detach().clone())
        xyz = (torch.einsum("kij,nj->kni", H[:, :3, :3], model.xyz) +
               H[:, None, :3, 3]).reshape(-1, 3)
        xyz.retain_grad()
        model._last_transformed_xyz = xyz
        pose = (xyz-model.anchors1.repeat(2, 1)).square().mean()
        desired = q.new_tensor([[.1, .9], [.9, .1], [.1, .9]])
        ownership = (q-desired).square().mean()
        return pose + ownership, dict(pose=pose, ownership=ownership)

    model.loss = loss
    return model, observed_q


class FrozenMotionOwnershipTests(unittest.TestCase):
    def test_feedback_refresh_preserves_frozen_probabilities_without_aliasing(self):
        initial = np.array([[.1, .9], [.8, .2]], np.float32)
        cfg = dict(joint=dict(freeze_motion_ownership=True))
        result = warm_ownership(dict(q=initial), [None, None], None, None,
                                None, None, cfg)
        np.testing.assert_array_equal(result, initial)
        self.assertFalse(np.shares_memory(result, initial))

    def test_feedback_rejects_changing_frozen_motion_count(self):
        cfg = dict(joint=dict(freeze_motion_ownership=True))
        for count in (1, 3):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "motion count"):
                warm_ownership(dict(q=np.ones((2, 2))), [None]*count, None,
                               None, None, None, cfg)

    def test_frozen_probabilities_survive_pose_updates_and_temperature_schedule(self):
        model, observed_q = cpu_model(dict(freeze_motion_ownership=True, tau_final=.05))
        initial = model.q_initial.clone()
        result = model.optimize(iterations=8, downsample=1, pose_lr=.03, q_lr=10.)
        torch.testing.assert_close(result["q"], initial, rtol=0, atol=0)
        torch.testing.assert_close(model.q_initial, initial, rtol=0, atol=0)
        for probability in observed_q:
            torch.testing.assert_close(probability, initial, rtol=0, atol=0)
        self.assertGreater(float(result["H"][:, 0, 3].mean()), .05)
        self.assertLess(result["history"][-1]["pose"], result["history"][0]["pose"])
        self.assertTrue(np.isfinite(result["grad_q"]).all())
        self.assertGreater(result["grad_q"][0], 0)
        self.assertTrue(np.isfinite(result["grad_motion"]).all())

    def test_zero_iterations_do_not_sharpen_frozen_initialization(self):
        model, _ = cpu_model(dict(freeze_motion_ownership=True, tau_final=.01))
        initial = model.q_initial.clone()
        result = model.optimize(iterations=0, downsample=1, pose_lr=.03, q_lr=.5)
        torch.testing.assert_close(result["q"], initial, rtol=0, atol=0)
        self.assertEqual(result["history"], [])
        self.assertGreater(result["grad_q"][0], 0)

    def test_legacy_default_still_optimizes_motion_ownership(self):
        model, observed_q = cpu_model(dict(tau_final=1.))
        initial = model.q_initial.clone()
        result = model.optimize(iterations=8, downsample=1, pose_lr=.03, q_lr=.4)
        self.assertGreater(float(result["q"][0, 1]), float(initial[0, 1])+.3)
        self.assertLess(result["history"][-1]["ownership"],
                        result["history"][0]["ownership"])
        self.assertFalse(torch.equal(observed_q[-1], initial))
        torch.testing.assert_close(model.q_initial, initial, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

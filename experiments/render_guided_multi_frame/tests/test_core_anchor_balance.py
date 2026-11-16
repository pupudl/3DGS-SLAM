"""A large object's extra anchors must not erase a small object's gradient."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ownership.object_render_optimizer import core_anchor_loss


class CoreAnchorBalanceTests(unittest.TestCase):
    def values(self, repeats):
        # Repeat identical evidence for object 1, retaining one anchor for 2.
        logits = torch.tensor([[0., 1., 0.], [0., 0., .5]], requires_grad=True)
        logp = logits.log_softmax(1)
        rows = torch.tensor([0]*repeats+[1])
        columns = torch.tensor([1]*repeats+[2])
        result = core_anchor_loss(logp, rows, columns, torch.ones(len(rows)), 'objects')
        result.backward()
        return result.detach(), logits.grad

    def test_duplicate_large_object_evidence_preserves_small_object_loss_and_gradient(self):
        one, gradient_one = self.values(1)
        many, gradient_many = self.values(100)
        torch.testing.assert_close(one, many)
        torch.testing.assert_close(gradient_one, gradient_many)

    def test_object_column_permutation_preserves_loss(self):
        logp = torch.tensor([[.1, .3, .6], [.1, .8, .1]]).log()
        rows, columns = torch.tensor([0, 1]), torch.tensor([2, 1])
        a = core_anchor_loss(logp, rows, columns, torch.ones(2), 'objects')
        b = core_anchor_loss(logp[:, [2, 0, 1]], rows, torch.tensor([0, 2]), torch.ones(2), 'objects')
        torch.testing.assert_close(a, b)

    def test_empty_anchor_loss_is_differentiable_zero(self):
        logp = torch.zeros((2, 3), requires_grad=True)
        empty = torch.empty(0, dtype=torch.long)
        value = core_anchor_loss(logp, empty, empty, torch.empty(0), 'objects')
        value.backward()
        torch.testing.assert_close(logp.grad, torch.zeros_like(logp))


if __name__ == '__main__':
    unittest.main()

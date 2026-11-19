import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from weak_data import WeakSalObjDataset
from weak_loss import partial_bce_with_logits


class PartialBCETests(unittest.TestCase):
    def test_only_valid_pixels_contribute(self):
        logits = torch.tensor([[[[0.0, 10.0], [-10.0, 2.0]]]], requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        valid = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = partial_bce_with_logits(logits, target, valid)
        expected = F.binary_cross_entropy_with_logits(
            logits.detach()[valid.bool()], target[valid.bool()])
        self.assertTrue(torch.allclose(loss.detach(), expected))
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 0, 1].item(), 0.0)
        self.assertEqual(logits.grad[0, 0, 1, 0].item(), 0.0)

    def test_empty_mask_has_zero_gradient(self):
        logits = torch.randn(1, 1, 2, 2, requires_grad=True)
        loss = partial_bce_with_logits(logits, torch.zeros_like(logits),
                                       torch.zeros_like(logits))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_ignored_target_can_be_undefined(self):
        logits = torch.zeros(1, 1, 1, 2, requires_grad=True)
        target = torch.tensor([[[[1.0, float("nan")]]]])
        valid = torch.tensor([[[[1.0, 0.0]]]])
        loss = partial_bce_with_logits(logits, target, valid)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertEqual(logits.grad[0, 0, 0, 1].item(), 0.0)


class DatasetTests(unittest.TestCase):
    def test_triplet_and_nearest_neighbor_masks(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            Image.new("RGB", (2, 2), "red").save(root / "image.png")
            Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(root / "target.png")
            Image.fromarray(np.array([[255, 0], [0, 255]], dtype=np.uint8)).save(root / "valid.png")
            sample = WeakSalObjDataset([(root / "image.png", root / "target.png",
                                         root / "valid.png")], 4)[0]
            self.assertEqual(set(sample), {"image", "target", "valid_mask"})
            self.assertEqual(tuple(sample["image"].shape), (3, 4, 4))
            self.assertEqual(sample["target"].sum().item(), 4.0)
            self.assertEqual(sample["valid_mask"].sum().item(), 8.0)

    def test_rejects_unknown_encoding(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            Image.new("RGB", (2, 2)).save(root / "image.png")
            Image.new("L", (2, 2), 128).save(root / "target.png")
            Image.new("L", (2, 2), 255).save(root / "valid.png")
            with self.assertRaises(ValueError):
                WeakSalObjDataset([(root / "image.png", root / "target.png",
                                    root / "valid.png")], 4)[0]


if __name__ == "__main__":
    unittest.main()

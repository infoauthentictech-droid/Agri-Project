"""Unit tests for GPU image loading.

Run:  python -m unittest discover -s ml/data/tests -v
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from gpu_loader import GpuBatchLoader, classifier_transform, cyclegan_transform, decode_file


@unittest.skipUnless(torch.cuda.is_available(), "GPU only")
class GpuLoaderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.samples = []
        for index in range(10):
            # Different sizes per image, as in the real datasets.
            path = self.root / f"{index}.jpg"
            array = np.full((60 + index * 4, 80, 3), index * 20, dtype=np.uint8)
            Image.fromarray(array).save(path, quality=95)
            self.samples.append((str(path), index % 2))
        self.png = self.root / "plain.png"
        Image.fromarray(np.full((40, 40, 3), 200, dtype=np.uint8)).save(self.png)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_batches_are_on_the_gpu_with_the_right_shape(self):
        loader = GpuBatchLoader(self.samples, 4, classifier_transform(32, train=True), shuffle=True)
        self.assertEqual(len(loader), 3)
        inputs, labels = next(iter(loader))
        self.assertEqual(inputs.shape, (4, 3, 32, 32))
        self.assertEqual(inputs.device.type, "cuda")
        self.assertEqual(labels.device.type, "cuda")

    def test_every_sample_appears_exactly_once_with_its_label(self):
        loader = GpuBatchLoader(self.samples, 3, classifier_transform(32, train=False), shuffle=False)
        batches = list(loader)
        self.assertEqual(sum(len(y) for _, y in batches), 10)
        self.assertEqual(torch.cat([y for _, y in batches]).tolist(), [s[1] for s in self.samples])

    def test_shuffle_keeps_each_label_with_its_image(self):
        # Brightness identifies each image, so a mismatch would show up as a wrong label.
        expected = {index * 20 // 10: index % 2 for index in range(10)}
        transform = classifier_transform(16, train=False)
        loader = GpuBatchLoader(self.samples, 3, lambda image: transform(image) * 0 + image.float().mean(), shuffle=True)
        for inputs, labels in loader:
            for value, label in zip(inputs[:, 0, 0, 0].tolist(), labels.tolist()):
                self.assertEqual(expected[round(value / 10)], label)

    def test_validation_transform_applies_imagenet_normalization(self):
        loader = GpuBatchLoader(self.samples[:1], 1, classifier_transform(16, train=False), shuffle=False)
        inputs, _ = next(iter(loader))
        self.assertAlmostEqual(inputs[0, 0].mean().item(), (0 - 0.485) / 0.229, delta=0.05)

    def test_cyclegan_transform_scales_to_minus_one_one(self):
        loader = GpuBatchLoader(self.samples[:1], 1, cyclegan_transform(16), shuffle=False)
        inputs, _ = next(iter(loader))
        self.assertAlmostEqual(inputs.mean().item(), -1.0, delta=0.05)

    def test_unsupported_file_in_a_batch_falls_back_and_is_counted(self):
        samples = self.samples[:2] + [(str(self.png), 1)]
        loader = GpuBatchLoader(samples, 3, classifier_transform(16, train=False), shuffle=False)
        inputs, _ = next(iter(loader))
        self.assertEqual(inputs.shape[0], 3)
        self.assertEqual(loader.cpu_decoded, 1)
        self.assertEqual(inputs.device.type, "cuda")

    def test_decode_file_returns_rgb_on_the_gpu(self):
        path = self.root / "red.jpg"
        Image.fromarray(np.dstack([np.full((32, 32), 220, np.uint8), np.zeros((32, 32), np.uint8),
                                   np.zeros((32, 32), np.uint8)])).save(path, quality=95)
        image, on_gpu = decode_file(path)
        self.assertTrue(on_gpu)
        self.assertEqual(image.device.type, "cuda")
        self.assertGreater(image[0].float().mean(), image[1].float().mean() + 100)


if __name__ == "__main__":
    unittest.main()

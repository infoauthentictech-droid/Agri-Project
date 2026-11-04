"""Unit tests for the Groundnut AI augmentation pipeline (stdlib unittest).

Run:  python -m unittest discover -s ml/augmentation/tests -v
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.pipeline import AugmentationPipeline, load_config
from src.transforms import GeometryState, op_flip_h, op_rotate

CONFIGS = ROOT / "configs"


def _img(h=64, w=48, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def _tiny_cfg(**overrides):
    cfg = {"name": "test", "apply_order": [], "transforms": {}}
    cfg.update(overrides)
    return cfg


class LoadConfigTests(unittest.TestCase):
    def test_provided_configs_parse(self):
        for name in ("none", "standard", "field_condition"):
            cfg = load_config(CONFIGS / f"{name}.json")
            self.assertIn("apply_order", cfg)

    def test_no_vertical_flip_in_provided_configs(self):
        for name in ("standard", "field_condition"):
            cfg = load_config(CONFIGS / f"{name}.json")
            v = cfg["transforms"].get("v_flip", {})
            self.assertFalse(v.get("enabled", False),
                             f"{name} enables vertical flip (unrealistic)")


class PipelineTests(unittest.TestCase):
    def test_determinism(self):
        cfg = load_config(CONFIGS / "field_condition.json")
        img = _img()
        p1 = AugmentationPipeline(cfg, seed=7)
        p2 = AugmentationPipeline(cfg, seed=7)
        np.testing.assert_array_equal(p1.apply(img), p2.apply(img))

    def test_seed_change_changes_output(self):
        cfg = load_config(CONFIGS / "field_condition.json")
        img = _img()
        a = AugmentationPipeline(cfg, seed=1).apply(img)
        b = AugmentationPipeline(cfg, seed=2).apply(img)
        self.assertFalse(np.array_equal(a, b))

    def test_no_op_config_returns_identical(self):
        cfg = load_config(CONFIGS / "none.json")
        img = _img()
        out = AugmentationPipeline(cfg, seed=0).apply(img)
        np.testing.assert_array_equal(out, img)

    def test_shape_preserved_all_configs(self):
        img = _img()
        for name in ("none", "standard", "field_condition"):
            cfg = load_config(CONFIGS / f"{name}.json")
            out = AugmentationPipeline(cfg, seed=3).apply(img)
            self.assertEqual(out.shape, img.shape, name)
            self.assertEqual(out.dtype, np.uint8)

    def test_standard_pipeline_describes(self):
        cfg = load_config(CONFIGS / "standard.json")
        p = AugmentationPipeline(cfg, seed=1)
        self.assertIn("standard", p.describe())


class SyncTests(unittest.TestCase):
    def _hflip_pipeline(self):
        cfg = _tiny_cfg(
            apply_order=["h_flip"],
            transforms={"h_flip": {"enabled": True, "p": 1.0, "params": {}}},
        )
        return AugmentationPipeline(cfg, seed=0)

    def test_hflip_image_and_boxes_sync(self):
        img = _img()
        boxes = np.array([[0.25, 0.5, 0.2, 0.3]])
        p = self._hflip_pipeline()
        out, boxes_out = p.apply_with_boxes(img, boxes)
        np.testing.assert_array_equal(out, op_flip_h(img))
        self.assertAlmostEqual(boxes_out[0, 0], 0.75, places=6)  # x mirrored
        self.assertAlmostEqual(boxes_out[0, 1], 0.5, places=6)   # y unchanged

    def test_geometry_mask_replay_matches_image(self):
        # A class-index mask (0 bg, 1 fg) rendered as colors; after geometric
        # ops the augmented image should agree with the replayed mask.
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 15:55] = 1
        color = np.where(mask[..., None] == 1, np.array([200, 30, 30]),
                         np.array([10, 10, 10])).astype(np.uint8)

        cfg = load_config(CONFIGS / "standard.json")
        p = AugmentationPipeline(cfg, seed=5)
        aug_color = p.apply(color)
        aug_mask = p.geometry.apply_to_mask(mask)
        self.assertEqual(aug_mask.shape, mask.shape)
        # Where mask == fg, image should be red-ish; where bg, dark.
        fg = aug_mask == 1
        self.assertGreater(fg.sum(), 500)  # foreground preserved (identity-safe)
        if fg.any():
            red = aug_color[fg][:, 0].mean()
            green = aug_color[fg][:, 1].mean()
            blue = aug_color[fg][:, 2].mean()
            self.assertGreater(red, green)
            self.assertGreater(red, blue)

    def test_geometry_empty_for_noop(self):
        p = AugmentationPipeline(_tiny_cfg(), seed=0)
        p.apply(_img())
        self.assertTrue(p.geometry.empty)


class GeometryStateTests(unittest.TestCase):
    def test_manual_rotation_mask_replay(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[20:44, 20:44] = (0, 200, 0)  # green square
        rng = np.random.default_rng(0)
        rotated, op = op_rotate(img, rng, (12.0, 12.0), (64, 64))
        self.assertEqual(rotated.shape, img.shape)
        g = GeometryState()
        g.append(op)
        mask = np.where(img[..., 1] > 100, 1, 0).astype(np.uint8)
        mask_out = g.apply_to_mask(mask)
        self.assertEqual(mask_out.shape, mask.shape)
        # green square must still be present (rotated)
        self.assertGreater(int(mask_out.sum()), 200)


if __name__ == "__main__":
    unittest.main()

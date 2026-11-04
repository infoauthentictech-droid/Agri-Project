"""Unit tests for the four-dataset generator's label and image transforms.

Run:  python -m unittest discover -s ml/data/tests -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from build_all_datasets import (
    RECIPES,
    Sample,
    check_leakage,
    clip_polygon,
    recipe_geometry,
    transform_box_line,
    transform_image,
    transform_points,
    transform_polygon_line,
)

GEOMETRIES = [("hflip",), ("vflip",), ("rot90ccw",), ("rot90cw",), ("rot180",), ("zoom", 0.1, 0.05, 0.8)]


def _sample(stem, split, base=None, group="g", source="s"):
    return Sample(stem, Path(f"{stem}.jpg"), split, source, base or stem, group)


class GeometryTests(unittest.TestCase):
    def test_labels_follow_the_pixels_for_every_edit(self):
        # A bright block on a dark, non-square image: its box must land where the pixels land.
        height, width = 300, 400
        image = torch.zeros((3, height, width), dtype=torch.uint8)
        image[:, 60:120, 200:280] = 255
        box = f"0 {240 / width:.6f} {90 / height:.6f} {80 / width:.6f} {60 / height:.6f}"
        for geometry in GEOMETRIES:
            with self.subTest(geometry=geometry[0]):
                moved = transform_image(image, geometry)
                new_h, new_w = moved.shape[1:]
                _, cx, cy, w, h = (float(v) for v in transform_box_line(box, geometry).split())
                ys, xs = np.nonzero(moved[0].numpy() > 127)
                self.assertAlmostEqual((xs.min() + xs.max() + 1) / 2 / new_w, cx, delta=0.01)
                self.assertAlmostEqual((ys.min() + ys.max() + 1) / 2 / new_h, cy, delta=0.01)
                self.assertAlmostEqual((xs.max() + 1 - xs.min()) / new_w, w, delta=0.02)
                self.assertAlmostEqual((ys.max() + 1 - ys.min()) / new_h, h, delta=0.02)

    def test_points_stay_in_frame_after_lossless_edits(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.3, 0.7]])
        for geometry in GEOMETRIES[:5]:
            moved = transform_points(points, geometry)
            self.assertTrue(((moved >= 0) & (moved <= 1)).all())


class ZoomClippingTests(unittest.TestCase):
    def test_box_mostly_out_of_frame_is_dropped(self):
        self.assertIsNone(transform_box_line("1 0.02 0.02 0.04 0.04", ("zoom", 0.2, 0.2, 0.8)))

    def test_box_partly_out_of_frame_is_clipped(self):
        line = transform_box_line("1 0.25 0.5 0.2 0.2", ("zoom", 0.2, 0.1, 0.8))
        _, cx, _, w, _ = (float(v) for v in line.split())
        self.assertAlmostEqual(cx - w / 2, 0.0, places=5)

    def test_polygon_is_clipped_to_the_frame(self):
        clipped = clip_polygon(np.array([[-0.5, 0.2], [0.5, 0.2], [0.5, 0.8], [-0.5, 0.8]]))
        self.assertTrue((clipped[:, 0] >= 0).all())
        self.assertAlmostEqual(clipped[:, 0].max(), 0.5)

    def test_polygon_outside_the_frame_is_dropped(self):
        self.assertIsNone(transform_polygon_line("0 0.01 0.01 0.05 0.01 0.05 0.05", ("zoom", 0.2, 0.2, 0.8)))

    def test_every_recipe_keeps_zoom_inside_the_image(self):
        for recipe in RECIPES:
            geometry = recipe_geometry(recipe, random.Random(recipe))
            if geometry[0] == "zoom":
                _, x0, y0, side = geometry
                self.assertLessEqual(x0 + side, 1.0)
                self.assertLessEqual(y0 + side, 1.0)


class LeakageCheckTests(unittest.TestCase):
    def test_flags_a_photo_in_two_splits(self):
        errors = check_leakage("d", [_sample("a", "train", group="g1"), _sample("a__aug", "test", base="a", group="g2")])
        self.assertTrue(any("photo a" in e for e in errors))

    def test_flags_a_group_in_two_splits(self):
        errors = check_leakage("d", [_sample("a", "train"), _sample("b", "val")])
        self.assertTrue(any("group" in e for e in errors))

    def test_clean_split_passes(self):
        self.assertEqual(check_leakage("d", [_sample("a", "train"), _sample("a__aug", "train", base="a")]), [])


if __name__ == "__main__":
    unittest.main()

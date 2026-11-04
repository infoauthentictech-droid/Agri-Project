"""Unit tests for the Groundnut AI preprocessing pipeline (stdlib unittest).

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
from PIL import Image

from src.common import file_fingerprint, read_json, write_json
from src.duplicates import cross_dataset_overlap, group_duplicates
from src.image_utils import copy_image, inspect_image, normalize_image, parse_size, resize_image
from src.labels_seg import inspect_mask
from src.labels_yolo import parse_yolo_label
from src.splits import make_split


class BaseTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="ga_pipeline_test_")
        self.root = Path(self._tmp)

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)

    def make_image(self, name, size=(32, 24), color=(120, 40, 10), fmt="JPEG"):
        p = self.root / name
        Image.new("RGB", size, color).save(p, format=fmt)
        return p


class ImageValidationTests(BaseTestCase):
    def test_valid_image(self):
        info = inspect_image(self.make_image("ok.jpg"))
        self.assertTrue(info.valid)
        self.assertEqual((info.width, info.height), (32, 24))
        self.assertEqual(info.mode, "RGB")

    def test_corrupted_image(self):
        p = self.root / "bad.jpg"
        p.write_bytes(b"this is definitely not a valid jpeg file")
        info = inspect_image(p)
        self.assertFalse(info.valid)
        self.assertIsNotNone(info.error)

    def test_parse_size_and_normalize(self):
        self.assertEqual(parse_size("512x512"), (512, 512))
        arr = normalize_image(Image.new("RGB", (4, 4), (255, 255, 255)))
        self.assertEqual(arr.shape, (4, 4, 3))
        self.assertAlmostEqual(float(arr[0, 0, 0]), 1.0, places=5)

    def test_resize_preserves_aspect(self):
        src = self.make_image("src.jpg", (100, 50))
        dst = self.root / "out" / "resized.jpg"
        resize_image(src, dst, (50, 50), keep_aspect=True)
        info = inspect_image(dst)
        self.assertTrue(info.valid)
        self.assertEqual((info.width, info.height), (50, 25))

    def test_copy_image(self):
        src = self.make_image("src2.jpg")
        dst = self.root / "nested" / "copy.jpg"
        info = copy_image(src, dst)
        self.assertTrue(info.valid)
        self.assertTrue(dst.exists())
        self.assertEqual(src.read_bytes(), dst.read_bytes())


class DuplicateTests(BaseTestCase):
    def test_duplicates_detected(self):
        a = self.make_image("a.jpg")
        b = self.root / "b.jpg"
        b.write_bytes(a.read_bytes())  # byte-identical
        c = self.make_image("c.jpg", color=(1, 2, 3))
        groups = group_duplicates([a, b, c])
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(next(iter(groups.values()))), 2)

    def test_cross_dataset_overlap(self):
        a = self.make_image("shared.jpg")
        fp = file_fingerprint(a)
        overlap = cross_dataset_overlap(
            {"ds1": {str(a): fp}, "ds2": {str(self.root / "other.jpg"): fp}}
        )
        self.assertEqual(len(overlap), 1)
        self.assertEqual(len(overlap[0]["locations"]), 2)



class YoloLabelTests(BaseTestCase):
    def test_valid_label(self):
        p = self.root / "ok.txt"
        p.write_text("1 0.5 0.5 0.2 0.2\n0 0.1 0.1 0.3 0.3\n", encoding="utf-8")
        res = parse_yolo_label(p, num_classes=2)
        self.assertTrue(res.valid)
        self.assertEqual(res.box_count, 2)
        self.assertEqual(dict(res.classes), {0: 1, 1: 1})

    def test_wrong_field_count(self):
        p = self.root / "bad.txt"
        p.write_text("1 0.5 0.5 0.2\n", encoding="utf-8")
        res = parse_yolo_label(p, num_classes=2)
        self.assertFalse(res.valid)
        self.assertTrue(any("5 fields" in e for e in res.errors))

    def test_class_range_and_coords(self):
        p = self.root / "bad2.txt"
        p.write_text("5 0.5 0.5 0.2 0.2\n0 1.5 0.5 0.2 0.2\n0 0.5 0.5 -1 0.2\n", encoding="utf-8")
        res = parse_yolo_label(p, num_classes=2)
        self.assertFalse(res.valid)
        joined = " ".join(res.errors)
        self.assertIn("class id 5 out of range", joined)
        self.assertIn("outside [0,1]", joined)
        self.assertIn("non-positive", joined)

    def test_empty_label(self):
        p = self.root / "empty.txt"
        p.write_text("", encoding="utf-8")
        res = parse_yolo_label(p, num_classes=2)
        self.assertTrue(res.empty)
        self.assertFalse(res.valid)


class MaskTests(BaseTestCase):
    def test_mask_with_crop_and_weed(self):
        img = self.make_image("m.jpg", (64, 48))
        mask = self.root / "m.png"
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[0:20, 0:30] = (0, 200, 0)    # crop (green)
        arr[20:40, 30:60] = (200, 0, 0)  # weed (red)
        Image.fromarray(arr).save(mask)
        res = inspect_mask(img, mask)
        self.assertTrue(res.valid)
        self.assertTrue(res.size_match)
        self.assertIn("crop", res.class_presence)
        self.assertIn("weed", res.class_presence)

    def test_empty_mask(self):
        img = self.make_image("m2.jpg", (64, 48))
        mask = self.root / "m2.png"
        Image.new("RGB", (64, 48), (0, 0, 0)).save(mask)
        res = inspect_mask(img, mask)
        self.assertTrue(res.empty)


class SplitTests(BaseTestCase):
    def test_deterministic(self):
        items = [f"img_{i:03d}" for i in range(100)]
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        s1 = make_split(items, ratios, seed=42)
        s2 = make_split(items, ratios, seed=42)
        self.assertEqual(s1, s2)
        self.assertEqual(sum(len(v) for v in s1.values()), 100)

    def test_dup_groups_stay_together(self):
        items = ["a1", "a2", "b1", "c1", "c2", "c3"]
        dup = {"a1": "gA", "a2": "gA", "c1": "gC", "c2": "gC", "c3": "gC"}
        s = make_split(items, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1, dup_groups=dup)
        bucket_of = {v: k for k, vals in s.items() for v in vals}
        for grp in (("a1", "a2"), ("c1", "c2", "c3")):
            buckets = {bucket_of[g] for g in grp}
            self.assertEqual(len(buckets), 1, f"group {grp} split across buckets")

    def test_json_roundtrip(self):
        p = self.root / "x.json"
        write_json(p, {"a": [1, 2, 3]})
        self.assertEqual(read_json(p), {"a": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
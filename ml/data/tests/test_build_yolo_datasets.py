"""Unit tests for the YOLO dataset builder's conversion and splitting logic.

Run:  python -m unittest discover -s ml/data/tests -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from build_yolo_datasets import (
    Item,
    assign_groups,
    close_mask,
    dsc_group,
    link,
    manual_label_paths,
    mask_to_polygons,
    place_field_items,
    polygon_line,
    rasterize,
    ratio_matches,
    sam_agrees,
    vegetation_mask,
    weak_labels,
)


def _item(name, group, stratum="s"):
    return Item(name, Path(f"{name}.jpg"), "src", group, stratum, "test")


class PolygonConversionTests(unittest.TestCase):
    def test_round_trip_keeps_the_mask(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[20:60, 30:120] = True
        polygons = mask_to_polygons(mask)
        self.assertEqual(len(polygons), 1)
        line = polygon_line(1, polygons[0], 200, 100)
        self.assertTrue(line.startswith("1 "))
        drawn = rasterize([line], 1, 200, 100)
        self.assertGreater((drawn & mask).sum() / (drawn | mask).sum(), 0.95)

    def test_rasterize_only_draws_the_requested_class(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:50, 10:50] = True
        line = polygon_line(0, mask_to_polygons(mask)[0], 100, 100)
        self.assertFalse(rasterize([line], 1, 100, 100).any())

    def test_small_regions_are_dropped(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[5:8, 5:8] = True
        self.assertEqual(mask_to_polygons(mask, min_area=100), [])


@unittest.skipUnless(torch.cuda.is_available(), "GPU only")
class GpuMaskTests(unittest.TestCase):
    def test_vegetation_mask_matches_the_opencv_method(self):
        rng = np.random.default_rng(0)
        image = np.clip(rng.normal((60, 90, 120), 12, (400, 600, 3)), 0, 255).astype(np.uint8)
        cv2.circle(image, (300, 200), 80, (40, 180, 40), -1)
        cv2.rectangle(image, (0, 0), (60, 60), (40, 180, 40), -1)  # plant touching the frame edge

        b, g, r = cv2.split(image.astype(np.float32))
        norm = cv2.normalize(2 * g - r - b, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, expected = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        expected = cv2.morphologyEx(expected, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
        expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

        mask = vegetation_mask(image)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater((mask == expected).mean(), 0.995)

    def test_closing_joins_nearby_parts_only(self):
        mask = np.zeros((100, 300), dtype=bool)
        mask[40:60, 20:50] = True
        mask[40:60, 58:90] = True  # 8 px gap: same plant
        mask[40:60, 200:240] = True  # far away: another plant
        count, _ = cv2.connectedComponents(close_mask(mask, 15, 1), connectivity=8)
        self.assertEqual(count - 1, 2)


class SamAgreementTests(unittest.TestCase):
    def setUp(self):
        self.colour = np.zeros((100, 100), dtype=bool)
        self.colour[20:80, 20:80] = True

    def test_close_masks_are_accepted(self):
        sam = np.zeros((100, 100), dtype=bool)
        sam[18:82, 18:82] = True
        self.assertTrue(sam_agrees(sam, self.colour))

    def test_result_can_be_counted_and_saved_in_the_report(self):
        # A numpy result makes the running total a numpy integer, which JSON refuses.
        accepted = 0
        accepted += sam_agrees(self.colour.copy(), self.colour)
        self.assertEqual(json.dumps({"accepted": accepted}), '{"accepted": 1}')

    def test_whole_frame_is_rejected(self):
        self.assertFalse(sam_agrees(np.ones((100, 100), dtype=bool), self.colour))

    def test_soil_instead_of_the_plant_is_rejected(self):
        self.assertFalse(sam_agrees(~self.colour, self.colour))

    def test_single_leaf_of_the_plant_is_rejected(self):
        sam = np.zeros((100, 100), dtype=bool)
        sam[20:40, 20:40] = True
        self.assertFalse(sam_agrees(sam, self.colour))


class LinkTests(unittest.TestCase):
    def test_jpeg_with_trailing_data_is_copied_so_a_resave_cannot_reach_the_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "phone.jpg", Path(tmp) / "out" / "phone.jpg"
            src.write_bytes(b"\xff\xd8image\xff\xd9extra data from the phone")
            link(src, dst)
            self.assertEqual(os.stat(src).st_nlink, 1)
            dst.write_bytes(b"re-saved")
            self.assertTrue(src.read_bytes().endswith(b"from the phone"))

    def test_normal_jpeg_is_hard_linked_to_save_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "a.jpg", Path(tmp) / "out" / "a.jpg"
            src.write_bytes(b"\xff\xd8image\xff\xd9")
            link(src, dst)
            self.assertEqual(os.stat(src).st_nlink, 2)


class FieldPlacementTests(unittest.TestCase):
    def _items(self, prefix, groups, split=""):
        return [Item(f"{prefix}{i}", Path("x.jpg"), prefix, g, prefix, "test", split=split) for i, g in enumerate(groups)]

    def test_machine_labels_only_ever_train(self):
        field = self._items("field", [f"g{i}" for i in range(40)])
        machine = self._items("machine", [f"m{i}" for i in range(10)])
        _, kept = place_field_items([], field, machine, seed=1)
        self.assertEqual({item.split for item in kept}, {"train"})
        self.assertEqual(len(kept), 10)

    def test_machine_label_sharing_a_photo_with_a_held_out_label_is_dropped(self):
        mixed = self._items("mixed", ["shared_with_mixed"], split="test")
        field = self._items("field", [f"g{i}" for i in range(40)])
        held_out = next(g for g in [f"g{i}" for i in range(40)]
                        if place_field_items(mixed, self._items("field", [f"g{i}" for i in range(40)]), [], seed=1)[0][int(g[1:])].split != "train")
        machine = self._items("machine", ["shared_with_mixed", held_out, "unique"])
        _, kept = place_field_items(mixed, field, machine, seed=1)
        self.assertEqual([item.group for item in kept], ["unique"])

    def test_human_label_sharing_a_photo_with_a_mixed_frame_joins_test(self):
        mixed = self._items("mixed", ["same"], split="test")
        field = self._items("field", ["same"] + [f"g{i}" for i in range(20)])
        placed, _ = place_field_items(mixed, field, [], seed=1)
        self.assertEqual(placed[0].split, "test")

    def test_field_groups_stay_whole(self):
        field = self._items("field", ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"])
        placed, _ = place_field_items([], field, [], seed=3)
        for group in "abcde":
            self.assertEqual(len({i.split for i in placed if i.group == group}), 1)


class ManualLabelTests(unittest.TestCase):
    def test_labels_sit_beside_the_image_folder(self):
        image = Path("data/batch_1/IMG 1.jpg")
        boxes, polygons = manual_label_paths(image)
        self.assertEqual(boxes, Path("data/batch_1/labels/IMG 1.txt"))
        self.assertEqual(polygons, Path("data/batch_1/labels_seg/IMG 1.txt"))

    def test_output_root_redirects_labels_for_testing(self):
        boxes, _ = manual_label_paths(Path("data/batch_1/a.jpg"), Path("tmp"))
        self.assertEqual(boxes, Path("tmp/batch_1/labels/a.txt"))


class PlantMaskTests(unittest.TestCase):
    def _plant_on_soil(self):
        image = np.full((400, 600, 3), (60, 90, 120), dtype=np.uint8)  # brown soil, BGR
        cv2.rectangle(image, (200, 120), (300, 280), (40, 180, 40), -1)  # main leaf
        cv2.rectangle(image, (340, 180), (400, 240), (40, 180, 40), -1)  # detached part
        return image

    def test_box_covers_every_part_of_the_plant(self):
        boxes, polygons = weak_labels(self._plant_on_soil(), 1, 1)
        _, cx, cy, w, h = (float(v) for v in boxes[0].split())
        x0, x1 = (cx - w / 2) * 600, (cx + w / 2) * 600
        y0, y1 = (cy - h / 2) * 400, (cy + h / 2) * 400
        self.assertLessEqual(x0, 205)
        self.assertGreaterEqual(x1, 395)
        self.assertLessEqual(y0, 125)
        self.assertGreaterEqual(y1, 275)
        self.assertGreaterEqual(len(polygons), 1)

    def test_image_without_vegetation_gives_no_labels(self):
        rng = np.random.default_rng(0)
        soil = np.clip(rng.normal((60, 90, 120), 2, (400, 600, 3)), 0, 255).astype(np.uint8)
        labels = weak_labels(soil, 0, 0)
        if labels is not None:
            # Otsu always splits something; noise must at least never produce a plant-sized box.
            _, _, _, w, h = (float(v) for v in labels[0][0].split())
            self.assertLess(w * h, 0.5)


class RatioMatchTests(unittest.TestCase):
    def _bf_count(self, query, train):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        return sum(1 for m in matcher.knnMatch(query, train, k=2)
                   if len(m) == 2 and m[0].distance < 0.75 * m[1].distance)

    def test_agrees_with_opencv_on_unequal_sized_batches(self):
        rng = np.random.default_rng(0)
        trains = [rng.integers(0, 256, (n, 32), dtype=np.uint8) for n in (300, 120)]
        queries = []
        for train, n in zip(trains, (200, 80)):
            query = rng.integers(0, 256, (n, 32), dtype=np.uint8)
            query[:40] = train[:40]  # planted exact matches
            queries.append(query)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = ratio_matches(queries, trains, device)
        for query, train, good in zip(queries, trains, results):
            self.assertEqual(len(good), self._bf_count(query, train))
            self.assertGreaterEqual(len(good), 40)
            for q, t in good[:40]:
                if q < 40:
                    self.assertEqual(q, t)

    def test_padding_is_never_matched(self):
        rng = np.random.default_rng(1)
        short = rng.integers(0, 256, (10, 32), dtype=np.uint8)
        long = rng.integers(0, 256, (200, 32), dtype=np.uint8)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = ratio_matches([long, long], [short, long], device)
        self.assertTrue(all(t < 10 for _, t in results[0]))


class SplitTests(unittest.TestCase):
    def test_groups_never_cross_splits(self):
        items = [_item(f"i{g}_{n}", f"g{g}") for g in range(50) for n in range(4)]
        assign_groups(items, seed=42)
        by_group = {}
        for item in items:
            by_group.setdefault(item.group, set()).add(item.split)
        self.assertTrue(all(len(splits) == 1 for splits in by_group.values()))
        counts = {s: sum(i.split == s for i in items) for s in ("train", "val", "test")}
        self.assertGreater(counts["train"], counts["val"])
        self.assertGreater(counts["val"], 0)
        self.assertGreater(counts["test"], 0)

    def test_group_spanning_two_strata_stays_whole(self):
        items = []
        for g in range(30):
            items.append(_item(f"a{g}", f"g{g}", stratum="stage_a"))
            items.append(_item(f"b{g}", f"g{g}", stratum="stage_b"))
        for seed in range(5):
            assign_groups(items, seed=seed)
            for g in range(30):
                self.assertEqual(len({i.split for i in items if i.group == f"g{g}"}), 1)

    def test_uneven_group_sizes_still_reach_the_target_ratios(self):
        rng = np.random.default_rng(3)
        items = [_item(f"i{g}_{n}", f"g{g}") for g in range(120) for n in range(int(rng.integers(1, 60)))]
        assign_groups(items, seed=42)
        for split, target in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            share = sum(i.split == split for i in items) / len(items)
            self.assertAlmostEqual(share, target, delta=0.03)

    def test_assignment_is_reproducible(self):
        first = [_item(f"i{g}", f"g{g}") for g in range(40)]
        second = [_item(f"i{g}", f"g{g}") for g in range(40)]
        assign_groups(first, seed=7)
        assign_groups(second, seed=7)
        self.assertEqual([i.split for i in first], [i.split for i in second])

    def test_dsc_suffix_variants_share_a_burst_group(self):
        # DSC043031 is a re-saved copy of DSC04303, so both must stay together.
        self.assertEqual(dsc_group("DSC043031"), dsc_group("DSC04303"))
        self.assertNotEqual(dsc_group("DSC04303"), dsc_group("DSC09000"))


if __name__ == "__main__":
    unittest.main()

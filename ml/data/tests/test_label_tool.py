"""Unit tests for the click-to-label tool's label writing.

Run:  python -m unittest discover -s ml/data/tests -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_tool import ToolState, label_lines, validate_objects

SQUARE = [0.2, 0.2, 0.4, 0.2, 0.4, 0.4, 0.2, 0.4]


class ValidationTests(unittest.TestCase):
    def test_accepts_a_good_object(self):
        self.assertEqual(validate_objects([{"cls": 1, "polygons": [SQUARE]}])[0]["cls"], 1)

    def test_rejects_an_unmarked_or_unknown_class(self):
        for cls in (None, 2, "weed"):
            with self.assertRaises(ValueError):
                validate_objects([{"cls": cls, "polygons": [SQUARE]}])

    def test_rejects_a_polygon_with_too_few_points(self):
        with self.assertRaises(ValueError):
            validate_objects([{"cls": 0, "polygons": [[0.1, 0.1, 0.2, 0.2]]}])

    def test_rejects_an_object_without_polygons(self):
        with self.assertRaises(ValueError):
            validate_objects([{"cls": 0, "polygons": []}])

    def test_clamps_coordinates_into_the_frame(self):
        clean = validate_objects([{"cls": 0, "polygons": [[-0.5, 0.1, 1.5, 0.1, 0.5, 0.9]]}])
        self.assertEqual(clean[0]["polygons"][0][:3], [0.0, 0.1, 1.0])

    def test_no_plants_is_a_valid_answer(self):
        self.assertEqual(validate_objects([]), [])


class LabelLineTests(unittest.TestCase):
    def test_box_encloses_the_polygon(self):
        boxes, polygons = label_lines([{"cls": 1, "polygons": [SQUARE]}])
        self.assertEqual(boxes, ["1 0.300000 0.300000 0.200000 0.200000"])
        self.assertTrue(polygons[0].startswith("1 0.200000 0.200000"))

    def test_object_with_two_parts_gets_one_box_and_two_polygons(self):
        other = [0.6, 0.6, 0.8, 0.6, 0.8, 0.8]
        boxes, polygons = label_lines([{"cls": 0, "polygons": [SQUARE, other]}])
        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(polygons), 2)
        self.assertEqual(boxes[0], "0 0.500000 0.500000 0.600000 0.600000")

    def test_reopening_a_saved_image_keeps_each_plants_parts_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "batch" / "a.jpg"
            state = ToolState.__new__(ToolState)
            state.out_root = None
            state.images = [image]
            objects = [{"cls": 1, "polygons": [SQUARE, [0.6, 0.6, 0.8, 0.6, 0.8, 0.8]]}]
            self.assertFalse(state.done(0))
            state.save(0, objects)
            self.assertTrue(state.done(0))
            self.assertEqual(state.saved_objects(0), objects)
            first_boxes = state.paths(0)[0].read_text(encoding="utf-8")
            state.save(0, state.saved_objects(0))
            self.assertEqual(state.paths(0)[0].read_text(encoding="utf-8"), first_boxes)
            self.assertEqual(len(first_boxes.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()

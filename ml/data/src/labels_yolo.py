"""YOLO-format annotation parsing and validation.

Validates: field count, numeric class id in range, normalized coordinates in
[0, 1], positive width/height, empty files, and image-label correspondence
(the latter is handled by the dataset validator).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class YoloLabelResult:
    path: str
    valid: bool = True
    line_count: int = 0
    box_count: int = 0
    empty: bool = False
    classes: Counter = field(default_factory=Counter)
    errors: List[str] = field(default_factory=list)


def parse_yolo_label(path: Path, num_classes: int) -> YoloLabelResult:
    """Parse one YOLO .txt label file and validate every line."""
    res = YoloLabelResult(path=str(path))
    if not path.exists():
        res.valid = False
        res.errors.append("file not found")
        return res

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    res.line_count = len(lines)

    if not lines:
        res.empty = True
        res.valid = False
        res.errors.append("empty annotation file")
        return res

    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) != 5:
            res.valid = False
            res.errors.append(f"line {i + 1}: expected 5 fields, got {len(parts)}: {parts!r}")
            continue
        try:
            cid = int(float(parts[0]))
            x, y, w, h = (float(p) for p in parts[1:5])
        except ValueError as exc:
            res.valid = False
            res.errors.append(f"line {i + 1}: non-numeric values {parts!r} ({exc})")
            continue
        if not (0 <= cid < num_classes):
            res.valid = False
            res.errors.append(f"line {i + 1}: class id {cid} out of range [0, {num_classes})")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            res.valid = False
            res.errors.append(f"line {i + 1}: coordinates outside [0,1]: {parts[1:]!r}")
        if w <= 0 or h <= 0:
            res.valid = False
            res.errors.append(f"line {i + 1}: non-positive width/height")
        res.classes[cid] += 1
        res.box_count += 1

    if res.box_count == 0 and not res.errors:
        res.valid = False
        res.errors.append("no valid boxes parsed")
    return res

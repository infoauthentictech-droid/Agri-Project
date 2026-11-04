"""Click-to-label tool for the own field photos that have no per-object labels.

Colour outlines every plant it can find. You click an outline to mark it crop or weed, or click
a plant with no outline and SAM 2.1 draws one on the GPU. Saving writes YOLO box labels to
``labels/`` and polygon labels to ``labels_seg/`` beside each image folder, and
``build_yolo_datasets.py`` picks them up on its next run.

Runs only on this computer (127.0.0.1). Usage, from the project root:
    python ml/data/label_tool.py
Then open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "ml" / "data"))

from build_yolo_datasets import (  # noqa: E402
    FIELD_DIR,
    MIN_COMPONENT_AREA,
    MIXED_DIR,
    PEANUT_IMAGES,
    PLANT_JOIN_KERNEL,
    SAM_WEIGHTS,
    close_mask,
    manual_label_paths,
    mask_to_polygons,
    read_bgr_gpu,
    sha256,
    vegetation_mask,
)

PAGE = Path(__file__).with_name("label_tool.html")
MAX_PROPOSALS = 80
MIN_PROPOSAL_FRACTION = 0.0003  # of the frame
FINE_CLOSE_KERNEL = 5  # px; only fills tiny gaps, unlike the close-up labels' heavy smoothing
MIN_PART_AREA = 60  # px^2; smallest leaf fragment worth keeping in an outline
CLASS_NAMES = {0: "groundnut", 1: "weed"}


def list_images() -> List[Path]:
    """Mixed frames, then wide field shots, without exact copies of each other or of public images."""
    seen = {sha256(p) for p in PEANUT_IMAGES.glob("*.jpg")}
    images = sorted(MIXED_DIR.glob("*.jpeg"))
    for batch in sorted(p for p in FIELD_DIR.iterdir() if p.is_dir()):
        images += sorted(p for p in batch.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    kept = []
    for path in images:
        digest = sha256(path)
        if digest not in seen:
            seen.add(digest)
            kept.append(path)
    return kept


def polygon_to_list(polygon: np.ndarray, width: int, height: int) -> List[float]:
    return [round(float(v), 5) for point in polygon for v in (point[0] / width, point[1] / height)]


def propose_objects(path: Path) -> List[List[List[float]]]:
    """Outline vegetation by colour, largest plants first, with no class assigned.

    Nearby leaves are grouped into one plant, but the outlines keep following the fine mask,
    so a saved label does not include the soil between leaves.
    """
    bgr = read_bgr_gpu(path)
    height, width = bgr.shape[:2]
    fine = vegetation_mask(bgr, close_kernel=FINE_CLOSE_KERNEL, close_iterations=1) > 0
    count, groups, stats, _ = cv2.connectedComponentsWithStats(close_mask(fine, PLANT_JOIN_KERNEL, 1), connectivity=8)
    min_area = max(MIN_COMPONENT_AREA, MIN_PROPOSAL_FRACTION * width * height)
    largest_first = sorted(range(1, count), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    objects = []
    for i in largest_first:
        region = (groups == i) & fine
        if region.sum() < min_area:
            continue
        polygons = mask_to_polygons(region, min_area=MIN_PART_AREA, epsilon=1.5)
        if polygons:
            objects.append([polygon_to_list(p, width, height) for p in polygons])
        if len(objects) == MAX_PROPOSALS:
            break
    return objects


def validate_objects(objects: object) -> List[Dict]:
    """Reject anything that would write a broken label file."""
    if not isinstance(objects, list):
        raise ValueError("objects must be a list")
    clean = []
    for obj in objects:
        cls = obj["cls"]
        if cls not in CLASS_NAMES:
            raise ValueError(f"bad class {cls!r}")
        polygons = []
        for polygon in obj["polygons"]:
            values = [min(max(float(v), 0.0), 1.0) for v in polygon]
            if len(values) < 6 or len(values) % 2:
                raise ValueError("a polygon needs at least 3 points")
            polygons.append(values)
        if not polygons:
            raise ValueError("an object needs a polygon")
        clean.append({"cls": cls, "polygons": polygons})
    return clean


def label_lines(objects: List[Dict]) -> tuple:
    """Return (box lines, polygon lines) in YOLO format. One box per object."""
    boxes, polygons = [], []
    for obj in objects:
        points = np.array([v for polygon in obj["polygons"] for v in polygon]).reshape(-1, 2)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        boxes.append(f"{obj['cls']} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        polygons += [f"{obj['cls']} " + " ".join(f"{v:.6f}" for v in polygon) for polygon in obj["polygons"]]
    return boxes, polygons


def objects_path(polygons_path: Path) -> Path:
    """YOLO polygon files cannot say which parts form one plant, so that is kept alongside them."""
    return polygons_path.with_suffix(".objects.json")


class ToolState:
    def __init__(self, out_root: Optional[Path]) -> None:
        self.out_root = out_root
        self.images = list_images()
        self.sam = None
        self.gpu_lock = threading.Lock()

    def paths(self, index: int) -> tuple:
        return manual_label_paths(self.images[index], self.out_root)

    def done(self, index: int) -> bool:
        return objects_path(self.paths(index)[1]).exists()

    def save(self, index: int, objects: List[Dict]) -> None:
        boxes_path, polygons_path = self.paths(index)
        boxes, polygons = label_lines(objects)
        for path, lines in ((boxes_path, boxes), (polygons_path, polygons)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        objects_path(polygons_path).write_text(json.dumps(objects), encoding="utf-8")

    def saved_objects(self, index: int) -> List[Dict]:
        return json.loads(objects_path(self.paths(index)[1]).read_text(encoding="utf-8"))

    def segment_at(self, index: int, x: float, y: float) -> List[List[float]]:
        """Ask SAM for the object under a normalized click."""
        if self.sam is None:
            from ultralytics import SAM
            self.sam = SAM(str(SAM_WEIGHTS))
        bgr = read_bgr_gpu(self.images[index])
        height, width = bgr.shape[:2]
        result = self.sam(bgr.cpu().numpy(), points=[[x * width, y * height]], labels=[1], device=0, verbose=False)[0]
        if result.masks is None:
            return []
        mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        polygons = mask_to_polygons(mask.astype(bool), min_area=MIN_COMPONENT_AREA / 4, epsilon=2.0)
        return [polygon_to_list(p, width, height) for p in polygons]


def make_handler(state: ToolState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _send(self, body: bytes, content_type: str = "application/json", status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, status: int = 200) -> None:
            self._send(json.dumps(payload).encode("utf-8"), status=status)

        def _index(self, text: str) -> int:
            index = int(text)
            if not 0 <= index < len(state.images):
                raise ValueError("no such image")
            return index

        def do_GET(self) -> None:
            try:
                if self.path == "/":
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif self.path == "/api/images":
                    self._json([{"index": i, "name": p.name, "folder": p.parent.name, "done": state.done(i)}
                                for i, p in enumerate(state.images)])
                elif match := re.fullmatch(r"/image/(\d+)", self.path):
                    self._send(state.images[self._index(match[1])].read_bytes(), "image/jpeg")
                elif match := re.fullmatch(r"/api/objects/(\d+)", self.path):
                    index = self._index(match[1])
                    if state.done(index):
                        self._json({"saved": True, "objects": state.saved_objects(index)})
                    else:
                        with state.gpu_lock:
                            proposals = propose_objects(state.images[index])
                        self._json({"saved": False, "objects": [{"cls": None, "polygons": obj} for obj in proposals]})
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if match := re.fullmatch(r"/api/save/(\d+)", self.path):
                    state.save(self._index(match[1]), validate_objects(body["objects"]))
                    self._json({"ok": True})
                elif match := re.fullmatch(r"/api/segment/(\d+)", self.path):
                    with state.gpu_lock:
                        polygons = state.segment_at(self._index(match[1]), float(body["x"]), float(body["y"]))
                    self._json({"polygons": polygons})
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self._json({"error": str(exc)}, 400)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out-root", default=None, help="save labels here instead of beside the images (for testing)")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA GPU available. This tool refuses to run on the CPU.")

    state = ToolState(Path(args.out_root) if args.out_root else None)
    done = sum(state.done(i) for i in range(len(state.images)))
    print(f"{len(state.images)} images, {done} already labelled. Open http://127.0.0.1:{args.port}", flush=True)
    # Threads, because a browser's idle spare connection would block a single-threaded server.
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the trained CustomCNN on the held-out test split.

Writes (into the run dir): metrics.json, classification_report.txt,
confusion_matrix.csv, confusion_matrix.png and prints a summary.

Usage:
    python ml/evaluation/evaluate_classifier.py --run-dir experiments/custom_cnn
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
import yaml

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from ml.models.registry import build_model, count_parameters  # noqa: E402
from ml.training.common import PatchDataset  # noqa: E402

CLASS_NAMES = ["groundnut", "weed"]


def render_confusion_matrix(cm: np.ndarray, class_names: list, path: Path) -> None:
    """Render a confusion matrix PNG using only PIL (no matplotlib)."""
    n = len(class_names)
    cell, pad = 170, 150
    img = Image.new("RGB", (pad + n * cell, pad + n * cell), "white")
    draw = ImageDraw.Draw(img)
    maxv = max(float(cm.max()), 1.0)
    for i in range(n):
        for j in range(n):
            t = float(cm[i, j]) / maxv
            color = (int(250 * (1 - t)), int(250 * (1 - t)), 255)
            x0, y0 = pad + j * cell, pad + i * cell
            draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=color, outline="black")
            txt = str(int(cm[i, j]))
            draw.text((x0 + cell // 2 - len(txt) * 8 // 2, y0 + cell // 2 - 8),
                      txt, fill="black")
        draw.text((12, pad + i * cell + cell // 2 - 8), class_names[i], fill="black")
        draw.text((pad + i * cell + cell // 2 - 8, 12), class_names[i], fill="black")
    draw.text((pad + n * cell // 2 - 40, pad + n * cell - 26), "True class", fill="black")
    draw.text((pad + n * cell // 2 - 46, 0), "Predicted class", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="experiments/custom_cnn")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default="cuda",
                    help="compute device: 'cuda' (GPU, default) or 'cpu'")
    args = ap.parse_args(argv)

    run_dir = PROJECT / args.run_dir
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    norm = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
    mean = np.array(norm["mean"], dtype=np.float32)
    std = np.array(norm["std"], dtype=np.float32)
    image_size = int(cfg["dataset"]["image_size"])
    root = PROJECT / cfg["dataset"]["root"]

    test_ds = PatchDataset(root, "test", image_size, mean, std, None, int(cfg["seed"]))
    batch = args.batch_size or int(cfg["evaluation"]["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=0)

    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device)
    model.eval()

    def predict_all():
        preds, targets = [], []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for x, y in test_loader:
                x = x.to(args.device)
                out = model(x)
                preds.append(out.argmax(1).cpu().numpy())
                targets.append(y.numpy())
        elapsed = time.perf_counter() - t0
        n = sum(len(p) for p in preds)
        return np.concatenate(preds), np.concatenate(targets), elapsed, n

    pred = target = None
    elapsed = 0.0
    n_total = 0
    for _ in range(3):  # one warm + two timed passes; report the last
        pred, target, elapsed, n_total = predict_all()
    per_image_ms = elapsed / max(1, n_total) * 1000.0

    cm = confusion_matrix(target, pred, labels=list(range(len(CLASS_NAMES))))
    acc = accuracy_score(target, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        target, pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    macro_f1 = float(f1_score(target, pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(target, pred, average="weighted", zero_division=0))
    report = classification_report(
        target, pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )

    n_params = int(count_parameters(model))
    metrics = {
        "test_samples": int(n_total),
        "test_accuracy": round(float(acc), 6),
        "per_class": {
            cls: {"precision": round(float(p), 6), "recall": round(float(r), 6),
                  "f1": round(float(f), 6)}
            for cls, p, r, f in zip(CLASS_NAMES, prec, rec, f1)
        },
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "confusion_matrix": cm.tolist(),
        "inference_time_seconds": round(elapsed, 4),
        "inference_time_per_image_ms": round(per_image_ms, 4),
        "parameters": n_params,
        "model_file_bytes": {
            "best_model.pt": (run_dir / "best_model.pt").stat().st_size,
            "final_model.pt": (run_dir / "final_model.pt").stat().st_size,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(run_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\predicted"] + CLASS_NAMES)
        for i, row in enumerate(cm):
            w.writerow([CLASS_NAMES[i]] + [int(v) for v in row])
    render_confusion_matrix(cm, CLASS_NAMES, run_dir / "confusion_matrix.png")

    print(f"=== {cfg['model']['name']} test results ===")
    print(report)
    print(f"accuracy={metrics['test_accuracy']}  macro_f1={macro_f1}  weighted_f1={weighted_f1}")
    print(f"confusion_matrix={cm.tolist()}")
    print(f"inference_time_total={elapsed:.3f}s  per_image={per_image_ms:.3f}ms")
    print(f"parameters={n_params}  model_file={metrics['model_file_bytes']}")
    print(f"metrics saved to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
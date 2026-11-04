"""Evaluate the 5 single and 3 hybrid classifiers on one test split, on the GPU.

Every model is scored on the same test images with the same settings: its trained input size,
ImageNet normalisation, no augmentation, and the class with the highest softmax probability as
the prediction. All metrics come from these actual test predictions.

Outputs (in --output):
    predictions/<model>.csv            one row per test image: true class, predicted class, probabilities
    confusion_matrices/<model>.png     confusion matrix (counts and row percentages)
    roc_curves/<model>.png             one-vs-rest ROC curve for each class, with AUC
    combined_confusion_matrices.png    all 8 confusion matrices in one figure
    combined_roc_curves.png            ROC curves of all 8 models in one graph
    comparison_table.csv / .md         all models ranked by performance
    results.json                       every metric, per model and per class

Usage (from the project root):
    python ml/evaluation/evaluate_classifiers.py
    python ml/evaluation/evaluate_classifiers.py --dataset-type primary_augmented --output experiments/evaluation/pa
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, classification_report, cohen_kappa_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score, roc_curve)
from torchvision.datasets import ImageFolder

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ml.classification.train_classifier import build_model  # noqa: E402
from ml.data.gpu_loader import GpuBatchLoader, classifier_transform  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset  # noqa: E402

MODELS = [  # (folder name, display name, type)
    ("cnn", "CNN", "Single"),
    ("vgg16", "VGG16", "Single"),
    ("resnet50", "ResNet50", "Single"),
    ("mobilenet", "MobileNetV2", "Single"),
    ("efficientnet", "EfficientNet-B0", "Single"),
    ("resnet50_mobilenet", "ResNet50 + MobileNetV2", "Hybrid"),
    ("resnet50_efficientnet", "ResNet50 + EfficientNet-B0", "Hybrid"),
    ("resnet50_efficientnet_mobilenet", "ResNet50 + MobileNetV2 + EfficientNet-B0", "Hybrid"),
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CM_MAP = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec3ef", "#2a78d6", "#123f7a"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def models_root(dataset: str) -> Path:
    # Hybrid models live where the backend loads them; the others under experiments/by_dataset.
    if dataset == "hybrid":
        return PROJECT / "experiments" / "classification"
    return PROJECT / "experiments" / "by_dataset" / dataset / "classification"


@torch.inference_mode()
def predict(key: str, folder: Path, test: ImageFolder, batch: int) -> Dict:
    size = json.loads((folder / "results.json").read_text(encoding="utf-8"))["image_size"]
    model = build_model(key, num_classes=len(test.classes), pretrained=False)
    model.load_state_dict(torch.load(folder / "best_model.pth", map_location="cuda"))
    model.cuda().eval()
    probs, labels = [], []
    for images, y in GpuBatchLoader(test.samples, batch, classifier_transform(size, train=False), shuffle=False):
        probs.append(torch.softmax(model(images).float(), dim=1).cpu())
        labels.append(y.cpu())
    del model
    torch.cuda.empty_cache()
    return {"image_size": size, "probs": torch.cat(probs).numpy(), "labels": torch.cat(labels).numpy()}


def metrics(labels: np.ndarray, probs: np.ndarray, classes: List[str]) -> Dict:
    preds = probs.argmax(1)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    per_class_auc = {c: roc_auc_score(labels == i, probs[:, i]) for i, c in enumerate(classes)}
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_weighted": p_w, "recall_weighted": r_w, "f1_weighted": f_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "mcc": matthews_corrcoef(labels, preds), "cohen_kappa": cohen_kappa_score(labels, preds),
        "auc_per_class": per_class_auc, "auc_macro": float(np.mean(list(per_class_auc.values()))),
        "errors": int((preds != labels).sum()), "n_test": int(len(labels)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in classes},
    }


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, length=0)


def draw_confusion(ax, cm: np.ndarray, classes: List[str], title: str, small: bool = False) -> None:
    rows = cm.sum(1, keepdims=True)
    ax.imshow(cm / rows, cmap=CM_MAP, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            share = cm[i, j] / rows[i, 0]
            ax.text(j, i, f"{cm[i, j]}\n({100 * share:.1f}%)", ha="center", va="center",
                    fontsize=10 if small else 13, fontweight="bold", color="white" if share > 0.5 else INK)
    names = [c.capitalize() for c in classes]
    ax.set_xticks(range(len(classes)), names)
    ax.set_yticks(range(len(classes)), names)
    ax.set_xlabel("Predicted class", color=INK2)
    ax.set_ylabel("True class", color=INK2)
    ax.set_title(title, fontsize=10 if small else 12, fontweight="bold", color=INK)
    ax.tick_params(length=0, colors=INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_confusion(path: Path, cm: np.ndarray, classes: List[str], name: str, m: Dict) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    draw_confusion(ax, cm, classes, f"{name}: confusion matrix")
    fig.text(0.5, 0.02, f"Accuracy {100 * m['accuracy']:.2f}%   ·   {m['errors']} errors of {m['n_test']} test images",
             ha="center", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.9, bottom=0.16)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def save_roc(path: Path, labels: np.ndarray, probs: np.ndarray, classes: List[str], name: str, m: Dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], ":", color="#9a9993", lw=1.5, label="Random guess (AUC = 0.500)")
    for i, (cls, color) in enumerate(zip(classes, COLORS)):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        ax.plot(fpr, tpr, color=color, lw=2.2, drawstyle="steps-post", linestyle="-" if i == 0 else "--",
                label=f"{cls.capitalize()} vs rest, AUC = {auc(fpr, tpr):.5f}")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlabel("False positive rate", color=INK2)
    ax.set_ylabel("True positive rate", color=INK2)
    ax.set_title(f"{name}: ROC curves (macro AUC = {m['auc_macro']:.5f})", fontsize=11.5, fontweight="bold",
                 color=INK, loc="left")
    ax.legend(loc="lower right", fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID)
    inset = inset_axes(ax, width="55%", height="45%", loc="center right", borderpad=1.5)
    for i, (cls, color) in enumerate(zip(classes, COLORS)):
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
        inset.plot(fpr, tpr, color=color, lw=2, drawstyle="steps-post", linestyle="-" if i == 0 else "--")
    inset.set_xlim(-0.002, 0.05)
    inset.set_ylim(0.95, 1.002)
    inset.set_facecolor("white")
    inset.grid(color=GRID, lw=0.8)
    inset.tick_params(labelsize=7, colors=INK2, length=0)
    inset.set_title("Zoom: FPR 0–0.05, TPR 0.95–1.00", fontsize=8, color=INK2)
    mark_inset(ax, inset, loc1=2, loc2=3, fc="none", ec="#9a9993", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def save_combined_confusion(path: Path, rows: List[Dict], classes: List[str], dataset: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 9.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    best = max(r["accuracy"] for r in rows)
    worst = min(r["accuracy"] for r in rows)
    for ax, r in zip(axes.flat, rows):
        tag = "  ★ BEST" if r["accuracy"] == best else ("  ▼ WORST" if r["accuracy"] == worst else "")
        draw_confusion(ax, np.array(r["confusion_matrix"]), classes,
                       f"{textwrap.fill(r['name'], 30)}\n{r['type']} · accuracy {100 * r['accuracy']:.2f}%, "
                       f"{r['errors']} errors{tag}", small=True)
    fig.suptitle(f"Confusion matrices of all models ({dataset} dataset test set, {rows[0]['n_test']} images)",
                 x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.05, right=0.955, top=0.86, bottom=0.07, wspace=0.5, hspace=0.6)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def save_combined_roc(path: Path, rows: List[Dict], positive: str, n_pos: int, n_neg: int) -> None:
    fig, ax = plt.subplots(figsize=(15, 8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], ":", color="#9a9993", lw=1.5, label="Random guess (AUC = 0.500)")
    best = max(r["auc_positive"] for r in rows)
    worst = min(r["auc_positive"] for r in rows)
    ranked = sorted(rows, key=lambda r: -r["auc_positive"])

    def draw(axis, legend: bool) -> None:
        for r in ranked:
            tag = "  ★ BEST" if r["auc_positive"] == best else ("  ▼ WORST" if r["auc_positive"] == worst else "")
            axis.plot(r["fpr"], r["tpr"], color=r["color"], drawstyle="steps-post",
                      lw=2.6 if tag else 1.8, linestyle="--" if r["type"] == "Hybrid" else "-", zorder=3,
                      label=f"{r['name']} ({r['type'].lower()}), AUC = {r['auc_positive']:.5f}{tag}" if legend else None)

    draw(ax, True)
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlabel(f"False positive rate ({'crop' if positive == 'weed' else 'other'} images wrongly called {positive})",
                  color=INK2)
    ax.set_ylabel(f"True positive rate ({positive} images correctly found)", color=INK2)
    inset = inset_axes(ax, width="60%", height="60%", loc="lower right", borderpad=3.2)
    draw(inset, False)
    inset.set_xlim(-0.002, 0.05)
    inset.set_ylim(0.95, 1.002)
    inset.set_facecolor("white")
    inset.grid(color=GRID, lw=0.8)
    inset.tick_params(labelsize=8, colors=INK2, length=0)
    inset.set_title("Zoom: top-left corner (FPR 0–0.05, TPR 0.95–1.00)", fontsize=9, color=INK2)
    mark_inset(ax, inset, loc1=2, loc2=3, fc="none", ec="#9a9993", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    order = list(range(1, len(handles))) + [0]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, frameon=True, facecolor=SURFACE, edgecolor=GRID,
              title="Model (sorted by AUC; dashed = hybrid)", title_fontsize=9)
    fig.suptitle("Combined ROC curves of all models", x=0.06, ha="left", fontsize=14, fontweight="bold", color=INK)
    negative = "crop" if positive == "weed" else "other"
    fig.text(0.06, 0.915, f"Test set: {n_pos} {positive} images (positive class) and {n_neg} {negative} images. "
             "Curves closer to the top-left corner are better.", ha="left", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.62, top=0.88, bottom=0.08)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-type", default="hybrid", choices=["primary", "primary_augmented", "secondary", "hybrid"])
    ap.add_argument("--output", default=str(PROJECT / "experiments" / "evaluation" / "classifiers"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--positive", default="weed", help="class treated as positive in the combined ROC graph")
    args = ap.parse_args(argv)

    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Evaluation is GPU-only.")
        return 1
    out = Path(args.output)
    for sub in ("predictions", "confusion_matrices", "roc_curves"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    test = ImageFolder(str(PROJECT / "datasets" / args.dataset_type / "classification" / "test"))
    classes = test.classes
    positive = test.class_to_idx[args.positive]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test set: {len(test)} images, classes {test.class_to_idx}")

    rows = []
    for (key, name, kind), color in zip(MODELS, COLORS):
        folder = models_root(args.dataset_type) / key
        if not (folder / "best_model.pth").exists():
            print(f"Skipping {name}: no trained model in {folder}")
            continue
        p = predict(key, folder, test, args.batch)
        labels, probs = p["labels"], p["probs"]
        m = metrics(labels, probs, classes)
        fpr, tpr, _ = roc_curve(labels == positive, probs[:, positive])
        rows.append({"key": key, "name": name, "type": kind, "color": color, "image_size": p["image_size"], **m,
                     "auc_positive": m["auc_per_class"][args.positive], "fpr": fpr.tolist(), "tpr": tpr.tolist()})

        with open(out / "predictions" / f"{key}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "true_class", "predicted_class", "correct"] + [f"prob_{c}" for c in classes])
            for (path, _), y, pr in zip(test.samples, labels, probs):
                writer.writerow([path, classes[y], classes[pr.argmax()], int(pr.argmax() == y)]
                                + [f"{v:.6f}" for v in pr])
        save_confusion(out / "confusion_matrices" / f"{key}.png", np.array(m["confusion_matrix"]), classes, name, m)
        save_roc(out / "roc_curves" / f"{key}.png", labels, probs, classes, name, m)
        print(f"{name:42s} accuracy {100 * m['accuracy']:6.2f}%  F1 {100 * m['f1_weighted']:6.2f}%  "
              f"AUC {m['auc_macro']:.5f}  errors {m['errors']}", flush=True)

    if not rows:
        print("Error: no trained models found.")
        return 1

    # Rank by accuracy; ties (same number of errors) are broken by AUC, which uses every probability.
    ranked = sorted(rows, key=lambda r: (-r["accuracy"], -r["auc_macro"]))
    header = ["Rank", "Model", "Type", "Input size", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1 (%)",
              "Macro F1 (%)", "MCC", "AUC", "Errors"]
    table = [[i, r["name"], r["type"], f"{r['image_size']}x{r['image_size']}", f"{100 * r['accuracy']:.2f}",
              f"{100 * r['precision_weighted']:.2f}", f"{100 * r['recall_weighted']:.2f}",
              f"{100 * r['f1_weighted']:.2f}", f"{100 * r['f1_macro']:.2f}", f"{r['mcc']:.4f}",
              f"{r['auc_macro']:.5f}", f"{r['errors']}/{r['n_test']}"] for i, r in enumerate(ranked, 1)]
    with open(out / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + table)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table]
    (out / "comparison_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    save_combined_confusion(out / "combined_confusion_matrices.png", rows, classes, args.dataset_type)
    n_pos = int(sum(1 for _, y in test.samples if y == positive))
    save_combined_roc(out / "combined_roc_curves.png", rows, args.positive, n_pos, len(test) - n_pos)

    saved = [{k: v for k, v in r.items() if k not in ("fpr", "tpr", "color")} for r in ranked]
    (out / "results.json").write_text(json.dumps({"dataset": args.dataset_type, "test_images": len(test),
                                                  "classes": classes, "models": saved}, indent=2), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Retrain an existing hybrid classifier (ml/models/hybrid.py) with an improved recipe.

The architecture is unchanged: the saved best_model.pth has exactly the same layers as a
checkpoint from train_classifier.py, so the backend and evaluation scripts load it as before.
Only the training and evaluation procedure differ:

* per-backbone auxiliary loss (``--aux-weight``): during training a throwaway linear head
  on each backbone's features gets its own loss, so every backbone stays discriminative on
  its own and their errors are less correlated. These heads are discarded, not saved.
* discriminative learning rates - pretrained backbones learn slower than the fusion head
* linear warmup followed by cosine decay, per optimisation step
* label smoothing
* an exponential moving average (EMA) of the weights; the EMA copy is validated and saved
* bf16 mixed precision
* model selection by validation accuracy, ties broken by lower validation loss
* optional flip test-time augmentation (TTA), decided before training via ``--tta``

The test split is only touched once, after training, with the checkpoint chosen on val.
Both the plain and the TTA test accuracy are recorded; ``test_accuracy`` is the one
selected by ``--tta`` so the reported number is fixed in advance.

Results go to experiments/classification_improved/<model>/ by default, so the existing
experiments/classification/<model>/ run is kept for comparison.

GPU-only: the script stops if CUDA is unavailable.

Usage:
    .venv-gpu/Scripts/python.exe ml/classification/train_hybrid.py --model resnet50_efficientnet_mobilenet
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.datasets import ImageFolder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.gpu_loader import GpuBatchLoader, classifier_transform  # noqa: E402
from ml.models.hybrid import build_hybrid_model, list_hybrid_models  # noqa: E402

IMAGE_SIZE = 224


def forward_with_aux(model: nn.Module, aux_heads: nn.ModuleDict, x: torch.Tensor):
    """Same computation as HybridClassifier.forward (concat fusion), plus per-backbone logits."""
    hybrid = model.model
    feats = [hybrid.backbones[name](x) for name in hybrid.models]
    logits = hybrid.fusion(torch.cat(feats, dim=1))
    return logits, [aux_heads[name](f) for name, f in zip(hybrid.models, feats)]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model: nn.Module, loader, tta: bool):
    """Return (class-1 probabilities, labels, mean cross-entropy) for a loader."""
    model.eval()
    probs, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x).float()
            p = torch.softmax(logits, 1)
            if tta:
                p = p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)
                p = p + torch.softmax(model(torch.flip(x, dims=[2])).float(), 1)
                p = p / 3
        loss_sum += F.nll_loss(torch.log(p.clamp_min(1e-8)), y, reduction="sum").item()
        probs.append(p.cpu())
        labels.append(y.cpu())
    probs, labels = torch.cat(probs), torch.cat(labels)
    return probs, labels, loss_sum / len(labels)


def accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    return 100.0 * (probs.argmax(1) == labels).float().mean().item()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, choices=list_hybrid_models())
    parser.add_argument("--dataset-type", default="hybrid",
                        choices=["primary", "primary_augmented", "secondary", "hybrid"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=0.3,
                        help="weight of the mean per-branch auxiliary loss")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="report flip-TTA test accuracy as test_accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100, help="print progress every N batches")
    parser.add_argument("--device", default="0", help="GPU number; training is GPU-only")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "experiments" / "classification_improved"))
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Training is GPU-only.")
        return 1
    device = f"cuda:{int(args.device)}"
    torch.cuda.set_device(device)
    print(f"Using GPU {args.device}: {torch.cuda.get_device_name(device)}")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = PROJECT_ROOT / "datasets" / args.dataset_type / "classification"
    out_dir = Path(args.output_root).resolve() / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = (ImageFolder(data_root / s) for s in ("train", "val", "test"))
    classes = train_ds.classes
    print(f"Dataset {data_root}: train {len(train_ds)}, val {len(val_ds)}, test {len(test_ds)}, classes {classes}")

    train_loader = GpuBatchLoader(train_ds.samples, args.batch, classifier_transform(IMAGE_SIZE, True),
                                  shuffle=True, device=device)
    val_loader = GpuBatchLoader(val_ds.samples, 32, classifier_transform(IMAGE_SIZE, False),
                                shuffle=False, device=device)
    test_loader = GpuBatchLoader(test_ds.samples, 32, classifier_transform(IMAGE_SIZE, False),
                                 shuffle=False, device=device)

    model = build_hybrid_model(args.model, num_classes=len(classes), pretrained=True).to(device)
    hybrid = model.model
    if hybrid.fusion_method != "concat":
        print(f"Error: only concat fusion is supported, {args.model} uses {hybrid.fusion_method}")
        return 1
    backbones = hybrid.models
    aux_heads = nn.ModuleDict({name: nn.Linear(dim, len(classes))
                               for name, dim in zip(backbones, hybrid.feature_dims)}).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True)

    optimizer = torch.optim.AdamW([
        {"params": list(hybrid.backbones.parameters()), "lr": args.lr_backbone},
        {"params": list(hybrid.fusion.parameters()) + list(aux_heads.parameters()), "lr": args.lr_head},
    ], weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_epochs * steps_per_epoch))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = (-1.0, float("inf"))  # (val_acc, -val_loss) compared lexicographically
    best_epoch, patience, history = 0, 0, []
    ckpt_path = out_dir / "best_model.pth"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        aux_heads.train()
        loss_sum, correct, seen = 0.0, 0, 0
        aux_correct = torch.zeros(len(backbones))
        epoch_start = time.time()
        for batch_idx, (x, y) in enumerate(train_loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, aux = forward_with_aux(model, aux_heads, x)
                loss = criterion(logits.float(), y)
                if args.aux_weight > 0:
                    loss = loss + args.aux_weight * sum(criterion(a.float(), y) for a in aux) / len(aux)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(aux_heads.parameters()), 5.0)
            optimizer.step()
            scheduler.step()
            ema.update_parameters(model)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            aux_correct += torch.tensor([(a.argmax(1) == y).sum().item() for a in aux], dtype=torch.float)
            if batch_idx % args.log_every == 0 or batch_idx == steps_per_epoch:
                elapsed = time.time() - epoch_start
                eta = elapsed / batch_idx * (steps_per_epoch - batch_idx)
                print(f"  epoch {epoch} batch {batch_idx}/{steps_per_epoch} "
                      f"loss {loss_sum / seen:.4f} acc {100 * correct / seen:.2f}% "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} "
                      f"elapsed {elapsed / 60:.1f}m epoch-eta {eta / 60:.1f}m", flush=True)

        val_probs, val_labels, val_loss = predict(ema.module, val_loader, tta=False)
        val_acc = accuracy(val_probs, val_labels)
        branch_acc = {m: round(100 * v, 2) for m, v in zip(backbones, (aux_correct / seen).tolist())}
        row = {"epoch": epoch, "train_loss": round(loss_sum / seen, 4), "train_acc": round(100 * correct / seen, 2),
               "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 2), "train_branch_acc": branch_acc}
        history.append(row)
        print(f"Epoch {epoch}/{args.epochs} train_loss {row['train_loss']} train_acc {row['train_acc']} "
              f"val_loss {row['val_loss']} val_acc {row['val_acc']} branch_acc {branch_acc}", flush=True)

        if (val_acc, -val_loss) > best:
            best, best_epoch, patience = (val_acc, -val_loss), epoch, 0
            torch.save(ema.module.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    val_tta_probs, val_labels, _ = predict(model, val_loader, tta=True)
    test_plain, test_labels, _ = predict(model, test_loader, tta=False)
    test_tta, _, _ = predict(model, test_loader, tta=True)
    plain_acc, tta_acc = accuracy(test_plain, test_labels), accuracy(test_tta, test_labels)
    chosen = test_tta if args.tta else test_plain
    preds = chosen.argmax(1).numpy()
    labels = test_labels.numpy()
    report = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    errors = [{"path": str(Path(test_ds.samples[i][0]).relative_to(PROJECT_ROOT)),
               "label": classes[labels[i]], "predicted": classes[preds[i]],
               "confidence": round(float(chosen[i].max()), 4)}
              for i in np.flatnonzero(preds != labels)]

    n = len(labels)
    acc = 100.0 * (preds == labels).mean()
    # Wilson 95% interval: with a few hundred test images one image moves accuracy by ~0.16 points.
    z, phat = 1.96, (preds == labels).mean()
    centre = (phat + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / (1 + z * z / n)

    results = {
        "model": args.model,
        "backbones": backbones,
        "recipe": "train_hybrid.py",
        "dataset_root": str(data_root),
        "image_size": IMAGE_SIZE,
        "class_names": classes,
        "seed": args.seed,
        "hyperparameters": {k: v for k, v in vars(args).items() if k not in ("model", "device", "output_root")},
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best[0], 2),
        "val_accuracy_tta": round(accuracy(val_tta_probs, val_labels), 2),
        "test_accuracy": round(acc, 2),
        "test_accuracy_plain": round(plain_acc, 2),
        "test_accuracy_tta": round(tta_acc, 2),
        "test_accuracy_ci95": [round(100 * (centre - half), 2), round(100 * (centre + half), 2)],
        "test_errors": len(errors),
        "test_size": n,
        "test_precision": round(report["weighted avg"]["precision"], 4),
        "test_recall": round(report["weighted avg"]["recall"], 4),
        "test_f1": round(report["weighted avg"]["f1-score"], 4),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "misclassified": errors,
        "training_time_seconds": round(train_time, 2),
        "model_size_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20, 2),
        "model_parameters": int(sum(p.numel() for p in model.parameters())),
        "class_metrics": {c: {k: round(v, 4) if isinstance(v, float) else v for k, v in report[c].items()}
                          for c in classes},
        "history": history,
        "classification_report": report,
    }
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\nBest epoch {best_epoch}, val acc {best[0]:.2f}")
    print(f"Test accuracy plain {plain_acc:.2f}  TTA {tta_acc:.2f}  reported {acc:.2f} "
          f"({len(errors)}/{n} wrong, 95% CI {results['test_accuracy_ci95']})")
    for e in errors:
        print(f"  wrong: {e['path']} label={e['label']} predicted={e['predicted']} conf={e['confidence']}")
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train the plant-vs-other gate classifier with data leakage prevention.

Reads ``datasets/<type>/plant_gate/{train,val,test}/{plant,non_plant}``, built by
``ml/data/build_all_datasets.py``, which splits by source photo so augmented
variants stay with their originals.

Output: ``backend/app/utils/plant_gate_model.joblib`` unless ``--output`` is given
        (RandomForest + threshold + feature names), used by the API gate.

Usage:
    python ml/gate/train_plant_gate.py --dataset-type hybrid
    python ml/gate/train_plant_gate.py --dataset-type primary --output experiments/by_dataset/primary/plant_gate/plant_gate_model.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.gate.features import FEATURE_NAMES  # noqa: E402
from ml.gate.features_gpu import GpuGateFeatures  # noqa: E402

OUT_MODEL = ROOT / "backend" / "app" / "utils" / "plant_gate_model.joblib"


def _image_files(*roots: Path) -> List[Path]:
    out: List[Path] = []
    exts = ("*.jpg", "*.jpeg", "*.png")
    for root in roots:
        if not root.exists():
            continue
        for ext in exts:
            out.extend(root.rglob(ext))
    # De-duplicate by absolute path, keep stable order.
    seen, ordered = set(), []
    for p in sorted(out):
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            ordered.append(p)
    return ordered


def _detect_duplicate_files(paths: List[Path]) -> Dict[str, List[Path]]:
    """Detect duplicate files by content hash"""
    hash_to_files = {}
    for p in paths:
        try:
            with open(p, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash not in hash_to_files:
                hash_to_files[file_hash] = []
            hash_to_files[file_hash].append(p)
        except:
            continue
    return {h: files for h, files in hash_to_files.items() if len(files) > 1}


def _load_images(paths: List[Path], label: int,
                 featurizer: GpuGateFeatures) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load images and return features, labels, and list of successfully loaded paths"""
    X, y, loaded_paths = [], [], []
    excluded_paths = []
    for p in paths:
        try:
            X.append(featurizer.extract(p))
        except Exception as e:
            excluded_paths.append((str(p), f"Image load error: {e}"))
            continue
        y.append(label)
        loaded_paths.append(str(p))
        if len(loaded_paths) % 1000 == 0:
            print(f"  extracted {len(loaded_paths)}/{len(paths)}")
    print(f"  loaded {len(loaded_paths)} images (label={label}), excluded {len(excluded_paths)}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), loaded_paths


def pick_threshold(proba_pos: np.ndarray, y_val: np.ndarray,
                   max_fnr: float = 0.005,
                   max_fpr: Optional[float] = None) -> Tuple[float, float, float]:
    """Choose a decision threshold on validation scores.

    Returns: (threshold, actual_fnr, actual_fpr)
    """
    pos_scores = proba_pos[y_val == 1]
    if pos_scores.size == 0:
        return 0.5, 1.0, 0.0
    allowed_pos = int(np.ceil((1.0 - max_fnr) * pos_scores.size))
    if allowed_pos < 1:
        allowed_pos = 1
    th = float(np.sort(pos_scores)[-allowed_pos])

    # Calculate actual FNR and FPR at this threshold
    y = y_val
    p = proba_pos
    preds = (p >= th).astype(int)
    actual_fnr = float(((preds == 0) & (y == 1)).sum()) / max(1, int((y == 1).sum()))
    actual_fpr = float(((preds == 1) & (y == 0)).sum()) / max(1, int((y == 0).sum()))

    if max_fpr is not None:
        cands = sorted(float(s) for s in np.unique(p) if s >= 0.5)
        for cand in cands:
            if cand < th:
                continue
            fpr = float(((p >= cand) & (y == 0)).sum()) / max(1, int((y == 0).sum()))
            if fpr <= max_fpr:
                # Recalculate FNR at this threshold
                preds_cand = (p >= cand).astype(int)
                actual_fnr = float(((preds_cand == 0) & (y == 1)).sum()) / max(1, int((y == 1).sum()))
                actual_fpr = fpr
                return cand, actual_fnr, actual_fpr

    return th, actual_fnr, actual_fpr


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train plant gate classifier with data leakage prevention",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--dataset-type", default="hybrid",
                    choices=["primary", "primary_augmented", "secondary", "hybrid"],
                    help="reads datasets/<type>/plant_gate/{train,val,test}/{plant,non_plant}")
    ap.add_argument("--output", default=str(OUT_MODEL), help="where to save the gate model")
    ap.add_argument("--max-fnr", type=float, default=0.005,
                    help="maximum allowed false-negative rate on plant images")
    ap.add_argument("--max-fpr", type=float, default=0.015,
                    help="maximum allowed false-positive rate on non-plant images")
    ap.add_argument("--imagenet", default="resnet50", help="ImageNet model for feature extraction")
    args = ap.parse_args()

    # GPU only: feature extraction and the forest both refuse to fall back to the CPU
    import torch
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Plant gate training is GPU-only.")
        return 1
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Validate parameters
    if not (0 <= args.max_fnr <= 1):
        print(f"Error: max_fnr must be between 0 and 1, got {args.max_fnr}")
        return 1
    if not (0 <= args.max_fpr <= 1):
        print(f"Error: max_fpr must be between 0 and 1, got {args.max_fpr}")
        return 1

    data_root = ROOT / "datasets" / args.dataset_type / "plant_gate"
    print(f"Using dataset: {data_root}")
    pos_paths, neg_paths = [], []
    splits: Dict[str, Tuple[List[Path], List[Path]]] = {}
    for split in ("train", "val", "test"):
        plant = _image_files(data_root / split / "plant")
        non_plant = _image_files(data_root / split / "non_plant")
        if not plant or not non_plant:
            print(f"ERROR: {data_root / split} needs images in both plant/ and non_plant/.")
            print("Run: python ml/data/build_all_datasets.py")
            return 1
        splits[split] = (plant, non_plant)
        pos_paths.extend(plant)
        neg_paths.extend(non_plant)
    print(f"  positive files: {len(pos_paths)}")
    print(f"  negative files: {len(neg_paths)}")

    # Detect duplicates
    pos_duplicates = _detect_duplicate_files(pos_paths)
    if pos_duplicates:
        print(f"  WARNING: Found {len(pos_duplicates)} duplicate groups in positives")
        for hash_val, files in list(pos_duplicates.items())[:3]:
            print(f"    {len(files)} files with same hash")
    neg_duplicates = _detect_duplicate_files(neg_paths)
    if neg_duplicates:
        print(f"  WARNING: Found {len(neg_duplicates)} duplicate groups in negatives")

    featurizer = GpuGateFeatures(args.imagenet)

    def _load_split(split: str) -> Tuple[np.ndarray, np.ndarray]:
        plant, non_plant = splits[split]
        print(f"Extracting features ({split})...")
        Xp, yp, _ = _load_images(plant, 1, featurizer)
        Xn, yn, _ = _load_images(non_plant, 0, featurizer)
        return np.vstack([Xp, Xn]), np.concatenate([yp, yn])

    # The folders are already split by source photo, so near-duplicates and
    # augmented copies can never cross from train into val or test.
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")
    X = np.vstack([X_train, X_val, X_test])
    y = np.concatenate([y_train, y_val, y_test])

    print(f"\nTotal samples: {X.shape[0]} (positives: {(y == 1).sum()}, negatives: {(y == 0).sum()})")

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples (positives: {(y_train == 1).sum()}, negatives: {(y_train == 0).sum()})")
    print(f"  Val:   {X_val.shape[0]} samples (positives: {(y_val == 1).sum()}, negatives: {(y_val == 0).sum()})")
    print(f"  Test:  {X_test.shape[0]} samples (positives: {(y_test == 1).sum()}, negatives: {(y_test == 0).sum()})")

    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    # Train on TRAINING data only. scikit-learn's forest has no GPU version, so this is
    # XGBoost's Random Forest on the GPU: one boosting round of 400 parallel trees, with
    # matching depth, leaf size and class balancing.
    clf = XGBClassifier(
        n_estimators=1, num_parallel_tree=400, learning_rate=1.0, max_depth=18, min_child_weight=2,
        subsample=0.8, colsample_bynode=0.3, reg_lambda=1e-5, tree_method="hist", device="cuda",
        random_state=42,
    )
    print(f"\nTraining GPU Random Forest on {X_train.shape[0]} training samples...")
    clf.fit(X_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))

    # Use validation set to select threshold
    print("\nSelecting threshold on validation set...")
    proba_val = clf.predict_proba(X_val)[:, 1]
    threshold, val_fnr, val_fpr = pick_threshold(proba_val, y_val, max_fnr=args.max_fnr, max_fpr=args.max_fpr)
    print(f"  Selected threshold: {threshold:.4f}")
    print(f"  Validation metrics: FNR={val_fnr:.4f}, FPR={val_fpr:.4f}")

    # Check if constraints are met
    if val_fnr > args.max_fnr:
        print(f"  WARNING: Validation FNR ({val_fnr:.4f}) exceeds target ({args.max_fnr:.4f})")
    if val_fpr > args.max_fpr:
        print(f"  WARNING: Validation FPR ({val_fpr:.4f}) exceeds target ({args.max_fpr:.4f})")

    # Final evaluation on test set (independent check)
    print("\nFinal evaluation on test set...")
    proba_test = clf.predict_proba(X_test)[:, 1]
    test_preds = (proba_test >= threshold).astype(int)
    test_fn = ((test_preds == 0) & (y_test == 1)).sum()
    test_fp = ((test_preds == 1) & (y_test == 0)).sum()
    test_fnr = test_fn / max(1, (y_test == 1).sum())
    test_fpr = test_fp / max(1, (y_test == 0).sum())
    print(f"  Test metrics: FNR={test_fnr:.4f}, FPR={test_fpr:.4f}")

    print("\n=== Gate Training Complete ===")
    print(f"  n_train={X_train.shape[0]}")
    print(f"  n_val={X_val.shape[0]}")
    print(f"  n_test={X_test.shape[0]}")
    print(f"  threshold={threshold:.4f}")
    print(f"  validation: FNR={val_fnr:.4f}, FPR={val_fpr:.4f}")
    print(f"  test: FNR={test_fnr:.4f}, FPR={test_fpr:.4f}")

    metrics = {
        "threshold": float(threshold),
        "target_max_fnr": args.max_fnr,
        "target_max_fpr": args.max_fpr,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_train_positives": int((y_train == 1).sum()),
        "n_train_negatives": int((y_train == 0).sum()),
        "validation_fnr": float(val_fnr),
        "validation_fpr": float(val_fpr),
        "test_fnr": float(test_fnr),
        "test_fpr": float(test_fpr),
        "validation_targets_met": bool(val_fnr <= args.max_fnr and val_fpr <= args.max_fpr),
        "test_targets_met": bool(test_fnr <= args.max_fnr and test_fpr <= args.max_fpr),
        "feature_extractor": args.imagenet,
        "n_positives_total": int((y == 1).sum()),
        "n_negatives_total": int((y == 0).sum()),
        "positive_duplicates": len(pos_duplicates),
        "negative_duplicates": len(neg_duplicates),
    }

    payload = {
        "model": clf,
        "threshold": float(threshold),
        "feature_names": FEATURE_NAMES,
        "positive_class": 1,
        "metrics": metrics,
    }
    import joblib
    # The backend predicts one image at a time and may run on a machine without a GPU.
    clf.set_params(device="cpu")
    out_model = Path(args.output)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_model)
    meta = out_model.with_name("plant_gate_model.json")
    meta.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved gate model -> {out_model}")
    print(f"Saved metrics -> {meta}")

    # Warn if targets not met
    if not metrics['test_targets_met']:
        print("\nWARNING: Test set did not meet FNR/FPR targets.")
        print("Consider adjusting thresholds or collecting more training data.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

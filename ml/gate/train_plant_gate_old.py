"""Train the plant-vs-other gate classifier.

Positives:
  - ``datasets/processed/groundnut_weed_detection/images/**/*``
  - ``datasets/processed/classification_groundnut_weed/{train,val,test}/*/*``
Negatives:
  - ``datasets/negatives_synthetic/**/*`` (generated stand-ins)
  - ``datasets/negatives_real/**/*`` (REAL photos you provide: people, objects,
    animals, roads, buildings, indoor scenes - strongly recommended)

IMPORTANT: This script now properly splits data into train/val/test sets to
prevent contamination. Training uses only train split, threshold selection
uses val split, and final evaluation uses test split.

Output: ``backend/app/utils/plant_gate_model.joblib``
        (RandomForest + threshold + feature names), used by the API gate.

Usage:
    python ml/gate/train_plant_gate.py
    python ml/gate/train_plant_gate.py --negatives-dir datasets/negatives_real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.gate.features import (  # noqa: E402
    FEATURE_NAMES, ImageNetFeaturizer, extract_features, feature_vector,
)

OUT_MODEL = ROOT / "backend" / "app" / "utils" / "plant_gate_model.joblib"

RESIZE = 256  # cap longest side to keep feature extraction fast


def _cap_size(img) -> np.ndarray:
    from PIL import Image
    w, h = img.size
    mx = max(w, h)
    if mx > RESIZE:
        scale = RESIZE / mx
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.BILINEAR)
    return np.asarray(img.convert("RGB"))


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


def _load_images(paths: List[Path], label: int,
                 featurizer: ImageNetFeaturizer) -> Tuple[np.ndarray, np.ndarray]:
    from PIL import Image
    X, y = [], []
    loaded = 0
    for p in paths:
        try:
            rgb = _cap_size(Image.open(p))
        except Exception:
            continue
        if rgb.size == 0 or rgb.shape[0] < 8 or rgb.shape[1] < 8:
            continue
        feats = extract_features(rgb, None, (rgb.shape[1], rgb.shape[0]),
                                 featurizer)
        X.append(feature_vector(feats))
        y.append(label)
        loaded += 1
        if loaded % 1000 == 0:
            print(f"  extracted {loaded}/{len(paths)}")
    print(f"  loaded {loaded} images (label={label})")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32)


def pick_threshold(proba_pos: np.ndarray, y_val: np.ndarray,
                   max_fnr: float = 0.005,
                   max_fpr: Optional[float] = None) -> float:
    """Choose a decision threshold on validation scores.

    Start from the highest threshold with positive false-negative rate
    <= ``max_fnr``. If ``max_fpr`` is set and the resulting threshold admits
    more false positives than allowed, push the threshold up until the
    non-plant false-positive rate <= ``max_fpr`` (this may raise FNR on plant
    images; the caller reports both numbers).
    """
    pos_scores = proba_pos[y_val == 1]
    if pos_scores.size == 0:
        return 0.5
    allowed_pos = int(np.ceil((1.0 - max_fnr) * pos_scores.size))
    if allowed_pos < 1:
        allowed_pos = 1
    th = float(np.sort(pos_scores)[-allowed_pos])

    if max_fpr is not None:
        y = y_val
        p = proba_pos
        # candidate thresholds: unique scores in the (0.5, 1.0] region.
        cands = sorted(float(s) for s in np.unique(p) if s >= 0.5)
        for cand in cands:
            if cand < th:
                continue
            fpr = float(((p >= cand) & (y == 0)).sum()) / max(1, int((y == 0).sum()))
            if fpr <= max_fpr:
                return cand
    return th
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--negatives-dir", default=None,
                    help="folder of REAL non-plant photos")
    ap.add_argument("--max-fnr", type=float, default=0.005,
                    help="maximum allowed false-negative rate on plant images")
    ap.add_argument("--max-fpr", type=float, default=0.015,
                    help="maximum allowed false-positive rate on non-plant "
                         "images (pushes the threshold up if needed)")
    ap.add_argument("--imagenet", default="resnet50")
    ap.add_argument("--test-split", type=float, default=0.2,
                    help="fraction of data reserved for final test evaluation")
    ap.add_argument("--val-split", type=float, default=0.2,
                    help="fraction of training data reserved for validation (threshold selection)")
    args = ap.parse_args()

    print("Collecting positives...")
    pos_paths = _image_files(
        ROOT / "datasets/processed/groundnut_weed_detection/images",
        ROOT / "datasets/processed/classification_groundnut_weed",
    )
    print(f"  positive files: {len(pos_paths)}")

    print("Collecting negatives...")
    neg_paths = _image_files(ROOT / "datasets/negatives_synthetic")
    if args.negatives_dir:
        extra = _image_files(Path(args.negatives_dir))
        print(f"  real negatives found: {len(extra)}")
        neg_paths.extend(extra)
    print(f"  negative files: {len(neg_paths)}")

    featurizer = ImageNetFeaturizer(args.imagenet)
    print("Extracting features (positives)...")
    Xp, yp = _load_images(pos_paths, 1, featurizer)
    print("Extracting features (negatives)...")
    Xn, yn = _load_images(neg_paths, 0, featurizer)

    X = np.vstack([Xp, Xn])
    y = np.concatenate([yp, yn])

    print(f"\nTotal samples: {X.shape[0]} (positives: {(y == 1).sum()}, negatives: {(y == 0).sum()})")

    # Split data: test set first, then train/val split from remaining
    from sklearn.model_selection import train_test_split

    # First split: separate test set (never used in training or threshold selection)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )

    # Second split: separate validation set from training (for threshold selection)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=args.val_split, random_state=42, stratify=y_trainval
    )

    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples (positives: {(y_train == 1).sum()}, negatives: {(y_train == 0).sum()})")
    print(f"  Val:   {X_val.shape[0]} samples (positives: {(y_val == 1).sum()}, negatives: {(y_val == 0).sum()})")
    print(f"  Test:  {X_test.shape[0]} samples (positives: {(y_test == 1).sum()}, negatives: {(y_test == 0).sum()})")

    from sklearn.ensemble import RandomForestClassifier

    # Train on TRAINING data only
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=18, min_samples_leaf=2,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )
    print(f"\nTraining RandomForest on {X_train.shape[0]} training samples...")
    clf.fit(X_train, y_train)

    # Use validation set to select threshold
    print("\nSelecting threshold on validation set...")
    proba_val = clf.predict_proba(X_val)[:, 1]
    threshold = pick_threshold(proba_val, y_val, max_fnr=args.max_fnr, max_fpr=args.max_fpr)
    print(f"  Selected threshold: {threshold:.4f}")

    # Evaluate on validation set with selected threshold
    val_preds = (proba_val >= threshold).astype(int)
    val_fn = ((val_preds == 0) & (y_val == 1)).sum()
    val_fp = ((val_preds == 1) & (y_val == 0)).sum()
    val_fnr = val_fn / max(1, (y_val == 1).sum())
    val_fpr = val_fp / max(1, (y_val == 0).sum())
    print(f"  Validation metrics: FNR={val_fnr:.4f}, FPR={val_fpr:.4f}")

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
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_train_positives": int((y_train == 1).sum()),
        "n_train_negatives": int((y_train == 0).sum()),
        "validation_fnr": float(val_fnr),
        "validation_fpr": float(val_fpr),
        "test_fnr": float(test_fnr),
        "test_fpr": float(test_fpr),
    }
    payload = {
        "model": clf,
        "threshold": float(threshold),
        "feature_names": FEATURE_NAMES,
        "positive_class": 1,
        "metrics": metrics,
    }
    import joblib
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, OUT_MODEL)
    meta = OUT_MODEL.with_name("plant_gate_model.json")
    meta.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved gate model -> {OUT_MODEL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
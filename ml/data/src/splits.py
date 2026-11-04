"""Reproducible, duplicate-aware train/validation/test splitting.

Splitting operates on *groups*: every item that shares a duplicate fingerprint
is treated as one atomic group so that byte-identical images can never appear
in more than one split. Determinism is guaranteed by a fixed seed and a sorted,
stable shuffle.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional


def make_split(
    items: List[str],
    ratios: Dict[str, float],
    seed: int,
    dup_groups: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    """Split item ids into train/val/test respecting duplicate groups.

    ``dup_groups`` maps item id -> shared group key (fingerprint). Items not in
    the map are treated as their own group.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {ratios}")

    groups: Dict[str, List[str]] = {}
    for item in items:
        key = (dup_groups or {}).get(item, item)
        groups.setdefault(key, []).append(item)

    keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    total = len(items)
    n_train = int(round(total * ratios["train"]))
    n_val = int(round(total * ratios["val"]))

    buckets: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}
    for key in keys:
        target = "train"
        if counts["train"] >= n_train:
            target = "val" if counts["val"] < n_val else "test"
        buckets[target].extend(groups[key])
        counts[target] += len(groups[key])

    # Deterministic ordering inside each split.
    return {name: sorted(paths) for name, paths in buckets.items()}


def split_manifest(dataset_key: str, version: str, ratios: Dict[str, float],
                   seed: int, split: Dict[str, List[str]]) -> dict:
    """Serializable manifest describing one reproducible split."""
    return {
        "dataset": dataset_key,
        "version": version,
        "seed": seed,
        "ratios": ratios,
        "counts": {k: len(v) for k, v in split.items()},
        "split": split,
    }

"""Duplicate detection via content fingerprints.

Duplicate groups are used both for reporting and for split-time deduplication
so that byte-identical images always land in the same split (no leakage).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from .common import file_fingerprint


def group_duplicates(files: List[Path], full_hash: bool = False) -> Dict[str, List[str]]:
    """Return {fingerprint: [paths]} for every group with >1 member."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        groups[file_fingerprint(f, full=full_hash)].append(str(f))
    return {fp: paths for fp, paths in groups.items() if len(paths) > 1}


def duplicate_group_map(files: List[Path], full_hash: bool = False) -> Dict[str, str]:
    """Map each duplicate file path to a shared group id (fingerprint).

    Files that are not part of any duplicate group are absent from the map.
    """
    out: Dict[str, str] = {}
    for fp, paths in group_duplicates(files, full_hash=full_hash).items():
        for p in paths:
            out[p] = fp
    return out


def cross_dataset_overlap(fingerprint_sets: Dict[str, Dict[str, str]]) -> List[dict]:
    """Detect images whose fingerprint appears in more than one dataset.

    ``fingerprint_sets`` maps dataset key -> {absolute path: fingerprint}.
    Returns a list of overlap records (fingerprint + locations) for hashes
    shared by at least two distinct datasets.
    """
    by_fp: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for dkey, mapping in fingerprint_sets.items():
        for path, fp in mapping.items():
            by_fp[fp].append((dkey, path))
    overlaps: List[dict] = []
    for fp, locs in by_fp.items():
        if len({d for d, _ in locs}) > 1:
            overlaps.append(
                {
                    "fingerprint": fp[:16],
                    "count": len(locs),
                    "locations": [{"dataset": d, "path": p} for d, p in locs],
                }
            )
    overlaps.sort(key=lambda o: -o["count"])
    return overlaps

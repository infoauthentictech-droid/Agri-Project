"""Shared utilities for the Groundnut AI preprocessing pipeline.

Pure-stdlib helpers: logging, JSON IO, deterministic content fingerprints.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string (second resolution)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure a reusable 'groundnut.preprocessing' logger (idempotent)."""
    logger = logging.getLogger("groundnut.preprocessing")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = utc_now().replace(":", "-").replace("T", "_")
        fh = logging.FileHandler(log_dir / f"pipeline_{ts}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def read_json(path: Any) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Any, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    return path


_SAMPLE = 64 * 1024


def file_fingerprint(path: Path, full: bool = False) -> str:
    """Deterministic content fingerprint for a file.

    For files <= 3 sample blocks the whole file is hashed (exact). For larger
    files three 64 KiB blocks (head / middle / tail) plus the byte size are
    hashed — a fast, near-unique signature that never requires full reads of
    multi-GB datasets. Pass ``full=True`` to force a whole-file SHA-256.
    """
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        if full or size <= 3 * _SAMPLE:
            h.update(fh.read())
        else:
            h.update(fh.read(_SAMPLE))
            fh.seek(size // 2)
            h.update(fh.read(_SAMPLE))
            fh.seek(max(0, size - _SAMPLE))
            h.update(fh.read(_SAMPLE))
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()

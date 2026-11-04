"""Groundnut AI preprocessing pipeline — CLI orchestrator.

Usage (from the workspace root):
    python ml/data/run_pipeline.py --stage all
    python ml/data/run_pipeline.py --stage validate
    python ml/data/run_pipeline.py --stage splits
    python ml/data/run_pipeline.py --stage prepare [--target-size 512x512]
    python ml/data/run_pipeline.py --stage report

Raw data is read-only. Derived artifacts are written under datasets/processed
and the two markdown reports under docs/.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common import read_json, setup_logging, utc_now, write_json  # noqa: E402
from src.duplicates import cross_dataset_overlap  # noqa: E402
from src.image_utils import copy_image, parse_size, resize_mask  # noqa: E402
from src.report import render_preprocessing_report, render_statistics_report  # noqa: E402
from src.splits import make_split, split_manifest  # noqa: E402
from src.stats import build_statistics  # noqa: E402
from src.validate import validate_dataset  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Groundnut AI dataset preprocessing pipeline")
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets.json"))
    ap.add_argument("--datasets", default="", help="comma-separated subset (default: all in config)")
    ap.add_argument("--stage", choices=["all", "validate", "splits", "prepare", "report"], default="all")
    ap.add_argument("--copy-images", action="store_true", help="force materializing processed copies")
    ap.add_argument("--no-copy", action="store_true", help="disable materialization even if config enables it")
    ap.add_argument("--target-size", default=None, help="resize processed copies to WxH (e.g. 512x512)")
    ap.add_argument("--seed", type=int, default=None, help="override split seed")
    args = ap.parse_args(argv)

    cfg = read_json(args.config)
    processed_root = Path(cfg["processed_root"])
    logger = setup_logging(processed_root / "logs")
    run_id = utc_now()

    selected = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(cfg["datasets"])
    for k in selected:
        if k not in cfg["datasets"]:
            logger.error("unknown dataset '%s'", k)
            return 1
    seed = args.seed if args.seed is not None else int(cfg["split_seed"])
    target = parse_size(args.target_size) if args.target_size else None
    version = cfg["version"]
    logger.info("run=%s stage=%s datasets=%s seed=%d target=%s", run_id, args.stage, selected, seed, target)

    records = {}
    fps = {}
    for key in selected:
        rec, fp = validate_dataset(cfg, key, logger)
        records[key] = rec
        fps[key] = {str(p): h for p, h in fp.items()}

    overlaps = cross_dataset_overlap(fps)
    stats = build_statistics(records, overlaps)
    if overlaps:
        logger.warning("cross-dataset overlap: %d groups", len(overlaps))
    write_json(processed_root / "manifests" / f"statistics_{version}.json", stats)
    for key, rec in records.items():
        write_json(processed_root / "manifests" / f"{key}_validation_{version}.json", rec)

    splits = {}
    if args.stage in ("all", "splits", "prepare", "report"):
        for key in selected:
            rec = records[key]
            if rec["task"] == "field_eval":
                logger.info("[%s] no split generated (field-eval set)", key)
                continue
            items = sorted(rec["file_map"])
            dup_map = {st: g for st, g in rec["duplicate_map"].items() if st in rec["file_map"]}
            split = make_split(items, dict(cfg["split_ratios"]), seed, dup_map)
            man = split_manifest(key, version, dict(cfg["split_ratios"]), seed, split)
            write_json(processed_root / "splits" / f"{key}_split_{version}.json", man)
            rec["split"] = {k: {"count": len(v), "files": v} for k, v in split.items()}
            splits[key] = split
            logger.info("[%s] split %s", key, {k: len(v) for k, v in split.items()})

    if args.stage in ("all", "prepare", "report"):
        for key in selected:
            if args.no_copy:
                continue
            d = cfg["datasets"][key]
            if d.get("copy", False) or args.copy_images:
                _prepare_dataset(cfg, key, records[key], splits.get(key), processed_root, target, logger)

    if args.stage in ("all", "report"):
        report_dir = Path(cfg["report_dir"])
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "dataset_statistics.md").write_text(
            render_statistics_report(cfg, stats), encoding="utf-8"
        )
        (report_dir / "dataset_preprocessing.md").write_text(
            render_preprocessing_report(cfg, records, stats, args, seed, version, target, run_id),
            encoding="utf-8",
        )
        logger.info("reports written to %s", report_dir)

    logger.info("pipeline finished (run %s)", run_id)
    return 0



def _prepare_dataset(cfg: dict, key: str, rec: dict, split: dict | None,
                     processed_root: Path, target, logger: logging.Logger) -> None:
    """Materialize processed copies for a dataset (native or resized)."""
    d = cfg["datasets"][key]
    dst = processed_root / key
    logger.info("[%s] materializing processed dataset -> %s (target=%s)", key, dst, target or "native")
    if d["task"] == "detection":
        _prepare_yolo(cfg, d, rec, split, dst, target, logger)
    elif d["task"] == "segmentation":
        _prepare_seg(cfg, d, rec, split, dst, target, logger)
    else:
        logger.info("[%s] task '%s': no materialization configured", key, d["task"])


def _copy_label(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def _prepare_yolo(cfg: dict, d: dict, rec: dict, split: dict, dst: Path,
                  target, logger: logging.Logger) -> None:
    raw_root = Path(cfg["raw_root"])
    labels_dir = raw_root / d["labels_dir"]
    label_ext = d["label_ext"]
    for sname in SPLIT_NAMES:
        img_dir = dst / "images" / sname
        lab_dir = dst / "labels" / sname
        for stem in split.get(sname, []):
            src = rec["file_map"].get(stem)
            if not src:
                continue
            ext = Path(src).suffix
            copy_image(Path(src), img_dir / f"{stem}{ext}", target)
            _copy_label(labels_dir / f"{stem}{label_ext}", lab_dir / f"{stem}{label_ext}")
        logger.info("[%s] prepared split '%s' (%d images)", d.get("_key", "?"), sname, len(split.get(sname, [])))

    names = {int(k): v for k, v in d.get("classes", {}).items()}
    # Write a PORTABLE, cwd-relative path (scripts always run from the project
    # root), never a machine-specific absolute path like D:/Flutter/...
    try:
        rel_path = os.path.relpath(dst).replace("\\", "/")
    except Exception:
        rel_path = "."
    yaml = [f"# Auto-generated by ml/data/run_pipeline.py (version {cfg['version']})"]
    yaml.append("# NOTE: 'agri_data' both classes are WEEDS (non-groundnut plants/crops)")
    yaml.append(f"path: {rel_path}")
    yaml.append("train: images/train")
    yaml.append("val: images/val")
    yaml.append("test: images/test")
    yaml.append("names:")
    for i in sorted(names):
        yaml.append(f"  {i}: {names[i]}")
    (dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")


def _prepare_seg(cfg: dict, d: dict, rec: dict, split: dict, dst: Path,
                 target, logger: logging.Logger) -> None:
    raw_root = Path(cfg["raw_root"])
    masks_dir = raw_root / d["masks_dir"]
    mask_ext = d["mask_ext"]
    for sname in SPLIT_NAMES:
        img_dir = dst / "images" / sname
        msk_dir = dst / "masks" / sname
        for stem in split.get(sname, []):
            src = rec["file_map"].get(stem)
            if not src:
                continue
            ext = Path(src).suffix
            copy_image(Path(src), img_dir / f"{stem}{ext}", target)
            mask_src = masks_dir / f"{stem}{mask_ext}"
            if mask_src.exists():
                mask_dst = msk_dir / f"{stem}{mask_ext}"
                mask_dst.parent.mkdir(parents=True, exist_ok=True)
                if target:
                    resize_mask(mask_src, mask_dst, target)
                else:
                    shutil.copy2(mask_src, mask_dst)
        logger.info("[%s] prepared split '%s' (%d images)", d.get("_key", "?"), sname, len(split.get(sname, [])))


if __name__ == "__main__":
    raise SystemExit(main())
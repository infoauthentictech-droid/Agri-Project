"""Markdown report rendering for the preprocessing pipeline (docs/ outputs)."""
from __future__ import annotations

from typing import Any, Dict, List


def _table(headers: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_statistics_report(cfg: Dict[str, Any], stats: Dict[str, Any]) -> str:
    """Human-readable dataset statistics report (docs/dataset_statistics.md)."""
    lines: List[str] = []
    lines.append("# Dataset Statistics Report")
    lines.append("")
    lines.append(f"* Generated: `{stats['generated_at']}`")
    lines.append(f"* Config version: `{cfg.get('version')}`")
    lines.append(f"* Raw root: `{cfg.get('raw_root')}`")
    lines.append("")

    rows = []
    for key, s in stats["datasets"].items():
        rows.append(
            [
                key,
                s["task"],
                str(s["total_images"]),
                str(s["valid_images"]),
                str(s["invalid_images"]),
                str(s["corrupted_files"]),
                str(s["duplicate_groups"]),
                str(s["duplicate_extra_copies"]),
            ]
        )
    lines.append("## Overall summary")
    lines.append("")
    lines.append(
        _table(
            ["dataset", "task", "total", "valid", "invalid", "corrupted", "dup groups", "dup copies"],
            rows,
        )
    )

    lines.append("")
    lines.append("## Per-dataset details")
    for key, s in stats["datasets"].items():
        lines.append("")
        lines.append(f"### {key}")
        lines.append("")
        dims = ", ".join(f"{k} x {s['dimensions'][k]}" for k in sorted(s["dimensions"], key=lambda k: -s["dimensions"][k])[:6])
        lines.append(f"* Image dimensions (top 6): {dims or 'n/a'}")
        lines.append(f"* Color modes: {s['modes'] or 'n/a'}")
        if s.get("annotation_type") == "yolo_detection":
            lines.append("* Annotations: YOLO detection")
            lines.append(f"  * Images with labels: {s['images_with_labels']} / {s['total_images']}")
            lines.append(f"  * Total boxes: {s['total_boxes']}")
            lines.append(
                f"  * Boxes/image: min {s['boxes_per_image_min']}, mean {s['boxes_per_image_mean']}, max {s['boxes_per_image_max']}"
            )
            lines.append(f"  * Boxes per class: {s['class_boxes']}")
            lines.append(f"  * Images containing each class: {s['images_per_class']}")
            lines.append(f"  * Class imbalance ratio (max/min boxes): {s['class_imbalance_ratio']}")
            lines.append(
                f"  * Empty annotations: {s['empty_annotations']}, invalid labels: {s['invalid_labels']}, malformed lines: {s['malformed_lines']}"
            )
        if s.get("annotation_type") == "segmentation":
            lines.append("* Annotations: segmentation masks")
            lines.append(f"  * Images with mask: {s['images_with_mask']} / {s['total_images']}")
            lines.append(f"  * Masks with no image: {s['masks_without_image']}, images with no mask: {s['images_without_mask']}")
            lines.append(f"  * Size mismatches: {s['size_mismatches']}, mode issues: {s['mode_issues']}, empty masks: {s['empty_masks']}")
            lines.append(f"  * Masks containing each class: {s['mask_class_presence']}")
            lines.append(f"  * Class imbalance ratio (crop vs weed presence): {s['class_imbalance_ratio']}")

    overlaps = stats.get("cross_dataset_overlaps", [])
    lines.append("")
    lines.append("## Cross-dataset duplicate (leakage) check")
    lines.append("")
    if not overlaps:
        lines.append("No content-identical images shared between datasets.")
    else:
        lines.append(f"**{len(overlaps)} fingerprint groups are shared by two or more datasets.**")
        lines.append("")
        rows = []
        for o in overlaps:
            locs = "; ".join(f"{l['dataset']}: {l['path'].split('/')[-1]}" for l in o["locations"][:3])
            rows.append([o["fingerprint"][:12], str(o["count"]), locs])
        lines.append(_table(["fingerprint", "files", "locations (first 3)"], rows))
    return "\n".join(lines)



def render_preprocessing_report(cfg: Dict[str, Any], records: Dict[str, Any],
                                stats: Dict[str, Any], args: Any,
                                seed: int, version: str, target: Any, run_id: str) -> str:
    """Pipeline execution report (docs/dataset_preprocessing.md)."""
    lines: List[str] = []
    lines.append("# Dataset Preprocessing Report — Groundnut AI")
    lines.append("")
    lines.append(f"* Run id: `{run_id}`")
    lines.append(f"* Config: `{cfg.get('version')}` | split seed `{seed}` | ratios `{cfg.get('split_ratios')}`")
    lines.append(f"* Target size: `{target or 'native (no resize)'}`")
    lines.append(f"* Datasets processed: `{', '.join(records)}`")
    lines.append("")

    lines.append("## 1. Pipeline architecture")
    lines.append("")
    lines.append("```")
    lines.append("ml/data/")
    lines.append("├── configs/datasets.json      # dataset registry (raw paths, tasks, classes)")
    lines.append("├── src/")
    lines.append("│   ├── common.py              # logging, JSON IO, fingerprints")
    lines.append("│   ├── image_utils.py         # image validation, resize, normalize")
    lines.append("│   ├── duplicates.py          # duplicate + cross-dataset overlap detection")
    lines.append("│   ├── labels_yolo.py         # YOLO annotation validation")
    lines.append("│   ├── labels_seg.py          # segmentation mask validation")
    lines.append("│   ├── stats.py               # class distribution / imbalance statistics")
    lines.append("│   ├── splits.py              # reproducible duplicate-aware splits")
    lines.append("│   ├── validate.py            # per-dataset validation orchestrator")
    lines.append("│   └── report.py              # markdown report renderers")
    lines.append("├── run_pipeline.py            # CLI orchestrator")
    lines.append("└── tests/                     # unit tests (stdlib unittest)")
    lines.append("```")
    lines.append("")
    lines.append("Raw data is **never written to**; all derived artifacts live under `datasets/processed/`.")
    lines.append("")

    lines.append("## 2. Validation results per dataset")
    lines.append("")
    for key, rec in records.items():
        lines.append(f"### {key} ({rec['task']})")
        lines.append("")
        issues = "; ".join(rec["issues"]) if rec["issues"] else "none"
        lines.append(f"* Images: total `{rec['total_images']}`, valid `{rec['valid_images']}`, invalid `{rec['invalid_images']}`")
        lines.append(f"* Duplicate groups: `{len(rec['duplicate_groups'])}` (extra copies `{rec['duplicate_extra_copies']}`)")
        lines.append(f"* Issues found: {issues}")
        if rec["labels"]:
            lab = rec["labels"]
            lines.append(
                f"* Labels: `{lab['images_with_labels']}` paired, boxes `{lab['total_boxes']}`, "
                f"classes `{lab['class_boxes']}`, empty `{len(lab['empty_annotations'])}`, "
                f"invalid `{len(lab['invalid_labels'])}`"
            )
        if rec["masks"]:
            msk = rec["masks"]
            lines.append(
                f"* Masks: paired `{msk['images_with_mask']}`, size mismatches `{len(msk['size_mismatches'])}`, "
                f"empty `{len(msk['empty_masks'])}`, class presence `{msk['class_presence']}`"
            )
        if rec.get("split"):
            sc = {k: v["count"] for k, v in rec["split"].items()}
            lines.append(f"* Split (seed {seed}): {sc}")
        lines.append("")

    lines.append("## 3. Reproducibility")
    lines.append("")
    lines.append("* Deterministic splits: fixed seed + sorted duplicate-aware grouping.")
    lines.append("* Fingerprints: SHA-256 (whole file for small files; sampled 3-block signature for large).")
    lines.append("* All per-file records and statistics are stored as JSON under `datasets/processed/manifests/`.")
    lines.append("")

    lines.append("## 4. How to reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append("# 1) validate + statistics")
    lines.append("python ml/data/run_pipeline.py --stage validate")
    lines.append("# 2) reproducible splits")
    lines.append("python ml/data/run_pipeline.py --stage splits")
    lines.append("# 3) materialize processed datasets (resize with --target-size 512x512)")
    lines.append("python ml/data/run_pipeline.py --stage prepare")
    lines.append("# 4) regenerate this report")
    lines.append("python ml/data/run_pipeline.py --stage report")
    lines.append("# 5) unit tests")
    lines.append("python -m unittest discover -s ml/data/tests -v")
    lines.append("```")
    lines.append("")
    lines.append("## 5. Known issues / decisions")
    lines.append("")
    lines.append("* `agri_data` class ids `0`/`1` are stored as `class_0`/`class_1` — crop/weed mapping is **pending user confirmation**.")
    lines.append("* 18 `Primary` images overlap the Peanut segmentation dataset (cross-dataset leakage); exclude them before using `Primary` as an independent field test set.")
    lines.append("* 118 Peanut masks contain no crop (weed-only) and crop:weed foreground is imbalanced; use per-class IoU/F1 at evaluation time.")
    lines.append("* Normalization is implemented as a load-time transform (`normalize_image`) and is never written back to disk.")
    lines.append("")
    return "\n".join(lines)
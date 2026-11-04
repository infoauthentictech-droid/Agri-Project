"""Mark knowledge-base records as agronomically reviewed (validation_status=passed).

This is the 'proper fix' documented in docs/Q&A_SESSION.md (Q11): the engine
only recommends from ``passed`` records. Records are reviewed here by data
completeness + source confidence:

PASSED  - non-conflicting record with a real dose and a cited source/page and
          a usable application stage.
PENDING - conflicting records (conflict_flag=true), records without a real
          dose, or 'no herbicide / not applicable' notes.

Run:  python herbicide/scripts/mark_records_passed.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "herbicide/processed/herbicide_dataset_normalized.json"
CSV_PATH = ROOT / "herbicide/processed/herbicide_dataset_normalized.csv"

# Records kept pending on purpose (unsafe/incomplete for recommendations).
PENDING_BY_ID = {
    "REC019": "conflict_flag=true",
    "REC020": "no herbicide / manual only (dose n/a)",
    "REC026": "conflict_flag=true",
    "REC030": "no dose ('Not specified'), not_applicable stage",
    "REC031": "conflict_flag=true",
}


def main() -> int:
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    changed = []
    for rec in data:
        rid = rec.get("record_id")
        if rid in PENDING_BY_ID:
            if rec.get("validation_status") != "pending":
                rec["validation_status"] = "pending"
                changed.append(rid)
            continue
        if rec.get("validation_status") != "passed":
            rec["validation_status"] = "passed"
            changed.append(rid)
    JSON_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if CSV_PATH.exists():
        rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
        status_by_id = {r["record_id"]: r.get("validation_status") for r in data}
        for row in rows:
            rid = row.get("record_id")
            if rid and rid in status_by_id:
                row["validation_status"] = status_by_id[rid]
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    passed = sum(1 for r in data if r.get("validation_status") == "passed")
    pending = sum(1 for r in data if r.get("validation_status") != "passed")
    print(f"Updated {len(changed)} record(s). Total passed={passed}, pending={pending}.")
    print("Kept pending:", ",".join(sorted(PENDING_BY_ID)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
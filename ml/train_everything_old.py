"""Train every trainable algorithm on all four dataset types, one run at a time, on the GPU.

Runs go shortest first, so results arrive early and problems show up quickly. CycleGAN comes
last because it is by far the slowest. Progress is saved after every run, so if the computer
restarts, running this again skips what finished and continues the rest (YOLO and CycleGAN
continue from their last checkpoint). A failed run is recorded and the next one still starts;
running this again retries it.

Usage (from the project root):
    python ml/train_everything.py              start or continue
    python ml/train_everything.py --status     show progress and exit
    python ml/train_everything.py --no-cyclegan
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "experiments" / "by_dataset"
STATUS = OUT / "training_status.json"
LOGS = OUT / "logs"

DATASET_ORDER = ("secondary", "primary", "primary_augmented", "hybrid")
CLASSIFIERS = ("cnn", "mobilenet", "efficientnet", "resnet50", "vgg16", "resnet50_mobilenet",
               "resnet50_efficientnet", "resnet50_efficientnet_mobilenet")
YOLO_LR0 = "0.001"  # the scripts' default 0.01 is about 10 times the usual rate for AdamW


@dataclass
class Run:
    name: str
    args: List[str]


def classification_root(dataset: str) -> Path:
    # Hybrid models go where the backend looks for them.
    return PROJECT / "experiments" / "classification" if dataset == "hybrid" else OUT / dataset / "classification"


def yolo_output(dataset: str, folder: str, name: str) -> Path:
    return PROJECT / "experiments" / folder / name if dataset == "hybrid" else OUT / dataset / folder / name


def build_runs(cyclegan: bool = True) -> List[Run]:
    runs: List[Run] = []
    for dataset in DATASET_ORDER:
        runs.append(Run(f"{dataset}/plant_gate", [
            "ml/gate/train_plant_gate.py", "--dataset-type", dataset,
            "--output", str(OUT / dataset / "plant_gate" / "plant_gate_model.joblib")]))
        for model in CLASSIFIERS:
            runs.append(Run(f"{dataset}/classification/{model}", [
                "ml/classification/train_classifier.py", "--model", model, "--dataset-type", dataset,
                "--epochs", "30", "--batch", "32", "--device", "0", "--output-root", str(classification_root(dataset))]))
        runs.append(Run(f"{dataset}/detection", [
            "ml/detection/train_yolo.py", "--dataset-type", dataset, "--epochs", "100", "--batch", "16",
            "--lr0", YOLO_LR0, "--device", "0", "--resume",
            "--output", str(yolo_output(dataset, "yolo", "agri_yolov8n"))]))
        runs.append(Run(f"{dataset}/segmentation", [
            "ml/segmentation/train_yoloseg.py", "--dataset-type", dataset, "--epochs", "100", "--batch", "16",
            "--lr0", YOLO_LR0, "--device", "0", "--resume",
            "--output", str(yolo_output(dataset, "yoloseg", "agri_yolov8n_seg"))]))
    if cyclegan:
        for dataset in DATASET_ORDER:
            domains = PROJECT / "datasets" / dataset / "cyclegan"
            runs.append(Run(f"{dataset}/cyclegan", [
                "ml/synthesis/cyclegan.py", "--domain_a", str(domains / "domain_a" / "train"),
                "--domain_b", str(domains / "domain_b" / "train"), "--epochs", "200", "--batch", "1",
                "--device", "0", "--resume", "--output", str(OUT / dataset / "cyclegan")]))
    return runs


def load_status(path: Path = STATUS) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_status(status: dict, path: Path = STATUS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
    temporary.replace(path)


def runs_to_do(runs: Sequence[Run], status: dict) -> List[Run]:
    """Everything not finished: never started, interrupted or failed."""
    return [run for run in runs if status.get(run.name, {}).get("state") != "done"]


def print_status(runs: Sequence[Run], status: dict) -> None:
    done = [r for r in runs if status.get(r.name, {}).get("state") == "done"]
    failed = [r for r in runs if status.get(r.name, {}).get("state") == "failed"]
    print(f"{len(done)} of {len(runs)} runs finished, {len(failed)} failed")
    for run in runs:
        info = status.get(run.name, {})
        state = info.get("state", "waiting")
        minutes = f"{info['minutes']:.1f} min" if "minutes" in info else ""
        print(f"  {state:8s} {run.name:58s} {minutes}")


def run_one(run: Run, status: dict) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / (run.name.replace("/", "__") + ".log")
    status[run.name] = {"state": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "log": str(log)}
    save_status(status)
    print(f"[{time.strftime('%H:%M:%S')}] START {run.name}", flush=True)
    start = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as out:
        # No stdin: a script that stops to ask a question fails instead of waiting for days.
        code = subprocess.run([sys.executable, "-W", "ignore", "-u", *run.args], cwd=PROJECT,
                              stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    minutes = (time.time() - start) / 60
    status[run.name].update(state="done" if code == 0 else "failed", exit_code=code, minutes=round(minutes, 1),
                            finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    save_status(status)
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE ' if code == 0 else 'FAILED'} {run.name} ({minutes:.1f} min)", flush=True)
    return code == 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--no-cyclegan", action="store_true", help="leave out the slowest runs")
    args = ap.parse_args(argv)

    runs = build_runs(cyclegan=not args.no_cyclegan)
    status = load_status()
    if args.status:
        print_status(runs, status)
        return 0

    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available. Training is GPU-only.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    failures = 0
    for run in runs_to_do(runs, status):
        failures += not run_one(run, status)
    print_status(runs, status)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

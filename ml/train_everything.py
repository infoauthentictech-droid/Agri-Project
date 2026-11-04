"""Train every trainable algorithm on all four dataset types, one run at a time, on the GPU.

Runs go shortest first, so results arrive early and problems show up quickly. CycleGAN comes
last because it is by far the slowest. Progress is saved after every run, so if the computer
restarts, running this again skips what finished and continues the rest (YOLO and CycleGAN
continue from their last checkpoint). A failed run is tried again a few times, because a GPU
error can be brief. If it still fails it is recorded and the next one starts; running this
again retries it.

This version shows FULL training output in real-time.

Usage (from the project root):
    python ml/train_everything.py              start or continue
    python ml/train_everything.py --status     show progress and exit
    python ml/train_everything.py --no-cyclegan
    python ml/train_everything.py --hybrid-only  train only on hybrid dataset
"""
from __future__ import annotations

import argparse
import json
import os
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
CLASSIFIER_LR = {"vgg16": "0.0001"}  # at 0.001 VGG16 stayed at chance level and predicted one class for every image
CLASSIFIER_BATCH = {"resnet50_efficientnet_mobilenet": "16"}  # at 32 it peaks at 8.06 GiB on this 7.96 GiB GPU
RETRIES = 3
RETRY_WAIT_SECONDS = 60


@dataclass
class Run:
    name: str
    args: List[str]


def classification_root(dataset: str) -> Path:
    # Hybrid models go where the backend looks for them.
    return PROJECT / "experiments" / "classification" if dataset == "hybrid" else OUT / dataset / "classification"


def yolo_output(dataset: str, folder: str, name: str) -> Path:
    return PROJECT / "experiments" / folder / name if dataset == "hybrid" else OUT / dataset / folder / name


def build_runs(cyclegan: bool = True, hybrid_only: bool = False) -> List[Run]:
    runs: List[Run] = []
    
    if hybrid_only:
        datasets = ("hybrid",)
    else:
        datasets = DATASET_ORDER
    
    for dataset in datasets:
        # Gate output stays out of the backend folder until it is copied there on purpose.
        runs.append(Run(f"{dataset}/plant_gate", [
            "ml/gate/train_plant_gate.py", "--dataset-type", dataset,
            "--output", str(OUT / dataset / "plant_gate" / "plant_gate_model.joblib")]))

        # Each dataset needs its own output root, or its models overwrite the previous dataset's.
        for model in CLASSIFIERS:
            runs.append(Run(f"{dataset}/classification/{model}", [
                "ml/classification/train_classifier.py", "--model", model, "--dataset-type", dataset,
                "--epochs", "30", "--batch", CLASSIFIER_BATCH.get(model, "32"), "--lr", CLASSIFIER_LR.get(model, "0.001"),
                "--device", "0", "--output-root", str(classification_root(dataset))]))

        # --resume continues from the last checkpoint after an interruption, if there is one.
        runs.append(Run(f"{dataset}/detection", [
            "ml/detection/train_yolo.py", "--dataset-type", dataset, "--epochs", "100", "--batch", "16",
            "--lr0", YOLO_LR0, "--device", "0", "--resume",
            "--output", str(yolo_output(dataset, "yolo", "agri_yolov8n"))]))

        runs.append(Run(f"{dataset}/segmentation", [
            "ml/segmentation/train_yoloseg.py", "--dataset-type", dataset, "--epochs", "100", "--batch", "16",
            "--lr0", YOLO_LR0, "--device", "0", "--resume",
            "--output", str(yolo_output(dataset, "yoloseg", "agri_yolov8n_seg"))]))
    
    if cyclegan and not hybrid_only:
        for dataset in datasets:
            domains = PROJECT / "datasets" / dataset / "cyclegan"
            if domains.exists():
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
    
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%H:%M:%S')}] STARTING: {run.name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(run.args)}")
    print(f"{'='*80}\n")
    
    start = time.time()
    
    # FIXED: Stream output to console AND log file for real-time visibility
    # Removed -W ignore to show all training information including warnings
    # Added environment variables to disable progress bars and force plain text output
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered Python output
    env["PYTHONIOENCODING"] = "utf-8"  # Ultralytics writes UTF-8; the Windows default cp1252 cannot decode its output
    env["NO_PROGRESS_BAR"] = "1"  # Disable progress bars for YOLO
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow optimizations that might interfere
    
    with open(log, "a", encoding="utf-8", errors="replace") as out:
        # Use Popen to stream output in real-time with full buffering disabled
        process = subprocess.Popen(
            [sys.executable, "-u", *run.args],
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",  # A character that cannot be decoded must not stop the queue
            bufsize=0,  # No buffering for immediate output
            env=env  # Pass environment variables
        )
        
        # Stream output line by line with immediate flush
        for line in process.stdout:
            line = line.rstrip()
            if line:  # Only print non-empty lines
                print(line)  # Show in console
                out.write(line + "\n")  # Write to log file
                out.flush()  # Ensure log is written immediately
            sys.stdout.flush()  # Force console flush
        
        process.wait()
        code = process.returncode
    
    minutes = (time.time() - start) / 60
    status[run.name].update(
        state="done" if code == 0 else "failed",
        exit_code=code,
        minutes=round(minutes, 1),
        finished=time.strftime("%Y-%m-%d %H:%M:%S")
    )
    save_status(status)
    
    print(f"\n{'='*80}")
    print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if code == 0 else 'FAILED'}: {run.name} ({minutes:.1f} min)")
    print(f"Exit code: {code}")
    print(f"{'='*80}\n")
    
    return code == 0


def run_with_retries(run: Run, status: dict, wait: float = RETRY_WAIT_SECONDS) -> bool:
    for attempt in range(RETRIES + 1):
        if run_one(run, status):
            return True
        if attempt < RETRIES:
            print(f"Trying {run.name} again in {wait:.0f} s (retry {attempt + 1} of {RETRIES})")
            time.sleep(wait)
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--no-cyclegan", action="store_true", help="leave out the slowest runs")
    ap.add_argument("--hybrid-only", action="store_true", help="train only on hybrid dataset (recommended)")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # A character the console cannot show must not stop the queue

    runs = build_runs(cyclegan=not args.no_cyclegan, hybrid_only=args.hybrid_only)
    status = load_status()
    
    if args.status:
        print_status(runs, status)
        return 0

    import torch
    if not torch.cuda.is_available():
        print("ERROR: no CUDA GPU available. Training is GPU-only.")
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"\nTotal runs: {len(runs)}")
    print(f"Runs to do: {len(runs_to_do(runs, status))}")
    print()

    failures = 0
    for run in runs_to_do(runs, status):
        failures += not run_with_retries(run, status)
    
    print("\n" + "="*80)
    print("FINAL STATUS")
    print("="*80)
    print_status(runs, status)
    print("="*80)
    
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

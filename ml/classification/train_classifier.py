"""Classification training script for groundnut vs weed classification.

Supports multiple model architectures: cnn, vgg16, resnet50, mobilenet, efficientnet, and hybrid models.

Dataset Types:
- primary: Original field images
- primary_augmented: Augmented versions of primary dataset
- secondary: External public datasets
- hybrid: Combined dataset (default for training)

Usage:
    python ml/classification/train_classifier.py --model cnn --epochs 30 --dataset_type hybrid
    python ml/classification/train_classifier.py --model efficientnet --epochs 30 --batch 32 --dataset_type primary_augmented
    python ml/classification/train_classifier.py --model resnet50_mobilenet --epochs 30 --dataset_type hybrid
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, resnet50, mobilenet_v2, efficientnet_b0
from sklearn.metrics import classification_report

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.gpu_loader import GpuBatchLoader, classifier_transform  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "datasets" / "hybrid" / "classification"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "classification"


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """Build a model by name."""
    model_name = model_name.lower()

    # Check for hybrid models first
    if model_name in ("resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"):
        from ml.models.hybrid import build_hybrid_model
        return build_hybrid_model(model_name, num_classes, pretrained)

    if model_name == "cnn":
        # Import CNN directly from local path
        cnn_path = Path(__file__).parent.parent / "models" / "cnn.py"
        spec = __import__('importlib.util').util.spec_from_file_location("cnn_module", cnn_path)
        cnn_module = __import__('importlib.util').util.module_from_spec(spec)
        sys.modules['ml.models.cnn'] = cnn_module
        spec.loader.exec_module(cnn_module)
        CNN = cnn_module.CNN
        return CNN(num_classes=num_classes, input_channels=3)

    elif model_name == "vgg16":
        model = vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[5] = nn.Dropout(p=0.5)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    elif model_name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model

    elif model_name == "mobilenet":
        model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        return model

    elif model_name == "efficientnet":
        model = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_model(
    model_name: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    dataset_root: Path,
    early_stop_patience: int = 10,
) -> Optional[Dict]:
    """Train a single model with comprehensive validation."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Validate hyperparameters
    if epochs <= 0:
        print(f"Error: epochs must be positive, got {epochs}")
        return None
    if batch_size <= 0:
        print(f"Error: batch_size must be positive, got {batch_size}")
        return None
    if learning_rate <= 0:
        print(f"Error: learning_rate must be positive, got {learning_rate}")
        return None

    # GPU only: never fall back to the CPU
    try:
        device_idx = int(device)
    except ValueError:
        print(f"Invalid device: {device}. Training is GPU-only, use a device number (0, 1, ...)")
        return None
    if not torch.cuda.is_available():
        print("Error: CUDA GPU not available. Training is GPU-only.")
        return None
    if device_idx >= torch.cuda.device_count():
        print(f"Error: GPU {device_idx} not available. Available GPUs: {torch.cuda.device_count()}")
        return None
    print(f"Using GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
    device = f"cuda:{device_idx}"

    # Validate dataset structure
    if not dataset_root.exists():
        print(f"Error: Dataset directory {dataset_root} does not exist")
        return None

    for split in ['train', 'val', 'test']:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"Error: {split} directory not found: {split_dir}")
            return None

    # Determine appropriate image size per model
    # FIX: Use 224 for ResNet50 and hybrids, 128 for others
    if model_name in ['resnet50', 'resnet50_mobilenet', 'resnet50_efficientnet', 'resnet50_efficientnet_mobilenet']:
        image_size = 224
    else:
        image_size = 128

    try:
        train_dataset = ImageFolder(dataset_root / "train")
        val_dataset = ImageFolder(dataset_root / "val")
        test_dataset = ImageFolder(dataset_root / "test")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None

    # Validate class balance
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    if len(train_dataset.classes) != 2:
        print(f"Warning: Expected 2 classes, found {len(train_dataset.classes)}")

    # Images are decoded and augmented on the GPU, not by CPU DataLoader workers.
    train_loader = GpuBatchLoader(train_dataset.samples, batch_size, classifier_transform(image_size, train=True),
                                  shuffle=True, device=device)
    val_loader = GpuBatchLoader(val_dataset.samples, batch_size, classifier_transform(image_size, train=False),
                                shuffle=False, device=device)
    test_loader = GpuBatchLoader(test_dataset.samples, batch_size, classifier_transform(image_size, train=False),
                                 shuffle=False, device=device)

    # Model
    try:
        model = build_model(model_name, num_classes=2, pretrained=True)
        model = model.to(device)
    except Exception as e:
        print(f"Error building model: {e}")
        return None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    history = []
    actual_epochs_trained = 0

    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

            scheduler.step()

            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
            })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_output_dir / "best_model.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            # FIX: Enable early stopping
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            actual_epochs_trained = epoch

        except Exception as e:
            print(f"Error during epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            return None  # A run cut short must count as failed, not as a finished model

    # FIX: Ensure best_model.pth exists
    if not (model_output_dir / "best_model.pth").exists():
        print("Warning: best_model.pth not created during training. Saving current model.")
        torch.save(model.state_dict(), model_output_dir / "best_model.pth")

    # Test evaluation
    try:
        model.load_state_dict(torch.load(model_output_dir / "best_model.pth"))
        test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    except Exception as e:
        print(f"Error during test evaluation: {e}")
        return None

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Classification report with explicit target names
    target_names = train_dataset.classes
    try:
        report = classification_report(
            test_labels,
            test_preds,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    except Exception as e:
        print(f"Error generating classification report: {e}")
        report = {}

    # Save results
    train_time = time.time() - start_time

    class_metrics = {}
    for class_name in target_names:
        if class_name in report:
            class_metrics[class_name] = {
                "precision": round(report[class_name]["precision"], 4),
                "recall": round(report[class_name]["recall"], 4),
                "f1-score": round(report[class_name]["f1-score"], 4),
                "support": int(report[class_name]["support"]),
            }

    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    model_params = sum(p.numel() for p in model.parameters())

    results = {
        "model": model_name,
        "dataset_root": str(dataset_root),
        "image_size": image_size,
        "epochs_trained": actual_epochs_trained,
        "best_val_accuracy": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "test_precision": round(report.get("weighted avg", {}).get("precision", 0), 4),
        "test_recall": round(report.get("weighted avg", {}).get("recall", 0), 4),
        "test_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
        "training_time_seconds": round(train_time, 2),
        "model_size_mb": round(model_size_mb, 2),
        "model_parameters": int(model_params),
        "class_metrics": class_metrics,
        "history": history,
        "classification_report": report,
    }

    (model_output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (model_output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Results saved to {model_output_dir}")
    print(f"{'='*60}\n")

    return results


def main(argv=None) -> int:
    """Main entry point with comprehensive validation."""
    global DATA_ROOT, OUTPUT_DIR
    parser = argparse.ArgumentParser(
        description="Train classification models for groundnut vs weed detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="cnn",
        choices=["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet",
                 "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"],
        help="Model architecture"
    )
    parser.add_argument(
        "--dataset-type",
        default="hybrid",
        choices=["primary", "primary_augmented", "secondary", "hybrid"],
        help="Dataset type to use"
    )
    parser.add_argument("--all-models", action="store_true", help="Train all models")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device",
        default="0",
        help="GPU device number (0, 1, ...); training is GPU-only"
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_DIR),
        help="Folder for results, so runs on different dataset types do not overwrite each other"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )

    args = parser.parse_args(argv)

    # Set dataset root based on dataset_type
    DATA_ROOT = PROJECT_ROOT / "datasets" / args.dataset_type / "classification"
    OUTPUT_DIR = Path(args.output_root).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Using dataset: {args.dataset_type}")
    print(f"Dataset path: {DATA_ROOT}")

    models_to_train = (
        ["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet",
         "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"]
        if args.all_models
        else [args.model]
    )

    all_results = {}
    failed_models = []

    for model_name in models_to_train:
        try:
            results = train_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch,
                learning_rate=args.lr,
                device=args.device,
                dataset_root=DATA_ROOT,
                early_stop_patience=args.early_stop_patience,
            )
            if results:
                all_results[model_name] = results
            else:
                failed_models.append(model_name)
                print(f"Failed to train {model_name}")
        except Exception as e:
            failed_models.append(model_name)
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Report failures
    if failed_models:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(failed_models)} model(s) failed to train:")
        for model in failed_models:
            print(f"  - {model}")
        print(f"{'='*60}")

    # Save summary if training multiple models
    if len(models_to_train) > 1 and all_results:
        comparison = {
            "timestamp": str(time.time()),
            "dataset_type": args.dataset_type,
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "early_stop_patience": args.early_stop_patience,
            "device": args.device,
            "models": {}
        }

        for model_name, results in all_results.items():
            comparison["models"][model_name] = {
                "test_accuracy": results["test_accuracy"],
                "test_precision": results["test_precision"],
                "test_recall": results["test_recall"],
                "test_f1": results["test_f1"],
                "model_size_mb": results.get("model_size_mb", "N/A"),
                "model_parameters": results.get("model_parameters", "N/A"),
                "training_time_seconds": results["training_time_seconds"],
                "epochs_trained": results["epochs_trained"],
                "class_metrics": results.get("class_metrics", {}),
            }

        # Add ranking
        by_accuracy = sorted(comparison["models"].items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
        by_size = sorted(comparison["models"].items(), key=lambda x: x[1].get("model_size_mb", float('inf')))
        by_speed = sorted(comparison["models"].items(), key=lambda x: x[1]["training_time_seconds"])

        comparison["rankings"] = {
            "by_accuracy": [m[0] for m in by_accuracy],
            "by_size": [m[0] for m in by_size],
            "by_training_speed": [m[0] for m in by_speed],
        }

        summary_path = OUTPUT_DIR / "benchmark_summary.json"
        summary_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<30} {'Accuracy':<10} {'Size(MB)':<10} {'Time(s)':<10}")
        print("-" * 60)
        for model_name, _ in by_accuracy:
            metrics = comparison["models"][model_name]
            print(f"{model_name:<30} {metrics['test_accuracy']:<10} {metrics.get('model_size_mb', 'N/A'):<10} {metrics['training_time_seconds']:<10}")
        print(f"\nBest accuracy: {by_accuracy[0][0]} ({by_accuracy[0][1]['test_accuracy']}%)")
        print(f"Smallest model: {by_size[0][0]} ({by_size[0][1].get('model_size_mb', 'N/A')} MB)")
        print(f"Fastest training: {by_speed[0][0]} ({by_speed[0][1]['training_time_seconds']}s)")
        print(f"{'='*60}")
        print(f"Benchmark summary saved to {summary_path}")

    # Return non-zero if any models failed
    return 1 if failed_models else 0


if __name__ == "__main__":
    sys.exit(main())

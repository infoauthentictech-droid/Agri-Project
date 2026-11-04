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
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vgg16, resnet50, mobilenet_v2, efficientnet_b0
from sklearn.metrics import classification_report

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / "datasets" / "hybrid" / "classification"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "classification"


def main():
    parser = argparse.ArgumentParser(description="Train classification models")
    parser.add_argument("--model", type=str, required=True, help="Model name (cnn, vgg16, resnet50, mobilenet, efficientnet, resnet50_mobilenet, resnet50_efficientnet, resnet50_efficientnet_mobilenet)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dataset_type", type=str, default="hybrid", choices=["primary", "primary_augmented", "secondary", "hybrid"], help="Dataset type to use")
    parser.add_argument("--all-models", action="store_true", help="Train all models")

    args = parser.parse_args()

    global DATA_ROOT
    DATA_ROOT = PROJECT_ROOT / "datasets" / args.dataset_type / "classification"

    print(f"Using dataset: {args.dataset_type}")
    print(f"Dataset path: {DATA_ROOT}")

    if not DATA_ROOT.exists():
        print(f"Error: Dataset directory {DATA_ROOT} does not exist")
        print("Please prepare the dataset first using the appropriate script:")
        print("  - Primary: python ml/data/prepare_primary_dataset.py --task classification")
        print("  - Augmented: python ml/data/prepare_augmented_dataset.py --task classification")
        print("  - Secondary: python ml/data/prepare_secondary_dataset.py --task classification")
        print("  - Hybrid: python ml/data/create_hybrid_dataset.py --task classification")
        sys.exit(1)

    if args.all_models:
        models = ["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet", "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"]
    else:
        models = [args.model]

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        train_model(model_name, args.epochs, args.batch, args.lr, args.device)


if __name__ == "__main__":
    main()


def get_transforms(image_size: int, augmentation: str = "standard"):
    """Get data transforms."""
    if augmentation == "none":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif augmentation == "standard":
        # Problem 7, 8, 9: Enhanced augmentation to prevent overfitting
        train_transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),  # Slightly larger for random crop
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """Build a model by name."""
    model_name = model_name.lower()

    # Check for hybrid models first
    if model_name in ("resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"):
        from ml.models.hybrid import build_hybrid_model
        return build_hybrid_model(model_name, num_classes, pretrained)

    if model_name == "cnn":
        # Import CNN directly from local path
        import sys
        from pathlib import Path
        cnn_path = Path(__file__).parent.parent / "models" / "cnn.py"
        spec = __import__('importlib.util').util.spec_from_file_location("cnn_module", cnn_path)
        cnn_module = __import__('importlib.util').util.module_from_spec(spec)
        sys.modules['ml.models.cnn'] = cnn_module
        spec.loader.exec_module(cnn_module)
        CNN = cnn_module.CNN
        return CNN(num_classes=num_classes, input_channels=3)

    elif model_name == "vgg16":
        model = vgg16(weights="IMAGENET1K_V1" if pretrained else None)
        # Problem 7: Add dropout for regularization
        model.classifier[5] = nn.Dropout(p=0.5)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model

    elif model_name == "resnet50":
        model = resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        # Problem 7: Add dropout to final layer
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        return model

    elif model_name == "mobilenet":
        model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        # Problem 7: Add dropout to final layer
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
):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()}")
    print(f"{'='*60}")

    # Check GPU
    if device != "cpu":
        device_idx = int(device)
        if torch.cuda.is_available():
            if device_idx >= torch.cuda.device_count():
                print(f"Error: GPU {device_idx} not available. Available GPUs: {torch.cuda.device_count()}")
                return None
            print(f"Using GPU {device_idx}: {torch.cuda.get_device_name(device_idx)}")
            device = f"cuda:{device_idx}"
        else:
            print("CUDA not available, falling back to CPU")
            device = "cpu"

    # Data
    image_size = 128
    train_transform, val_transform = get_transforms(image_size, "standard")

    train_dataset = ImageFolder(DATA_ROOT / "train", transform=train_transform)
    val_dataset = ImageFolder(DATA_ROOT / "val", transform=val_transform)
    test_dataset = ImageFolder(DATA_ROOT / "test", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Model
    model = build_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    # Problem 7: Increased weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # Problem 8: Use cosine annealing for smoother learning rate decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_acc = 0.0
    patience_counter = 0
    history = []

    model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
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

        # Early stopping disabled - train for full epochs
        # if patience_counter >= early_stop_patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break

    # Test evaluation
    model.load_state_dict(torch.load(model_output_dir / "best_model.pth"))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)

    print(f"\nTest Accuracy: {test_acc:.2f}%")

    # Classification report
    target_names = train_dataset.classes
    report = classification_report(test_labels, test_preds, target_names=target_names, output_dict=True)

    # Save results
    train_time = time.time() - start_time

    # Problem 6, 18: Add class-wise metrics
    class_metrics = {}
    for class_name in target_names:
        if class_name in report:
            class_metrics[class_name] = {
                "precision": round(report[class_name]["precision"], 4),
                "recall": round(report[class_name]["recall"], 4),
                "f1-score": round(report[class_name]["f1-score"], 4),
                "support": int(report[class_name]["support"]),
            }

    # Problem 20: Add model size and compute metrics
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    model_params = sum(p.numel() for p in model.parameters())

    results = {
        "model": model_name,
        "epochs_trained": epoch,
        "best_val_accuracy": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "test_precision": round(report["weighted avg"]["precision"], 4),
        "test_recall": round(report["weighted avg"]["recall"], 4),
        "test_f1": round(report["weighted avg"]["f1-score"], 4),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cnn",
                    choices=["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet",
                             "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"],
                    help="Model architecture")
    ap.add_argument("--all-models", action="store_true", help="Train all models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--device", default="0", help="GPU device (0, 1, ...) or 'cpu'")
    args = ap.parse_args(argv)

    models_to_train = ["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet",
                       "resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"] if args.all_models else [args.model]

    all_results = {}

    for model_name in models_to_train:
        try:
            results = train_model(
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch,
                learning_rate=args.lr,
                device=args.device,
            )
            if results:
                all_results[model_name] = results
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save summary if training multiple models
    if len(models_to_train) > 1:
        # Problem 20: Create detailed comparison summary
        comparison = {
            "timestamp": str(time.time()),
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
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
        print(f"{'Model':<15} {'Accuracy':<10} {'Size(MB)':<10} {'Time(s)':<10}")
        print("-" * 60)
        for model_name, _ in by_accuracy:
            metrics = comparison["models"][model_name]
            print(f"{model_name:<15} {metrics['test_accuracy']:<10} {metrics.get('model_size_mb', 'N/A'):<10} {metrics['training_time_seconds']:<10}")
        print(f"\nBest accuracy: {by_accuracy[0][0]} ({by_accuracy[0][1]['test_accuracy']}%)")
        print(f"Smallest model: {by_size[0][0]} ({by_size[0][1].get('model_size_mb', 'N/A')} MB)")
        print(f"Fastest training: {by_speed[0][0]} ({by_speed[0][1]['training_time_seconds']}s)")
        print(f"{'='*60}")
        print(f"Benchmark summary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Comprehensive evaluation of all trained models.

This script evaluates:
1. YOLO detection model on test set
2. All classification models on test set
3. Plant gate model on test set
4. Generates detailed metrics, confusion matrices, and prediction visualizations
"""

import sys
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not installed, using matplotlib for confusion matrices")
import torch
import cv2
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
YOLO_MODEL_PATH = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
YOLO_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "data.yaml"
CLASSIFICATION_RESULTS_DIR = PROJECT_ROOT / "experiments" / "classification"
EVALUATION_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation"
EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Build a classification model."""
    import torch.nn as nn
    from torchvision.models import vgg16, resnet50, mobilenet_v2, efficientnet_b0
    
    if model_name == "cnn":
        # Load CNN from the actual CNN module
        import importlib.util
        cnn_path = PROJECT_ROOT / "ml" / "models" / "cnn.py"
        spec = importlib.util.spec_from_file_location("cnn_module", cnn_path)
        cnn_module = importlib.util.module_from_spec(spec)
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


def evaluate_yolo():
    """Evaluate YOLO detection model on test set."""
    print("=" * 60)
    print("EVALUATING YOLO DETECTION MODEL")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Load model
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    model = YOLO(str(YOLO_MODEL_PATH))
    
    # Evaluate on test set
    print("Evaluating on TEST set...")
    metrics = model.val(
        data=str(YOLO_DATA_PATH),
        split='test',
        device=0,
        verbose=True,
        save_json=True,
        save_hybrid=True
    )
    
    # Extract metrics
    results = {
        "model": "YOLOv8n",
        "split": "test",
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "class_metrics": {
            "crop": {
                "precision": float(metrics.box.p[0]),
                "recall": float(metrics.box.r[0]),
            },
            "weed": {
                "precision": float(metrics.box.p[1]),
                "recall": float(metrics.box.r[1]),
            }
        }
    }
    
    print("\n" + "=" * 60)
    print("YOLO TEST SET RESULTS")
    print("=" * 60)
    print(f"mAP50: {results['mAP50']:.4f}")
    print(f"mAP50-95: {results['mAP50-95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"\nClass-wise metrics:")
    print(f"  Crop - Precision: {results['class_metrics']['crop']['precision']:.4f}, Recall: {results['class_metrics']['crop']['recall']:.4f}")
    print(f"  Weed - Precision: {results['class_metrics']['weed']['precision']:.4f}, Recall: {results['class_metrics']['weed']['recall']:.4f}")
    
    # Save results
    output_file = EVALUATION_OUTPUT_DIR / "yolo_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


def evaluate_classification_models():
    """Evaluate all classification models on test set."""
    print("\n" + "=" * 60)
    print("EVALUATING CLASSIFICATION MODELS")
    print("=" * 60)
    
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    
    # Data paths
    test_data_path = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test data
    test_dataset = ImageFolder(test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {test_dataset.classes}")
    
    # Models to evaluate
    models_to_eval = ["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet"]
    all_results = {}
    
    for model_name in models_to_eval:
        print(f"\n--- Evaluating {model_name.upper()} ---")
        
        model_path = CLASSIFICATION_RESULTS_DIR / model_name / "best_model.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue
        
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        
        # For all models, try state_dict loading
        try:
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model = build_model(model_name, num_classes=2, pretrained=False)
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model = build_model(model_name, num_classes=2, pretrained=False)
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model = build_model(model_name, num_classes=2, pretrained=False)
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model = model.to(device)
            model.eval()
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Evaluate
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Per-class metrics
        report = classification_report(all_labels, all_preds, target_names=test_dataset.classes, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            "model": model_name,
            "test_accuracy": float(accuracy),
            "test_precision": float(report["weighted avg"]["precision"]),
            "test_recall": float(report["weighted avg"]["recall"]),
            "test_f1": float(report["weighted avg"]["f1-score"]),
            "class_metrics": {
                class_name: {
                    "precision": float(report[class_name]["precision"]),
                    "recall": float(report[class_name]["recall"]),
                    "f1-score": float(report[class_name]["f1-score"]),
                    "support": int(report[class_name]["support"]),
                }
                for class_name in test_dataset.classes
            },
            "confusion_matrix": cm.tolist()
        }
        
        all_results[model_name] = results
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {results['test_precision']:.4f}")
        print(f"Recall: {results['test_recall']:.4f}")
        print(f"F1-Score: {results['test_f1']:.4f}")
        print(f"\nClass-wise metrics:")
        for class_name in test_dataset.classes:
            metrics = results['class_metrics'][class_name]
            print(f"  {class_name}: P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
        else:
            # Use matplotlib if seaborn is not available
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'{model_name.upper()} Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(test_dataset.classes))
            plt.xticks(tick_marks, test_dataset.classes, rotation=45)
            plt.yticks(tick_marks, test_dataset.classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        cm_path = EVALUATION_OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
    
    # Save all results
    output_file = EVALUATION_OUTPUT_DIR / "classification_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll classification results saved to: {output_file}")
    
    return all_results


def evaluate_plant_gate():
    """Evaluate plant gate model."""
    print("\n" + "=" * 60)
    print("EVALUATING PLANT GATE MODEL")
    print("=" * 60)
    
    import joblib
    from PIL import Image
    import numpy as np
    
    # Load model
    model_path = PROJECT_ROOT / "backend" / "app" / "utils" / "plant_gate_model.joblib"
    metadata_path = PROJECT_ROOT / "backend" / "app" / "utils" / "plant_gate_model.json"
    
    if not model_path.exists():
        print("Plant gate model not found")
        return None
    
    rf_model = joblib.load(model_path)
    
    # Handle case where model was saved as dict
    if isinstance(rf_model, dict):
        if 'model' in rf_model:
            rf_model = rf_model['model']
        elif 'rf_model' in rf_model:
            rf_model = rf_model['rf_model']
        else:
            print("Error: Loaded dict but cannot find model key")
            return None
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata.get("threshold", 0.5)
    print(f"Using threshold: {threshold}")
    
    # Import the plant gate feature extraction
    from ml.gate.features import ImageNetFeaturizer, extract_features, feature_vector
    
    # Initialize featurizer
    featurizer = ImageNetFeaturizer(name="resnet50")
    
    # Load test data
    # Positive samples (plants)
    positive_dir = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
    negative_dir = PROJECT_ROOT / "datasets" / "negatives_real"
    
    # Sample some images for evaluation
    positive_images = list(positive_dir.glob("*/*.jpg"))[:50]
    negative_images = list(negative_dir.glob("*.jpg"))[:50]
    
    print(f"Positive samples: {len(positive_images)}")
    print(f"Negative samples: {len(negative_images)}")
    
    # Evaluate
    all_preds = []
    all_labels = []
    
    # Positive samples
    for img_path in positive_images:
        try:
            img = Image.open(img_path).convert('RGB')
            rgb = np.array(img)
            feats = extract_features(rgb, None, (img.width, img.height), featurizer)
            features = feature_vector(feats).reshape(1, -1)
            proba = rf_model.predict_proba(features)[0, 1]
            pred = 1 if proba >= threshold else 0
            all_preds.append(pred)
            all_labels.append(1)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Negative samples
    for img_path in negative_images:
        try:
            img = Image.open(img_path).convert('RGB')
            rgb = np.array(img)
            feats = extract_features(rgb, None, (img.width, img.height), featurizer)
            features = feature_vector(feats).reshape(1, -1)
            proba = rf_model.predict_proba(features)[0, 1]
            pred = 1 if proba >= threshold else 0
            all_preds.append(pred)
            all_labels.append(0)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    
    # False positive and false negative rates
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    results = {
        "threshold": threshold,
        "accuracy": float(accuracy),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "total_samples": len(all_labels)
    }
    
    print(f"\nPlant Gate Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"False Negative Rate: {fnr:.4f}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    
    # Save results
    output_file = EVALUATION_OUTPUT_DIR / "plant_gate_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


def generate_summary_report(yolo_results, classification_results, plant_gate_results):
    """Generate a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)
    
    summary = {
        "evaluation_date": str(Path(__file__).stat().st_mtime),
        "yolo_detection": yolo_results,
        "classification": classification_results,
        "plant_gate": plant_gate_results,
    }
    
    # Save summary
    output_file = EVALUATION_OUTPUT_DIR / "comprehensive_evaluation_report.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComprehensive report saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print("\n--- YOLO Detection ---")
    print(f"mAP50: {yolo_results['mAP50']:.4f}")
    print(f"mAP50-95: {yolo_results['mAP50-95']:.4f}")
    print(f"Precision: {yolo_results['precision']:.4f}")
    print(f"Recall: {yolo_results['recall']:.4f}")
    
    print("\n--- Classification Models ---")
    for model_name, results in classification_results.items():
        print(f"{model_name.upper()}: Acc={results['test_accuracy']:.4f}, P={results['test_precision']:.4f}, R={results['test_recall']:.4f}")
    
    print("\n--- Plant Gate ---")
    if plant_gate_results:
        print(f"Accuracy: {plant_gate_results['accuracy']:.4f}")
        print(f"FPR: {plant_gate_results['false_positive_rate']:.4f}")
        print(f"FNR: {plant_gate_results['false_negative_rate']:.4f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nAll results saved to: {EVALUATION_OUTPUT_DIR}")
    print("Review the following files:")
    print(f"  - {EVALUATION_OUTPUT_DIR / 'comprehensive_evaluation_report.json'}")
    print(f"  - {EVALUATION_OUTPUT_DIR / 'yolo_evaluation.json'}")
    print(f"  - {EVALUATION_OUTPUT_DIR / 'classification_evaluation.json'}")
    print(f"  - {EVALUATION_OUTPUT_DIR / 'plant_gate_evaluation.json'}")
    print(f"  - Confusion matrices: {EVALUATION_OUTPUT_DIR / '*_confusion_matrix.png'}")


def main():
    """Run all evaluations."""
    print("Starting comprehensive model evaluation...")
    print(f"Output directory: {EVALUATION_OUTPUT_DIR}")
    
    # Evaluate YOLO
    yolo_results = evaluate_yolo()
    
    # Evaluate classification models
    classification_results = evaluate_classification_models()
    
    # Evaluate plant gate
    plant_gate_results = evaluate_plant_gate()
    
    # Generate summary
    generate_summary_report(yolo_results, classification_results, plant_gate_results)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

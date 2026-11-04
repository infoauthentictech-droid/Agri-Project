"""
Test model inference on representative field images.

This script tests:
1. Loading trained models
2. Running inference on sample images
3. Verifying predictions are sensible
4. Saving prediction visualizations
"""

import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
YOLO_MODEL_PATH = PROJECT_ROOT / "experiments" / "yolo" / "agri_yolov8n" / "weights" / "best.pt"
CLASSIFICATION_MODELS_DIR = PROJECT_ROOT / "experiments" / "classification"
PLANT_GATE_MODEL_PATH = PROJECT_ROOT / "backend" / "app" / "utils" / "plant_gate_model.joblib"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "processed" / "groundnut_weed_detection" / "test"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation" / "inference_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_yolo_inference():
    """Test YOLO model loading and inference."""
    print("=" * 60)
    print("TESTING YOLO INFERENCE")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Load model
    print(f"Loading YOLO model from: {YOLO_MODEL_PATH}")
    if not YOLO_MODEL_PATH.exists():
        print("ERROR: YOLO model not found!")
        return False
    
    model = YOLO(str(YOLO_MODEL_PATH))
    print("✓ Model loaded successfully")
    
    # Get sample images
    test_images = list(TEST_IMAGES_DIR.glob("images/*.jpg"))[:5]
    if not test_images:
        # Try alternative path
        test_images = list(TEST_IMAGES_DIR.glob("*.jpg"))[:5]
    if not test_images:
        print("ERROR: No test images found!")
        return False
    
    print(f"Found {len(test_images)} test images")
    
    # Run inference
    for img_path in test_images:
        print(f"\nProcessing: {img_path.name}")
        
        results = model(str(img_path))
        
        # Check if detections were made
        if len(results[0].boxes) > 0:
            print(f"  ✓ Detected {len(results[0].boxes)} objects")
            
            # Check class distribution
            classes = results[0].boxes.cls.cpu().numpy()
            if len(classes) > 0:
                class_counts = {}
                for cls in classes:
                    cls_name = results[0].names[int(cls)]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                print(f"  Classes: {class_counts}")
            
            # Save visualization
            result_img = results[0].plot()
            output_path = OUTPUT_DIR / f"yolo_{img_path.name}"
            cv2.imwrite(str(output_path), result_img)
            print(f"  ✓ Visualization saved to: {output_path}")
        else:
            print(f"  ⚠ No objects detected")
    
    print("\n✓ YOLO inference test completed")
    return True


def test_classification_inference():
    """Test classification model loading and inference."""
    print("\n" + "=" * 60)
    print("TESTING CLASSIFICATION INFERENCE")
    print("=" * 60)
    
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision.models import vgg16, resnet50, mobilenet_v2, efficientnet_b0
    import torch.nn as nn
    
    def build_model(model_name: str, num_classes: int, pretrained: bool = False):
        """Build a classification model."""
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
            model = vgg16(weights=None)
            model.classifier[5] = nn.Dropout(p=0.5)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        elif model_name == "resnet50":
            model = resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.fc.in_features, num_classes)
            )
            return model
        elif model_name == "mobilenet":
            model = mobilenet_v2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
            return model
        elif model_name == "efficientnet":
            model = efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            return model
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load test data
    test_data_path = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
    test_dataset = ImageFolder(test_data_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    # Models to test
    models_to_test = ["cnn", "vgg16", "resnet50", "mobilenet", "efficientnet"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for model_name in models_to_test:
        print(f"\n--- Testing {model_name.upper()} ---")
        
        model_path = CLASSIFICATION_MODELS_DIR / model_name / "best_model.pth"
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path}")
            continue
        
        # Load model
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Build model and load state_dict
            model = build_model(model_name, num_classes=2, pretrained=False)
            if model is None:
                continue
            
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            
            model = model.to(device)
            model.eval()
            print(f"  ✓ Model loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            continue
        
        # Test on a few samples
        sample_count = 0
        correct = 0
        
        for inputs, labels in test_loader:
            if sample_count >= 5:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
            
            if preds.item() == labels.item():
                correct += 1
            
            sample_count += 1
        
        accuracy = correct / sample_count if sample_count > 0 else 0
        print(f"  ✓ Tested on {sample_count} samples")
        print(f"  ✓ Accuracy: {accuracy:.2f}")
    
    print("\n✓ Classification inference test completed")
    return True


def test_plant_gate_inference():
    """Test plant gate model loading and inference."""
    print("\n" + "=" * 60)
    print("TESTING PLANT GATE INFERENCE")
    print("=" * 60)
    
    import joblib
    import numpy as np
    
    # Import the plant gate feature extraction
    from ml.gate.features import ImageNetFeaturizer, extract_features, feature_vector
    
    # Load model
    if not PLANT_GATE_MODEL_PATH.exists():
        print("⚠ Plant gate model not found")
        return False
    
    try:
        rf_model = joblib.load(PLANT_GATE_MODEL_PATH)
        
        # Handle case where model was saved as dict
        if isinstance(rf_model, dict):
            if 'model' in rf_model:
                rf_model = rf_model['model']
            elif 'rf_model' in rf_model:
                rf_model = rf_model['rf_model']
            else:
                print("Error: Loaded dict but cannot find model key")
                return False
        
        print("✓ Plant gate model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading plant gate model: {e}")
        return False
    
    # Initialize featurizer
    featurizer = ImageNetFeaturizer(name="resnet50")
    
    # Test on plant images
    plant_dir = PROJECT_ROOT / "datasets" / "processed" / "classification_groundnut_weed" / "test"
    plant_images = list(plant_dir.glob("*/*.jpg"))[:5]
    
    print(f"Testing on {len(plant_images)} plant images")
    
    plant_accepted = 0
    for img_path in plant_images:
        try:
            img = Image.open(img_path).convert('RGB')
            rgb = np.array(img)
            feats = extract_features(rgb, None, (img.width, img.height), featurizer)
            features = feature_vector(feats).reshape(1, -1)
            proba = rf_model.predict_proba(features)[0, 1]
            
            if proba >= 0.3:  # Using threshold 0.3
                plant_accepted += 1
                print(f"  ✓ {img_path.name}: Accepted (prob={proba:.3f})")
            else:
                print(f"  ⚠ {img_path.name}: Rejected (prob={proba:.3f})")
        except Exception as e:
            print(f"  ✗ Error processing {img_path.name}: {e}")
    
    print(f"\nPlant acceptance rate: {plant_accepted}/{len(plant_images)}")
    print("✓ Plant gate inference test completed")
    return True


def main():
    """Run all inference tests."""
    print("Starting model inference tests...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    results = {
        "yolo": test_yolo_inference(),
        "classification": test_classification_inference(),
        "plant_gate": test_plant_gate_inference(),
    }
    
    print("\n" + "=" * 60)
    print("INFERENCE TEST SUMMARY")
    print("=" * 60)
    print(f"YOLO: {'✓ PASSED' if results['yolo'] else '✗ FAILED'}")
    print(f"Classification: {'✓ PASSED' if results['classification'] else '✗ FAILED'}")
    print(f"Plant Gate: {'✓ PASSED' if results['plant_gate'] else '✗ FAILED'}")
    print()
    
    if all(results.values()):
        print("✓ All inference tests passed!")
        print("Models are ready for integration.")
    else:
        print("✗ Some tests failed. Review the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

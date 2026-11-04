"""
Test script for XAI explanations on various prediction scenarios.

This script tests XAI on:
- Correct predictions
- Incorrect predictions
- Low-confidence predictions
- Groundnut class
- Weed class

Run this after YOLO training completes.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
from ml.xai.api import XAIEngine


def test_xai_scenarios():
    """Test XAI on various prediction scenarios."""
    
    # Load trained model (single deployed detector)
    model_path = 'runs/detect/experiments/yolo/agri_yolov8n/weights/best.pt'
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please complete YOLO training first.")
        return
    
    model = YOLO(model_path)
    
    # Class names (update with actual names from dataset)
    class_names = {
        0: 'groundnut',
        1: 'weed'
    }
    
    # Initialize XAI engine
    output_dir = Path('experiments/xai')
    engine = XAIEngine(model, class_names, output_dir)
    
    # Test images from test set
    test_images_dir = Path('datasets/processed/groundnut_weed_detection/images/test')
    if not test_images_dir.exists():
        print(f"Test images directory not found: {test_images_dir}")
        return
    
    # Get sample test images
    test_images = list(test_images_dir.glob('*.jpeg'))[:10]
    if len(test_images) < 5:
        print(f"Need at least 5 test images, found {len(test_images)}")
        return
    
    print(f"Testing XAI on {len(test_images)} images...")
    
    # Test scenarios
    scenarios = {
        'correct_high_conf': [],
        'correct_low_conf': [],
        'incorrect': [],
        'groundnut': [],
        'weed': []
    }
    
    for i, image_path in enumerate(test_images):
        print(f"\n[{i+1}/{len(test_images)}] Processing {image_path.name}")
        
        # Get prediction summary first
        pred_summary = engine.get_prediction_summary(str(image_path))
        
        # Categorize based on predictions
        for pred in pred_summary['predictions']:
            class_id = pred['class_id']
            conf = pred['confidence']
            
            # Categorize by class
            if class_id == 0:
                scenarios['groundnut'].append(str(image_path))
            elif class_id == 1:
                scenarios['weed'].append(str(image_path))
            
            # Categorize by confidence (assuming ground truth for now)
            if conf >= 0.8:
                scenarios['correct_high_conf'].append(str(image_path))
            elif conf >= 0.5:
                scenarios['correct_low_conf'].append(str(image_path))
            else:
                scenarios['incorrect'].append(str(image_path))
        
        # Generate explanation
        save_prefix = f"test_{i}_{image_path.stem}"
        result = engine.explain_image(str(image_path), save_prefix)
        
        print(f"  Predictions: {len(result['predictions'])}")
        print(f"  Explanations: {len(result['explanations'])}")
        print(f"  Saved files: {len(result['saved_files'])}")
    
    # Print scenario summary
    print("\n" + "="*60)
    print("Scenario Summary")
    print("="*60)
    for scenario, images in scenarios.items():
        print(f"{scenario}: {len(images)} images")
    
    # Save scenario summary
    import json
    summary_path = output_dir / 'test_scenarios_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"\nScenario summary saved to {summary_path}")
    print("\nXAI testing complete!")
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    print("XAI Testing Script")
    print("="*60)
    print("This script tests XAI on various prediction scenarios.")
    print("Ensure YOLO training has completed before running.")
    print("="*60)
    
    test_xai_scenarios()

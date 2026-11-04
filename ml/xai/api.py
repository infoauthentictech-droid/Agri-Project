"""
API-ready functions for XAI explanations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from .gradcam import GradCAMExplainer
from .visualization import XAIVisualizer
from .explanation import ExplanationGenerator


class XAIEngine:
    """
    API-ready XAI engine for generating explanations.
    
    This class provides a simple API for:
    - Generating Grad-CAM explanations
    - Creating visualizations
    - Generating text explanations
    - Saving results
    """
    
    def __init__(
        self,
        model,
        class_names: Optional[Dict[int, str]] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize XAI engine.
        
        Args:
            model: Trained YOLO model
            class_names: Dictionary mapping class IDs to names
            output_dir: Directory to save outputs
        """
        self.model = model
        self.class_names = class_names or {}
        self.output_dir = output_dir or Path('experiments/xai')
        
        # Initialize components
        self.explainer = GradCAMExplainer(model)
        self.visualizer = XAIVisualizer(self.output_dir)
        self.explanation_generator = ExplanationGenerator(self.class_names)
    
    def explain_image(
        self,
        image_path: str,
        save_prefix: Optional[str] = None,
        return_visualizations: bool = True
    ) -> Dict:
        """
        Generate complete explanation for an image.
        
        Args:
            image_path: Path to input image
            save_prefix: Prefix for saved files
            return_visualizations: Whether to return visualization data
        
        Returns:
            Dictionary with explanation results
        """
        # Run prediction
        yolo_results = self.model(image_path, verbose=False)
        
        # Generate explanation
        explanation_data = self.explainer.explain_prediction(
            image_path,
            yolo_results,
            self.class_names
        )
        
        # Generate visualizations if requested
        saved_files = {}
        if return_visualizations:
            save_prefix = save_prefix or Path(image_path).stem
            saved_files = self.visualizer.visualize_explanation(
                explanation_data,
                save_prefix
            )
        
        # Generate text explanations
        explanations = []
        for exp in explanation_data['explanations']:
            if exp['success']:
                method = exp.get('method', 'gradcam')
                text_exp = self.explanation_generator.generate_explanation(
                    exp['prediction'], {'method': method}
                )
                explanations.append({
                    'prediction': text_exp.prediction,
                    'explanation_text': text_exp.explanation_text,
                    'visualization_description': text_exp.visualization_description,
                    'limitations': text_exp.limitations,
                    'method': method,
                })
        
        # Save summary (saved_files paths are converted to strings so the result
        # is directly JSON-serializable for the API layer)
        saved_files = {k: str(v) for k, v in saved_files.items()}

        # Convert predictions to plain JSON-safe primitive types (numpy arrays
        # and numpy scalars are not guaranteed JSON-serializable).
        safe_predictions = []
        for pred in explanation_data['predictions']:
            box = pred.get('box')
            safe_predictions.append({
                'class_id': int(pred.get('class_id', 0)),
                'class_name': str(pred.get('class_name', '')),
                'confidence': round(float(pred.get('confidence', 0.0)), 6),
                'box': [float(v) for v in box] if box is not None else None,
            })

        return {
            'image_path': image_path,
            'predictions': safe_predictions,
            'explanations': explanations,
            'saved_files': saved_files,
            'success': True
        }
    
    def explain_batch(
        self,
        image_paths: List[str],
        return_visualizations: bool = True
    ) -> List[Dict]:
        """
        Generate explanations for multiple images.
        
        Args:
            image_paths: List of image paths
            return_visualizations: Whether to return visualization data
        
        Returns:
            List of explanation results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            save_prefix = f"batch_{i}_{Path(image_path).stem}"
            result = self.explain_image(image_path, save_prefix, return_visualizations)
            results.append(result)
        
        return results
    
    def get_prediction_summary(self, image_path: str) -> Dict:
        """
        Get prediction summary without full explanation.
        
        Args:
            image_path: Path to input image
        
        Returns:
            Dictionary with prediction summary
        """
        yolo_results = self.model(image_path, verbose=False)
        
        predictions = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    predictions.append({
                        'class_id': cls_id,
                        'class_name': self.class_names.get(cls_id, f'class_{cls_id}'),
                        'confidence': conf,
                        'bbox': xyxy.tolist()
                    })
        
        return {
            'image_path': image_path,
            'num_predictions': len(predictions),
            'predictions': predictions
        }


# Convenience API functions

def explain_single_image(
    model,
    image_path: str,
    class_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    API function to explain a single image.
    
    Args:
        model: Trained YOLO model
        image_path: Path to input image
        class_names: Dictionary mapping class IDs to names
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with explanation results
    """
    engine = XAIEngine(model, class_names, output_dir)
    return engine.explain_image(image_path)


def explain_images_batch(
    model,
    image_paths: List[str],
    class_names: Optional[Dict[int, str]] = None,
    output_dir: Optional[Path] = None
) -> List[Dict]:
    """
    API function to explain multiple images.
    
    Args:
        model: Trained YOLO model
        image_paths: List of image paths
        class_names: Dictionary mapping class IDs to names
        output_dir: Directory to save outputs
    
    Returns:
        List of explanation results
    """
    engine = XAIEngine(model, class_names, output_dir)
    return engine.explain_batch(image_paths)


def get_predictions(
    model,
    image_path: str,
    class_names: Optional[Dict[int, str]] = None
) -> Dict:
    """
    API function to get predictions without explanation.
    
    Args:
        model: Trained YOLO model
        image_path: Path to input image
        class_names: Dictionary mapping class IDs to names
    
    Returns:
        Dictionary with prediction summary
    """
    engine = XAIEngine(model, class_names)
    return engine.get_prediction_summary(image_path)


if __name__ == '__main__':
    print("XAI API functions ready")
    print("Available functions:")
    print("- explain_single_image()")
    print("- explain_images_batch()")
    print("- get_predictions()")

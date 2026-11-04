"""
Visualization module for XAI explanations.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import json


class XAIVisualizer:
    """
    Visualizer for XAI explanations.
    
    Generates:
    - Original image
    - Grad-CAM heatmap
    - Overlay visualization
    - Combined comparison plots
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or Path('experiments/xai')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def apply_colormap(self, heatmap: np.ndarray, colormap: str = 'jet') -> np.ndarray:
        """
        Apply colormap to heatmap.
        
        Args:
            heatmap: Normalized heatmap (0-1)
            colormap: Colormap name (default: jet)
        
        Returns:
            Colored heatmap
        """
        # Convert to 0-255
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        return heatmap_color
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (BGR)
            heatmap: Normalized heatmap (0-1)
            alpha: Transparency of heatmap (0-1)
        
        Returns:
            Overlay image
        """
        # Apply colormap
        heatmap_color = self.apply_colormap(heatmap)
        
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap_color, (image.shape[1], image.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_resized, alpha, 0)
        
        return overlay
    
    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        boxes: list,
        class_names: Optional[Dict[int, str]] = None,
        confidences: Optional[list] = None
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
            class_names: Dictionary mapping class IDs to names
            confidences: List of confidence scores
        
        Returns:
            Image with bounding boxes
        """
        image_copy = image.copy()
        
        colors = {
            0: (0, 255, 0),    # Green
            1: (0, 0, 255),    # Red
        }
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Get class ID (assuming first element is class if box is from detection)
            if len(box) == 6:  # (class_id, x1, y1, x2, y2, conf)
                class_id = int(box[0])
                x1, y1, x2, y2 = map(int, box[1:5])
            else:
                class_id = 0  # Default
            
            color = colors.get(class_id, (255, 0, 0))
            
            # Draw box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Class {class_id}"
            if class_names and class_id in class_names:
                label = class_names[class_id]
            
            if confidences and i < len(confidences):
                label += f" {confidences[i]:.2f}"
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_copy
    
    def create_comparison_plot(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        title: str = "XAI Visualization",
        save_path: Optional[Path] = None
    ) -> None:
        """
        Create comparison plot with original, heatmap, and overlay.
        
        Args:
            original_image: Original image
            heatmap: Grad-CAM heatmap
            overlay: Overlay visualization
            title: Plot title
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison plot to {save_path}")
        
        plt.close()
    
    def visualize_explanation(
        self,
        explanation_data: Dict,
        save_prefix: str
    ) -> Dict[str, Path]:
        """
        Visualize complete explanation.
        
        Args:
            explanation_data: Explanation data from GradCAMExplainer
            save_prefix: Prefix for saved files
        
        Returns:
            Dictionary of saved file paths
        """
        saved_files = {}
        
        # Load original image
        image_path = explanation_data['image_path']
        original_image = cv2.imread(image_path)
        
        if original_image is None:
            print(f"Warning: Could not load image {image_path}")
            return saved_files
        
        # Process each prediction
        for i, exp in enumerate(explanation_data['explanations']):
            if not exp['success']:
                continue
            
            pred = exp['prediction']
            cam = exp['cam']
            
            # Generate heatmap visualization
            heatmap_color = self.apply_colormap(cam)
            heatmap_path = self.output_dir / f"{save_prefix}_heatmap_{i}.png"
            cv2.imwrite(str(heatmap_path), heatmap_color)
            saved_files[f'heatmap_{i}'] = heatmap_path
            
            # Generate overlay
            overlay = self.overlay_heatmap(original_image, cam)
            overlay_path = self.output_dir / f"{save_prefix}_overlay_{i}.png"
            cv2.imwrite(str(overlay_path), overlay)
            saved_files[f'overlay_{i}'] = overlay_path
            
            # Generate comparison plot
            comparison_path = self.output_dir / f"{save_prefix}_comparison_{i}.png"
            self.create_comparison_plot(
                original_image,
                cam,
                overlay,
                title=f"Prediction: {pred['class_name']} (Conf: {pred['confidence']:.2f})",
                save_path=comparison_path
            )
            saved_files[f'comparison_{i}'] = comparison_path
        
        return saved_files
    
    def save_explanation_summary(
        self,
        explanation_data: Dict,
        saved_files: Dict[str, Path],
        output_path: Path
    ) -> None:
        """
        Save explanation summary as JSON.
        
        Args:
            explanation_data: Explanation data
            saved_files: Dictionary of saved file paths
            output_path: Path to save summary
        """
        def _json_default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Path):
                return str(o)
            return str(o)

        summary = {
            'image_path': explanation_data['image_path'],
            'image_shape': list(explanation_data.get('image_shape', [])),
            'num_predictions': len(explanation_data['predictions']),
            'predictions': explanation_data['predictions'],
            'explanations': [
                {
                    'success': exp['success'],
                    'error': exp.get('error'),
                    'prediction': exp['prediction']
                }
                for exp in explanation_data['explanations']
            ],
            'saved_files': {k: str(v) for k, v in saved_files.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=_json_default)
        
        print(f"Saved explanation summary to {output_path}")


if __name__ == '__main__':
    # Example usage
    visualizer = XAIVisualizer()
    print("XAI Visualizer initialized")

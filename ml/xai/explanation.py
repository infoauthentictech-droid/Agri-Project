"""
Text explanation generation for XAI visualizations.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class Explanation:
    """Text explanation for model prediction."""
    prediction: str
    confidence: float
    explanation_text: str
    visualization_description: str
    limitations: List[str]


class ExplanationGenerator:
    """
    Generate text explanations for XAI visualizations.
    
    The explanations describe what the visualization shows without claiming
    that the highlighted region proves biological causation.
    """
    
    def __init__(self, class_names: Optional[Dict[int, str]] = None):
        """
        Initialize explanation generator.
        
        Args:
            class_names: Dictionary mapping class IDs to names
        """
        self.class_names = class_names or {}
    
    def generate_explanation(
        self,
        prediction_data: Dict,
        heatmap_data: Optional[Dict] = None
    ) -> Explanation:
        """
        Generate text explanation for a prediction.
        
        Args:
            prediction_data: Dictionary with prediction info
            heatmap_data: Optional dictionary with heatmap statistics
        
        Returns:
            Explanation object
        """
        class_id = prediction_data.get('class_id', 0)
        confidence = prediction_data.get('confidence', 0.0)
        class_name = self.class_names.get(class_id, f'class_{class_id}')
        
        # Generate main explanation
        explanation_text = self._generate_main_explanation(class_name, confidence)
        
        # Generate visualization description
        viz_description = self._generate_visualization_description(heatmap_data)
        
        # Generate limitations
        limitations = self._generate_limitations()
        
        return Explanation(
            prediction=class_name,
            confidence=confidence,
            explanation_text=explanation_text,
            visualization_description=viz_description,
            limitations=limitations
        )
    
    def _generate_main_explanation(self, class_name: str, confidence: float) -> str:
        """
        Generate main explanation text.
        
        Args:
            class_name: Predicted class name
            confidence: Confidence score
        
        Returns:
            Explanation text
        """
        if confidence >= 0.9:
            confidence_level = "very high"
        elif confidence >= 0.7:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        explanation = (
            f"The model predicted '{class_name}' with {confidence_level} confidence "
            f"({confidence:.2f}). This prediction is based on the model's learned "
            f"features from the training dataset."
        )
        
        return explanation
    
    def _generate_visualization_description(self, heatmap_data: Optional[Dict]) -> str:
        """
        Generate description of the visualization.

        Args:
            heatmap_data: Optional heatmap statistics
        """
        method = (heatmap_data or {}).get('method', 'gradcam')
        if method == 'fallback_activation':
            return (
                "The gradient path could not produce a genuine Grad-CAM map for "
                "this image, so a non-gradient activation/blur overlay is shown "
                "instead. It is NOT a model-gradient explanation and should be "
                "interpreted only as a rough visual emphasis."
            )

        description = (
            "The heatmap highlights image regions the model's feature maps "
            "weighted most strongly for this prediction. Warmer colors "
            "(red/orange) indicate regions with higher contribution, while "
            "cooler colors (blue) indicate lower contribution. This is a "
            "gradient-computed Grad-CAM heatmap: the map is derived from the "
            "gradient of the predicted class score with respect to the last "
            "convolutional feature maps."
        )

        if heatmap_data:
            if 'max_activation' in heatmap_data:
                description += f" The maximum activation intensity is {heatmap_data['max_activation']:.2f}."
            if 'dominant_region' in heatmap_data:
                description += f" The dominant activation region is {heatmap_data['dominant_region']}."

        return description

    def _generate_limitations(self) -> List[str]:
        """
        Generate limitations of the explanation.
        
        Returns:
            List of limitation statements
        """
        limitations = [
            "The highlighted regions show where the model attended, but do not prove biological causation.",
            "The visualization is based on the model's internal representations, which may not align with human reasoning.",
            "Grad-CAM for object detection is an adaptation and may not capture all relevant features.",
            "The explanation is specific to this model and may not generalize to other models.",
            "High attention in a region does not necessarily mean that region contains the target object."
        ]
        
        return limitations
    
    def generate_batch_explanation(
        self,
        predictions: List[Dict],
        heatmap_data_list: Optional[List[Dict]] = None
    ) -> List[Explanation]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
            heatmap_data_list: Optional list of heatmap data
        
        Returns:
            List of Explanation objects
        """
        explanations = []
        
        for i, pred in enumerate(predictions):
            heatmap_data = heatmap_data_list[i] if heatmap_data_list and i < len(heatmap_data_list) else None
            explanation = self.generate_explanation(pred, heatmap_data)
            explanations.append(explanation)
        
        return explanations
    
    def format_explanation_for_display(self, explanation: Explanation) -> str:
        """
        Format explanation for display.
        
        Args:
            explanation: Explanation object
        
        Returns:
            Formatted text
        """
        lines = [
            f"Prediction: {explanation.prediction}",
            f"Confidence: {explanation.confidence:.2f}",
            "",
            "Explanation:",
            explanation.explanation_text,
            "",
            "Visualization:",
            explanation.visualization_description,
            "",
            "Limitations:",
        ]
        
        for limitation in explanation.limitations:
            lines.append(f"- {limitation}")
        
        return "\n".join(lines)


if __name__ == '__main__':
    # Example usage
    generator = ExplanationGenerator({0: 'groundnut', 1: 'weed'})
    
    # Test explanation generation
    pred_data = {'class_id': 1, 'confidence': 0.85}
    explanation = generator.generate_explanation(pred_data)
    
    print(generator.format_explanation_for_display(explanation))

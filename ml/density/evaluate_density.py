"""
Evaluation script for weed density estimation vs field observations.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FieldObservation:
    """Field observation data for comparison with density estimates."""
    image_id: str
    manual_weed_count: Optional[int] = None
    manual_coverage_percent: Optional[float] = None
    manual_density_category: Optional[str] = None
    notes: Optional[str] = None


class DensityEvaluator:
    """
    Evaluate weed density estimates against manually verified field observations.
    """
    
    def __init__(self):
        self.observations: List[FieldObservation] = []
    
    def load_observations_from_csv(self, csv_path: Path) -> None:
        """
        Load field observations from CSV file.
        
        Expected CSV format:
        image_id,manual_weed_count,manual_coverage_percent,manual_density_category,notes
        """
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                obs = FieldObservation(
                    image_id=row['image_id'],
                    manual_weed_count=int(row['manual_weed_count']) if row['manual_weed_count'] else None,
                    manual_coverage_percent=float(row['manual_coverage_percent']) if row['manual_coverage_percent'] else None,
                    manual_density_category=row['manual_density_category'] if row['manual_density_category'] else None,
                    notes=row.get('notes', None)
                )
                self.observations.append(obs)
        
        print(f"Loaded {len(self.observations)} field observations from {csv_path}")
    
    def load_density_results(self, results_path: Path) -> Dict:
        """
        Load density estimation results from JSON file.
        
        Args:
            results_path: Path to density results JSON file
        
        Returns:
            Dictionary with results and summary
        """
        with open(results_path, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded density results from {results_path}")
        return data
    
    def compare_weed_counts(
        self,
        density_results: Dict,
        min_samples: int = 10
    ) -> Dict:
        """
        Compare estimated weed counts with manual field counts.
        
        Args:
            density_results: Density estimation results
            min_samples: Minimum number of paired samples required
        
        Returns:
            Dictionary with comparison metrics
        """
        # Create lookup for field observations
        obs_lookup = {obs.image_id: obs for obs in self.observations}
        
        # Collect paired data
        estimated_counts = []
        manual_counts = []
        image_ids = []
        
        for result in density_results.get('results', []):
            image_path = result['image_path']
            image_id = Path(image_path).stem
            
            if image_id in obs_lookup:
                obs = obs_lookup[image_id]
                if obs.manual_weed_count is not None:
                    estimated_counts.append(result['weed_count'])
                    manual_counts.append(obs.manual_weed_count)
                    image_ids.append(image_id)
        
        if len(estimated_counts) < min_samples:
            print(f"Warning: Only {len(estimated_counts)} paired samples (minimum {min_samples} required)")
            return {}
        
        # Calculate metrics
        estimated_counts = np.array(estimated_counts)
        manual_counts = np.array(manual_counts)
        
        # Absolute errors
        absolute_errors = np.abs(estimated_counts - manual_counts)
        
        # Relative errors
        relative_errors = absolute_errors / (manual_counts + 1e-6)  # Avoid division by zero
        
        # Correlation
        correlation = np.corrcoef(estimated_counts, manual_counts)[0, 1]
        
        return {
            'num_samples': len(estimated_counts),
            'mean_absolute_error': float(np.mean(absolute_errors)),
            'std_absolute_error': float(np.std(absolute_errors)),
            'mean_relative_error': float(np.mean(relative_errors)),
            'std_relative_error': float(np.std(relative_errors)),
            'correlation': float(correlation),
            'mean_estimated': float(np.mean(estimated_counts)),
            'mean_manual': float(np.mean(manual_counts)),
            'image_ids': image_ids
        }
    
    def compare_coverage_percent(
        self,
        density_results: Dict,
        min_samples: int = 10
    ) -> Dict:
        """
        Compare estimated coverage percentages with manual field measurements.
        
        Args:
            density_results: Density estimation results
            min_samples: Minimum number of paired samples required
        
        Returns:
            Dictionary with comparison metrics
        """
        # Create lookup for field observations
        obs_lookup = {obs.image_id: obs for obs in self.observations}
        
        # Collect paired data
        estimated_coverage = []
        manual_coverage = []
        image_ids = []
        
        for result in density_results.get('results', []):
            image_path = result['image_path']
            image_id = Path(image_path).stem
            
            if image_id in obs_lookup:
                obs = obs_lookup[image_id]
                if obs.manual_coverage_percent is not None:
                    estimated_coverage.append(result['weed_coverage_percent'])
                    manual_coverage.append(obs.manual_coverage_percent)
                    image_ids.append(image_id)
        
        if len(estimated_coverage) < min_samples:
            print(f"Warning: Only {len(estimated_coverage)} paired samples (minimum {min_samples} required)")
            return {}
        
        # Calculate metrics
        estimated_coverage = np.array(estimated_coverage)
        manual_coverage = np.array(manual_coverage)
        
        # Absolute errors
        absolute_errors = np.abs(estimated_coverage - manual_coverage)
        
        # Relative errors
        relative_errors = absolute_errors / (manual_coverage + 1e-6)
        
        # Correlation
        correlation = np.corrcoef(estimated_coverage, manual_coverage)[0, 1]
        
        return {
            'num_samples': len(estimated_coverage),
            'mean_absolute_error': float(np.mean(absolute_errors)),
            'std_absolute_error': float(np.std(absolute_errors)),
            'mean_relative_error': float(np.mean(relative_errors)),
            'std_relative_error': float(np.std(relative_errors)),
            'correlation': float(correlation),
            'mean_estimated': float(np.mean(estimated_coverage)),
            'mean_manual': float(np.mean(manual_coverage)),
            'image_ids': image_ids
        }
    
    def compare_density_categories(
        self,
        density_results: Dict
    ) -> Dict:
        """
        Compare estimated density categories with manual field classifications.
        
        Args:
            density_results: Density estimation results
        
        Returns:
            Dictionary with confusion matrix and accuracy metrics
        """
        # Create lookup for field observations
        obs_lookup = {obs.image_id: obs for obs in self.observations}
        
        # Collect paired data

        categories = ['low', 'medium', 'high']
        
        # Initialize confusion matrix
        confusion_matrix = {pred: {actual: 0 for actual in categories} for pred in categories}
        
        correct_predictions = 0
        total_predictions = 0
        
        for result in density_results.get('results', []):
            image_path = result['image_path']
            image_id = Path(image_path).stem
            
            if image_id in obs_lookup:
                obs = obs_lookup[image_id]
                if obs.manual_density_category is not None:
                    predicted = result['density_category']
                    actual = obs.manual_density_category
                    
                    if predicted in confusion_matrix and actual in confusion_matrix[predicted]:
                        confusion_matrix[predicted][actual] += 1
                        total_predictions += 1
                        
                        if predicted == actual:
                            correct_predictions += 1
        
        if total_predictions == 0:
            print("Warning: No paired density category comparisons found")
            return {}
        
        # Calculate per-class accuracy
        per_class_accuracy = {}
        for category in categories:
            if category in confusion_matrix:
                total = sum(confusion_matrix[category].values())
                correct = confusion_matrix[category].get(category, 0)
                per_class_accuracy[category] = correct / total if total > 0 else 0.0
        
        overall_accuracy = correct_predictions / total_predictions
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': overall_accuracy,
            'per_class_accuracy': per_class_accuracy,
            'confusion_matrix': confusion_matrix
        }
    
    def generate_evaluation_report(
        self,
        density_results_path: Path,
        field_observations_path: Optional[Path] = None,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report comparing density estimates with field observations.
        
        Args:
            density_results_path: Path to density estimation results
            field_observations_path: Path to field observations CSV (optional)
            output_path: Path to save evaluation report (optional)
        
        Returns:
            Dictionary with all evaluation metrics
        """
        # Load density results
        density_results = self.load_density_results(density_results_path)
        
        # Load field observations if provided
        if field_observations_path and field_observations_path.exists():
            self.load_observations_from_csv(field_observations_path)
        
        # Run comparisons
        report = {
            'density_results_summary': density_results.get('summary', {}),
            'weed_count_comparison': self.compare_weed_counts(density_results),
            'coverage_comparison': self.compare_coverage_percent(density_results),
            'category_comparison': self.compare_density_categories(density_results)
        }
        
        # Save report if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Evaluation report saved to {output_path}")
        
        return report
    
    def plot_comparison_scatter(
        self,
        density_results_path: Path,
        field_observations_path: Path,
        output_path: Path,
        metric: str = 'coverage'
    ) -> None:
        """
        Create scatter plot comparing estimated vs manual measurements.
        
        Args:
            density_results_path: Path to density estimation results
            field_observations_path: Path to field observations CSV
            output_path: Path to save plot
            metric: 'coverage' or 'count'
        """
        # Load data
        density_results = self.load_density_results(density_results_path)
        self.load_observations_from_csv(field_observations_path)
        
        # Get comparison data
        if metric == 'coverage':
            comparison = self.compare_coverage_percent(density_results)
            xlabel = 'Estimated Coverage (%)'
            ylabel = 'Manual Coverage (%)'
            title = 'Weed Coverage: Estimated vs Manual'
        else:
            comparison = self.compare_weed_counts(density_results)
            xlabel = 'Estimated Weed Count'
            ylabel = 'Manual Weed Count'
            title = 'Weed Count: Estimated vs Manual'
        
        if not comparison or comparison.get('num_samples', 0) == 0:
            print(f"No data available for {metric} comparison plot")
            return
        
        # Recreate the paired data for plotting
        obs_lookup = {obs.image_id: obs for obs in self.observations}
        estimated = []
        manual = []
        
        for result in density_results.get('results', []):
            image_id = Path(result['image_path']).stem
            if image_id in obs_lookup:
                obs = obs_lookup[image_id]
                if metric == 'coverage':
                    if obs.manual_coverage_percent is not None:
                        estimated.append(result['weed_coverage_percent'])
                        manual.append(obs.manual_coverage_percent)
                else:
                    if obs.manual_weed_count is not None:
                        estimated.append(result['weed_count'])
                        manual.append(obs.manual_weed_count)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(manual, estimated, alpha=0.6)
        
        # Add diagonal line for perfect prediction
        min_val = min(min(manual), min(estimated))
        max_val = max(max(manual), max(estimated))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add correlation text
        correlation = comparison.get('correlation', 0)
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Scatter plot saved to {output_path}")
    
    def create_sample_observations_template(self, output_path: Path) -> None:
        """
        Create a template CSV file for field observations.
        
        Args:
            output_path: Path to save template
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'image_id',
                'manual_weed_count',
                'manual_coverage_percent',
                'manual_density_category',
                'notes'
            ])
            # Add example row
            writer.writerow([
                'agri_0_1009',
                '15',
                '8.5',
                'medium',
                'Example observation'
            ])
        
        print(f"Observations template created at {output_path}")


if __name__ == '__main__':
    # Create sample observations template
    evaluator = DensityEvaluator()
    evaluator.create_sample_observations_template(
        Path('experiments/yolo/field_observations_template.csv')
    )
    
    print("\nTo evaluate density estimates:")
    print("1. Fill in the field_observations_template.csv with manual observations")
    print("2. Run: python ml/density/evaluate_density.py")

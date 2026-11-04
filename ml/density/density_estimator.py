"""
Weed density estimation from object detection results.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class DensityConfig:
    """
    Configuration for weed density estimation.
    
    Thresholds are configurable for calibration with field data.
    These are NOT scientifically justified thresholds and should be
    calibrated based on actual field observations.
    """
    # Density classification thresholds (percentage coverage)
    low_threshold: float = 5.0      # Below 5% = low density
    medium_threshold: float = 15.0  # 5-15% = medium density
    # Above 15% = high density
    
    # Confidence threshold for detections
    detection_confidence: float = 0.25
    
    # IoU threshold for NMS
    nms_iou_threshold: float = 0.45
    
    # Class ID for weed detection (may need adjustment based on dataset)
    weed_class_id: int = 1  # Assuming class 1 is weed, configurable
    
    # Minimum bounding box area (pixels) to filter small detections
    min_box_area: int = 100
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DensityConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'DensityConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


@dataclass
class DensityResult:
    """Result of weed density estimation for a single image."""
    image_path: str
    image_width: int
    image_height: int
    total_image_area: int
    weed_count: int
    weed_boxes: List[Tuple[int, int, int, int]]  # List of (x1, y1, x2, y2)
    weed_box_areas: List[int]
    total_weed_area: int
    weed_coverage_percent: float
    density_category: str
    confidence_scores: List[float]
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return asdict(self)


class DensityEstimator:
    """
    Estimate weed density from object detection results.
    
    This class calculates:
    - Number of detected weeds
    - Bounding box areas
    - Total image area
    - Weed coverage percentage
    - Density category (low/medium/high)
    """
    
    def __init__(self, config: Optional[DensityConfig] = None):
        """
        Initialize density estimator.
        
        Args:
            config: DensityConfig object. If None, uses default config.
        """
        self.config = config or DensityConfig()
    
    def calculate_box_area(self, box: Tuple[int, int, int, int]) -> int:
        """
        Calculate area of a bounding box.
        
        Args:
            box: Tuple of (x1, y1, x2, y2)
        
        Returns:
            Box area in pixels
        """
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        return width * height
    
    def classify_density(self, coverage_percent: float) -> str:
        """
        Classify weed density based on coverage percentage.
        
        Args:
            coverage_percent: Weed coverage percentage (0-100)
        
        Returns:
            Density category: 'low', 'medium', or 'high'
        """
        if coverage_percent < self.config.low_threshold:
            return 'low'
        elif coverage_percent < self.config.medium_threshold:
            return 'medium'
        else:
            return 'high'
    
    def estimate_from_detections(
        self,
        image_path: str,
        detections: List[Dict],
        image_width: int,
        image_height: int
    ) -> DensityResult:
        """
        Estimate weed density from detection results.
        
        Args:
            image_path: Path to the image
            detections: List of detection results from YOLO model
            image_width: Image width in pixels
            image_height: Image height in pixels
        
        Returns:
            DensityResult object with all metrics
        """
        # Filter detections for weed class and confidence threshold
        weed_detections = []
        for det in detections:
            if (det.get('class_id') == self.config.weed_class_id and 
                det.get('confidence', 0) >= self.config.detection_confidence):
                weed_detections.append(det)
        
        # Extract bounding boxes and calculate areas
        weed_boxes = []
        weed_box_areas = []
        confidence_scores = []
        
        for det in weed_detections:
            box = det.get('box')
            if box is not None:
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Filter by minimum area
                box_area = self.calculate_box_area((x1, y1, x2, y2))
                if box_area >= self.config.min_box_area:
                    weed_boxes.append((x1, y1, x2, y2))
                    weed_box_areas.append(box_area)
                    confidence_scores.append(det.get('confidence', 0.0))
        
        # Calculate metrics
        weed_count = len(weed_boxes)
        total_image_area = image_width * image_height
        total_weed_area = sum(weed_box_areas)
        weed_coverage_percent = (total_weed_area / total_image_area * 100) if total_image_area > 0 else 0.0
        
        # Classify density
        density_category = self.classify_density(weed_coverage_percent)
        
        return DensityResult(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            total_image_area=total_image_area,
            weed_count=weed_count,
            weed_boxes=weed_boxes,
            weed_box_areas=weed_box_areas,
            total_weed_area=total_weed_area,
            weed_coverage_percent=weed_coverage_percent,
            density_category=density_category,
            confidence_scores=confidence_scores
        )
    
    def estimate_from_yolo_results(
        self,
        image_path: str,
        yolo_results,
        image: Optional[np.ndarray] = None
    ) -> DensityResult:
        """
        Estimate weed density from YOLO model results.
        
        Args:
            image_path: Path to the image
            yolo_results: Results from YOLO model prediction
            image: Optional image array (numpy array). If None, will load from path.
        
        Returns:
            DensityResult object with all metrics
        """
        # Load image if not provided
        if image is None:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
        
        image_height, image_width = image.shape[:2]
        
        # Convert YOLO results to detection format
        detections = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'class_id': cls_id,
                        'confidence': conf,
                        'box': xyxy
                    })
        
        return self.estimate_from_detections(
            image_path=image_path,
            detections=detections,
            image_width=image_width,
            image_height=image_height
        )
    
    def estimate_from_directory(
        self,
        image_dir: Path,
        yolo_model,
        max_images: Optional[int] = None
    ) -> List[DensityResult]:
        """
        Estimate weed density for all images in a directory.
        
        Args:
            image_dir: Path to directory containing images
            yolo_model: Trained YOLO model
            max_images: Maximum number of images to process (None for all)
        
        Returns:
            List of DensityResult objects
        """
        image_dir = Path(image_dir)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f'*{ext}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        results = []
        for i, image_path in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] Processing {image_path.name}")
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load {image_path}")
                continue
            
            # Run YOLO prediction
            yolo_results = yolo_model(str(image_path), conf=self.config.detection_confidence, verbose=False)
            
            # Estimate density
            density_result = self.estimate_from_yolo_results(
                str(image_path),
                yolo_results,
                image
            )
            
            results.append(density_result)
        
        return results
    
    def calculate_pixel_level_coverage(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Calculate pixel-level weed coverage if segmentation masks are available.
        
        Args:
            image: Input image (numpy array)
            mask: Segmentation mask where weed pixels are marked
        
        Returns:
            Weed coverage percentage based on pixel count
        """
        if mask is None:
            return 0.0
        
        # Count weed pixels (assuming mask is binary where 1 = weed)
        weed_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        
        coverage_percent = (weed_pixels / total_pixels * 100) if total_pixels > 0 else 0.0
        return coverage_percent
    
    def save_results(self, results: List[DensityResult], output_path: Path) -> None:
        """
        Save density estimation results to JSON file.
        
        Args:
            results: List of DensityResult objects
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            'config': self.config.to_dict(),
            'results': [r.to_dict() for r in results],
            'summary': self.calculate_summary(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def calculate_summary(self, results: List[DensityResult]) -> Dict:
        """
        Calculate summary statistics from density results.
        
        Args:
            results: List of DensityResult objects
        
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        weed_counts = [r.weed_count for r in results]
        coverage_percents = [r.weed_coverage_percent for r in results]
        
        density_counts = {'low': 0, 'medium': 0, 'high': 0}
        for r in results:
            density_counts[r.density_category] += 1
        
        return {
            'total_images': len(results),
            'total_weeds_detected': sum(weed_counts),
            'avg_weeds_per_image': np.mean(weed_counts),
            'std_weeds_per_image': np.std(weed_counts),
            'min_weeds_per_image': min(weed_counts),
            'max_weeds_per_image': max(weed_counts),
            'avg_coverage_percent': np.mean(coverage_percents),
            'std_coverage_percent': np.std(coverage_percents),
            'min_coverage_percent': min(coverage_percents),
            'max_coverage_percent': max(coverage_percents),
            'density_distribution': density_counts
        }


if __name__ == '__main__':
    # Example usage
    from ultralytics import YOLO
    
    # Load configuration
    config = DensityConfig()
    print("Default density thresholds:")
    print(f"  Low: < {config.low_threshold}%")
    print(f"  Medium: {config.low_threshold}% - {config.medium_threshold}%")
    print(f"  High: > {config.medium_threshold}%")
    
    # Save default config
    config.save(Path('ml/density/default_config.json'))
    print("\nDefault config saved to ml/density/default_config.json")

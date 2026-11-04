"""Dataset Manifest System

Maintains a dataset manifest with original image IDs, source, capture session,
labels, split, and annotation provenance to establish dataset lineage and
prevent leakage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatasetSplit(str, Enum):
    """Dataset split types."""
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNSPLIT = "unsplit"


class DatasetType(str, Enum):
    """Dataset types."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    NEGATIVES = "negatives"


@dataclass
class ImageRecord:
    """Record for a single image in the dataset."""
    # Identification
    image_id: str  # Unique identifier (hash of original file)
    original_filename: str
    relative_path: str  # Path relative to dataset root

    # Provenance
    source: str  # Where the image came from (e.g., "field_capture_2024_10_01")
    capture_session: str  # Session ID for grouping related images
    capture_date: Optional[str] = None
    capture_device: Optional[str] = None
    location: Optional[str] = None

    # Content
    dataset_type: DatasetType = DatasetType.CLASSIFICATION
    labels: Optional[List[int]] = None  # Class IDs
    class_names: Optional[List[str]] = None
    annotation_type: Optional[str] = None  # "classification", "detection", "segmentation"

    # Annotation metadata
    num_objects: int = 0  # For detection/segmentation
    has_mask: bool = False
    annotation_version: str = "1.0"
    annotation_date: Optional[str] = None
    annotator: Optional[str] = None

    # Split assignment
    split: DatasetSplit = DatasetSplit.UNSPLIT
    split_method: str = "manual"  # "manual", "random", "stratified"
    split_seed: Optional[int] = None

    # File metadata
    file_hash: str = ""  # SHA256 of current file
    file_size: int = 0
    width: int = 0
    height: int = 0
    channels: int = 3

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Set default timestamps."""
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of image file.

        Args:
            file_path: Path to image file

        Returns:
            SHA256 hash as hex string
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        self.file_hash = sha256.hexdigest()
        return self.file_hash


@dataclass
class DatasetManifest:
    """Manifest for an entire dataset."""

    dataset_name: str
    dataset_version: str
    dataset_type: DatasetType
    root_path: str  # Absolute path to dataset root

    # Statistics
    total_images: int = 0
    split_counts: Optional[Dict[str, int]] = None
    class_distribution: Optional[Dict[str, int]] = None

    # Records
    images: Dict[str, ImageRecord] = None  # image_id -> ImageRecord

    # Metadata
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    created_by: str = "dataset_manifest.py"

    def __post_init__(self):
        """Initialize defaults."""
        if self.images is None:
            self.images = {}
        if self.split_counts is None:
            self.split_counts = {}
        if self.class_distribution is None:
            self.class_distribution = {}
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        if not self.updated_at:
            self.updated_at = datetime.utcnow().isoformat()

    def add_image(self, record: ImageRecord) -> None:
        """Add an image record to the manifest.

        Args:
            record: Image record to add
        """
        record.updated_at = datetime.utcnow().isoformat()
        self.images[record.image_id] = record
        self.total_images = len(self.images)
        self._update_statistics()
        self.updated_at = datetime.utcnow().isoformat()

    def _update_statistics(self) -> None:
        """Update dataset statistics."""
        # Split counts
        self.split_counts = {}
        for record in self.images.values():
            split = record.split.value
            self.split_counts[split] = self.split_counts.get(split, 0) + 1

        # Class distribution
        self.class_distribution = {}
        for record in self.images.values():
            if record.labels:
                for label in record.labels:
                    self.class_distribution[str(label)] = self.class_distribution.get(str(label), 0) + 1

    def get_images_by_split(self, split: DatasetSplit) -> List[ImageRecord]:
        """Get all images in a specific split.

        Args:
            split: Dataset split

        Returns:
            List of image records
        """
        return [r for r in self.images.values() if r.split == split]

    def get_images_by_session(self, session: str) -> List[ImageRecord]:
        """Get all images from a capture session.

        Args:
            session: Capture session ID

        Returns:
            List of image records
        """
        return [r for r in self.images.values() if r.capture_session == session]

    def check_session_leakage(self) -> Dict[str, Dict[str, Set[str]]]:
        """Check if capture sessions span multiple splits (potential leakage).

        Returns:
            Dictionary mapping sessions to their splits
        """
        session_splits: Dict[str, Set[str]] = {}
        for record in self.images.values():
            if record.capture_session:
                if record.capture_session not in session_splits:
                    session_splits[record.capture_session] = set()
                session_splits[record.capture_session].add(record.split.value)

        # Find sessions with multiple splits
        leaked_sessions = {
            session: splits
            for session, splits in session_splits.items()
            if len(splits) > 1
        }

        return {"leaked_sessions": leaked_sessions, "all_sessions": session_splits}

    def check_duplicate_hashes(self) -> Dict[str, List[str]]:
        """Check for duplicate file hashes (exact duplicates).

        Returns:
            Dictionary mapping hash to list of image IDs
        """
        hash_to_ids: Dict[str, List[str]] = {}
        for image_id, record in self.images.items():
            if record.file_hash:
                if record.file_hash not in hash_to_ids:
                    hash_to_ids[record.file_hash] = []
                hash_to_ids[record.file_hash].append(image_id)

        # Find duplicates
        duplicates = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
        return duplicates

    def get_session_distribution(self) -> Dict[str, Dict[str, int]]:
        """Get distribution of images by session and split.

        Returns:
            Dictionary mapping session to split counts
        """
        session_dist: Dict[str, Dict[str, int]] = {}
        for record in self.images.values():
            if record.capture_session:
                if record.capture_session not in session_dist:
                    session_dist[record.capture_session] = {}
                split = record.split.value
                session_dist[record.capture_session][split] = (
                    session_dist[record.capture_session].get(split, 0) + 1
                )
        return session_dist


class DatasetManifestManager:
    """Manager for dataset manifests."""

    def __init__(self, manifest_dir: Optional[Path] = None):
        """Initialize the manifest manager.

        Args:
            manifest_dir: Directory to store manifest files
        """
        if manifest_dir is None:
            manifest_dir = PROJECT_ROOT / "datasets" / "manifests"

        self.manifest_dir = manifest_dir
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifests: Dict[str, DatasetManifest] = {}

    def create_manifest(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        root_path: str,
        version: str = "1.0",
        description: str = "",
    ) -> DatasetManifest:
        """Create a new dataset manifest.

        Args:
            dataset_name: Name of the dataset
            dataset_type: Type of dataset
            root_path: Root path of dataset
            version: Dataset version
            description: Description

        Returns:
            New DatasetManifest
        """
        manifest = DatasetManifest(
            dataset_name=dataset_name,
            dataset_version=version,
            dataset_type=dataset_type,
            root_path=root_path,
            description=description,
        )
        self.manifests[dataset_name] = manifest
        return manifest

    def load_manifest(self, dataset_name: str) -> Optional[DatasetManifest]:
        """Load a manifest from disk.

        Args:
            dataset_name: Name of dataset

        Returns:
            DatasetManifest or None if not found
        """
        manifest_path = self.manifest_dir / f"{dataset_name}.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r") as f:
            data = json.load(f)

        # Convert enums
        data["dataset_type"] = DatasetType(data["dataset_type"])

        # Convert image records
        images = {}
        for image_id, record_data in data["images"].items():
            record_data["dataset_type"] = DatasetType(record_data["dataset_type"])
            record_data["split"] = DatasetSplit(record_data["split"])
            images[image_id] = ImageRecord(**record_data)

        data["images"] = images
        manifest = DatasetManifest(**data)
        self.manifests[dataset_name] = manifest
        return manifest

    def save_manifest(self, dataset_name: str) -> bool:
        """Save a manifest to disk.

        Args:
            dataset_name: Name of dataset

        Returns:
            True if saved, False if manifest not found
        """
        if dataset_name not in self.manifests:
            return False

        manifest = self.manifests[dataset_name]
        data = asdict(manifest)

        # Convert enums to strings
        data["dataset_type"] = manifest.dataset_type.value

        # Convert image records
        data["images"] = {}
        for image_id, record in manifest.images.items():
            record_data = asdict(record)
            record_data["dataset_type"] = record.dataset_type.value
            record_data["split"] = record.split.value
            data["images"][image_id] = record_data

        manifest_path = self.manifest_dir / f"{dataset_name}.json"
        with open(manifest_path, "w") as f:
            json.dump(data, f, indent=2)

        return True

    def generate_manifest_from_directory(
        self,
        dataset_name: str,
        dataset_type: DatasetType,
        root_path: str,
        version: str = "1.0",
        source: str = "unknown",
    ) -> DatasetManifest:
        """Generate a manifest by scanning a directory structure.

        Args:
            dataset_name: Name of dataset
            dataset_type: Type of dataset
            root_path: Root path to scan
            version: Dataset version
            source: Source identifier

        Returns:
            Generated DatasetManifest
        """
        from PIL import Image

        root = Path(root_path)
        manifest = self.create_manifest(dataset_name, dataset_type, str(root), version)

        # Scan for images
        image_extensions = [".jpg", ".jpeg", ".png"]
        for ext in image_extensions:
            for img_path in root.rglob(f"*{ext}"):
                try:
                    # Generate image ID from relative path
                    rel_path = img_path.relative_to(root)
                    image_id = hashlib.sha256(str(rel_path).encode()).hexdigest()[:16]

                    # Get image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        channels = len(img.getbands())

                    # Determine split from path
                    split = DatasetSplit.UNSPLIT
                    for split_name in ["train", "val", "test"]:
                        if split_name in str(rel_path):
                            split = DatasetSplit(split_name)
                            break

                    # Determine class from path (for classification)
                    labels = None
                    class_names = None
                    if dataset_type == DatasetType.CLASSIFICATION:
                        for class_name in ["crop", "weed", "groundnut"]:
                            if class_name in str(rel_path):
                                labels = [0 if class_name in ["crop", "groundnut"] else 1]
                                class_names = ["groundnut", "weed"]
                                break

                    record = ImageRecord(
                        image_id=image_id,
                        original_filename=img_path.name,
                        relative_path=str(rel_path),
                        source=source,
                        capture_session=source,  # Use source as session for now
                        dataset_type=dataset_type,
                        labels=labels,
                        class_names=class_names,
                        split=split,
                        width=width,
                        height=height,
                        channels=channels,
                        file_size=img_path.stat().st_size,
                    )
                    record.compute_file_hash(img_path)
                    manifest.add_image(record)

                except Exception as e:
                    print(f"Warning: Could not process {img_path}: {e}")

        self.save_manifest(dataset_name)
        return manifest

    def audit_leakage(self, dataset_name: str) -> Dict:
        """Audit a dataset for leakage issues.

        Args:
            dataset_name: Name of dataset to audit

        Returns:
            Audit results
        """
        manifest = self.load_manifest(dataset_name)
        if not manifest:
            return {"error": f"Dataset {dataset_name} not found"}

        session_leakage = manifest.check_session_leakage()
        duplicate_hashes = manifest.check_duplicate_hashes()
        session_distribution = manifest.get_session_distribution()

        return {
            "dataset_name": dataset_name,
            "total_images": manifest.total_images,
            "split_counts": manifest.split_counts,
            "session_leakage": session_leakage,
            "duplicate_hashes": duplicate_hashes,
            "session_distribution": session_distribution,
            "has_leakage": len(session_leakage["leaked_sessions"]) > 0,
            "has_duplicates": len(duplicate_hashes) > 0,
        }


# Global manager instance
_manager: Optional[DatasetManifestManager] = None


def get_manager() -> DatasetManifestManager:
    """Get the global dataset manifest manager.

    Returns:
        DatasetManifestManager instance
    """
    global _manager
    if _manager is None:
        _manager = DatasetManifestManager()
    return _manager

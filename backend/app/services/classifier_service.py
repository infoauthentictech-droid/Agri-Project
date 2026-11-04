"""Classifier inference service — loads any registered PyTorch classifier that
was produced by the experimental pipeline (``ml/training``) and runs single
image inference.

Each experiment directory (``experiments/classification/<name>/``) is self-describing:
``config.yaml`` (architecture + hyper-parameters), ``normalization.json``
(channel mean/std), ``best_model.pt`` or ``best_model.pth`` (weights). The service re-uses
``ml.models.registry.build_model`` so serving code can never diverge from the
code that trained the model.

If config.yaml is missing, it will attempt to derive configuration from results.json.

Preprocessing is centralized via ``app.utils.preprocessing`` to ensure consistency
across training, evaluation, and serving.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.models.registry import build_model  # noqa: E402
from app.utils.preprocessing import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImagePreprocessor,
    PreprocessingConfig,
    ResizeMethod,
)

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 128
DEFAULT_CLASSIFIER = "mobilenet"

#: Experiment directories whose weights are training artifacts ONLY — they are
#: never served to the app through ``/models``. ``custom_cnn`` is a duplicate
#: training run of the same ``cnn`` architecture (identical parameter count and
#: metrics); ``cnn`` is the canonical name and is the single CNN model exposed to the app.
HIDDEN_CLASSIFIERS: frozenset[str] = frozenset({
    "custom_cnn",
})


def _decode_rgb(image_data: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def discover_classifiers(experiment_root: Path) -> List[str]:
    """Names of experiment dirs that hold a config + weights (available).

    Directories listed in ``HIDDEN_CLASSIFIERS`` are excluded from the registry
    even though their artifacts remain on disk — they stay usable for offline
    experiments but are never served to the app.

    Looks for models in experiments/classification/<name>/ with either
    best_model.pt or best_model.pth. If config.yaml is missing, will attempt
    to use results.json for configuration.
    """
    available: List[str] = []
    classification_root = experiment_root / "classification"
    if not classification_root.exists():
        return available
    for d in sorted(classification_root.iterdir()):
        if not d.is_dir():
            continue
        if d.name in HIDDEN_CLASSIFIERS:
            continue
        # Accept either .pt or .pth checkpoint format
        has_weights = (d / "best_model.pt").exists() or (d / "best_model.pth").exists()
        # Accept config.yaml OR results.json for configuration
        has_config = (d / "config.yaml").exists() or (d / "results.json").exists()
        if has_weights and has_config:
            available.append(d.name)
    return available


class ClassifierService:
    """Lazy per-model loader + inference for the classifier zoo."""

    def __init__(self, experiment_root: Optional[Path] = None,
                 default_model: str = DEFAULT_CLASSIFIER) -> None:
        self.experiment_root = experiment_root or (project_root / "experiments")
        self.default_model = default_model
        self._models: Dict[str, dict] = {}
        # Class names derived from classification task
        self.default_class_names = ["groundnut", "weed"]

    @property
    def available_models(self) -> List[str]:
        return discover_classifiers(self.experiment_root)

    def is_loaded(self, model_name: Optional[str] = None) -> bool:
        name = model_name or self.default_model
        return name in self._models

    def test_accuracy(self, model_name: str) -> Optional[float]:
        """Held-out test accuracy (0-1) from the run's results.json, if recorded."""
        path = self.experiment_root / "classification" / model_name / "results.json"
        try:
            # results.json stores a percentage (e.g. 99.53).
            return float(json.loads(path.read_text(encoding="utf-8"))["test_accuracy"]) / 100.0
        except (OSError, KeyError, TypeError, ValueError):
            return None

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Eagerly load a classifier (default when omitted).

        Used by startup warm-up so the first ``/analyze`` request does not pay
        the one-time torch import + checkpoint load cost that can push the
        request past the client's timeout. Returns True on success; never
        raises.
        """
        name = model_name or self.default_model
        try:
            self._resolve_model(name)
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Pre-warm of classifier '%s' failed: %s", name, exc)
            return False

    def _resolve_model(self, model_name: str) -> dict:
        if model_name in self._models:
            return self._models[model_name]
        # Tight allow-list: only names that were actually discovered in the
        # experiments directory are accepted. This prevents a user-supplied
        # model_name (e.g. "../../etc/passwd") from traversing directories.
        # Hybrid models are also allowed if they exist in the registry
        if model_name not in discover_classifiers(self.experiment_root):
            # Check if it's a hybrid model that hasn't been trained yet
            hybrid_models = ["resnet50_mobilenet", "resnet50_efficientnet", "resnet50_efficientnet_mobilenet"]
            if model_name in hybrid_models:
                raise ValueError(
                    f"Hybrid model '{model_name}' is registered but not yet trained. "
                    f"Train the model first using the training pipeline."
                )
            raise ValueError(f"classifier '{model_name}' is not available")
        run_dir = self.experiment_root / "classification" / model_name
        cfg_path = run_dir / "config.yaml"
        results_path = run_dir / "results.json"
        # Try both .pt and .pth formats
        ckpt_path_pt = run_dir / "best_model.pt"
        ckpt_path_pth = run_dir / "best_model.pth"
        ckpt_path = ckpt_path_pt if ckpt_path_pt.exists() else ckpt_path_pth
        norm_path = run_dir / "normalization.json"

        if not ckpt_path.exists():
            raise ValueError(f"classifier '{model_name}' is not available - no checkpoint found")

        # Load configuration from config.yaml if available, otherwise from results.json
        if cfg_path.exists():
            cfg = _load_yaml(cfg_path)
        elif results_path.exists():
            # Derive minimal config from results.json
            results = json.loads(results_path.read_text(encoding="utf-8"))
            cfg = {
                "model": {
                    "name": model_name,
                    "num_classes": 2,  # Binary classification for groundnut/weed
                    "class_names": self.default_class_names,
                },
                "dataset": {
                    "image_size": results.get("image_size", DEFAULT_IMAGE_SIZE),
                },
            }
        else:
            raise ValueError(f"classifier '{model_name}' is not available - no config found")

        import torch  # deferred: heavy import

        model = build_model(cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats and extract class_names
        checkpoint_class_names = None
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
                checkpoint_class_names = ckpt.get("class_names")
            elif "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                checkpoint_class_names = ckpt.get("class_names")
            else:
                # Assume the dict itself is the state_dict
                model.load_state_dict(ckpt)
        else:
            # Assume checkpoint is the state_dict itself
            model.load_state_dict(ckpt)
        model.eval()

        # Training normalises with ImageNet statistics (ml/data/gpu_loader.py)
        # unless the run recorded its own normalization.json.
        norm = {"mean": np.array(IMAGENET_MEAN, dtype=np.float32),
                "std": np.array(IMAGENET_STD, dtype=np.float32)}
        if norm_path.exists():
            raw = json.loads(norm_path.read_text(encoding="utf-8"))
            norm = {"mean": np.array(raw["mean"], dtype=np.float32),
                    "std": np.array(raw["std"], dtype=np.float32)}

        image_size = int(cfg.get("dataset", {}).get("image_size", DEFAULT_IMAGE_SIZE))
        # Use class_names from checkpoint if available, otherwise from config
        if checkpoint_class_names is not None:
            class_names = [str(n) for n in checkpoint_class_names]
        else:
            class_names = list(cfg.get("model", {}).get(
                "class_names", self.default_class_names))

        entry = {
            "model": model,
            "image_size": image_size,
            "mean": norm["mean"],
            "std": norm["std"],
            "class_names": class_names,
            "architecture": cfg.get("model", {}).get("name", model_name),
        }
        self._models[model_name] = entry
        logger.info("classifier '%s' loaded (%s, %s classes, %dx%d)",
                    model_name, entry["architecture"], len(class_names),
                    image_size, image_size)
        return entry

    def classify(
        self,
        image_data: str,
        model_name: Optional[str] = None,
        top_k: int = 2,
    ) -> Tuple[bool, Dict, float, Optional[str]]:
        """Run single-image classification.

        Returns (success, result, inference_ms, error). ``result`` keys:
        ``model``, ``architecture``, ``class_name``, ``class_id``,
        ``confidence`` and ``probabilities``.
        """
        name = model_name or self.default_model
        try:
            entry = self._resolve_model(name)
        except ValueError as e:
            return False, {}, 0.0, str(e)

        try:
            import torch  # deferred: heavy import

            # Match this model's evaluation transform: plain resize to its own
            # training size + its normalisation (not the shared 128px letterbox).
            size = entry["image_size"]
            preprocessor = ImagePreprocessor(PreprocessingConfig(
                target_size=(size, size),
                resize_method=ResizeMethod.RESIZE,
                mean=entry["mean"].tolist(),
                std=entry["std"].tolist(),
            ))
            image_array = preprocessor.preprocess(image_data)

            # Convert to tensor (C, H, W)
            tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)

            start = time.time()
            with torch.inference_mode():
                logits = entry["model"](tensor)
                probs = torch.softmax(logits, dim=1)[0]
            inference_ms = (time.time() - start) * 1000.0

            top_idx = torch.argsort(probs, descending=True)[: max(1, int(top_k))]
            names = entry["class_names"]
            probabilities = [
                {"class_id": int(i),
                 "class_name": names[i] if i < len(names) else f"class_{int(i)}",
                 "probability": round(float(probs[i]), 6)}
                for i in top_idx
            ]
            best = probabilities[0]
            return True, {
                "model": name,
                "architecture": entry["architecture"],
                "class_name": best["class_name"],
                "class_id": best["class_id"],
                "confidence": best["probability"],
                "probabilities": probabilities,
            }, inference_ms, None
        except Exception as e:
            logger.error("Classification error (%s): %s", name, e)
            return False, {}, 0.0, f"Classification failed: {e}"


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
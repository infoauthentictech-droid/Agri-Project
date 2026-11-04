"""Model Lifecycle and Concurrency Management

Manages model loading, unloading, and concurrency to prevent memory exhaustion
and slow responses from loading large models for every request.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


@dataclass
class ModelInstance:
    """Represents a loaded model instance."""
    name: str
    status: ModelStatus
    model: Optional[object] = None
    load_time: Optional[float] = None
    last_used: Optional[float] = None
    error_message: Optional[str] = None
    ref_count: int = 0


class ModelLifecycleManager:
    """Manages model lifecycle with concurrency control."""

    def __init__(self, max_concurrent_loads: int = 2, enable_lazy_loading: bool = True):
        """Initialize model lifecycle manager.

        Args:
            max_concurrent_loads: Maximum number of models loading concurrently
            enable_lazy_loading: Whether to enable lazy loading
        """
        self.max_concurrent_loads = max_concurrent_loads
        self.enable_lazy_loading = enable_lazy_loading

        self.models: Dict[str, ModelInstance] = {}
        self.loading_lock = threading.Lock()
        self.semaphore = threading.Semaphore(max_concurrent_loads)

    def register_model(self, name: str, loader_func) -> None:
        """Register a model with its loader function.

        Args:
            name: Model name
            loader_func: Function to load the model (takes no args, returns model)
        """
        if name not in self.models:
            self.models[name] = ModelInstance(
                name=name,
                status=ModelStatus.UNLOADED,
            )
        self.models[name].loader_func = loader_func

    def acquire_model(self, name: str) -> Optional[object]:
        """Acquire a model for use, loading if necessary.

        Args:
            name: Model name

        Returns:
            Model instance or None if loading failed
        """
        with self.loading_lock:
            if name not in self.models:
                logger.error(f"Model {name} not registered")
                return None

            model_instance = self.models[name]

            # Increment reference count
            model_instance.ref_count += 1
            model_instance.last_used = None  # Will be set on use

            # Load if not loaded
            if model_instance.status == ModelStatus.UNLOADED:
                if self.enable_lazy_loading:
                    return self._load_model(name)
                else:
                    logger.warning(f"Lazy loading disabled, model {name} not preloaded")
                    return None

            return model_instance.model

    def release_model(self, name: str) -> None:
        """Release a model after use.

        Args:
            name: Model name
        """
        with self.loading_lock:
            if name in self.models:
                model_instance = self.models[name]
                model_instance.ref_count = max(0, model_instance.ref_count - 1)

                # Optionally unload if no references and memory pressure is high
                # For now, we keep models loaded for performance

    def _load_model(self, name: str) -> Optional[object]:
        """Load a model with concurrency control.

        Args:
            name: Model name

        Returns:
            Loaded model or None
        """
        model_instance = self.models[name]

        if model_instance.status == ModelStatus.LOADING:
            # Wait for loading to complete
            logger.info(f"Model {name} already loading, waiting...")
            # In production, use asyncio properly or proper synchronization
            return None

        model_instance.status = ModelStatus.LOADING

        with self.semaphore:
            try:
                import time

                start_time = time.time()
                model = model_instance.loader_func()
                load_time = time.time() - start_time

                model_instance.model = model
                model_instance.status = ModelStatus.LOADED
                model_instance.load_time = load_time
                model_instance.error_message = None

                logger.info(f"Model {name} loaded in {load_time:.2f}s")
                return model

            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")
                model_instance.status = ModelStatus.ERROR
                model_instance.error_message = str(e)
                return None

    def unload_model(self, name: str) -> bool:
        """Unload a model to free memory.

        Args:
            name: Model name

        Returns:
            True if unloaded, False if not found or in use
        """
        with self.loading_lock:
            if name not in self.models:
                return False

            model_instance = self.models[name]

            if model_instance.ref_count > 0:
                logger.warning(f"Cannot unload model {name}, still in use (ref_count={model_instance.ref_count})")
                return False

            model_instance.status = ModelStatus.UNLOADING

            try:
                # Clear model reference
                model_instance.model = None
                model_instance.status = ModelStatus.UNLOADED
                model_instance.load_time = None
                model_instance.last_used = None

                logger.info(f"Model {name} unloaded")
                return True

            except Exception as e:
                logger.error(f"Error unloading model {name}: {e}")
                model_instance.status = ModelStatus.ERROR
                return False

    def get_status(self, name: str) -> Optional[ModelStatus]:
        """Get model status.

        Args:
            name: Model name

        Returns:
            Model status or None if not found
        """
        if name in self.models:
            return self.models[name].status
        return None

    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all registered models.

        Returns:
            Dictionary of model statuses
        """
        return {
            name: {
                "status": instance.status.value,
                "loaded": instance.status == ModelStatus.LOADED,
                "ref_count": instance.ref_count,
                "load_time": instance.load_time,
                "error": instance.error_message,
            }
            for name, instance in self.models.items()
        }

    def preload_models(self, model_names: Optional[Set[str]] = None) -> None:
        """Preload models (e.g., at startup).

        Args:
            model_names: Set of model names to preload. If None, preload all.
        """
        if model_names is None:
            model_names = set(self.models.keys())

        logger.info(f"Preloading {len(model_names)} models...")

        for name in model_names:
            if name in self.models:
                self._load_model(name)

    def cleanup(self) -> None:
        """Unload all models (e.g., at shutdown)."""
        logger.info("Cleaning up all models...")
        for name in list(self.models.keys()):
            self.unload_model(name)


@contextmanager
def model_context(manager: ModelLifecycleManager, model_name: str):
    """Context manager for model usage.

    Args:
        manager: ModelLifecycleManager instance
        model_name: Model name

    Yields:
        Model instance
    """
    model = manager.acquire_model(model_name)
    try:
        yield model
    finally:
        manager.release_model(model_name)


# Global lifecycle manager
_lifecycle_manager: Optional[ModelLifecycleManager] = None


def get_lifecycle_manager() -> ModelLifecycleManager:
    """Get the global model lifecycle manager.

    Returns:
        ModelLifecycleManager instance
    """
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ModelLifecycleManager()
    return _lifecycle_manager

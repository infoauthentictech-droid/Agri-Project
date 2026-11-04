"""Performance Benchmarking

Benchmarks model loading, inference latency, peak RAM/VRAM, throughput,
and concurrent requests to ensure models perform well in production.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    model_name: str
    load_time_ms: float
    inference_time_ms: float
    peak_memory_mb: float
    peak_gpu_memory_mb: Optional[float] = None
    throughput_requests_per_second: Optional[float] = None
    concurrent_capacity: Optional[int] = None


class PerformanceBenchmark:
    """Performance benchmark for ML models."""

    def __init__(self):
        """Initialize performance benchmark."""
        self.process = psutil.Process()

    def measure_memory(self) -> float:
        """Measure current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        return self.process.memory_info().rss / 1024 / 1024

    def measure_gpu_memory(self) -> Optional[float]:
        """Measure current GPU memory usage in MB.

        Returns:
            GPU memory usage in MB or None if no GPU
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return None

    def benchmark_model_loading(
        self, loader_func, model_name: str
    ) -> float:
        """Benchmark model loading time.

        Args:
            loader_func: Function to load the model
            model_name: Name of model

        Returns:
            Load time in milliseconds
        """
        # Force garbage collection before benchmark
        import gc
        gc.collect()

        start_mem = self.measure_memory()
        start_time = time.time()

        model = loader_func()

        end_time = time.time()
        end_mem = self.measure_memory()

        load_time_ms = (end_time - start_time) * 1000
        memory_delta = end_mem - start_mem

        logger.info(
            f"{model_name} load time: {load_time_ms:.2f}ms, "
            f"memory delta: {memory_delta:.2f}MB"
        )

        return load_time_ms

    def benchmark_inference(
        self, model, inference_func, input_data, num_runs: int = 10
    ) -> Dict:
        """Benchmark inference latency.

        Args:
            model: Model instance
            inference_func: Function to run inference
            input_data: Input data for inference
            num_runs: Number of inference runs

        Returns:
            Dictionary with latency statistics
        """
        # Warmup
        for _ in range(3):
            _ = inference_func(model, input_data)

        # Measure latency
        latencies = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = inference_func(model, input_data)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        return {
            "avg_ms": sum(latencies) / len(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "std_ms": (sum((x - sum(latencies) / len(latencies)) ** 2 for x in latencies) / len(latencies)) ** 0.5,
        }

    def benchmark_concurrent_requests(
        self, inference_func, input_data, max_concurrent: int = 10
    ) -> int:
        """Benchmark concurrent request capacity.

        Args:
            inference_func: Function to run inference
            input_data: Input data for inference
            max_concurrent: Maximum concurrent requests to test

        Returns:
            Maximum concurrent requests that succeed
        """
        results = []
        threads = []

        def run_inference(result_list, index):
            try:
                start = time.time()
                _ = inference_func(input_data)
                end = time.time()
                result_list.append((index, end - start, True))
            except Exception as e:
                result_list.append((index, 0, False))

        for i in range(max_concurrent):
            t = threading.Thread(target=run_inference, args=(results, i))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        successful = sum(1 for _, _, success in results if success)
        return successful

    def run_full_benchmark(
        self, model_name: str, loader_func, inference_func, input_data
    ) -> BenchmarkResult:
        """Run full performance benchmark.

        Args:
            model_name: Name of model
            loader_func: Function to load model
            inference_func: Function to run inference
            input_data: Input data

        Returns:
            BenchmarkResult
        """
        logger.info(f"Starting full benchmark for {model_name}...")

        # Benchmark loading
        load_time = self.benchmark_model_loading(loader_func, model_name)

        # Load model
        model = loader_func()

        # Benchmark inference
        latency_stats = self.benchmark_inference(model, inference_func, input_data)

        # Benchmark concurrency
        concurrent_capacity = self.benchmark_concurrent_requests(
            lambda data: inference_func(model, data), input_data
        )

        # Measure memory
        peak_memory = self.measure_memory()
        peak_gpu_memory = self.measure_gpu_memory()

        return BenchmarkResult(
            model_name=model_name,
            load_time_ms=load_time,
            inference_time_ms=latency_stats["avg_ms"],
            peak_memory_mb=peak_memory,
            peak_gpu_memory_mb=peak_gpu_memory,
            concurrent_capacity=concurrent_capacity,
        )


def get_benchmark() -> PerformanceBenchmark:
    """Get the global performance benchmark instance.

    Returns:
        PerformanceBenchmark instance
    """
    return PerformanceBenchmark()

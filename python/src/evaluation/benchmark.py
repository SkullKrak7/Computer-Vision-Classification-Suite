"""Model benchmarking utilities"""

import time
from collections.abc import Callable

import numpy as np


def benchmark_inference(
    model_predict: Callable, X: np.ndarray, warmup: int = 5, runs: int = 10
) -> dict:
    """Benchmark model inference speed"""
    # Warmup
    for _ in range(warmup):
        _ = model_predict(X[:1])

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model_predict(X)
        times.append(time.time() - start)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "throughput": len(X) / np.mean(times),
    }

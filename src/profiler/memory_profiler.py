import time
import tracemalloc
from typing import Optional

import torch


def get_current_memory_mb(device: torch.device) -> float:
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            pass
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    current, _peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def get_peak_memory_mb(device: torch.device, sampled_peak_mb: Optional[float] = None) -> float:
    if device.type == "mps":
        if sampled_peak_mb is not None:
            return float(sampled_peak_mb)
        return get_current_memory_mb(device)
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    _current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


def timed_step(fn):
    start = time.perf_counter()
    out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return out, elapsed_ms

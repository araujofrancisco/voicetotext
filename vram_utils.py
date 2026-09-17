"""
Shared VRAM utilities for GPU memory checking and automatic model fallback.
Used by both app.py (CLI) and web_app.py (Flask).
"""

import gc
import torch

GB = 1024 ** 3

MODEL_VRAM_ESTIMATES = {
    "tiny":     int(1.5 * GB),
    "base":     int(2.0 * GB),
    "small":    int(3.5 * GB),
    "medium":   int(9.0 * GB),
    "large":   int(16.0 * GB),
}

MODEL_FALLBACK_ORDER = ["tiny", "base", "small", "medium", "large"]


def get_gpu_info() -> list[dict]:
    """Return a list of dicts with GPU info for each available CUDA device."""
    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        used_bytes = total_bytes - free_bytes
        gpus.append({
            "index": i,
            "name": props.name,
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
            "used_bytes": used_bytes,
        })
    return gpus


def get_model_vram(model_name: str) -> int:
    """Return estimated VRAM in bytes for a model name (with buffer)."""
    base = MODEL_VRAM_ESTIMATES.get(model_name)
    if base is None:
        return MODEL_VRAM_ESTIMATES["small"]  # fallback default
    return base


def format_vram(bytes_val: int) -> str:
    """Convert bytes to human-readable string (e.g., '5.5 GB')."""
    gb = bytes_val / GB
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = bytes_val / (1024 ** 2)
    return f"{mb:.0f} MB"


def format_gpu_info(gpus: list[dict]) -> str:
    """Human-readable GPU string for logging."""
    parts = []
    for g in gpus:
        parts.append(f"{g['name']} #{g['index']} ({format_vram(g['free_bytes'])} free)")
    return ", ".join(parts)


def find_best_gpu_from_list(gpus: list[dict]) -> dict | None:
    """Return the GPU with the most free memory from a list."""
    if not gpus:
        return None
    return max(gpus, key=lambda g: g["free_bytes"])


def find_best_gpu(preferred_model: str) -> tuple[int, str, str | None]:
    """Find the best GPU and model combo.

    1. Sort GPUs by free memory descending.
    2. Check if preferred_model fits on any GPU.
    3. If not, try smaller models in fallback order (tiny → base → small → medium → large).
    4. Return (gpu_index, model_used, None | downgraded_from_original).

    Returns gpu_idx=0 with 'tiny' as last resort even if it doesn't fit.
    """
    gpus = get_gpu_info()
    if not gpus:
        return (0, preferred_model, None)

    # Sort by free memory descending
    sorted_gpus = sorted(gpus, key=lambda g: g["free_bytes"], reverse=True)

    vram_needed = get_model_vram(preferred_model)

    # Check if preferred model fits on any GPU
    for gpu in sorted_gpus:
        if gpu["free_bytes"] >= vram_needed:
            return (gpu["index"], preferred_model, None)

    # Preferred doesn't fit — try smaller models in fallback order
    for candidate in MODEL_FALLBACK_ORDER:
        candidate_vram = get_model_vram(candidate)
        for gpu in sorted_gpus:
            if gpu["free_bytes"] >= candidate_vram:
                return (gpu["index"], candidate, preferred_model)

    # Nothing fits — fall back to smallest model on best GPU anyway
    # Let torch handle the OOM and report it clearly
    return (sorted_gpus[0]["index"], "tiny", preferred_model)


def get_next_smaller_model(current_model: str) -> str | None:
    """Return the next smaller model in fallback order, or None if already at smallest."""
    idx = MODEL_FALLBACK_ORDER.index(current_model) if current_model in MODEL_FALLBACK_ORDER else -1
    if idx <= 0:
        return None
    return MODEL_FALLBACK_ORDER[idx - 1]

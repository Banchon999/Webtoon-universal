"""Compute device selection.

Frozen (PyInstaller) builds ship CPU-only torch, so CUDA is only used when
running from source with a CUDA-enabled torch install.
"""

from __future__ import annotations

import functools

from ..paths import is_frozen


@functools.lru_cache(maxsize=1)
def pick_device(preference: str = "auto") -> str:
    if preference == "cpu":
        return "cpu"
    if is_frozen() and preference == "auto":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def onnx_providers() -> list[str]:
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

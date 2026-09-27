"""Device placement helpers. Nothing in this project called `.to(device)` anywhere
until now -- correct for this machine (CPU-only, so there was nothing to move to),
but a real gap the moment training moves to Kaggle/Colab, where doing nothing here
means silently running on CPU in a GPU notebook. `resolve_device(None)` auto-detects,
so every script keeps working unchanged locally and starts using the GPU automatically
the moment one is available -- no script-level changes needed either place.
"""
from __future__ import annotations

from typing import Optional

import torch


def resolve_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Moves every tensor value to `device`; leaves non-tensor metadata (e.g. the
    Collator's `domain` list of strings) untouched rather than erroring on it."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

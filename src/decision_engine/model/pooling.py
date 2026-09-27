"""Shared mean-pooling helper, used for both option-query pooling (option_encoder.py)
and state pooling for the MoE gate (gate.py needs a fixed-size state summary to route
on). Kept as its own module so the padding-exclusion behavior is tested once, in
isolation, rather than duplicated and re-verified in two places.
"""
from __future__ import annotations

import torch


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """hidden: [N, L, H]; attention_mask: [N, L] (1 = real token, 0 = padding).
    Returns [N, H], averaged over real tokens only."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

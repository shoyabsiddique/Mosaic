"""Soft top-k MoE gate -- see docs/architecture-decisions.md. Standard sparse gating
(Shazeer et al. 2017) and Switch-Transformer-style load balancing (Fedus, Zoph &
Shazeer, 2021): well-established, non-proprietary techniques, not borrowed from either
competitor (neither Jev nor Laya has a gate at all).

`hard=True` selects exactly one expert (softmax over a single logit is just 1.0) and
implements the `routing_mode: "hard"` latency fast-path from architecture-decisions.md;
the default (`hard=False`) is the soft top-k blend.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.proj = nn.Linear(hidden_size, num_experts)

    def forward(self, pooled_state: torch.Tensor, hard: bool = False):
        """pooled_state: [B, H].
        Returns (blend_weights [B, E] sparse -- zero outside the selected set,
                 gate_logits [B, E] dense, topk_idx [B, k])."""
        gate_logits = self.proj(pooled_state)
        k = 1 if hard else self.top_k
        topk_vals, topk_idx = gate_logits.topk(k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        blend_weights = torch.zeros_like(gate_logits).scatter(-1, topk_idx, topk_weights)
        return blend_weights, gate_logits, topk_idx


def load_balancing_loss(gate_logits: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Switch-Transformer-style auxiliary loss: penalizes experts whose dense routing
    probability and actual top-k selection frequency are both high at once, pushing the
    gate toward spreading load rather than collapsing onto a few experts. Its minimum
    isn't exactly zero (it scales with top_k/num_experts at perfect balance) -- what
    matters is that it's lower for balanced routing than for collapsed routing, which is
    what makes it a useful gradient signal, not a specific target value."""
    probs = torch.softmax(gate_logits, dim=-1)  # [B, E]
    mean_prob = probs.mean(0)  # [E]
    one_hot = torch.zeros_like(probs).scatter(-1, topk_idx, 1.0)  # [B, E]
    frac_routed = one_hot.mean(0)  # [E]
    return num_experts * (mean_prob * frac_routed).sum()

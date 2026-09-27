"""Phase 2 baseline head: exactly one instance of the poly-encoder-style cross-attention
mechanism from docs/expert-head-design.md, with no MoE gate and no multiple domain
experts. This is the accuracy/latency floor Phase 3's MoE router needs to beat -- see
PLAN.md's Phase 2 entry.
"""
from __future__ import annotations

import torch.nn as nn


class SingleHead(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.compare_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, option_queries, option_mask, state_hidden, state_attention_mask):
        """option_queries: [B, K, H]; option_mask: [B, K] bool (True = real option).
        state_hidden: [B, L, H]; state_attention_mask: [B, L] (1 = real token).
        Returns: logits [B, K], invalid option positions set to -1e4."""
        state_key_padding_mask = ~state_attention_mask.bool()
        cross_out, _ = self.cross_attn(
            query=option_queries, key=state_hidden, value=state_hidden,
            key_padding_mask=state_key_padding_mask,
        )
        x = self.norm1(option_queries + cross_out)

        option_key_padding_mask = ~option_mask
        compare_out, _ = self.compare_attn(
            query=x, key=x, value=x, key_padding_mask=option_key_padding_mask,
        )
        x = self.norm2(x + compare_out)

        logits = self.scorer(x).squeeze(-1)
        return logits.masked_fill(~option_mask, -1e4)

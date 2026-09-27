"""Encodes (instructions, option_text) pairs into per-option query vectors for the
cross-attention head -- see docs/expert-head-design.md.

Reuses the shared encoder's weights for the short option-pair pass (no separate
tower), mean-pools the resulting tokens, and adds a learned type embedding so the
query is both question-aware (via the instructions half of the pair) and type-aware
(choice/score/noul), matching the type-conditioning role Laya's own type embedding
plays -- but injected into the option query here rather than broadcast across every
state token.
"""
from __future__ import annotations

import torch.nn as nn

from decision_engine.model.pooling import mean_pool
from decision_engine.model.types import QTYPE_TO_IDX


class OptionEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder  # shared weights with the state encoder
        self.type_embedding = nn.Embedding(len(QTYPE_TO_IDX), hidden_size)

    def forward(self, input_ids, attention_mask, qtype_idx):
        """input_ids/attention_mask: [N, L] flattened over (batch, option) pairs.
        qtype_idx: [N] int tensor, one entry per flattened option.
        Returns: [N, hidden_size] pooled, type-conditioned query vectors."""
        h = self.encoder(input_ids, attention_mask)  # [N, L, H]
        pooled = mean_pool(h, attention_mask)
        return pooled + self.type_embedding(qtype_idx)

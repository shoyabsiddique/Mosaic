"""Phase 3: shared NeoBERT encoder + soft top-k gate + one SingleHead expert per
domain -- see docs/architecture-decisions.md. Each expert is literally a SingleHead
(Phase 2's baseline head, unchanged) -- the MoE moat is entirely in having several of
them plus a learned gate, not a different per-expert mechanism.

Known limitation, honestly scoped rather than hidden: this computes every expert
densely for every example, then masks/blends by gate weight. That's correct for
proving the mechanism (Phase 3's goal) but not how it should run in production --
serving should only forward through the top-k *selected* experts per example. Sparse
dispatch is real optimization work for Phase 6 (docs/local-serving.md), not attempted
here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, Encoder
from decision_engine.model.gate import Gate
from decision_engine.model.option_encoder import OptionEncoder
from decision_engine.model.pooling import mean_pool
from decision_engine.model.single_head import SingleHead


class MoEModel(nn.Module):
    def __init__(self, domain_names: list[str], model_name: str = DEFAULT_ENCODER_NAME, top_k: int = 2):
        super().__init__()
        self.domain_names = list(domain_names)
        self.encoder = Encoder(model_name)
        self.option_encoder = OptionEncoder(self.encoder, self.encoder.hidden_size)
        self.experts = nn.ModuleList([SingleHead(self.encoder.hidden_size) for _ in self.domain_names])
        self.gate = Gate(self.encoder.hidden_size, len(self.domain_names), top_k=top_k)

    def forward(self, batch: dict, hard_routing: bool = False) -> dict:
        state_hidden = self.encoder(batch["state_input_ids"], batch["state_attention_mask"])
        pooled_state = mean_pool(state_hidden, batch["state_attention_mask"])
        blend_weights, gate_logits, topk_idx = self.gate(pooled_state, hard=hard_routing)

        b, k, l_opt = batch["option_input_ids"].shape
        flat_ids = batch["option_input_ids"].view(b * k, l_opt)
        flat_mask = batch["option_attention_mask"].view(b * k, l_opt)
        flat_qtype = batch["qtype_idx"].unsqueeze(1).expand(b, k).reshape(b * k)
        option_queries = self.option_encoder(flat_ids, flat_mask, flat_qtype).view(b, k, -1)

        expert_logits = torch.stack([
            expert(option_queries, batch["option_mask"], state_hidden, batch["state_attention_mask"])
            for expert in self.experts
        ], dim=1)  # [B, E, K]
        expert_probs = torch.softmax(expert_logits, dim=-1)
        blended_probs = (expert_probs * blend_weights.unsqueeze(-1)).sum(1)  # [B, K]

        return {
            "blended_probs": blended_probs,
            "expert_probs": expert_probs,
            "gate_logits": gate_logits,
            "topk_idx": topk_idx,
            "blend_weights": blend_weights,
        }

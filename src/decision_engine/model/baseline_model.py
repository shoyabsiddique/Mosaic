"""Phase 2 baseline: shared NeoBERT encoder (state + options) + one SingleHead.
No MoE gate, no multiple domain experts yet -- see PLAN.md Phase 2.
"""
from __future__ import annotations

import torch.nn as nn

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, Encoder
from decision_engine.model.option_encoder import OptionEncoder
from decision_engine.model.single_head import SingleHead


class BaselineModel(nn.Module):
    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        self.encoder = Encoder(model_name)
        self.option_encoder = OptionEncoder(self.encoder, self.encoder.hidden_size)
        self.head = SingleHead(self.encoder.hidden_size)

    def forward(self, batch: dict):
        state_hidden = self.encoder(batch["state_input_ids"], batch["state_attention_mask"])

        b, k, l_opt = batch["option_input_ids"].shape
        flat_ids = batch["option_input_ids"].view(b * k, l_opt)
        flat_mask = batch["option_attention_mask"].view(b * k, l_opt)
        flat_qtype = batch["qtype_idx"].unsqueeze(1).expand(b, k).reshape(b * k)

        option_queries = self.option_encoder(flat_ids, flat_mask, flat_qtype).view(b, k, -1)
        return self.head(option_queries, batch["option_mask"], state_hidden, batch["state_attention_mask"])

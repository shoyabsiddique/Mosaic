"""NeoBERT encoder wrapper -- see docs/architecture-decisions.md for why this encoder
was chosen over Laya's ModernBERT. Confirmed live: 768 hidden size, 28 layers, 12
heads, MIT licensed, requires `trust_remote_code=True` (custom modeling file) and the
`xformers` package importable (its compiled CUDA ops aren't required on CPU -- the
modeling file's own import check just needs the package present; it falls back to a
plain-Python attention path when its extensions can't load).
"""
from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER_NAME = "chandar-lab/NeoBERT"


def load_tokenizer(model_name: str = DEFAULT_ENCODER_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


class Encoder(nn.Module):
    """Used for both the (long) state pass and the (short) option-query pass in
    option_encoder.py -- one set of weights, no separate towers."""

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

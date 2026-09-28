"""NeoBERT encoder wrapper -- see docs/architecture-decisions.md for why this encoder
was chosen over Laya's ModernBERT. Confirmed live: 768 hidden size, 28 layers, 12
heads, MIT licensed, requires `trust_remote_code=True` (custom modeling file) and the
`xformers` package importable.

FORCED EAGER SwiGLU -- root-caused a real production bug, not a defensive guess:
NeoBERT's FFN uses `xformers.ops.SwiGLU` with `op=None`, which lets xformers
auto-select "the best implementation" per its own docstring. On this CPU dev machine,
the installed xformers build has no compiled CUDA kernels, so its `SwiGLUOpDispatch.op`
property is hardcoded to always return the safe, plain-PyTorch `SwiGLUEagerOp` --
confirmed by reading `xformers/ops/swiglu_op.py` directly. A real CUDA install of
xformers (e.g. on Colab/Kaggle) has genuine dispatch logic and can select an actual
fused CUDA kernel instead -- a code path this dev machine cannot exercise or test at
all. Training on a real T4 produced a non-finite loss on literally 100% of batches
across every domain and question type, in plain fp32 *and* both AMP dtypes (ruling out
precision/overflow as the cause -- see train_moe_eval.py's docstring history) -- i.e.
the one thing that reliably differed between "works" (this machine) and "always
broken" (real GPU) was exactly this fused-vs-eager SwiGLU dispatch. Forcing eager
directly on every loaded instance, regardless of what xformers build ends up
installed, removes that variable entirely rather than hoping a future xformers
version's fused kernel happens to be correct.
"""
from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER_NAME = "chandar-lab/NeoBERT"


def load_tokenizer(model_name: str = DEFAULT_ENCODER_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _force_eager_swiglu(model: nn.Module) -> int:
    """Walks every submodule and forces xformers' SwiGLU dispatch to the plain
    eager implementation, sidestepping whatever fused CUDA kernel a real xformers
    build might otherwise auto-select. Returns how many instances were patched, so a
    silent 0 (e.g. after an xformers internal rename) is visible rather than assumed."""
    import xformers.ops as xops

    patched = 0
    for module in model.modules():
        if isinstance(module, xops.SwiGLU):
            module.op = xops.SwiGLUEagerOp
            patched += 1
    return patched


class Encoder(nn.Module):
    """Used for both the (long) state pass and the (short) option-query pass in
    option_encoder.py -- one set of weights, no separate towers."""

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size
        n_patched = _force_eager_swiglu(self.backbone)
        print(f"encoder.py: forced eager SwiGLU on {n_patched} module(s)")

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

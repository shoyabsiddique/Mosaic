"""NeoBERT encoder wrapper -- see docs/architecture-decisions.md for why this encoder
was chosen over Laya's ModernBERT. Confirmed live: 768 hidden size, 28 layers, 12
heads, MIT licensed, requires `trust_remote_code=True` (custom modeling file) and the
`xformers` package importable (its compiled CUDA ops aren't required on CPU -- the
modeling file's own import check just needs the package present; it falls back to a
plain-Python attention path when its extensions can't load).

Note: a prior version of this file force-set each SwiGLU submodule's `.op` to
`SwiGLUEagerOp` on the theory that a real CUDA xformers build might auto-dispatch to a
different (buggy) fused kernel. That was wrong and actively harmful -- confirmed by
reading `xformers/ops/swiglu_op.py`'s `SwiGLUOpDispatch.op` property, which
unconditionally returns `SwiGLUEagerOp` regardless of platform (it's a plain Python
property, not CUDA-gated), so the "different build, different dispatch" premise never
held. Worse, explicitly setting `.op` makes `SwiGLU.forward()` take its packed-weights
fast path, which then asserts `self.op.PACKED_WEIGHTS` -- `False` for the eager op --
crashing immediately on a real run.

REAL root cause of the NaN-on-CUDA bug, found via scripts/diagnose_nan.py plus direct
inspection of transformers' loading internals: `from_pretrained` constructs
`trust_remote_code` models under an internal meta-device context, then re-materializes
any *non-persistent* buffer -- like NeoBERT's own rotary-embedding table `freqs_cis`,
computed once in its `__init__` and registered with `persistent=False` since it's
derived, not learned -- via `torch.empty_like` (uninitialized memory) rather than by
recomputing it (see transformers' `_move_missing_keys_from_meta_to_device`). NeoBERT's
own `_init_weights` only initializes `nn.Linear`/`nn.Embedding` submodules, so this
buffer is left as uninitialized memory, which empirically comes out all-zeros.
Confirmed directly: `AutoModel.from_pretrained(...).freqs_cis` is exactly `0+0j`
everywhere, while constructing the same class fresh (bypassing `from_pretrained`) gives
the correct rotary table. A zeroed `freqs_cis` means every query/key vector is
multiplied by zero in `apply_rotary_emb`, so attention receives degenerate all-zero
query/key input at every layer -- something CUDA's SDPA kernel apparently cannot
handle without producing NaN, even though CPU's kernel tolerates the same degenerate
input (silently wrong, not NaN), which is why this was never reproducible locally.
Fixed by recomputing the buffer in place after loading, using NeoBERT's own
`precompute_freqs_cis` (reached via its dynamically-loaded module, so this stays in
sync with whatever formula NeoBERT itself uses rather than duplicating it here).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER_NAME = "chandar-lab/NeoBERT"


def load_tokenizer(model_name: str = DEFAULT_ENCODER_NAME):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _fix_zeroed_freqs_cis(backbone: nn.Module) -> None:
    freqs_cis = getattr(backbone, "freqs_cis", None)
    if freqs_cis is None or not torch.is_complex(freqs_cis):
        return
    precompute_freqs_cis = sys.modules[type(backbone).__module__].precompute_freqs_cis
    dim_head = backbone.config.hidden_size // backbone.config.num_attention_heads
    with torch.no_grad():
        backbone.freqs_cis.copy_(precompute_freqs_cis(dim_head, backbone.config.max_length))


class Encoder(nn.Module):
    """Used for both the (long) state pass and the (short) option-query pass in
    option_encoder.py -- one set of weights, no separate towers."""

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.hidden_size = self.backbone.config.hidden_size
        _fix_zeroed_freqs_cis(self.backbone)

    def forward(self, input_ids, attention_mask):
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

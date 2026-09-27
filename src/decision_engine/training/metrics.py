"""Interpretable evaluation metrics -- accuracy and calibration -- as scoped in
docs/code-structure.md but not yet built. Raw loss/reward numbers (what the Phase 2/3
smoke tests reported so far) prove the pipeline runs; these metrics are what actually
answer "is it any good," including on held-out data the model wasn't trained on.
"""
from __future__ import annotations

import torch


def top1_accuracy(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """probs, target, mask: [B, K]. Fraction of examples where the top predicted
    option matches the top target option. Applies to all three question types --
    for `score`, this is "predicted the exact right level," a stricter check than the
    expected-value error `score_mean_absolute_error` reports separately."""
    pred_idx = probs.argmax(-1)
    target_idx = target.argmax(-1)
    return (pred_idx == target_idx).float().mean()


def score_mean_absolute_error(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """For ordinal `score` questions only: |E[predicted level] - E[target level]|,
    in level units (e.g. 0.7 means "off by under one level" on average). Only
    meaningful where mask indicates real options -- call with score-only rows.

    No division by option count: `probs` and `target` are already proper distributions
    over the valid (masked-in) options, each summing to 1, so `(probs * levels *
    mask).sum(-1)` is already the expectation E[level] -- dividing by mask.sum() again
    would double-normalize (caught by a test asserting MAE scales with actual level
    distance, not 1/num_options)."""
    levels = torch.arange(probs.shape[-1], device=probs.device).float()
    pred_level = (probs * levels * mask).sum(-1)
    target_level = (target * levels * mask).sum(-1)
    return (pred_level - target_level).abs().mean()


def expected_calibration_error(confidences: torch.Tensor, correctness: torch.Tensor, n_bins: int = 10) -> float:
    """Standard ECE: bin predictions by confidence, compare each bin's mean confidence
    to its actual accuracy, weight by bin size. A well-known, standard formula
    (Guo et al. 2017) -- implemented independently, not borrowed from Laya's own
    `ece_score` we read during Phase 0 research, though the two necessarily agree
    since ECE has one standard definition."""
    if confidences.numel() == 0:
        return float("nan")
    edges = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(())
    n = confidences.numel()
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = correctness[in_bin].mean()
        ece = ece + (in_bin.sum().float() / n) * (bin_conf - bin_acc).abs()
    return ece.item()


def summarize_batch(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     qtype_idx: torch.Tensor, qtype_to_idx: dict) -> dict:
    """Per-question-type breakdown for one batch -- accumulate across batches for a
    real evaluation pass; see evaluate() in loop.py / moe_loop.py."""
    out: dict = {}
    for name, idx in qtype_to_idx.items():
        sel = qtype_idx == idx
        if sel.sum() == 0:
            continue
        p, t, m = probs[sel], target[sel], mask[sel]
        out[name] = {
            "n": int(sel.sum()),
            "accuracy": top1_accuracy(p, t, m).item(),
        }
        if name == "score":
            out[name]["mae"] = score_mean_absolute_error(p, t, m).item()
    return out

"""Reward/loss functions -- implements docs/reward-design.md.

One strictly proper scoring rule per question-type structure (Brier for unordered
choice/noul, Ranked Probability Score for ordinal score), not Laya's single fixed
log+spherical+RPS combination applied everywhere. Both are differentiable functions of
the reported distribution and a fixed target, so training is direct backprop -- no
policy gradient or exploration noise needed for these single-shot questions (see
reward-design.md's "Reframing RL" section).

MMCE (Kumar, Sarawagi & Jain, ICML 2018) is included as an optional in-training
calibration regularizer, off by default (`lambda_calib=0.0`) until Phase 2/3
experiments confirm whether it earns its keep over post-hoc temperature scaling alone.
"""
from __future__ import annotations

import torch

from decision_engine.model.gate import load_balancing_loss
from decision_engine.model.types import QTYPE_TO_IDX


def brier_score(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """probs, target, mask: [B, K]. Strictly proper (Brier, 1950). Bounded, unlike the
    log score -- returns a per-example score to MAXIMIZE, roughly in [-1, 1]."""
    diff2 = ((probs - target) ** 2) * mask
    return 1.0 - diff2.sum(-1)


def ranked_probability_score(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """RPS: the ordinal generalization of Brier over cumulative distributions.
    Strictly proper for ordered categories. Returns a per-example score to MAXIMIZE
    (negative RPS, since RPS itself is a penalty)."""
    cdf_p = torch.cumsum(probs * mask, dim=-1)
    cdf_t = torch.cumsum(target * mask, dim=-1)
    k = mask.sum(-1).clamp(min=2)
    rps = (((cdf_p - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
    return -rps


def typed_reward(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                  qtype_idx: torch.Tensor) -> torch.Tensor:
    """qtype_idx: [B] int, indexing QTYPE_TO_IDX. `score` -> RPS; `choice`/`noul` -> Brier."""
    is_score = qtype_idx == QTYPE_TO_IDX["score"]
    return torch.where(is_score, ranked_probability_score(probs, target, mask), brier_score(probs, target, mask))


def mmce(confidences: torch.Tensor, correctness: torch.Tensor, gamma: float = 0.2) -> torch.Tensor:
    """Maximum Mean Calibration Error (Kumar et al., ICML 2018), Laplacian kernel.
    confidences, correctness: [N] (top-1 confidence and 0/1 correctness per example)."""
    n = confidences.shape[0]
    if n < 2:
        return confidences.new_zeros(())
    diff = confidences - correctness
    kernel = torch.exp(-torch.abs(confidences.unsqueeze(0) - confidences.unsqueeze(1)) / gamma)
    outer = diff.unsqueeze(0) * diff.unsqueeze(1)
    return (outer * kernel).sum() / (n * n)


def compute_loss_from_probs(probs: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
                             qtype_idx: torch.Tensor, lambda_calib: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Same as compute_loss, but takes an already-computed probability distribution
    rather than logits -- needed for the MoE model, whose output is a *blend* of
    several experts' softmax distributions (see reward-design.md), not a single
    logit vector you can softmax directly. Returns (scalar loss to minimize,
    per-example reward for logging)."""
    mask = option_mask.to(probs.dtype)
    reward = typed_reward(probs, target, mask, qtype_idx)
    loss = -reward.mean()

    if lambda_calib > 0:
        top1_conf, top1_idx = probs.max(-1)
        target_idx = target.argmax(-1)
        correctness = (top1_idx == target_idx).to(probs.dtype)
        loss = loss + lambda_calib * mmce(top1_conf, correctness)

    return loss, reward


def compute_loss(logits: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor,
                  qtype_idx: torch.Tensor, lambda_calib: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-head (Phase 2) path: softmaxes logits, then delegates to
    compute_loss_from_probs. Logits are assumed already masked to -1e4 at invalid
    positions (SingleHead does this)."""
    probs = torch.softmax(logits, dim=-1)
    return compute_loss_from_probs(probs, target, option_mask, qtype_idx, lambda_calib)


def compute_moe_loss(
    blended_probs: torch.Tensor, expert_probs: torch.Tensor, gate_logits: torch.Tensor,
    topk_idx: torch.Tensor, target: torch.Tensor, option_mask: torch.Tensor, qtype_idx: torch.Tensor,
    lambda_expert: float = 0.1, lambda_calib: float = 0.0, lambda_balance: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MoE-aware loss per reward-design.md's "MoE-specific reward wiring": primary
    reward on the blended output (the actual API answer), a smaller auxiliary reward on
    each *selected* expert's own pre-blend distribution (so experts the gate doesn't
    pick this batch still get gradient signal and don't starve -- a different failure
    mode than gate collapse, which load_balancing_loss addresses separately), and the
    gate's own load-balancing term.

    blended_probs: [B, K]; expert_probs: [B, E, K]; gate_logits: [B, E]; topk_idx: [B, top_k].
    """
    loss, reward = compute_loss_from_probs(blended_probs, target, option_mask, qtype_idx, lambda_calib)

    if lambda_expert > 0:
        num_experts = expert_probs.shape[1]
        selected = torch.zeros(blended_probs.shape[0], num_experts, device=blended_probs.device)
        selected.scatter_(1, topk_idx, 1.0)  # [B, E], 1 where that expert was actually consulted

        mask = option_mask.to(expert_probs.dtype)
        expert_loss_total = expert_probs.new_zeros(())
        n_selected = selected.sum().clamp(min=1.0)
        for e in range(num_experts):
            sel_e = selected[:, e]
            if sel_e.sum() == 0:
                continue
            reward_e = typed_reward(expert_probs[:, e, :], target, mask, qtype_idx)
            expert_loss_total = expert_loss_total - (reward_e * sel_e).sum()
        loss = loss + lambda_expert * (expert_loss_total / n_selected)

    if lambda_balance > 0:
        loss = loss + lambda_balance * load_balancing_loss(gate_logits, topk_idx, gate_logits.shape[-1])

    return loss, reward

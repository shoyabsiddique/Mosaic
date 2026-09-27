"""Phase 3 smoke test loop -- proves the MoE mechanism (gate + multiple experts +
MoE-aware reward) runs end-to-end on real data, the same way Phase 2's loop.py proved
the single-head pipeline. Not a convergence run; see docs/phase2-notes.md's scoping
note, which applies here too.

Domain routing is printed per step purely as a diagnostic, not a claim: with a
randomly-initialized gate and only a handful of steps, routing should NOT be expected
to align with ground-truth domains yet -- that alignment (if it emerges at all) is a
Phase 3 research question, not something a smoke test can demonstrate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, load_tokenizer
from decision_engine.model.moe_model import MoEModel
from decision_engine.model.types import QTYPE_TO_IDX
from decision_engine.training.dataset import Collator, TypedDecisionDataset
from decision_engine.training.metrics import expected_calibration_error, summarize_batch
from decision_engine.training.reward import compute_moe_loss

DEFAULT_DOMAIN_NAMES = [
    "customer-support", "trust-safety", "email-communication",
    "sales-crm", "security-ops", "ecommerce",
]


@torch.no_grad()
def evaluate(model: MoEModel, loader: DataLoader) -> dict:
    """Aggregate accuracy/calibration over a held-out set -- what actually answers
    "is it any good," as opposed to the smoke test's raw loss/reward numbers."""
    model.eval()
    all_confidences: list[torch.Tensor] = []
    all_correctness: list[torch.Tensor] = []
    acc_sum: dict[str, float] = {}
    acc_n: dict[str, int] = {}
    mae_sum: dict[str, float] = {}
    mae_n: dict[str, int] = {}

    for batch in loader:
        out = model(batch)
        probs, target, mask, qtype_idx = out["blended_probs"], batch["target"], batch["option_mask"], batch["qtype_idx"]

        correct = (probs.argmax(-1) == target.argmax(-1)).float()
        all_confidences.append(probs.max(-1).values)
        all_correctness.append(correct)

        for name, stats in summarize_batch(probs, target, mask.float(), qtype_idx, QTYPE_TO_IDX).items():
            n = stats["n"]
            acc_sum[name] = acc_sum.get(name, 0.0) + stats["accuracy"] * n
            acc_n[name] = acc_n.get(name, 0) + n
            if "mae" in stats:
                mae_sum[name] = mae_sum.get(name, 0.0) + stats["mae"] * n
                mae_n[name] = mae_n.get(name, 0) + n

    model.train()

    confidences = torch.cat(all_confidences) if all_confidences else torch.tensor([])
    correctness = torch.cat(all_correctness) if all_correctness else torch.tensor([])

    per_type = {}
    for name, n in acc_n.items():
        entry = {"n": n, "accuracy": acc_sum[name] / n}
        if name in mae_sum:
            entry["mae"] = mae_sum[name] / mae_n[name]
        per_type[name] = entry

    return {
        "n_examples": int(confidences.numel()),
        "overall_accuracy": correctness.mean().item() if correctness.numel() else float("nan"),
        "ece": expected_calibration_error(confidences, correctness),
        "per_type": per_type,
    }


def run_moe_smoke_test(
    data_dir: Path,
    domain_names: list[str] = DEFAULT_DOMAIN_NAMES,
    n_steps: int = 10,
    batch_size: int = 2,
    lr: float = 1e-5,
    max_examples: Optional[int] = 200,
    max_state_len: int = 192,
    lambda_expert: float = 0.1,
    lambda_calib: float = 0.0,
    lambda_balance: float = 0.01,
    seed: int = 0,
) -> list[float]:
    torch.manual_seed(seed)

    tokenizer = load_tokenizer(DEFAULT_ENCODER_NAME)
    model = MoEModel(domain_names, DEFAULT_ENCODER_NAME)
    model.train()

    dataset = TypedDecisionDataset(data_dir, max_examples=max_examples)
    collate = Collator(tokenizer, max_state_len=max_state_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        out = model(batch)
        loss, reward = compute_moe_loss(
            out["blended_probs"], out["expert_probs"], out["gate_logits"], out["topk_idx"],
            batch["target"], batch["option_mask"], batch["qtype_idx"],
            lambda_expert=lambda_expert, lambda_calib=lambda_calib, lambda_balance=lambda_balance,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        routed0 = [domain_names[i] for i in out["topk_idx"][0].tolist()]
        print(f"step {step:3d}  loss {loss.item():.4f}  mean_reward {reward.mean().item():.4f}  "
              f"ex0_actual_domain={batch['domain'][0]!r}  ex0_routed_to={routed0}")

    return losses

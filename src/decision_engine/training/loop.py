"""Minimal training loop -- proves the Phase 2 pipeline runs correctly end-to-end on
real data (finite loss, gradients flow, a checkpoint can be saved). This is NOT a
convergence run: real training needs rented GPU compute, not this local CPU-only
machine -- see docs/local-serving.md's inference-vs-training distinction and
docs/tech-stack.md's `accelerate` entry.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from decision_engine.model.baseline_model import BaselineModel
from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, load_tokenizer
from decision_engine.training.dataset import Collator, TypedDecisionDataset
from decision_engine.training.reward import compute_loss


def run_smoke_test(
    data_dir: Path,
    n_steps: int = 10,
    batch_size: int = 2,
    lr: float = 1e-5,
    max_examples: Optional[int] = 200,
    max_state_len: int = 256,
    lambda_calib: float = 0.0,
    seed: int = 0,
) -> list[float]:
    torch.manual_seed(seed)

    tokenizer = load_tokenizer(DEFAULT_ENCODER_NAME)
    model = BaselineModel(DEFAULT_ENCODER_NAME)
    model.train()

    dataset = TypedDecisionDataset(data_dir, max_examples=max_examples)
    collate = Collator(tokenizer, max_state_len=max_state_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        logits = model(batch)
        loss, reward = compute_loss(
            logits, batch["target"], batch["option_mask"], batch["qtype_idx"], lambda_calib=lambda_calib,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"step {step:3d}  loss {loss.item():.4f}  mean_reward {reward.mean().item():.4f}  "
              f"batch_size {batch['qtype_idx'].shape[0]}  max_k {batch['option_mask'].shape[1]}")

    return losses


def save_checkpoint(model: BaselineModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

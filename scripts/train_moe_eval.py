"""Real before/after check on CPU: evaluate the MoE model on held-out data, train for
a modest number of steps, then evaluate again. Interpretable metrics (accuracy, ECE),
not just raw loss -- and critically, measured on examples the model did NOT train on,
so this actually checks generalization rather than memorization of the training batch.

Still not a convergence run (see docs/phase2-notes.md / phase3-notes.md's scoping
notes) -- just a more meaningful CPU check than a bare loss trend.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, load_tokenizer
from decision_engine.model.moe_model import MoEModel
from decision_engine.training.dataset import Collator, TypedDecisionDataset, train_val_split
from decision_engine.training.device_utils import move_batch_to_device, resolve_device
from decision_engine.training.moe_loop import DEFAULT_DOMAIN_NAMES, evaluate
from decision_engine.training.reward import compute_moe_loss


def main():
    torch.manual_seed(0)
    data_dir = Path(__file__).parents[1] / "data"
    device = resolve_device(None)
    print(f"using device: {device}")

    tokenizer = load_tokenizer(DEFAULT_ENCODER_NAME)
    model = MoEModel(DEFAULT_DOMAIN_NAMES, DEFAULT_ENCODER_NAME).to(device)

    dataset = TypedDecisionDataset(data_dir, max_examples=300)
    train_ds, val_ds = train_val_split(dataset, val_fraction=0.2, seed=0)
    print(f"train examples: {len(train_ds)}   val examples: {len(val_ds)}")

    collate = Collator(tokenizer, max_state_len=192)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)

    print("\n=== BEFORE training (random init) ===")
    before = evaluate(model, val_loader)
    print(before)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    n_steps = 20
    print(f"\n=== training for {n_steps} steps ===")
    step = 0
    for batch in train_loader:
        if step >= n_steps:
            break
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        loss, reward = compute_moe_loss(
            out["blended_probs"], out["expert_probs"], out["gate_logits"], out["topk_idx"],
            batch["target"], batch["option_mask"], batch["qtype_idx"],
            lambda_expert=0.1, lambda_calib=0.0, lambda_balance=0.01,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step {step:3d}  loss {loss.item():.4f}  mean_reward {reward.mean().item():.4f}")
        step += 1

    print("\n=== AFTER training ===")
    after = evaluate(model, val_loader)
    print(after)

    print("\n=== summary ===")
    print(f"overall accuracy: {before['overall_accuracy']:.4f} -> {after['overall_accuracy']:.4f}")
    print(f"ECE:              {before['ece']:.4f} -> {after['ece']:.4f}")
    for name in after["per_type"]:
        b = before["per_type"].get(name, {})
        a = after["per_type"][name]
        print(f"  {name}: n={a['n']}  accuracy {b.get('accuracy', float('nan')):.4f} -> {a['accuracy']:.4f}"
              + (f"  mae {b.get('mae', float('nan')):.4f} -> {a['mae']:.4f}" if "mae" in a else ""))


if __name__ == "__main__":
    main()

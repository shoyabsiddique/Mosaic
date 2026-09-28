"""Real before/after check: evaluate the MoE model on held-out data, train for N
steps, then evaluate again. Interpretable metrics (accuracy, ECE), not just raw loss --
and critically, measured on examples the model did NOT train on, so this checks
generalization rather than memorization of the training batch.

Sizing is deliberately configurable rather than hardcoded: the CPU-smoke-test defaults
(300 examples, 20 steps, batch size 2) exist because that's what a CPU could handle in
reasonable time. On a real GPU there's no reason to stay that small -- scale up via the
flags below. A non-finite (NaN/Inf) loss on any step is skipped rather than applied
(see the SKIPPED log line), so a bad batch can't corrupt the whole run; if you see
skips, root-cause them (scripts/diagnose_nan.py) before trusting the final numbers,
but the run itself will still complete and any good steps still train normally.
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, load_tokenizer
from decision_engine.model.moe_model import MoEModel
from decision_engine.training.dataset import Collator, TypedDecisionDataset, train_val_split
from decision_engine.training.device_utils import move_batch_to_device, resolve_device
from decision_engine.training.moe_loop import DEFAULT_DOMAIN_NAMES, evaluate
from decision_engine.training.reward import compute_moe_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-examples", type=int, default=300, help="total examples pulled before the train/val split")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--max-state-len", type=int, default=192, help="up to NeoBERT's real limit of 4096")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-expert", type=float, default=0.1)
    p.add_argument("--lambda-calib", type=float, default=0.0)
    p.add_argument("--lambda-balance", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    data_dir = Path(__file__).parents[1] / "data"
    device = resolve_device(None)
    print(f"using device: {device}")
    print(f"config: {vars(args)}")

    tokenizer = load_tokenizer(DEFAULT_ENCODER_NAME)
    model = MoEModel(DEFAULT_DOMAIN_NAMES, DEFAULT_ENCODER_NAME).to(device)

    dataset = TypedDecisionDataset(data_dir, max_examples=args.max_examples)
    train_ds, val_ds = train_val_split(dataset, val_fraction=args.val_fraction, seed=args.seed)
    print(f"train examples: {len(train_ds)}   val examples: {len(val_ds)}")

    collate = Collator(tokenizer, max_state_len=args.max_state_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)

    print("\n=== BEFORE training (random init) ===")
    before = evaluate(model, val_loader)
    print(before)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    print(f"\n=== training for {args.n_steps} steps ===")
    step = 0
    n_skipped = 0
    while step < args.n_steps:
        for batch in train_loader:
            if step >= args.n_steps:
                break
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            loss, reward = compute_moe_loss(
                out["blended_probs"], out["expert_probs"], out["gate_logits"], out["topk_idx"],
                batch["target"], batch["option_mask"], batch["qtype_idx"],
                lambda_expert=args.lambda_expert, lambda_calib=args.lambda_calib, lambda_balance=args.lambda_balance,
            )

            if not torch.isfinite(loss):
                # Skip rather than let a NaN/Inf loss reach optimizer.step(), which would
                # permanently corrupt every parameter with a NaN update -- see moe_loop.py's
                # run_moe_smoke_test for the same guard and scripts/diagnose_nan.py for a
                # deeper one-off root-cause dive.
                n_skipped += 1
                bad_rows = (~torch.isfinite(out["blended_probs"])).any(-1)
                print(f"step {step:4d}  SKIPPED: non-finite loss ({loss.item()})  "
                      f"bad_qtypes={batch['qtype_idx'][bad_rows].tolist()}  "
                      f"bad_domains={[d for d, bad in zip(batch['domain'], bad_rows.tolist()) if bad]}")
                optimizer.zero_grad()
                step += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % max(1, args.n_steps // 50) == 0 or step == args.n_steps - 1:
                print(f"step {step:4d}  loss {loss.item():.4f}  mean_reward {reward.mean().item():.4f}")
            step += 1

    if n_skipped:
        print(f"\n{n_skipped}/{args.n_steps} steps skipped due to non-finite loss -- "
              f"root-cause before fully trusting these numbers (see bad_qtypes/bad_domains above).")

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

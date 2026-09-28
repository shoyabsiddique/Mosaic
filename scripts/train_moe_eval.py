"""Real before/after check: evaluate the MoE model on held-out data, train for N
optimizer updates, then evaluate again. Interpretable metrics (accuracy, ECE), not just
raw loss -- and critically, measured on examples the model did NOT train on, so this
checks generalization rather than memorization of the training batch.

Sizing is configurable rather than hardcoded, since the CPU-smoke-test defaults exist
only because that's what a CPU could handle in reasonable time -- scale up on a GPU.
Two things make that scaling actually fit in GPU memory instead of just OOM-ing:
  - Mixed precision (`torch.amp`), enabled automatically on CUDA, defaulting to
    **bfloat16** -- roughly halves activation memory like fp16 does, but keeps fp32's
    dynamic range. fp16 was tried first and confirmed, on a real T4 run, to produce a
    non-finite loss on literally every single batch regardless of domain or question
    type -- a classic symptom of pretrained-model activations overflowing fp16's
    narrow range (max ~65504), not a bug in this codebase's own logic. bf16 doesn't
    have that ceiling. `--amp-dtype fp16` is kept available for comparison, not as the
    default. Note T4 (Turing) has no native bf16 Tensor Core acceleration -- expect the
    memory win without necessarily the full throughput win newer GPUs would give bf16.
  - Gradient accumulation (`--grad-accum-steps`): `--batch-size` is the micro-batch
    that must fit in memory; the *effective* batch size used for each optimizer update
    is `batch-size * grad-accum-steps`. This is the standard way to train at a large
    effective batch size without needing it to fit in memory all at once.

A non-finite (NaN/Inf) loss on any micro-batch is skipped rather than applied (see the
SKIPPED log line) -- so a bad batch can't corrupt the whole run; if you see skips,
root-cause them (scripts/diagnose_nan.py) before trusting the final numbers, but the
run itself will still complete and any good steps still train normally. Separately,
`torch.amp.GradScaler` also silently skips an optimizer step if fp16 *gradients*
overflow (a different, unrelated failure mode from a non-finite forward pass) -- that
is standard, expected fp16-AMP behavior. It is disabled (a no-op) under bf16, which
doesn't need loss scaling in the first place.
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-examples", type=int, default=300, help="total examples pulled before the train/val split")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=20, help="number of OPTIMIZER UPDATES, not micro-batches")
    p.add_argument("--batch-size", type=int, default=2, help="micro-batch size -- what must fit in GPU memory")
    p.add_argument("--grad-accum-steps", type=int, default=1,
                   help="micro-batches accumulated per update; effective batch = batch-size * this")
    p.add_argument("--val-batch-size", type=int, default=4)
    p.add_argument("--max-state-len", type=int, default=192, help="up to NeoBERT's real limit of 4096")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lambda-expert", type=float, default=0.1)
    p.add_argument("--lambda-calib", type=float, default=0.0)
    p.add_argument("--lambda-balance", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision even on CUDA")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16",
                   help="bf16 (default) has fp32's dynamic range so pretrained-model activations "
                        "don't overflow the way they commonly do under fp16 -- confirmed via a real "
                        "run where every single batch produced a non-finite loss under fp16 on a T4. "
                        "fp16 needs GradScaler to avoid gradient underflow; bf16 does not, so the "
                        "scaler is a no-op (enabled=False) whenever this is bf16.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    data_dir = Path(__file__).parents[1] / "data"
    device = resolve_device(None)
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16  # bf16 doesn't need loss scaling
    print(f"using device: {device}  mixed_precision: {use_amp} ({args.amp_dtype if use_amp else 'n/a'})")
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    model.train()

    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"\n=== training for {args.n_steps} optimizer updates "
          f"(micro-batch {args.batch_size} x accum {args.grad_accum_steps} = effective batch {effective_batch}) ===")

    step, micro_step, n_skipped = 0, 0, 0
    optimizer.zero_grad()
    train_iter = iter(train_loader)

    while step < args.n_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)  # exhausted the data -- start another epoch
            batch = next(train_iter)

        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            out = model(batch)
            loss, reward = compute_moe_loss(
                out["blended_probs"], out["expert_probs"], out["gate_logits"], out["topk_idx"],
                batch["target"], batch["option_mask"], batch["qtype_idx"],
                lambda_expert=args.lambda_expert, lambda_calib=args.lambda_calib, lambda_balance=args.lambda_balance,
            )

        if not torch.isfinite(loss):
            n_skipped += 1
            bad_rows = (~torch.isfinite(out["blended_probs"])).any(-1)
            print(f"step {step:4d}.{micro_step}  SKIPPED: non-finite loss ({loss.item()})  "
                  f"bad_qtypes={batch['qtype_idx'][bad_rows].tolist()}  "
                  f"bad_domains={[d for d, bad in zip(batch['domain'], bad_rows.tolist()) if bad]}")
            micro_step += 1
            if micro_step >= args.grad_accum_steps:
                optimizer.zero_grad()
                micro_step, step = 0, step + 1
            continue

        scaler.scale(loss / args.grad_accum_steps).backward()
        micro_step += 1

        if micro_step >= args.grad_accum_steps:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            micro_step = 0
            if step % max(1, args.n_steps // 50) == 0 or step == args.n_steps - 1:
                print(f"step {step:4d}  loss {loss.item():.4f}  mean_reward {reward.mean().item():.4f}")
            step += 1

    if n_skipped:
        print(f"\n{n_skipped} micro-batch(es) skipped due to non-finite loss -- "
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

"""Diagnoses a NaN forward pass by checking each stage for NaN and, if found, dumping
enough about the offending batch to pinpoint the cause -- rather than guessing.

Run this on the same machine/environment where train_moe_eval.py produced NaN losses.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from decision_engine.model.encoder import DEFAULT_ENCODER_NAME, load_tokenizer
from decision_engine.model.moe_model import MoEModel
from decision_engine.training.dataset import Collator, TypedDecisionDataset
from decision_engine.training.device_utils import move_batch_to_device, resolve_device


def has_nan(t: torch.Tensor, name: str) -> bool:
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan or n_inf:
        print(f"  BAD VALUES in {name}  shape={tuple(t.shape)}  nan={n_nan}  inf={n_inf} / {t.numel()}")
        return True
    return False


def main():
    torch.manual_seed(0)
    device = resolve_device(None)
    print(f"using device: {device}")

    data_dir = Path(__file__).parents[1] / "data"
    tokenizer = load_tokenizer(DEFAULT_ENCODER_NAME)
    model = MoEModel(
        ["customer-support", "trust-safety", "email-communication", "sales-crm", "security-ops", "ecommerce"],
        DEFAULT_ENCODER_NAME,
    ).to(device)
    model.eval()

    dataset = TypedDecisionDataset(data_dir, max_examples=300)
    collate = Collator(tokenizer, max_state_len=192)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate)

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        # -- check the batch's own masks for a fully-masked row before running anything --
        state_mask = batch["state_attention_mask"]
        option_mask = batch["option_mask"]
        fully_masked_state_rows = (state_mask.sum(-1) == 0).nonzero(as_tuple=True)[0]
        fully_masked_option_rows = (option_mask.sum(-1) == 0).nonzero(as_tuple=True)[0]
        if len(fully_masked_state_rows) > 0:
            print(f"batch {batch_idx}: ROW(S) WITH ZERO REAL STATE TOKENS: {fully_masked_state_rows.tolist()}")
        if len(fully_masked_option_rows) > 0:
            print(f"batch {batch_idx}: ROW(S) WITH ZERO REAL OPTIONS: {fully_masked_option_rows.tolist()}")

        with torch.no_grad():
            state_hidden = model.encoder(batch["state_input_ids"], batch["state_attention_mask"])
            stage1_nan = has_nan(state_hidden, "state_hidden (encoder output)")

            from decision_engine.model.pooling import mean_pool
            pooled_state = mean_pool(state_hidden, batch["state_attention_mask"])
            stage2_nan = has_nan(pooled_state, "pooled_state")

            b, k, l_opt = batch["option_input_ids"].shape
            flat_ids = batch["option_input_ids"].view(b * k, l_opt)
            flat_mask = batch["option_attention_mask"].view(b * k, l_opt)
            flat_qtype = batch["qtype_idx"].unsqueeze(1).expand(b, k).reshape(b * k)
            option_queries = model.option_encoder(flat_ids, flat_mask, flat_qtype).view(b, k, -1)
            stage3_nan = has_nan(option_queries, "option_queries")

            out = model(batch)
            stage4a_nan = has_nan(out["gate_logits"], "gate_logits")
            stage4b_nan = has_nan(out["expert_probs"], "expert_probs (per-expert, pre-blend)")
            stage4c_nan = has_nan(out["blended_probs"], "blended_probs (final output)")

        stage4_nan = stage4a_nan or stage4b_nan or stage4c_nan
        if stage1_nan or stage2_nan or stage3_nan or stage4_nan:
            print(f"\n--- batch {batch_idx} is where bad values first appear ---")
            decoded_states = tokenizer.batch_decode(batch["state_input_ids"], skip_special_tokens=False)
            for i, s in enumerate(decoded_states):
                print(f"  row {i} state (decoded, incl. special/pad tokens): {s[:200]!r}")
            print("domains in this batch:", batch["domain"])
            print("qtype_idx (0=choice 1=score 2=noul):", batch["qtype_idx"].tolist())
            print("option_mask sums (real option count per row):", option_mask.sum(-1).tolist())
            print("state_mask sums (real token count per row):", state_mask.sum(-1).tolist())
            print("state_input_ids shape:", batch["state_input_ids"].shape)
            print("option_input_ids shape:", batch["option_input_ids"].shape)
            if not (stage1_nan or stage2_nan or stage3_nan):
                # bad values appear only once experts/gate combine -- report it per-row
                # so we can see whether it's specifically the `score`-type rows (qtype 1)
                per_row_bad = torch.isnan(out["blended_probs"]).any(-1) | torch.isinf(out["blended_probs"]).any(-1)
                print("per-row bad-value flag in blended_probs:", per_row_bad.tolist())
                print("-> compare against qtype_idx above: if bad rows line up with qtype==1 (score),")
                print("   the issue is specific to score-type routing/RPS-adjacent computation.")
            return  # stop at the first offending batch -- that's enough to diagnose

        if batch_idx >= 20:
            print("no NaN found in the first 20 batches")
            return


if __name__ == "__main__":
    main()

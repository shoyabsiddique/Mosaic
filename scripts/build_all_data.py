"""Regenerates every domain's data/<domain>/*.jsonl from scratch by running all six
real-data loaders and four synthetic generators. Needed on any fresh checkout (data/
is gitignored on purpose -- see .gitignore's comment -- since it's a reproducible
pull/generation, not source code), including cloud notebooks: this is the step the
Kaggle/Colab instructions in docs/cloud-training.md were missing, caught when
train_moe_eval.py failed with "train examples: 0" on a fresh Colab clone.

Needs internet access (hits CFPB, NVD, HF datasets-server, SpamAssassin, and the VCDB
GitHub mirror) -- same requirement the loaders have when run individually.

Each source is independent, so one failing (e.g. CFPB's API rate-limiting/blocking a
run after repeated hits in one session -- confirmed to happen, see
customer_support.py's RATE LIMITING note) shouldn't prevent the other five domains from
building. Failures are caught, reported, and the script continues; the final summary
says exactly which domains built and which didn't, so a partial run is still usable
and obviously not silently incomplete.
"""
from pathlib import Path

from decision_engine.data.loaders import (
    customer_support,
    ecommerce,
    email_communication,
    sales_crm,
    security_ops,
    trust_safety,
)
from decision_engine.data.synth import (
    customer_support_urgency_generator,
    ecommerce_return_generator,
    email_urgency_generator,
    sales_crm_generator,
)

DATA_DIR = Path(__file__).parents[1] / "data"

REAL_LOADERS = [
    (customer_support, DATA_DIR / "customer-support" / "cfpb_sample.jsonl"),
    (trust_safety, DATA_DIR / "trust-safety" / "civil_comments_sample.jsonl"),
    (email_communication, DATA_DIR / "email-communication" / "combined_sample.jsonl"),
    (sales_crm, DATA_DIR / "sales-crm" / "online_shoppers.jsonl"),
    (security_ops, DATA_DIR / "security-ops" / "combined_sample.jsonl"),
    (ecommerce, DATA_DIR / "ecommerce" / "amazon_food_sample.jsonl"),
]

SYNTHETIC_GENERATORS = [
    (customer_support_urgency_generator, DATA_DIR / "customer-support" / "synthetic_urgency.jsonl"),
    (email_urgency_generator, DATA_DIR / "email-communication" / "synthetic_urgency.jsonl"),
    (sales_crm_generator, DATA_DIR / "sales-crm" / "synthetic_leads.jsonl"),
    (ecommerce_return_generator, DATA_DIR / "ecommerce" / "synthetic_returns.jsonl"),
]


def _run_all(entries: list, label: str) -> tuple[int, list[str], list[str]]:
    total, succeeded, failed = 0, [], []
    for module, out_path in entries:
        name = module.__name__
        print(f"[{label}] {name} -> {out_path}")
        try:
            n = module.build_and_save(out_path)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: one bad source must not stop the rest
            print(f"            FAILED: {type(e).__name__}: {e}")
            failed.append(name)
            continue
        print(f"            wrote {n} records")
        total += n
        succeeded.append(name)
    return total, succeeded, failed


def main():
    real_total, real_ok, real_failed = _run_all(REAL_LOADERS, "real")
    synth_total, synth_ok, synth_failed = _run_all(SYNTHETIC_GENERATORS, "synthetic")

    print(f"\ntotal records written: {real_total + synth_total}")
    print(f"succeeded ({len(real_ok) + len(synth_ok)}): {real_ok + synth_ok}")
    if real_failed or synth_failed:
        print(f"FAILED ({len(real_failed) + len(synth_failed)}): {real_failed + synth_failed}")
        print("-> re-run this script later to retry just the failed ones (existing")
        print("   output files for succeeded domains are left as-is either way).")


if __name__ == "__main__":
    main()

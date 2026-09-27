"""Regenerates every domain's data/<domain>/*.jsonl from scratch by running all six
real-data loaders and four synthetic generators. Needed on any fresh checkout (data/
is gitignored on purpose -- see .gitignore's comment -- since it's a reproducible
pull/generation, not source code), including cloud notebooks: this is the step the
Kaggle/Colab instructions in docs/cloud-training.md were missing, caught when
train_moe_eval.py failed with "train examples: 0" on a fresh Colab clone.

Needs internet access (hits CFPB, NVD, HF datasets-server, SpamAssassin, and the VCDB
GitHub mirror) -- same requirement the loaders have when run individually.
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


def main():
    total = 0
    for module, out_path in REAL_LOADERS:
        print(f"[real]      {module.__name__} -> {out_path}")
        n = module.build_and_save(out_path)
        print(f"            wrote {n} records")
        total += n

    for module, out_path in SYNTHETIC_GENERATORS:
        print(f"[synthetic] {module.__name__} -> {out_path}")
        n = module.build_and_save(out_path)
        print(f"            wrote {n} records")
        total += n

    print(f"\ntotal records written: {total}")


if __name__ == "__main__":
    main()

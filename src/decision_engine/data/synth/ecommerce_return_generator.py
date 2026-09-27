"""Synthetic e-commerce return-reason data, closing the confirmed gap for this domain.

research/datasets-ecommerce.md and docs/candidate-datasets.md confirmed no public
dataset -- at any license -- has genuine customer-reported return-reason labels tied to
real transactions; every one found is itself synthetic. This generator is that
confirmed, designed fix, not a stand-in for real data that could have been found.

Unlike the sales-crm and urgency generators, `choice` (not `score`) is the natural
question type here -- return reasons are categorical, not ordinal (a "wrong size"
return isn't "more" or "less" than a "changed my mind" return). `noul` questions
(refund vs. exchange preference, product-fault vs. buyer's-remorse) come from the same
underlying generated evidence.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

DOMAIN = "ecommerce"
SOURCE_DATASET = "ecommerce_return_generator (synthetic)"
LICENSE = "synthetic (generated for training, not derived from real transactions)"

PRODUCT_CATEGORIES = ["clothing", "electronics", "home goods", "footwear", "accessories"]

# (reason, message_template, is_product_fault, refund_preference_probability)
RETURN_REASONS = [
    ("wrong_size", "The {item} didn't fit -- I need a different size.", False, 0.4),
    ("defective", "The {item} arrived broken / stopped working after a few days.", True, 0.85),
    ("not_as_described", "The {item} doesn't match the photos or description on the listing.", False, 0.7),
    ("changed_mind", "I just don't need the {item} anymore, no issue with the product itself.", False, 0.6),
    ("wrong_item_shipped", "I ordered a different {item} than what showed up in the box.", True, 0.9),
    ("late_arrival", "The {item} arrived too late, I already found a replacement elsewhere.", False, 0.75),
]

ITEMS_BY_CATEGORY = {
    "clothing": ["jacket", "sweater", "pair of jeans", "dress"],
    "electronics": ["headphones", "charging cable", "bluetooth speaker", "webcam"],
    "home goods": ["set of towels", "desk lamp", "throw pillow", "cutting board"],
    "footwear": ["pair of sneakers", "pair of boots", "sandals"],
    "accessories": ["watch", "backpack", "sunglasses", "wallet"],
}

CRITERIA = {
    "wrong_size": "the product didn't fit as expected",
    "defective": "the product arrived damaged or stopped working",
    "not_as_described": "the product doesn't match its listing",
    "changed_mind": "the customer no longer wants the product, no product issue",
    "wrong_item_shipped": "a different item than ordered was received",
    "late_arrival": "the product arrived after it was still needed",
}


def _build_one(rng: random.Random) -> tuple[dict, str, bool, float]:
    """Returns (state_dict, reason_label, is_product_fault, refund_preference_probability)."""
    category = rng.choice(PRODUCT_CATEGORIES)
    item = rng.choice(ITEMS_BY_CATEGORY[category])
    reason, template, is_fault, refund_prob = rng.choice(RETURN_REASONS)
    message = template.format(item=item)

    state = {
        "product_category": category,
        "message": message,
    }
    return state, reason, is_fault, refund_prob


def _reason_question() -> Question:
    return Question(type=QuestionType.CHOICE, instructions="Why is the customer returning this product?",
                     criteria=CRITERIA)


def _fault_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Is this return due to a product defect or error, not buyer preference?")


def _refund_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Does the customer prefer a refund over an exchange?")


def generate(n: int, seed: int = 33, split: str = "train") -> Iterator[Record]:
    rng = random.Random(seed)
    reason_q = _reason_question()
    fault_q = _fault_question()
    refund_q = _refund_question()
    labels = reason_q.option_labels()

    for i in range(n):
        state, reason, is_fault, refund_prob = _build_one(rng)
        return_id = f"synth-return-{i}"

        idx = labels.index(reason)
        dist = tuple(1.0 if j == idx else 0.0 for j in range(len(labels)))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{return_id}-reason",
            question=reason_q, target=TypedTarget(distribution=dist, label_index=idx),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        fault_val = 1.0 if is_fault else 0.0
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{return_id}-fault",
            question=fault_q, target=TypedTarget(distribution=(1.0 - fault_val, fault_val), label_index=int(fault_val)),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        yield Record(
            state=state, domain=DOMAIN, question_id=f"{return_id}-refund-pref",
            question=refund_q, target=TypedTarget(distribution=(1.0 - refund_prob, refund_prob)),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )


def _record_to_dict(r: Record) -> dict:
    return {
        "state": r.state, "domain": r.domain, "question_id": r.question_id,
        "question_type": r.question.type.value, "instructions": r.question.instructions,
        "criteria": r.question.criteria, "target_distribution": list(r.target.distribution),
        "target_label_index": r.target.label_index, "source_dataset": r.source_dataset,
        "license": r.license, "split": r.split,
    }


def build_and_save(out_path: Path, n: int = 2000, seed: int = 33) -> int:
    records = list(generate(n, seed=seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "ecommerce" / "synthetic_returns.jsonl")
    print(f"wrote {n} records")

"""Synthetic customer-support ticket urgency, closing the confirmed `score` gap for
this domain -- the CFPB complaint data (loaders/customer_support.py) has real `choice`
and `noul` fields but no ordinal urgency field at all; research/datasets-customer-support.md
confirmed the one dataset with a native priority field (the multilingual ticket set) is
CC-BY-NC-4.0, eval-only per docs/architecture-decisions.md, not usable for training.

Same soft-target technique as the other two synthetic generators: urgency is derived
from a continuous weighted score over tone, repeat-contact count, and SLA/deadline
mentions, then interpolated between adjacent levels so ambiguous tickets (frustrated
tone but no explicit deadline) land as genuine soft splits, not confident one-hot labels.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget
from decision_engine.data.soft_targets import interpolated_ordinal_target

DOMAIN = "customer-support"
SOURCE_DATASET = "customer_support_urgency_generator (synthetic)"
LICENSE = "synthetic (generated for training, not derived from real customer data)"

URGENCY_LEVELS = ["low", "medium", "high", "critical"]

CUSTOMER_TIERS = ["free", "standard", "enterprise"]
CHANNELS = ["email", "chat", "phone"]

# (tone, weight, phrase)
TONES = [
    ("neutral", 0.0, "The customer described the issue in a neutral, matter-of-fact tone."),
    ("frustrated", 1.5, "The customer expressed clear frustration with the ongoing issue."),
    ("cancel_threat", 3.0, "The customer said they will cancel their subscription if this isn't resolved."),
    ("already_escalated", 2.5, "The customer mentioned this has already been escalated once before."),
]

# (mentioned, weight, phrase)
SLA_MENTIONS = [
    (False, 0.0, ""),
    (True, 2.0, " They referenced a service-level agreement that has already been breached."),
]

# (deadline, weight, phrase)
DEADLINES = [
    ("none", 0.0, ""),
    ("today", 2.0, " They need this resolved by end of day today."),
    ("overdue", 2.5, " This request is already past its promised resolution date."),
]


def _build_one(rng: random.Random) -> tuple[dict, float, float]:
    """Returns (state_dict, urgency_raw_normalized, cancel_risk_probability)."""
    tier = rng.choice(CUSTOMER_TIERS)
    channel = rng.choice(CHANNELS)
    tone_label, tone_w, tone_phrase = rng.choice(TONES)
    sla_mentioned, sla_w, sla_phrase = rng.choice(SLA_MENTIONS)
    deadline_label, deadline_w, deadline_phrase = rng.choice(DEADLINES)
    repeat_contacts = rng.randint(0, 5)
    repeat_w = min(2.0, repeat_contacts * 0.5)

    tier_w = {"free": -0.5, "standard": 0.0, "enterprise": 1.0}[tier]

    raw = tone_w + sla_w + deadline_w + repeat_w + tier_w
    # weights span roughly [-0.5, 11.0]; normalize onto [0, 1]
    normalized = max(0.0, min(1.0, (raw + 0.5) / 11.5))

    message = tone_phrase + sla_phrase + deadline_phrase
    cancel_prob = {"neutral": 0.05, "frustrated": 0.25, "cancel_threat": 0.9, "already_escalated": 0.4}[tone_label]

    state = {
        "customer_tier": tier,
        "channel": channel,
        "repeat_contacts": repeat_contacts,
        "message": message,
    }
    return state, normalized, cancel_prob


def _urgency_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="How urgent is this support ticket?",
                     criteria=URGENCY_LEVELS)


def _churn_risk_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Is this customer at risk of churning?")


def generate(n: int, seed: int = 21, split: str = "train") -> Iterator[Record]:
    rng = random.Random(seed)
    urgency_q = _urgency_question()
    churn_q = _churn_risk_question()

    for i in range(n):
        state, normalized_score, cancel_prob = _build_one(rng)
        ticket_id = f"synth-ticket-{i}"

        urgency_dist = interpolated_ordinal_target(normalized_score, len(URGENCY_LEVELS))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{ticket_id}-urgency",
            question=urgency_q, target=TypedTarget(distribution=urgency_dist),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        yield Record(
            state=state, domain=DOMAIN, question_id=f"{ticket_id}-churn",
            question=churn_q, target=TypedTarget(distribution=(1.0 - cancel_prob, cancel_prob)),
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


def build_and_save(out_path: Path, n: int = 2000, seed: int = 21) -> int:
    records = list(generate(n, seed=seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "customer-support" / "synthetic_urgency.jsonl")
    print(f"wrote {n} records")

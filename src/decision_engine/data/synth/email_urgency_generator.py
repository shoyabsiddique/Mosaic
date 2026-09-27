"""Synthetic email/message urgency, closing the confirmed `score` gap for this domain.

research/datasets-email-communication.md found no real, commercially-licensed dataset
with an ordinal urgency field (the one that has it, the multilingual ticket dataset, is
CC-BY-NC-4.0 -- eval-only, not usable for training per docs/architecture-decisions.md).
This generator is the designed fix, not a stand-in for real data -- every record is
tagged `license: "synthetic"` and a distinct `source_dataset`.

Same soft-target technique as sales_crm_generator.py and the real trust-safety loader:
urgency is derived from a continuous weighted score and interpolated between adjacent
levels, so ambiguous combinations (e.g. an "ASAP" subject with otherwise mild body text)
produce genuine uncertainty rather than an arbitrarily confident label.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget
from decision_engine.data.soft_targets import interpolated_ordinal_target

DOMAIN = "email-communication"
SOURCE_DATASET = "email_urgency_generator (synthetic)"
LICENSE = "synthetic (generated for training, not derived from real correspondence)"

URGENCY_LEVELS = ["not urgent", "somewhat urgent", "urgent", "critical"]

# (relationship, weight)
SENDERS = [
    ("an unknown sender", -0.5),
    ("a known colleague", 0.5),
    ("your direct manager", 1.5),
    ("an external client", 1.0),
]

# (marker, weight, subject_text)
SUBJECT_MARKERS = [
    ("none", 0.0, "Quick question"),
    ("follow_up", 0.5, "Following up on my last email"),
    ("asap", 2.0, "ASAP: need this today"),
    ("urgent", 3.0, "URGENT - action required"),
]

# (deadline, weight, phrasing)
DEADLINES = [
    ("none", 0.0, "no particular deadline"),
    ("end_of_week", 1.0, "needs a response by end of week"),
    ("today", 2.5, "needs a response today"),
    ("overdue", 3.0, "this is already overdue"),
]

# (tone, weight, phrasing)
TONES = [
    ("casual", -0.5, "Just checking in when you get a chance."),
    ("polite_request", 0.5, "Could you take a look when you have a moment?"),
    ("frustrated", 1.5, "This is the third time I'm reaching out about this."),
    ("consequence", 2.5, "If this isn't resolved soon, we'll need to escalate."),
]


def _build_one(rng: random.Random) -> tuple[str, float, str]:
    """Returns (email_text, urgency_raw_normalized, sender_relationship)."""
    sender, sender_w = rng.choice(SENDERS)
    marker, marker_w, subject_text = rng.choice(SUBJECT_MARKERS)
    deadline, deadline_w, deadline_phrase = rng.choice(DEADLINES)
    tone, tone_w, tone_text = rng.choice(TONES)
    exclamations = rng.randint(0, 3)
    exclaim_w = exclamations * 0.3

    raw = sender_w + marker_w + deadline_w + tone_w + exclaim_w
    # weights span roughly [-1, 11.4]; normalize onto [0, 1]
    normalized = max(0.0, min(1.0, (raw + 1.0) / 12.4))

    body = f"{tone_text} Regarding the deadline, {deadline_phrase}." + ("!" * exclamations)
    email_text = f"From: {sender}\nSubject: {subject_text}\n\n{body}"
    return email_text, normalized, sender


def _urgency_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="How urgent is this message?",
                     criteria=URGENCY_LEVELS)


def _external_sender_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Is this message from someone outside the team?")


def generate(n: int, seed: int = 7, split: str = "train") -> Iterator[Record]:
    rng = random.Random(seed)
    urgency_q = _urgency_question()
    external_q = _external_sender_question()

    for i in range(n):
        text, normalized_score, sender = _build_one(rng)
        msg_id = f"synth-email-{i}"

        urgency_dist = interpolated_ordinal_target(normalized_score, len(URGENCY_LEVELS))
        yield Record(
            state=text, domain=DOMAIN, question_id=f"{msg_id}-urgency",
            question=urgency_q, target=TypedTarget(distribution=urgency_dist),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        is_external = 1.0 if sender in ("an unknown sender", "an external client") else 0.0
        yield Record(
            state=text, domain=DOMAIN, question_id=f"{msg_id}-external",
            question=external_q, target=TypedTarget(distribution=(1.0 - is_external, is_external),
                                                       label_index=int(is_external)),
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


def build_and_save(out_path: Path, n: int = 2000, seed: int = 7) -> int:
    records = list(generate(n, seed=seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "email-communication" / "synthetic_urgency.jsonl")
    print(f"wrote {n} records")

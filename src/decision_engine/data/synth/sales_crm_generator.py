"""Synthetic sales/CRM lead states, closing the confirmed `score` gap for this domain.

research/datasets-sales-crm.md confirmed no public dataset -- at any license -- combines
free-text CRM notes with calibrated priority labels: real tabular lead data exists
(loaders/sales_crm.py covers `choice`/`noul` from it), but text-grounded, gradable
lead-priority data does not exist publicly under a commercial-usable license. This
generator is the designed fix for that gap, not a stand-in for real data -- every
record is tagged `license: "synthetic"` and a distinct `source_dataset`, so it can
never be mistaken for real signal downstream.

Follows the generation spec written during Phase 0 research: mixed structured + text
state, genuinely graded evidence (not just a label slapped on), and deliberately
ambiguous/conflicting cases so the `lead_priority` target isn't trivially separable --
soft targets reflect real uncertainty in the underlying evidence, they aren't just
label noise.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget
from decision_engine.data.soft_targets import interpolated_ordinal_target

DOMAIN = "sales-crm"
SOURCE_DATASET = "sales_crm_generator (synthetic)"
LICENSE = "synthetic (generated for training, not derived from real customer data)"

PRIORITY_LEVELS = ["cold", "warm", "hot", "urgent"]

COMPANY_SIZES = ["startup (under 50 employees)", "mid-market (50-500 employees)", "enterprise (500+ employees)"]
INDUSTRIES = ["SaaS", "healthcare", "financial services", "retail", "manufacturing", "education"]

# (title, seniority_weight)
TITLES = [
    ("individual contributor", 0.0),
    ("manager", 1.0),
    ("director", 2.0),
    ("VP", 3.0),
    ("C-suite executive", 4.0),
]

# (status, weight, phrasing)
BUDGET_STATUSES = [
    ("confirmed", 3.0, "budget has been confirmed for this initiative"),
    ("likely", 1.0, "budget looks likely but hasn't been formally approved"),
    ("unconfirmed", -0.5, "budget status hasn't come up yet"),
    ("no_budget", -3.0, "the contact explicitly said there's no budget this quarter"),
]

# (label, weight, phrasing)
RESPONSE_SPEEDS = [
    ("fast", 2.0, "responded within a few hours of the last outreach"),
    ("normal", 0.5, "responded within a day or two"),
    ("slow", -1.0, "took over a week to respond"),
    ("unresponsive", -2.5, "hasn't responded to the last two follow-ups"),
]

# (label, weight, phrasing)
TIMELINES = [
    ("immediate", 3.0, "wants to move forward immediately"),
    ("this_quarter", 1.5, "needs a solution by the end of this quarter"),
    ("next_year", -1.0, "is exploring options for next year, no urgency"),
    ("none", 0.0, "hasn't mentioned a specific timeline"),
]


def _build_one(rng: random.Random, idx: int) -> tuple[dict, float, float]:
    """Returns (state_dict, priority_raw_score in [0,1], budget_confirmed_probability)."""
    company_size = rng.choice(COMPANY_SIZES)
    industry = rng.choice(INDUSTRIES)
    title, seniority_w = rng.choice(TITLES)
    budget_status, budget_w, budget_phrase = rng.choice(BUDGET_STATUSES)
    response_label, response_w, response_phrase = rng.choice(RESPONSE_SPEEDS)
    timeline_label, timeline_w, timeline_phrase = rng.choice(TIMELINES)
    competitor_mentioned = rng.random() < 0.3
    num_touchpoints = rng.randint(1, 8)
    days_since_first_contact = rng.randint(1, 60)

    touchpoint_w = min(2.0, num_touchpoints / 4.0)
    recency_w = max(-1.0, 1.0 - days_since_first_contact / 30.0)
    competitor_w = -1.5 if competitor_mentioned else 0.0

    raw = seniority_w + budget_w + response_w + timeline_w + touchpoint_w + recency_w + competitor_w
    # weights span roughly [-9, 15.5]; normalize onto [0, 1] for the interpolation helper
    normalized = max(0.0, min(1.0, (raw + 9.0) / 24.5))

    notes = (
        f"Contact is a {title} at a company in the {industry} industry ({company_size}). "
        f"Regarding budget, {budget_phrase}. On timeline, the contact {timeline_phrase}. "
        f"They {response_phrase}. "
        + ("A competing vendor is also being evaluated. " if competitor_mentioned else "")
        + f"{num_touchpoints} touchpoints so far over {days_since_first_contact} days since first contact."
    )

    budget_prob = {"confirmed": 0.95, "likely": 0.7, "unconfirmed": 0.35, "no_budget": 0.05}[budget_status]

    state = {
        "company_size": company_size,
        "industry": industry,
        "contact_title": title,
        "num_touchpoints": num_touchpoints,
        "days_since_first_contact": days_since_first_contact,
        "notes": notes,
    }
    return state, normalized, budget_prob


def _priority_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="How should this lead be prioritized?",
                     criteria=PRIORITY_LEVELS)


def _budget_question() -> Question:
    return Question(type=QuestionType.NOUL,
                     instructions="Does the buyer have confirmed budget for this deal?")


def generate(n: int, seed: int = 42, split: str = "train") -> Iterator[Record]:
    rng = random.Random(seed)
    priority_q = _priority_question()
    budget_q = _budget_question()

    for i in range(n):
        state, normalized_score, budget_prob = _build_one(rng, i)
        lead_id = f"synth-lead-{i}"

        priority_dist = interpolated_ordinal_target(normalized_score, len(PRIORITY_LEVELS))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{lead_id}-priority",
            question=priority_q, target=TypedTarget(distribution=priority_dist),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        yield Record(
            state=state, domain=DOMAIN, question_id=f"{lead_id}-budget",
            question=budget_q, target=TypedTarget(distribution=(1.0 - budget_prob, budget_prob)),
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


def build_and_save(out_path: Path, n: int = 2000, seed: int = 42) -> int:
    records = list(generate(n, seed=seed))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "sales-crm" / "synthetic_leads.jsonl")
    print(f"wrote {n} records")

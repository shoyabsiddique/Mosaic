"""Trust & safety / content moderation loader: Jigsaw Civil Comments (CC0).

Chosen because it's the only Tier-1 dataset with genuinely continuous, calibrated
severity labels (not just hard majority-vote classes) -- see
research/datasets-trust-safety.md. Fetched via Hugging Face's public datasets-server
REST API, which needs no auth and no `datasets` library install for a modest sample.

Two question types covered:
  - `score` (severity): a soft ordinal target built by linear interpolation between
    adjacent severity levels from the continuous toxicity score -- demonstrates the
    schema's soft-target path on a real (not synthetic) calibrated label.
  - `noul`: `is_threat` / `is_insult`, directly from the dataset's own continuous
    per-attribute scores (already in [0, 1], no derivation needed).

NOT covered here: `choice`. Civil Comments has no natural category label (it's a
severity-scoring dataset, not a classification one, per research findings) -- category
classification for this domain needs HateXplain or Davidson's 3-class labels instead,
not forced here from a weak derived signal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import requests

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget
from decision_engine.data.soft_targets import interpolated_ordinal_target

ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET_ID = "google/civil_comments"
DOMAIN = "trust-safety"
SOURCE_DATASET = "Jigsaw Unintended Bias in Toxicity Classification (Civil Comments)"
LICENSE = "CC0-1.0 (public domain)"

SEVERITY_LEVELS = ["not toxic", "mildly toxic", "toxic", "severely toxic"]


def fetch_raw(n_rows: int = 1000, page_size: int = 100) -> list[dict]:
    """Page through the HF datasets-server rows API. No API key required."""
    rows: list[dict] = []
    offset = 0
    while len(rows) < n_rows:
        resp = requests.get(ROWS_API, params={
            "dataset": DATASET_ID, "config": "default", "split": "train",
            "offset": offset, "length": min(page_size, n_rows - len(rows)),
        }, timeout=30)
        resp.raise_for_status()
        batch = [r["row"] for r in resp.json()["rows"]]
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
    return rows


def _severity_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="How severe is this comment?",
                     criteria=SEVERITY_LEVELS)


def _noul_question(attribute: str, phrasing: str) -> Question:
    return Question(type=QuestionType.NOUL, instructions=phrasing)


def to_records(raw_rows: list[dict], split: str = "train") -> Iterator[Record]:
    severity_q = _severity_question()
    threat_q = _noul_question("threat", "Does this comment contain or threaten violence?")
    insult_q = _noul_question("insult", "Is this comment insulting toward a person or group?")

    for i, row in enumerate(raw_rows):
        text = row.get("text")
        toxicity = row.get("toxicity")
        threat = row.get("threat")
        insult = row.get("insult")
        if text is None or toxicity is None:
            continue

        row_id = f"civilcomments-{i}"

        severity_dist = interpolated_ordinal_target(float(toxicity), len(SEVERITY_LEVELS))
        yield Record(
            state=text, domain=DOMAIN, question_id=f"{row_id}-severity",
            question=severity_q, target=TypedTarget(distribution=severity_dist),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        if threat is not None:
            t = max(0.0, min(1.0, float(threat)))
            yield Record(
                state=text, domain=DOMAIN, question_id=f"{row_id}-threat",
                question=threat_q, target=TypedTarget(distribution=(1.0 - t, t)),
                source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
            )

        if insult is not None:
            ins = max(0.0, min(1.0, float(insult)))
            yield Record(
                state=text, domain=DOMAIN, question_id=f"{row_id}-insult",
                question=insult_q, target=TypedTarget(distribution=(1.0 - ins, ins)),
                source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
            )


def load(n_rows: int = 1000, split: str = "train") -> list[Record]:
    return list(to_records(fetch_raw(n_rows), split=split))


def build_and_save(out_path: Path, n_rows: int = 1000) -> int:
    records = load(n_rows=n_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "state": r.state, "domain": r.domain, "question_id": r.question_id,
                "question_type": r.question.type.value, "instructions": r.question.instructions,
                "criteria": r.question.criteria, "target_distribution": list(r.target.distribution),
                "target_label_index": r.target.label_index, "source_dataset": r.source_dataset,
                "license": r.license, "split": r.split,
            }) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "trust-safety" / "civil_comments_sample.jsonl")
    print(f"wrote {n} records")

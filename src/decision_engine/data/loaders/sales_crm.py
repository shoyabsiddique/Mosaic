"""Sales/CRM loader: UCI Online Shoppers Purchasing Intention (CC BY 4.0).

Chosen because it's a real, directly-downloadable, cleanly-licensed dataset with two
genuinely distinct real labels -- see research/datasets-sales-crm.md, which confirmed
tabular lead/deal data is NOT thin (unlike text-grounded sales conversations, which are
thin and non-commercially-licensed everywhere they exist).

State here is a JSON dict of session/behavioral features, not free text -- this is a
deliberate exercise of the schema's dict-state path (Jev/Laya both support structured
JSON state, not just text), distinct from the mostly-text other loaders.

Two question types covered, both from real (not derived/synthetic) fields:
  - `noul`: will this session end in a purchase (from `Revenue`).
  - `choice`: what type of visitor is this (from `VisitorType`).

NOT covered: `score`. This dataset has no ordinal field (lead-quality tier, engagement
score) -- per research findings, that gap is real for this domain and is the confirmed
candidate for synthetic generation, not forced here from a proxy like PageValues, which
was never labeled by a human as a graded signal.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import requests

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

ZIP_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
CSV_NAME = "online_shoppers_intention.csv"
DOMAIN = "sales-crm"
SOURCE_DATASET = "UCI Online Shoppers Purchasing Intention Dataset"
LICENSE = "CC-BY-4.0"

VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]
STATE_FIELDS = (
    "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues",
    "SpecialDay", "Month", "OperatingSystems", "Browser", "Region", "TrafficType", "Weekend",
)


def fetch_raw() -> list[dict]:
    """Download and parse the session CSV. No API key required."""
    resp = requests.get(ZIP_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(CSV_NAME).decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(raw)))


def _revenue_question() -> Question:
    return Question(type=QuestionType.NOUL,
                     instructions="Will this website visitor complete a purchase in this session?")


def _visitor_type_question() -> Question:
    return Question(type=QuestionType.CHOICE, instructions="What type of visitor is this?",
                     criteria={
                         "Returning_Visitor": "has visited this site before",
                         "New_Visitor": "first-time visitor",
                         "Other": "visitor type not otherwise classified",
                     })


def to_records(raw_rows: list[dict], split: str = "train") -> Iterator[Record]:
    revenue_q = _revenue_question()
    visitor_q = _visitor_type_question()
    labels = visitor_q.option_labels()

    for i, row in enumerate(raw_rows):
        visitor_type = row.get("VisitorType")
        revenue_raw = row.get("Revenue")
        if visitor_type not in VISITOR_TYPES or revenue_raw not in ("TRUE", "FALSE"):
            continue

        state = {k: row.get(k) for k in STATE_FIELDS}
        session_id = f"session-{i}"

        is_purchase = 1.0 if revenue_raw == "TRUE" else 0.0
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{session_id}-revenue",
            question=revenue_q, target=TypedTarget(distribution=(1.0 - is_purchase, is_purchase),
                                                     label_index=int(is_purchase)),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        idx = labels.index(visitor_type)
        dist = tuple(1.0 if j == idx else 0.0 for j in range(len(labels)))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{session_id}-visitor-type",
            question=visitor_q, target=TypedTarget(distribution=dist, label_index=idx),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )


def load(split: str = "train") -> list[Record]:
    return list(to_records(fetch_raw(), split=split))


def build_and_save(out_path: Path) -> int:
    records = load()
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
    n = build_and_save(Path(__file__).parents[4] / "data" / "sales-crm" / "online_shoppers.jsonl")
    print(f"wrote {n} records")

"""Customer support triage loader: CFPB Consumer Complaint Database.

Chosen as the first loader specifically because it's the commercially-clearest dataset
found in Phase 0 (US government public domain data, no license ambiguity) and because
it's reachable over a public, no-auth API -- see research/datasets-customer-support.md
for the full survey and why this beat the alternatives on license grounds.

Two questions per complaint:
  - `product` (choice): which financial product the complaint concerns.
  - `timely_response` (noul): did the company respond on time.

`score` (urgency) is covered separately by
`decision_engine.data.synth.customer_support_urgency_generator`, a synthetic generator
-- this dataset genuinely has no ordinal field, confirmed in research, not a gap left
unaddressed.

SAMPLING: the CFPB API's default sort returns the most recently *filed* complaints,
which is heavily skewed by product (a naive 1,000-row pull was 86% "Credit reporting").
`fetch_raw` instead queries each product category separately via the API's own
`product` filter, so every category is represented regardless of its true prevalence.

The `timely` field's imbalance is different in kind and is NOT corrected the same way:
checked directly against the full population (not just a sample) via the API's
`timely=No`/`timely=Yes` filters, the true split is approximately 0.6% No / 99.4% Yes
(109,886 vs. 17,930,879 complaints) -- a genuine population characteristic, not a
sampling artifact, so it would be wrong to pretend otherwise by force-balancing it to
50/50. Instead, `fetch_raw` deliberately oversamples `timely=No` for training (there
just aren't enough examples otherwise to learn from) while this docstring records the
true prior for anyone calibrating against it later.

RATE LIMITING: `fetch_raw` makes 14 sequential requests to this endpoint. During
development this triggered a `403 Forbidden` from CFPB's side after repeated runs in
one session (confirmed: it started failing even from the same machine that had pulled
this data successfully minutes earlier, so it's rate-limiting/abuse detection, not a
one-off). A small delay between requests is a courtesy that reduces the odds of
tripping it again -- see `sleep_s` below, same pattern security_ops.py already uses
for the NVD API. If you hit a 403 here, it is very likely temporary: wait a while
before retrying rather than hammering it again immediately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from decision_engine.data.http_utils import get_with_retry
from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
DOMAIN = "customer-support"
SOURCE_DATASET = "CFPB Consumer Complaint Database"
LICENSE = "public-domain (US government work, no commercial-use restriction)"

# Observed in a live sample; CFPB's full taxonomy has a couple more legacy/retired
# categories not seen here (e.g. "Consumer Loan" appears once above as a legacy label
# alongside "Vehicle loan or lease"). Short descriptions are standard, well-known
# definitions of these public financial-product categories, not sourced from the API.
PRODUCT_CRITERIA: dict[str, str] = {
    "Credit reporting or other personal consumer reports": "credit bureau reports, report accuracy, disputes",
    "Debt collection": "third-party or original-creditor debt collection practices",
    "Credit card": "credit card accounts, billing, rewards, fees",
    "Checking or savings account": "checking or savings account issues",
    "Mortgage": "home loans, mortgage servicing, foreclosure",
    "Vehicle loan or lease": "auto loans and leases",
    "Bank account or service": "general banking services outside checking/savings",
    "Payday loan, title loan, personal loan, or advance loan": "short-term or personal loans",
    "Money transfer, virtual currency, or money service": "money transfers, virtual currency, money services",
    "Credit reporting": "legacy credit-reporting category (pre-2022 taxonomy)",
    "Student loan": "federal or private student loans",
    "Consumer Loan": "legacy general consumer loan category",
    "Money transfers": "legacy money-transfer category (pre-2022 taxonomy)",
}

STATE_FIELDS = ("issue", "sub_issue", "company_response", "submitted_via", "company", "state", "tags")


def _fetch_page(params: dict) -> list[dict]:
    """The live API is occasionally slow enough to exceed a short timeout on some
    product queries -- fetch_raw makes 14 sequential calls, so one transient timeout
    shouldn't fail the whole stratified pull (see data/http_utils.py)."""
    resp = get_with_retry(API_URL, params=params, headers={"Accept": "application/json"})
    return [hit["_source"] for hit in resp.json()["hits"]["hits"]]


def fetch_raw(per_product: int = 150, n_timely_no: int = 150, sleep_s: float = 1.5) -> list[dict]:
    """Pull a stratified sample from the live CFPB API. No API key required.

    Queries each product category separately (`per_product` complaints each) instead
    of relying on the API's recency-skewed default sort, then tops up with
    `n_timely_no` explicitly `timely=No` complaints -- a real but rare (~0.6% of the
    population, verified) class that a plain stratified-by-product pull would otherwise
    barely include. `sleep_s` paces the 14 sequential requests -- see this module's
    RATE LIMITING note.
    """
    complaints: dict[str, dict] = {}
    products = list(PRODUCT_CRITERIA)

    for i, product in enumerate(products):
        for c in _fetch_page({"size": per_product, "product": product}):
            complaints[c["complaint_id"]] = c
        if i < len(products) - 1:
            time.sleep(sleep_s)

    time.sleep(sleep_s)
    for c in _fetch_page({"size": n_timely_no, "timely": "No"}):
        complaints[c["complaint_id"]] = c

    return list(complaints.values())


def _product_question() -> Question:
    return Question(
        type=QuestionType.CHOICE,
        instructions="Which financial product is this complaint about?",
        criteria=PRODUCT_CRITERIA,
    )


def _timely_question() -> Question:
    return Question(
        type=QuestionType.NOUL,
        instructions="Did the company respond to this complaint in a timely manner?",
    )


def to_records(raw_complaints: list[dict], split: str = "train") -> Iterator[Record]:
    product_q = _product_question()
    timely_q = _timely_question()
    product_labels = product_q.option_labels()

    for c in raw_complaints:
        product = c.get("product")
        if product not in PRODUCT_CRITERIA:
            continue  # unseen/legacy category not in our fixed criteria set -- skip rather than guess
        timely = c.get("timely")
        if timely not in ("Yes", "No"):
            continue  # missing/unknown -- skip rather than fabricate a target

        state = {k: c.get(k) for k in STATE_FIELDS if c.get(k) is not None}
        complaint_id = c.get("complaint_id", "unknown")

        product_idx = product_labels.index(product)
        product_dist = tuple(1.0 if i == product_idx else 0.0 for i in range(len(product_labels)))
        yield Record(
            state=state,
            domain=DOMAIN,
            question_id=f"{complaint_id}-product",
            question=product_q,
            target=TypedTarget(distribution=product_dist, label_index=product_idx),
            source_dataset=SOURCE_DATASET,
            license=LICENSE,
            split=split,
        )

        timely_idx = 1 if timely == "Yes" else 0
        timely_dist = (1.0 - timely_idx, float(timely_idx))
        yield Record(
            state=state,
            domain=DOMAIN,
            question_id=f"{complaint_id}-timely",
            question=timely_q,
            target=TypedTarget(distribution=timely_dist, label_index=timely_idx),
            source_dataset=SOURCE_DATASET,
            license=LICENSE,
            split=split,
        )


def load(per_product: int = 150, n_timely_no: int = 150, split: str = "train") -> list[Record]:
    return list(to_records(fetch_raw(per_product=per_product, n_timely_no=n_timely_no), split=split))


def build_and_save(out_path: Path, per_product: int = 150, n_timely_no: int = 150) -> int:
    """Fetch, convert, and write as JSONL. Returns the number of records written."""
    records = load(per_product=per_product, n_timely_no=n_timely_no)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "state": r.state,
                "domain": r.domain,
                "question_id": r.question_id,
                "question_type": r.question.type.value,
                "instructions": r.question.instructions,
                "criteria": r.question.criteria,
                "target_distribution": list(r.target.distribution),
                "target_label_index": r.target.label_index,
                "source_dataset": r.source_dataset,
                "license": r.license,
                "split": r.split,
            }) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "customer-support" / "cfpb_sample.jsonl")
    print(f"wrote {n} records")

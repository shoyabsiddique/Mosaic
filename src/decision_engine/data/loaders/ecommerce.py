"""E-commerce loader: Amazon Fine Food Reviews (SNAP/Stanford, CC0).

Chosen because it's a real, directly-downloadable, CC0-licensed dataset -- see
research/datasets-ecommerce.md, which explicitly ruled out Yelp (commercial use
banned by its own terms) and flagged the Amazon Reviews 2018/2023 academic releases as
license-ambiguous. The full file is ~122MB gzipped; this loader fetches only a byte
range (not the whole file) since a proof-of-concept sample doesn't need the full
corpus, and parses whatever complete records fall inside that range.

Two question types covered, both from real fields:
  - `score`: star rating (1-5), hard target.
  - `noul`: was the review found helpful, as a genuinely soft/calibrated target from
    the helpfulness vote ratio (num/denom) -- not derived or synthetic, a real
    calibrated signal the same way Civil Comments' scores are.

NOT covered: `choice`. This dataset is food-only (single vertical, per research
findings) -- product categorization needs the Flipkart dataset or similar, not forced
here from a field this dataset doesn't have.
"""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Iterator, Optional

from decision_engine.data.http_utils import get_with_retry
from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

GZ_URL = "https://snap.stanford.edu/data/finefoods.txt.gz"
DOMAIN = "ecommerce"
SOURCE_DATASET = "Amazon Fine Food Reviews (SNAP/Stanford)"
LICENSE = "CC0 (public domain, per Kaggle mirror listing)"

RATING_LEVELS = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
FIELD_RE = re.compile(r"^(product/productId|review/helpfulness|review/score|review/summary|review/text): (.*)$")


def fetch_raw(byte_range: int = 4_000_000) -> list[dict]:
    """Fetch a byte-range prefix of the gzip file (not the whole 122MB) and parse
    whatever complete review blocks fall inside it. No API key required."""
    resp = get_with_retry(GZ_URL, headers={"Range": f"bytes=0-{byte_range}"}, timeout=60)
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    text = decompressor.decompress(resp.content).decode("utf-8", errors="replace")

    reviews: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                reviews.append(current)
                current = {}
            continue
        m = FIELD_RE.match(line)
        if m:
            current[m.group(1)] = m.group(2)
    # `current` only accumulates fields for a block still awaiting its terminating blank
    # line; a block is appended to `reviews` above only once that blank line arrives, so
    # a block truncated mid-stream by the byte range is correctly left out of `reviews`
    # without any special-casing here.
    return reviews


def _rating_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="What star rating did the reviewer give?",
                     criteria=RATING_LEVELS)


def _helpful_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Did other users find this review helpful?")


def _parse_helpfulness(raw: Optional[str]) -> Optional[tuple[int, int]]:
    if not raw or "/" not in raw:
        return None
    num_s, _, den_s = raw.partition("/")
    try:
        num, den = int(num_s), int(den_s)
    except ValueError:
        return None
    return (num, den) if den > 0 else None


def to_records(raw_reviews: list[dict], split: str = "train") -> Iterator[Record]:
    rating_q = _rating_question()
    helpful_q = _helpful_question()

    for i, review in enumerate(raw_reviews):
        text = review.get("review/text")
        summary = review.get("review/summary", "")
        score_raw = review.get("review/score")
        if not text or not score_raw:
            continue

        state = f"{summary}\n\n{text}" if summary else text
        review_id = f"amazon-food-{i}"

        try:
            score = float(score_raw)
        except ValueError:
            continue
        idx = max(0, min(4, int(round(score)) - 1))
        dist = tuple(1.0 if j == idx else 0.0 for j in range(5))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"{review_id}-rating",
            question=rating_q, target=TypedTarget(distribution=dist, label_index=idx),
            source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
        )

        helpfulness = _parse_helpfulness(review.get("review/helpfulness"))
        if helpfulness:
            num, den = helpfulness
            p = num / den
            yield Record(
                state=state, domain=DOMAIN, question_id=f"{review_id}-helpful",
                question=helpful_q, target=TypedTarget(distribution=(1.0 - p, p)),
                source_dataset=SOURCE_DATASET, license=LICENSE, split=split,
            )


def load(byte_range: int = 4_000_000, split: str = "train") -> list[Record]:
    return list(to_records(fetch_raw(byte_range), split=split))


def build_and_save(out_path: Path, byte_range: int = 4_000_000) -> int:
    records = load(byte_range=byte_range)
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
    n = build_and_save(Path(__file__).parents[4] / "data" / "ecommerce" / "amazon_food_sample.jsonl")
    print(f"wrote {n} records")

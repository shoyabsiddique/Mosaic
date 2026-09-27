"""Email/communication loader: two real sources, combined.

1. UCI SMS Spam Collection (CC BY 4.0) -- `noul` ("is this spam").
2. Apache SpamAssassin Public Corpus (CC0 / public domain) -- `choice`, using the
   corpus's OWN native 3-way structure (spam / easy_ham / hard_ham), not an invented
   category. `hard_ham` is legitimate mail the corpus's own curators specifically chose
   because it resembles spam -- exposing that as its own category (rather than
   collapsing it into "legitimate") is a real, useful triage distinction, not a
   made-up one: it directly represents "genuine mail a naive spam filter would
   false-positive on," which is exactly a business-relevant category, not a research
   artifact.

`score` is still not covered by either source -- neither has a native ordinal field.
See `decision_engine.data.synth.email_urgency_generator` for how that gap is closed
(synthetic, and explicitly labeled as such -- not presented as real data).
"""
from __future__ import annotations

import email as email_parser
import io
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import requests

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

DOMAIN = "email-communication"

# -- Source 1: UCI SMS Spam Collection (noul) --------------------------------------
SMS_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_SOURCE_DATASET = "UCI SMS Spam Collection"
SMS_LICENSE = "CC-BY-4.0"


def fetch_raw_sms() -> list[tuple[str, str]]:
    """Download and extract (label, text) pairs. No API key required."""
    resp = requests.get(SMS_ZIP_URL, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read("SMSSpamCollection").decode("utf-8", errors="replace")
    pairs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        label, _, text = line.partition("\t")
        if label in ("ham", "spam") and text:
            pairs.append((label, text))
    return pairs


def _spam_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Is this message unsolicited spam?")


def to_records_sms(raw_pairs: list[tuple[str, str]], split: str = "train") -> Iterator[Record]:
    spam_q = _spam_question()
    for i, (label, text) in enumerate(raw_pairs):
        is_spam = 1.0 if label == "spam" else 0.0
        yield Record(
            state=text, domain=DOMAIN, question_id=f"sms-{i}-spam",
            question=spam_q, target=TypedTarget(distribution=(1.0 - is_spam, is_spam), label_index=int(is_spam)),
            source_dataset=SMS_SOURCE_DATASET, license=SMS_LICENSE, split=split,
        )


# -- Source 2: SpamAssassin Public Corpus (choice) -----------------------------------
SPAMASSASSIN_ARCHIVES = {
    "spam": "20021010_spam.tar.bz2",
    "easy_ham": "20021010_easy_ham.tar.bz2",
    "hard_ham": "20021010_hard_ham.tar.bz2",
}
SPAMASSASSIN_BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
SPAMASSASSIN_SOURCE_DATASET = "Apache SpamAssassin Public Corpus"
SPAMASSASSIN_LICENSE = "CC0 (Open Data Commons Public Domain Dedication)"

CATEGORY_CRITERIA = {
    "spam": "unsolicited bulk or commercial email",
    "legitimate": "genuine correspondence, clearly not spam",
    "legitimate_spam_like": "genuine correspondence that happens to resemble spam in style or wording",
}
FOLDER_TO_CATEGORY = {"spam": "spam", "easy_ham": "legitimate", "hard_ham": "legitimate_spam_like"}


def _safe_decode(payload: bytes, declared_charset: Optional[str]) -> str:
    """Real mail declares bogus/unregistered charsets often enough (e.g. "unknown-8bit")
    that trusting the header outright breaks on real corpora -- fall back to utf-8."""
    for charset in (declared_charset, "utf-8"):
        if not charset:
            continue
        try:
            return payload.decode(charset, errors="replace")
        except LookupError:
            continue
    return payload.decode("utf-8", errors="replace")


def _extract_subject_and_body(raw_bytes: bytes) -> Optional[tuple[str, str]]:
    try:
        msg = email_parser.message_from_bytes(raw_bytes)
    except Exception:
        return None
    subject = msg.get("Subject", "") or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    body = _safe_decode(payload, part.get_content_charset())
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = _safe_decode(payload, msg.get_content_charset())
    body = body.strip()
    if not body:
        return None
    return subject, body


def fetch_raw_spamassassin(per_category: int = 300) -> list[tuple[str, str, str]]:
    """Download and parse SpamAssassin messages. No API key required.

    Returns (folder_name, subject, body) tuples, capped at `per_category` messages per
    folder to keep processing fast -- the archives together hold several thousand.
    """
    results: list[tuple[str, str, str]] = []
    for folder, archive_name in SPAMASSASSIN_ARCHIVES.items():
        resp = requests.get(SPAMASSASSIN_BASE_URL + archive_name, timeout=60)
        resp.raise_for_status()
        count = 0
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:bz2") as tf:
            for member in tf.getmembers():
                if count >= per_category or not member.isfile() or "cmds" in member.name:
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                parsed = _extract_subject_and_body(extracted.read())
                if parsed is None:
                    continue
                results.append((folder, parsed[0], parsed[1]))
                count += 1
    return results


def _category_question() -> Question:
    return Question(type=QuestionType.CHOICE, instructions="How should this message be classified?",
                     criteria=CATEGORY_CRITERIA)


def to_records_spamassassin(raw_messages: list[tuple[str, str, str]], split: str = "train") -> Iterator[Record]:
    category_q = _category_question()
    labels = category_q.option_labels()

    for i, (folder, subject, body) in enumerate(raw_messages):
        category = FOLDER_TO_CATEGORY[folder]
        state = f"Subject: {subject}\n\n{body}" if subject else body
        idx = labels.index(category)
        dist = tuple(1.0 if j == idx else 0.0 for j in range(len(labels)))
        yield Record(
            state=state, domain=DOMAIN, question_id=f"spamassassin-{i}-category",
            question=category_q, target=TypedTarget(distribution=dist, label_index=idx),
            source_dataset=SPAMASSASSIN_SOURCE_DATASET, license=SPAMASSASSIN_LICENSE, split=split,
        )


# -- Combined --------------------------------------------------------------------
def load(split: str = "train", per_category: int = 300) -> list[Record]:
    records = list(to_records_sms(fetch_raw_sms(), split=split))
    records += list(to_records_spamassassin(fetch_raw_spamassassin(per_category), split=split))
    return records


def _record_to_dict(r: Record) -> dict:
    return {
        "state": r.state, "domain": r.domain, "question_id": r.question_id,
        "question_type": r.question.type.value, "instructions": r.question.instructions,
        "criteria": r.question.criteria, "target_distribution": list(r.target.distribution),
        "target_label_index": r.target.label_index, "source_dataset": r.source_dataset,
        "license": r.license, "split": r.split,
    }


def build_and_save(out_path: Path, per_category: int = 300) -> int:
    records = load(per_category=per_category)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "email-communication" / "combined_sample.jsonl")
    print(f"wrote {n} records")

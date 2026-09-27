"""Security operations / alert triage loader: two real sources, combined.

1. NVD CVE API (public domain) -- `score` (vulnerability severity). CVSS metrics come
   from different schema versions (v2, v3.0, v3.1) depending on when a CVE was scored;
   this loader prefers v3.1 > v3.0 > v2 and maps every version's severity onto one
   canonical 4-level scale so records stay comparable regardless of source version.

2. VERIS Community Database (VCDB), CC-BY-SA 4.0 -- `choice` and `noul`, using the
   corpus's own coded incident fields, not an invented taxonomy:
   - `choice`: VCDB's own top-level action category (Malware / Hacking / Social /
     Physical / Misuse / Error / Environmental) coded by the corpus's analysts from a
     real incident write-up. This is the deliberate alternative to forcing an
     unbounded CWE taxonomy, which research flagged as real curation work this loader
     doesn't attempt.
   - `noul`: whether an external actor was involved, derived from the presence of any
     `actor.external.*` (vs `actor.internal.*`) coded field -- a real fact the VERIS
     schema captures, not a redundant restating of the severity/category questions.

Fetched as VCDB's own flattened CSV export (2,683 one-hot columns per the VERIS
schema) rather than the 10,000+ individual per-incident JSON files, to avoid making
thousands of requests for a sample.
"""
from __future__ import annotations

import csv
import io
import json
import time
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import requests

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget

DOMAIN = "security-ops"

# -- Source 1: NVD CVE API (score) ---------------------------------------------------
NVD_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_SOURCE_DATASET = "NVD (National Vulnerability Database) CVE/CVSS data"
NVD_LICENSE = "public-domain (US government work)"

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
METRIC_KEYS_PREFERRED = ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2")


def fetch_raw_nvd(n_results: int = 500, page_size: int = 200, sleep_s: float = 6.0) -> list[dict]:
    """Page through the live NVD API. No API key required (rate-limited without one,
    hence the sleep between pages)."""
    cves: list[dict] = []
    start = 0
    while len(cves) < n_results:
        resp = requests.get(NVD_API_URL, params={
            "resultsPerPage": min(page_size, n_results - len(cves)), "startIndex": start,
        }, timeout=30)
        resp.raise_for_status()
        batch = [v["cve"] for v in resp.json().get("vulnerabilities", [])]
        if not batch:
            break
        cves.extend(batch)
        start += page_size
        if len(cves) < n_results:
            time.sleep(sleep_s)
    return cves


def _extract_severity(cve: dict) -> Optional[str]:
    metrics = cve.get("metrics", {})
    for key in METRIC_KEYS_PREFERRED:
        entries = metrics.get(key)
        if entries:
            sev = entries[0].get("baseSeverity")
            if sev:
                return sev.lower()
    return None


def _extract_description(cve: dict) -> Optional[str]:
    for d in cve.get("descriptions", []):
        if d.get("lang") == "en":
            return d.get("value")
    return None


def _severity_question() -> Question:
    return Question(type=QuestionType.SCORE, instructions="How severe is this vulnerability?",
                     criteria=SEVERITY_LEVELS)


def to_records_nvd(raw_cves: list[dict], split: str = "train") -> Iterator[Record]:
    severity_q = _severity_question()

    for cve in raw_cves:
        description = _extract_description(cve)
        severity = _extract_severity(cve)
        if not description or severity not in SEVERITY_LEVELS:
            continue  # no usable CVSS metric or description -- skip rather than guess

        idx = SEVERITY_LEVELS.index(severity)
        dist = tuple(1.0 if i == idx else 0.0 for i in range(len(SEVERITY_LEVELS)))
        yield Record(
            state=description, domain=DOMAIN, question_id=f"{cve['id']}-severity",
            question=severity_q, target=TypedTarget(distribution=dist, label_index=idx),
            source_dataset=NVD_SOURCE_DATASET, license=NVD_LICENSE, split=split,
        )


# -- Source 2: VERIS Community Database (choice, noul) --------------------------------
VCDB_CSV_URL = "https://raw.githubusercontent.com/vz-risk/VCDB/master/data/csv/vcdb.csv.zip"
VCDB_SOURCE_DATASET = "VERIS Community Database (VCDB)"
VCDB_LICENSE = "CC-BY-SA-4.0"

ACTION_CATEGORIES = {
    "Malware": "malicious software used to compromise a system",
    "Hacking": "unauthorized access or use of a system via a specific technique or vulnerability",
    "Social": "social engineering of a person, e.g. phishing or pretexting",
    "Physical": "physical actions, e.g. theft, tampering, or surveillance",
    "Misuse": "use of entrusted organizational resources or privileges for unintended purposes",
    "Error": "unintentional actions or omissions that directly compromised a security attribute",
    "Environmental": "environmental or natural events unrelated to human action",
}


def fetch_raw_vcdb(n_rows: int = 2000) -> list[dict]:
    """Download VCDB's flattened CSV export and return the first `n_rows` records with
    a non-empty summary. No API key required."""
    resp = requests.get(VCDB_CSV_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("vcdb.csv") as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text)
            rows = []
            for row in reader:
                if row.get("summary", "").strip():
                    rows.append(row)
                if len(rows) >= n_rows:
                    break
    return rows


def _is_true(value: Optional[str]) -> bool:
    return value in ("TRUE", "True", "1")


def _single_action_category(row: dict) -> Optional[str]:
    present = [cat for cat in ACTION_CATEGORIES if _is_true(row.get(f"action.{cat}"))]
    return present[0] if len(present) == 1 else None  # skip multi-cause or unclassified incidents


def _external_actor_signal(row: dict) -> Optional[bool]:
    external = any(_is_true(v) for k, v in row.items()
                    if k.startswith("actor.external.") and "notes" not in k)
    internal = any(_is_true(v) for k, v in row.items()
                    if k.startswith("actor.internal.") and "notes" not in k)
    if external and not internal:
        return True
    if internal and not external:
        return False
    return None  # both, or neither, coded -- ambiguous, skip rather than guess


def _category_question() -> Question:
    return Question(type=QuestionType.CHOICE, instructions="What type of action caused this incident?",
                     criteria=ACTION_CATEGORIES)


def _external_actor_question() -> Question:
    return Question(type=QuestionType.NOUL, instructions="Was an external party responsible for this incident?")


def to_records_vcdb(raw_rows: list[dict], split: str = "train") -> Iterator[Record]:
    category_q = _category_question()
    external_q = _external_actor_question()
    labels = category_q.option_labels()

    for i, row in enumerate(raw_rows):
        state = (row.get("summary") or "").strip()
        if not state:
            continue  # no narrative text to serve as state -- skip rather than use an empty string
        incident_id = f"vcdb-{i}"

        category = _single_action_category(row)
        if category is not None:
            idx = labels.index(category)
            dist = tuple(1.0 if j == idx else 0.0 for j in range(len(labels)))
            yield Record(
                state=state, domain=DOMAIN, question_id=f"{incident_id}-category",
                question=category_q, target=TypedTarget(distribution=dist, label_index=idx),
                source_dataset=VCDB_SOURCE_DATASET, license=VCDB_LICENSE, split=split,
            )

        is_external = _external_actor_signal(row)
        if is_external is not None:
            val = 1.0 if is_external else 0.0
            yield Record(
                state=state, domain=DOMAIN, question_id=f"{incident_id}-external",
                question=external_q, target=TypedTarget(distribution=(1.0 - val, val), label_index=int(val)),
                source_dataset=VCDB_SOURCE_DATASET, license=VCDB_LICENSE, split=split,
            )


# -- Combined --------------------------------------------------------------------
def load(n_nvd: int = 500, n_vcdb: int = 2000, split: str = "train") -> list[Record]:
    records = list(to_records_nvd(fetch_raw_nvd(n_nvd), split=split))
    records += list(to_records_vcdb(fetch_raw_vcdb(n_vcdb), split=split))
    return records


def _record_to_dict(r: Record) -> dict:
    return {
        "state": r.state, "domain": r.domain, "question_id": r.question_id,
        "question_type": r.question.type.value, "instructions": r.question.instructions,
        "criteria": r.question.criteria, "target_distribution": list(r.target.distribution),
        "target_label_index": r.target.label_index, "source_dataset": r.source_dataset,
        "license": r.license, "split": r.split,
    }


def build_and_save(out_path: Path, n_nvd: int = 500, n_vcdb: int = 2000) -> int:
    records = load(n_nvd=n_nvd, n_vcdb=n_vcdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(_record_to_dict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    n = build_and_save(Path(__file__).parents[4] / "data" / "security-ops" / "combined_sample.jsonl")
    print(f"wrote {n} records")

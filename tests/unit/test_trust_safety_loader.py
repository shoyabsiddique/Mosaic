from decision_engine.data.loaders.trust_safety import DOMAIN, SEVERITY_LEVELS, to_records

RAW_FIXTURE = [
    {"text": "this is a fine comment", "toxicity": 0.0, "threat": 0.0, "insult": 0.0},
    {"text": "very toxic comment", "toxicity": 1.0, "threat": 0.9, "insult": 0.8},
    {"text": "borderline comment", "toxicity": 0.5, "threat": None, "insult": 0.4},
    {"text": None, "toxicity": 0.5},  # missing text -- must be skipped
    {"text": "no toxicity score", "toxicity": None},  # missing toxicity -- must be skipped
]


def test_full_row_produces_three_records():
    records = list(to_records(RAW_FIXTURE[:1]))
    ids = {r.question_id for r in records}
    assert ids == {"civilcomments-0-severity", "civilcomments-0-threat", "civilcomments-0-insult"}


def test_missing_optional_noul_field_skips_only_that_question():
    records = {r.question_id: r for r in to_records([RAW_FIXTURE[2]])}
    assert "civilcomments-0-severity" in records
    assert "civilcomments-0-insult" in records
    assert "civilcomments-0-threat" not in records  # threat was None


def test_rows_missing_text_or_toxicity_are_skipped():
    records = list(to_records(RAW_FIXTURE[3:5]))
    assert records == []


def test_zero_toxicity_maps_fully_to_lowest_severity_level():
    records = {r.question_id: r for r in to_records([RAW_FIXTURE[0]])}
    sev = records["civilcomments-0-severity"]
    assert sev.question.criteria == SEVERITY_LEVELS
    assert sev.target.distribution[0] == 1.0
    assert sum(sev.target.distribution) == 1.0


def test_max_toxicity_maps_fully_to_highest_severity_level():
    records = {r.question_id: r for r in to_records([RAW_FIXTURE[1]])}
    sev = records["civilcomments-0-severity"]
    assert sev.target.distribution[-1] == 1.0


def test_midpoint_toxicity_produces_soft_split_target():
    records = {r.question_id: r for r in to_records([RAW_FIXTURE[2]])}
    sev = records["civilcomments-0-severity"]
    nonzero = [p for p in sev.target.distribution if p > 0]
    assert len(nonzero) == 2  # genuinely soft, not collapsed to one-hot
    assert abs(sum(sev.target.distribution) - 1.0) < 1e-9


def test_domain_and_license_tagged_correctly():
    for r in to_records(RAW_FIXTURE[:1]):
        assert r.domain == DOMAIN
        assert r.license == "CC0-1.0 (public domain)"

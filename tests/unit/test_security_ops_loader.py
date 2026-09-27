from decision_engine.data.loaders.security_ops import (
    ACTION_CATEGORIES,
    DOMAIN,
    SEVERITY_LEVELS,
    to_records_nvd,
    to_records_vcdb,
)

CVE_V31 = {
    "id": "CVE-2024-0001",
    "descriptions": [{"lang": "en", "value": "A buffer overflow allows remote code execution."}],
    "metrics": {"cvssMetricV31": [{"baseSeverity": "CRITICAL"}]},
}
CVE_V2_ONLY = {
    "id": "CVE-2010-0001",
    "descriptions": [{"lang": "en", "value": "The debug command in Sendmail is enabled."}],
    "metrics": {"cvssMetricV2": [{"baseSeverity": "MEDIUM"}]},
}
CVE_PREFERS_V31_OVER_V2 = {
    "id": "CVE-2024-0002",
    "descriptions": [{"lang": "en", "value": "Something with both metric versions."}],
    "metrics": {
        "cvssMetricV2": [{"baseSeverity": "LOW"}],
        "cvssMetricV31": [{"baseSeverity": "HIGH"}],
    },
}
CVE_NO_METRICS = {
    "id": "CVE-2024-0003",
    "descriptions": [{"lang": "en", "value": "No CVSS score assigned yet."}],
    "metrics": {},
}
CVE_NO_ENGLISH_DESCRIPTION = {
    "id": "CVE-2024-0004",
    "descriptions": [{"lang": "es", "value": "Solo en espanol."}],
    "metrics": {"cvssMetricV31": [{"baseSeverity": "LOW"}]},
}


def test_v31_metric_produces_correct_severity():
    records = list(to_records_nvd([CVE_V31]))
    assert len(records) == 1
    assert records[0].target.label_index == SEVERITY_LEVELS.index("critical")


def test_v2_only_cve_maps_onto_canonical_scale():
    records = list(to_records_nvd([CVE_V2_ONLY]))
    assert len(records) == 1
    assert records[0].target.label_index == SEVERITY_LEVELS.index("medium")


def test_v31_preferred_over_v2_when_both_present():
    records = list(to_records_nvd([CVE_PREFERS_V31_OVER_V2]))
    assert records[0].target.label_index == SEVERITY_LEVELS.index("high")


def test_cve_with_no_metrics_is_skipped_not_guessed():
    assert list(to_records_nvd([CVE_NO_METRICS])) == []


def test_cve_with_no_english_description_is_skipped():
    assert list(to_records_nvd([CVE_NO_ENGLISH_DESCRIPTION])) == []


def test_nvd_state_is_the_cve_description_text():
    records = list(to_records_nvd([CVE_V31]))
    assert records[0].state == CVE_V31["descriptions"][0]["value"]


def test_nvd_domain_and_criteria_tagged_correctly():
    records = list(to_records_nvd([CVE_V31]))
    assert records[0].domain == DOMAIN
    assert records[0].question.criteria == SEVERITY_LEVELS


def _vcdb_row(summary="An incident occurred.", **flags):
    row = {"summary": summary}
    row.update(flags)
    return row


VCDB_SINGLE_CATEGORY = _vcdb_row(
    summary="An attacker phished an employee for their credentials.",
    **{"action.Social": "TRUE", "actor.external.motive.Financial": "TRUE"},
)
VCDB_MULTI_CATEGORY = _vcdb_row(
    summary="Ambiguous incident with two coded causes.",
    **{"action.Hacking": "TRUE", "action.Malware": "TRUE"},
)
VCDB_NO_CATEGORY = _vcdb_row(summary="No action category was coded.")
VCDB_INTERNAL_ACTOR = _vcdb_row(
    summary="An employee misused their access.",
    **{"action.Misuse": "TRUE", "actor.internal.job_change.Let go": "TRUE"},
)
VCDB_BOTH_ACTORS = _vcdb_row(
    summary="Both an insider and an external party were involved.",
    **{"action.Error": "TRUE", "actor.external.motive.Financial": "TRUE",
       "actor.internal.job_change.Hired": "TRUE"},
)
VCDB_EMPTY_SUMMARY = _vcdb_row(summary="   ", **{"action.Hacking": "TRUE"})


def test_vcdb_single_category_produces_choice_record():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_SINGLE_CATEGORY])}
    r = records["vcdb-0-category"]
    labels = r.question.option_labels()
    assert r.target.label_index == labels.index("Social")


def test_vcdb_multi_category_is_skipped_not_guessed():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_MULTI_CATEGORY])}
    assert "vcdb-0-category" not in records


def test_vcdb_no_category_coded_is_skipped():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_NO_CATEGORY])}
    assert "vcdb-0-category" not in records


def test_vcdb_external_actor_only_targets_true():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_SINGLE_CATEGORY])}
    r = records["vcdb-0-external"]
    assert r.target.label_index == 1


def test_vcdb_internal_actor_only_targets_false():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_INTERNAL_ACTOR])}
    r = records["vcdb-0-external"]
    assert r.target.label_index == 0


def test_vcdb_both_actors_coded_is_skipped_as_ambiguous():
    records = {r.question_id: r for r in to_records_vcdb([VCDB_BOTH_ACTORS])}
    assert "vcdb-0-external" not in records
    assert "vcdb-0-category" in records  # category question is independent and still valid


def test_vcdb_empty_summary_row_produces_no_records():
    assert list(to_records_vcdb([VCDB_EMPTY_SUMMARY])) == []


def test_vcdb_domain_and_license_tagged_correctly():
    for r in to_records_vcdb([VCDB_SINGLE_CATEGORY]):
        assert r.domain == DOMAIN
        assert r.license == "CC-BY-SA-4.0"


def test_vcdb_category_criteria_match_declared_categories():
    r = list(to_records_vcdb([VCDB_SINGLE_CATEGORY]))[0]
    assert r.question.criteria == ACTION_CATEGORIES

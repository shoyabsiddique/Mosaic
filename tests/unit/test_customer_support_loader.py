"""Tests the CFPB loader's conversion logic against fixture data -- no network calls.
`fetch_raw`'s HTTP call (`_fetch_page`) is monkeypatched below rather than hit live.
"""
import decision_engine.data.loaders.customer_support as cs
from decision_engine.data.loaders.customer_support import DOMAIN, PRODUCT_CRITERIA, fetch_raw, to_records

RAW_FIXTURE = [
    {
        "complaint_id": "1",
        "product": "Credit card",
        "timely": "Yes",
        "issue": "Billing dispute",
        "sub_issue": "Charged twice",
        "company_response": "Closed with explanation",
        "submitted_via": "Web",
        "company": "ACME BANK",
        "state": "CA",
    },
    {
        "complaint_id": "2",
        "product": "Some Unmapped Future Product Category",
        "timely": "No",
    },
    {
        "complaint_id": "3",
        "product": "Mortgage",
        "timely": "Unknown",
    },
]


def test_known_complaint_produces_two_records():
    records = list(to_records(RAW_FIXTURE[:1]))
    assert len(records) == 2
    ids = {r.question_id for r in records}
    assert ids == {"1-product", "1-timely"}


def test_all_records_tagged_with_correct_domain_and_provenance():
    records = list(to_records(RAW_FIXTURE[:1]))
    for r in records:
        assert r.domain == DOMAIN
        assert r.source_dataset == "CFPB Consumer Complaint Database"
        assert "public-domain" in r.license


def test_target_leakage_excluded_from_state():
    records = list(to_records(RAW_FIXTURE[:1]))
    for r in records:
        assert "product" not in r.state
        assert "timely" not in r.state


def test_product_target_is_one_hot_at_correct_index():
    records = {r.question_id: r for r in to_records(RAW_FIXTURE[:1])}
    product_record = records["1-product"]
    labels = product_record.question.option_labels()
    idx = labels.index("Credit card")
    assert product_record.target.label_index == idx
    assert product_record.target.distribution[idx] == 1.0
    assert sum(product_record.target.distribution) == 1.0


def test_unmapped_product_category_is_skipped_not_guessed():
    records = list(to_records([RAW_FIXTURE[1]]))
    assert records == []


def test_unknown_timely_value_is_skipped_not_guessed():
    records = list(to_records([RAW_FIXTURE[2]]))
    assert records == []


def test_split_is_propagated():
    records = list(to_records(RAW_FIXTURE[:1], split="val"))
    assert all(r.split == "val" for r in records)


def test_fetch_raw_queries_every_product_category_and_a_timely_no_topup(monkeypatch):
    calls = []

    def fake_fetch_page(params):
        calls.append(dict(params))
        product = params.get("product", "unfiltered")
        cid = f"{product}-{params.get('timely', 'na')}-{len(calls)}"
        return [{"complaint_id": cid, "product": params.get("product", "Mortgage"),
                 "timely": params.get("timely", "Yes")}]

    monkeypatch.setattr(cs, "_fetch_page", fake_fetch_page)

    complaints = fetch_raw(per_product=5, n_timely_no=5)

    queried_products = {c["product"] for c in calls if "product" in c}
    assert queried_products == set(PRODUCT_CRITERIA.keys())
    assert any(c.get("timely") == "No" for c in calls)
    assert len(complaints) == len(PRODUCT_CRITERIA) + 1  # one complaint per call, no duplicate ids


def test_fetch_raw_deduplicates_by_complaint_id(monkeypatch):
    def fake_fetch_page(params):
        # every call "coincidentally" returns the same complaint_id
        return [{"complaint_id": "dup-1", "product": "Mortgage", "timely": "Yes"}]

    monkeypatch.setattr(cs, "_fetch_page", fake_fetch_page)

    complaints = fetch_raw(per_product=5, n_timely_no=5)
    assert len(complaints) == 1

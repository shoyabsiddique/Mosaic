from decision_engine.data.loaders.sales_crm import DOMAIN, to_records

RAW_FIXTURE = [
    {
        "Administrative": "0", "Administrative_Duration": "0", "Informational": "0",
        "Informational_Duration": "0", "ProductRelated": "5", "ProductRelated_Duration": "120",
        "BounceRates": "0.02", "ExitRates": "0.04", "PageValues": "10.5", "SpecialDay": "0",
        "Month": "Nov", "OperatingSystems": "2", "Browser": "2", "Region": "1", "TrafficType": "2",
        "VisitorType": "Returning_Visitor", "Weekend": "FALSE", "Revenue": "TRUE",
    },
    {
        "VisitorType": "New_Visitor", "Revenue": "FALSE", "Month": "Feb",
    },
    {
        "VisitorType": "Bogus_Type", "Revenue": "TRUE",  # invalid visitor type -- must be skipped
    },
    {
        "VisitorType": "Other", "Revenue": "Maybe",  # invalid revenue value -- must be skipped
    },
]


def test_valid_session_produces_two_records():
    records = list(to_records(RAW_FIXTURE[:1]))
    ids = {r.question_id for r in records}
    assert ids == {"session-0-revenue", "session-0-visitor-type"}


def test_state_is_a_dict_not_text():
    records = list(to_records(RAW_FIXTURE[:1]))
    assert all(isinstance(r.state, dict) for r in records)


def test_target_leakage_excluded_from_state():
    records = list(to_records(RAW_FIXTURE[:1]))
    for r in records:
        assert "Revenue" not in r.state
        assert "VisitorType" not in r.state


def test_revenue_true_targets_purchase():
    records = {r.question_id: r for r in to_records(RAW_FIXTURE[:1])}
    r = records["session-0-revenue"]
    assert r.target.label_index == 1


def test_revenue_false_targets_no_purchase():
    records = {r.question_id: r for r in to_records([RAW_FIXTURE[1]])}
    r = records["session-0-revenue"]
    assert r.target.label_index == 0


def test_visitor_type_one_hot_at_correct_index():
    records = {r.question_id: r for r in to_records(RAW_FIXTURE[:1])}
    r = records["session-0-visitor-type"]
    labels = r.question.option_labels()
    idx = labels.index("Returning_Visitor")
    assert r.target.distribution[idx] == 1.0


def test_invalid_visitor_type_is_skipped_not_guessed():
    assert list(to_records([RAW_FIXTURE[2]])) == []


def test_invalid_revenue_value_is_skipped_not_guessed():
    assert list(to_records([RAW_FIXTURE[3]])) == []


def test_domain_tagged_correctly():
    for r in to_records(RAW_FIXTURE[:1]):
        assert r.domain == DOMAIN

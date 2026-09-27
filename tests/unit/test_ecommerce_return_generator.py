from decision_engine.data.synth.ecommerce_return_generator import (
    CRITERIA,
    DOMAIN,
    generate,
)


def test_each_generated_return_produces_three_records():
    records = list(generate(n=5, seed=1))
    assert len(records) == 15
    ids = {r.question_id for r in records}
    assert "synth-return-0-reason" in ids
    assert "synth-return-0-fault" in ids
    assert "synth-return-0-refund-pref" in ids


def test_generation_is_reproducible_given_same_seed():
    a = [r.target.distribution for r in generate(n=20, seed=8)]
    b = [r.target.distribution for r in generate(n=20, seed=8)]
    assert a == b


def test_reason_target_is_one_hot():
    for r in generate(n=50, seed=5):
        if r.question_id.endswith("-reason"):
            assert sum(r.target.distribution) == 1.0
            assert r.target.distribution.count(1.0) == 1


def test_defective_reason_marks_product_fault_true():
    records = list(generate(n=200, seed=9))
    by_return = {}
    for r in records:
        base = "-".join(r.question_id.split("-")[:3])  # "synth-return-N" regardless of suffix
        by_return.setdefault(base, {})[r.question_id] = r

    for base, qs in by_return.items():
        reason_r = qs[f"{base}-reason"]
        fault_r = qs[f"{base}-fault"]
        labels = reason_r.question.option_labels()
        reason = labels[reason_r.target.label_index]
        if reason in ("defective", "wrong_item_shipped"):
            assert fault_r.target.label_index == 1
        elif reason in ("changed_mind", "wrong_size"):
            assert fault_r.target.label_index == 0


def test_all_reason_categories_reachable_across_many_draws():
    records = [r for r in generate(n=300, seed=12) if r.question_id.endswith("-reason")]
    seen_reasons = {r.question.option_labels()[r.target.label_index] for r in records}
    assert seen_reasons == set(CRITERIA.keys())


def test_state_shape_has_expected_fields():
    r = list(generate(n=1, seed=1))[0]
    assert isinstance(r.state, dict)
    assert "product_category" in r.state
    assert "message" in r.state


def test_domain_and_license_tagged_as_synthetic():
    for r in generate(n=3, seed=1):
        assert r.domain == DOMAIN
        assert "synthetic" in r.license
        assert "synthetic" in r.source_dataset


def test_reason_criteria_match_declared_categories():
    reason_records = [r for r in generate(n=1, seed=1) if r.question_id.endswith("-reason")]
    assert reason_records[0].question.criteria == CRITERIA

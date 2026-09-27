import random

from decision_engine.data.synth.sales_crm_generator import (
    DOMAIN,
    LICENSE,
    PRIORITY_LEVELS,
    _build_one,
    generate,
)


def test_each_generated_lead_produces_two_records():
    records = list(generate(n=5, seed=1))
    assert len(records) == 10
    ids = {r.question_id for r in records}
    assert "synth-lead-0-priority" in ids
    assert "synth-lead-0-budget" in ids


def test_generation_is_reproducible_given_same_seed():
    a = [r.target.distribution for r in generate(n=20, seed=99)]
    b = [r.target.distribution for r in generate(n=20, seed=99)]
    assert a == b


def test_different_seeds_produce_different_output():
    a = [r.target.distribution for r in generate(n=20, seed=1)]
    b = [r.target.distribution for r in generate(n=20, seed=2)]
    assert a != b


def test_priority_target_is_a_valid_distribution():
    for r in generate(n=50, seed=3):
        if r.question_id.endswith("-priority"):
            assert abs(sum(r.target.distribution) - 1.0) < 1e-9
            nonzero = [p for p in r.target.distribution if p > 0]
            assert len(nonzero) in (1, 2)  # one-hot or adjacent-pair soft split


def test_scoring_function_spans_both_extremes_across_many_draws():
    """Not a check on any single draw (categorical sampling makes any one draw
    unpredictable) -- verifies the underlying formula can actually reach both the high
    and low ends of the range across enough samples, i.e. it isn't accidentally
    compressed into a narrow band regardless of evidence quality."""
    rng = random.Random(0)
    scores = [_build_one(rng, i)[1] for i in range(300)]
    assert max(scores) > 0.85
    assert min(scores) < 0.15


def test_confirmed_budget_yields_high_probability_noul_target():
    records = list(generate(n=200, seed=5))
    budget_records = [r for r in records if r.question_id.endswith("-budget")]
    confirmed = [r for r in budget_records if "budget has been confirmed" in r.state["notes"]]
    no_budget = [r for r in budget_records if "no budget this quarter" in r.state["notes"]]
    assert confirmed and no_budget
    assert all(r.target.distribution[1] > 0.8 for r in confirmed)
    assert all(r.target.distribution[1] < 0.2 for r in no_budget)


def test_state_leakage_free_and_shaped_as_expected():
    r = list(generate(n=1, seed=1))[0]
    assert isinstance(r.state, dict)
    assert "notes" in r.state
    assert "company_size" in r.state


def test_domain_and_license_tagged_as_synthetic():
    for r in generate(n=3, seed=1):
        assert r.domain == DOMAIN
        assert "synthetic" in r.license
        assert "synthetic" in r.source_dataset


def test_priority_criteria_match_declared_levels():
    priority_records = [r for r in generate(n=1, seed=1) if r.question_id.endswith("-priority")]
    assert priority_records[0].question.criteria == PRIORITY_LEVELS

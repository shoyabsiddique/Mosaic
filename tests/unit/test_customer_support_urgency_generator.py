import random

from decision_engine.data.synth.customer_support_urgency_generator import (
    DOMAIN,
    URGENCY_LEVELS,
    _build_one,
    generate,
)


def test_each_generated_ticket_produces_two_records():
    records = list(generate(n=5, seed=1))
    assert len(records) == 10
    ids = {r.question_id for r in records}
    assert "synth-ticket-0-urgency" in ids
    assert "synth-ticket-0-churn" in ids


def test_generation_is_reproducible_given_same_seed():
    a = [r.target.distribution for r in generate(n=20, seed=17)]
    b = [r.target.distribution for r in generate(n=20, seed=17)]
    assert a == b


def test_urgency_target_is_a_valid_distribution():
    for r in generate(n=50, seed=4):
        if r.question_id.endswith("-urgency"):
            assert abs(sum(r.target.distribution) - 1.0) < 1e-9
            nonzero = [p for p in r.target.distribution if p > 0]
            assert len(nonzero) in (1, 2)


def test_scoring_function_spans_both_extremes_across_many_draws():
    rng = random.Random(0)
    scores = [_build_one(rng)[1] for _ in range(300)]
    assert max(scores) > 0.85
    assert min(scores) < 0.15


def test_cancel_threat_tone_yields_high_churn_probability():
    records = list(generate(n=200, seed=6))
    churn_records = [r for r in records if r.question_id.endswith("-churn")]
    threats = [r for r in churn_records if "will cancel their subscription" in r.state["message"]]
    neutrals = [r for r in churn_records if "neutral, matter-of-fact tone" in r.state["message"]]
    assert threats and neutrals
    assert all(r.target.distribution[1] > 0.8 for r in threats)
    assert all(r.target.distribution[1] < 0.2 for r in neutrals)


def test_state_shape_has_expected_fields():
    r = list(generate(n=1, seed=1))[0]
    assert isinstance(r.state, dict)
    for key in ("customer_tier", "channel", "repeat_contacts", "message"):
        assert key in r.state


def test_domain_and_license_tagged_as_synthetic():
    for r in generate(n=3, seed=1):
        assert r.domain == DOMAIN
        assert "synthetic" in r.license
        assert "synthetic" in r.source_dataset


def test_urgency_criteria_match_declared_levels():
    urgency_records = [r for r in generate(n=1, seed=1) if r.question_id.endswith("-urgency")]
    assert urgency_records[0].question.criteria == URGENCY_LEVELS

import random

from decision_engine.data.synth.email_urgency_generator import (
    DOMAIN,
    URGENCY_LEVELS,
    _build_one,
    generate,
)


def test_each_generated_message_produces_two_records():
    records = list(generate(n=5, seed=1))
    assert len(records) == 10
    ids = {r.question_id for r in records}
    assert "synth-email-0-urgency" in ids
    assert "synth-email-0-external" in ids


def test_generation_is_reproducible_given_same_seed():
    a = [r.target.distribution for r in generate(n=20, seed=11)]
    b = [r.target.distribution for r in generate(n=20, seed=11)]
    assert a == b


def test_urgency_target_is_a_valid_distribution():
    for r in generate(n=50, seed=2):
        if r.question_id.endswith("-urgency"):
            assert abs(sum(r.target.distribution) - 1.0) < 1e-9
            nonzero = [p for p in r.target.distribution if p > 0]
            assert len(nonzero) in (1, 2)


def test_scoring_function_spans_both_extremes_across_many_draws():
    rng = random.Random(0)
    scores = [_build_one(rng)[1] for _ in range(300)]
    assert max(scores) > 0.85
    assert min(scores) < 0.15


def test_external_sender_target_matches_sender_field():
    for r in generate(n=50, seed=3):
        if not r.question_id.endswith("-external"):
            continue
        is_external_marked = r.target.label_index == 1
        sender_line = r.state.splitlines()[0]
        mentions_external = "unknown sender" in sender_line or "external client" in sender_line
        assert is_external_marked == mentions_external


def test_state_is_email_shaped_text():
    r = list(generate(n=1, seed=1))[0]
    assert r.state.startswith("From: ")
    assert "Subject: " in r.state


def test_domain_and_license_tagged_as_synthetic():
    for r in generate(n=3, seed=1):
        assert r.domain == DOMAIN
        assert "synthetic" in r.license
        assert "synthetic" in r.source_dataset


def test_urgency_criteria_match_declared_levels():
    urgency_records = [r for r in generate(n=1, seed=1) if r.question_id.endswith("-urgency")]
    assert urgency_records[0].question.criteria == URGENCY_LEVELS

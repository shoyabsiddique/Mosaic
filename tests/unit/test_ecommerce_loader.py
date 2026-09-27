from decision_engine.data.loaders.ecommerce import DOMAIN, to_records

FULL_REVIEW = {
    "review/summary": "Great product",
    "review/text": "Loved this, would buy again.",
    "review/score": "5.0",
    "review/helpfulness": "8/10",
}
NO_HELPFULNESS_VOTES = {
    "review/summary": "It's ok",
    "review/text": "Fine I guess.",
    "review/score": "3.0",
    "review/helpfulness": "0/0",
}
MISSING_TEXT = {"review/score": "4.0"}
MISSING_SCORE = {"review/text": "no score given"}
BAD_SCORE = {"review/text": "bad score", "review/score": "not-a-number"}


def test_full_review_produces_rating_and_helpful_records():
    records = list(to_records([FULL_REVIEW]))
    ids = {r.question_id for r in records}
    assert ids == {"amazon-food-0-rating", "amazon-food-0-helpful"}


def test_review_with_zero_helpfulness_votes_skips_helpful_question():
    records = {r.question_id: r for r in to_records([NO_HELPFULNESS_VOTES])}
    assert "amazon-food-0-rating" in records
    assert "amazon-food-0-helpful" not in records


def test_five_star_rating_maps_to_last_index():
    records = {r.question_id: r for r in to_records([FULL_REVIEW])}
    r = records["amazon-food-0-rating"]
    assert r.target.label_index == 4
    assert r.target.distribution[4] == 1.0


def test_helpfulness_ratio_is_a_genuinely_soft_target():
    records = {r.question_id: r for r in to_records([FULL_REVIEW])}
    r = records["amazon-food-0-helpful"]
    assert abs(r.target.distribution[1] - 0.8) < 1e-9
    assert r.target.label_index is None  # soft target, not collapsed to a hard label


def test_state_combines_summary_and_text():
    records = {r.question_id: r for r in to_records([FULL_REVIEW])}
    r = records["amazon-food-0-rating"]
    assert "Great product" in r.state
    assert "Loved this" in r.state


def test_missing_text_or_score_is_skipped():
    assert list(to_records([MISSING_TEXT, MISSING_SCORE])) == []


def test_unparseable_score_is_skipped_not_guessed():
    assert list(to_records([BAD_SCORE])) == []


def test_domain_tagged_correctly():
    for r in to_records([FULL_REVIEW]):
        assert r.domain == DOMAIN

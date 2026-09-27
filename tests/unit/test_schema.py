import pytest

from decision_engine.data.schema import Question, QuestionType, Record, TypedTarget


def test_choice_option_labels_from_dict_criteria():
    q = Question(type=QuestionType.CHOICE, instructions="pick one",
                 criteria={"billing": "invoices", "technical": "bugs"})
    assert q.option_labels() == ["billing", "technical"]


def test_score_option_labels_from_list_criteria():
    q = Question(type=QuestionType.SCORE, instructions="how urgent",
                 criteria=["low", "medium", "high"])
    assert q.option_labels() == ["0", "1", "2"]


def test_noul_option_labels_are_fixed():
    q = Question(type=QuestionType.NOUL, instructions="is this true")
    assert q.option_labels() == ["false", "true"]


def test_typed_target_rejects_distribution_not_summing_to_one():
    with pytest.raises(ValueError, match="sum to ~1.0"):
        TypedTarget(distribution=(0.2, 0.2))


def test_typed_target_accepts_soft_distribution():
    t = TypedTarget(distribution=(0.3, 0.7))
    assert t.label_index is None


def _make_record(**overrides):
    defaults = dict(
        state="some ticket text",
        domain="customer-support",
        question_id="q1",
        question=Question(type=QuestionType.NOUL, instructions="is this urgent"),
        target=TypedTarget(distribution=(0.5, 0.5)),
        source_dataset="test",
        license="test-license",
    )
    defaults.update(overrides)
    return Record(**defaults)


def test_record_accepts_matching_option_count():
    r = _make_record()
    assert r.split == "train"


def test_record_rejects_mismatched_option_count():
    bad_question = Question(type=QuestionType.CHOICE, instructions="pick",
                             criteria={"a": "", "b": "", "c": ""})
    with pytest.raises(ValueError, match="options"):
        _make_record(question=bad_question, target=TypedTarget(distribution=(0.5, 0.5)))


def test_record_rejects_invalid_split():
    with pytest.raises(ValueError, match="split"):
        _make_record(split="bogus")


def test_record_accepts_dict_state():
    r = _make_record(state={"issue": "billed twice"})
    assert isinstance(r.state, dict)

import torch

from decision_engine.model.types import QTYPE_TO_IDX
from decision_engine.training.metrics import (
    expected_calibration_error,
    score_mean_absolute_error,
    summarize_batch,
    top1_accuracy,
)


def test_top1_accuracy_all_correct():
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    assert top1_accuracy(probs, target, mask).item() == 1.0


def test_top1_accuracy_all_wrong():
    probs = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    target = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    assert top1_accuracy(probs, target, mask).item() == 0.0


def test_top1_accuracy_partial():
    probs = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    assert top1_accuracy(probs, target, mask).item() == 0.5


def test_score_mae_zero_for_exact_match():
    probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert score_mean_absolute_error(probs, target, mask).item() < 1e-6


def test_score_mae_reflects_distance_not_just_hit_or_miss():
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    near = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # one level off
    far = torch.tensor([[0.0, 0.0, 0.0, 1.0]])   # three levels off
    mae_near = score_mean_absolute_error(near, target, mask).item()
    mae_far = score_mean_absolute_error(far, target, mask).item()
    assert mae_near == 1.0
    assert mae_far == 3.0
    assert mae_far > mae_near


def test_ece_zero_for_perfectly_calibrated_predictions():
    confidences = torch.full((100,), 0.8)
    correctness = torch.tensor([1.0] * 80 + [0.0] * 20)
    assert expected_calibration_error(confidences, correctness) < 1e-6


def test_ece_positive_for_overconfident_predictions():
    confidences = torch.full((100,), 0.99)
    correctness = torch.tensor([1.0] * 60 + [0.0] * 40)
    assert expected_calibration_error(confidences, correctness) > 0.1


def test_ece_handles_empty_input():
    result = expected_calibration_error(torch.tensor([]), torch.tensor([]))
    assert result != result  # nan check without importing math


def test_summarize_batch_splits_by_question_type():
    probs = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.1, 0.8, 0.1, 0.0], [0.0, 0.0, 0.6, 0.4]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[True, True, False, False], [True, True, False, False], [True, True, True, True]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["choice"], QTYPE_TO_IDX["noul"], QTYPE_TO_IDX["score"]])

    out = summarize_batch(probs, target, mask, qtype_idx, QTYPE_TO_IDX)
    assert set(out.keys()) == {"choice", "noul", "score"}
    assert out["choice"]["n"] == 1
    assert out["score"]["accuracy"] == 1.0
    assert "mae" in out["score"]
    assert "mae" not in out["choice"]

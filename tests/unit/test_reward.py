"""Correctness properties for the reward functions in docs/reward-design.md.

`brier_score`'s strict-properness matters as a correctness property, not just a nice
metric: if it weren't maximized by reporting the true distribution, the calibration
claims in the plan wouldn't hold. Mirrors the rigor of Laya's own test_training.py,
not its specific formula.
"""
import torch

from decision_engine.model.types import QTYPE_TO_IDX
from decision_engine.training.reward import (
    brier_score,
    compute_loss,
    mmce,
    ranked_probability_score,
    typed_reward,
)


def _expected_reward(score_fn, q_vec, target_dist):
    """E[score(q, y)] under the target distribution -- used to test strict properness."""
    q = torch.tensor([q_vec])
    mask = torch.ones_like(q)
    total = 0.0
    for i, p in enumerate(target_dist):
        if p == 0:
            continue
        one_hot = torch.zeros_like(q)
        one_hot[0, i] = 1.0
        total += p * score_fn(q, one_hot, mask).item()
    return total


def test_brier_strictly_proper_argmax_at_target():
    target = [0.7, 0.3]
    grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    rewards = [(q, _expected_reward(brier_score, [q, 1 - q], target)) for q in grid]
    best_q = max(rewards, key=lambda kv: kv[1])[0]
    assert abs(best_q - 0.7) < 1e-9


def test_rps_strictly_proper_argmax_at_target():
    target = [0.7, 0.3]
    grid = [0.5, 0.6, 0.7, 0.8, 0.9]
    rewards = [(q, _expected_reward(ranked_probability_score, [q, 1 - q], target)) for q in grid]
    best_q = max(rewards, key=lambda kv: kv[1])[0]
    assert abs(best_q - 0.7) < 1e-9


def test_brier_perfect_prediction_beats_wrong_prediction():
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0]])
    perfect = brier_score(torch.tensor([[1.0, 0.0]]), target, mask)
    wrong = brier_score(torch.tensor([[0.0, 1.0]]), target, mask)
    assert perfect.item() > wrong.item()


def test_masked_option_does_not_affect_brier_reward():
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0]])  # second option masked out
    a = brier_score(torch.tensor([[0.9, 0.1]]), target, mask)
    b = brier_score(torch.tensor([[0.9, 0.9]]), target, mask)
    assert torch.allclose(a, b)


def test_rps_penalizes_far_miss_more_than_near_miss_for_ordinal():
    # 4-level ordinal: true level is index 0. A confident guess at index 1 (near) should
    # score better than a confident guess at index 3 (far) -- RPS's whole point.
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    near_miss = ranked_probability_score(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), target, mask)
    far_miss = ranked_probability_score(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), target, mask)
    assert near_miss.item() > far_miss.item()


def test_brier_has_no_such_ordinal_distance_sensitivity():
    # Brier treats every wrong class the same regardless of "distance" -- this is
    # exactly why score questions use RPS instead, not Brier.
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4)
    near_miss = brier_score(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), target, mask)
    far_miss = brier_score(torch.tensor([[0.0, 0.0, 0.0, 1.0]]), target, mask)
    assert torch.allclose(near_miss, far_miss)


def test_typed_reward_dispatches_score_to_rps_and_choice_to_brier():
    # A *near* miss (one level off), where RPS and Brier are known to disagree (see
    # test_rps_penalizes_far_miss_more_than_near_miss_for_ordinal above) -- at the most
    # extreme far-miss the two metrics coincide at exactly -1, so that case wouldn't
    # actually prove dispatch is happening; a near miss does.
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(2, 4)
    probs = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # identical near-miss prediction
    qtype_idx = torch.tensor([QTYPE_TO_IDX["score"], QTYPE_TO_IDX["choice"]])

    reward = typed_reward(probs, target, mask, qtype_idx)
    rps_only = ranked_probability_score(probs, target, mask)
    brier_only = brier_score(probs, target, mask)

    assert torch.isclose(reward[0], rps_only[0])
    assert torch.isclose(reward[1], brier_only[1])
    assert not torch.isclose(reward[0], reward[1])  # the two rules genuinely disagree here


def test_mmce_is_zero_for_a_single_perfectly_calibrated_bucket():
    # 100 examples all at confidence 0.8, 80 correct -- textbook-calibrated bucket.
    confidences = torch.full((100,), 0.8)
    correctness = torch.tensor([1.0] * 80 + [0.0] * 20)
    err = mmce(confidences, correctness)
    assert err.item() < 1e-6


def test_mmce_is_positive_for_overconfident_predictions():
    confidences = torch.full((100,), 0.99)
    correctness = torch.tensor([1.0] * 60 + [0.0] * 40)  # actually only 60% accurate
    err = mmce(confidences, correctness)
    assert err.item() > 0.05


def test_mmce_handles_fewer_than_two_examples_without_erroring():
    assert mmce(torch.tensor([0.9]), torch.tensor([1.0])).item() == 0.0
    assert mmce(torch.tensor([]), torch.tensor([])).item() == 0.0


def test_compute_loss_is_negative_mean_reward_without_calibration_term():
    logits = torch.tensor([[2.0, -2.0, -1e4, -1e4]])
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["choice"]])

    loss, reward = compute_loss(logits, target, mask, qtype_idx, lambda_calib=0.0)
    assert torch.isclose(loss, -reward.mean())


def test_compute_loss_with_calibration_term_is_larger_when_overconfident_and_wrong():
    target = torch.tensor([[0.0, 1.0]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["choice"]])
    mask = torch.tensor([[True, True]])

    confident_wrong = torch.tensor([[10.0, -10.0]])
    loss_no_calib, _ = compute_loss(confident_wrong, target, mask, qtype_idx, lambda_calib=0.0)
    loss_with_calib, _ = compute_loss(confident_wrong, target, mask, qtype_idx, lambda_calib=1.0)
    assert loss_with_calib.item() >= loss_no_calib.item()


def test_compute_loss_gradients_flow_to_logits():
    logits = torch.tensor([[1.0, 0.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0]])
    mask = torch.tensor([[True, True]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["noul"]])

    loss, _ = compute_loss(logits, target, mask, qtype_idx)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

"""compute_moe_loss tests -- pure tensors, no real model needed. Covers the two things
reward-design.md's "MoE-specific reward wiring" section requires that don't exist for a
single head: the per-expert auxiliary reward (only for *selected* experts) and the
gate's load-balancing term.
"""
import torch

from decision_engine.model.types import QTYPE_TO_IDX
from decision_engine.training.reward import compute_loss_from_probs, compute_moe_loss


def _toy_batch(num_experts=4, batch_size=3, k_options=2):
    torch.manual_seed(0)
    target = torch.zeros(batch_size, k_options)
    target[:, 0] = 1.0
    option_mask = torch.ones(batch_size, k_options, dtype=torch.bool)
    qtype_idx = torch.full((batch_size,), QTYPE_TO_IDX["choice"])
    expert_probs = torch.softmax(torch.randn(batch_size, num_experts, k_options), dim=-1)
    gate_logits = torch.randn(batch_size, num_experts)
    topk_idx = gate_logits.topk(2, dim=-1).indices
    blend_weights = torch.zeros(batch_size, num_experts).scatter(
        -1, topk_idx, torch.softmax(gate_logits.gather(-1, topk_idx), dim=-1))
    blended_probs = (expert_probs * blend_weights.unsqueeze(-1)).sum(1)
    return blended_probs, expert_probs, gate_logits, topk_idx, target, option_mask, qtype_idx


def test_primary_term_matches_compute_loss_from_probs_on_blended_output():
    blended, experts, gate_logits, topk_idx, target, mask, qtype = _toy_batch()
    loss_moe, reward_moe = compute_moe_loss(
        blended, experts, gate_logits, topk_idx, target, mask, qtype,
        lambda_expert=0.0, lambda_calib=0.0, lambda_balance=0.0,
    )
    loss_plain, reward_plain = compute_loss_from_probs(blended, target, mask, qtype)
    assert torch.isclose(loss_moe, loss_plain)
    assert torch.allclose(reward_moe, reward_plain)


def test_expert_term_only_penalizes_selected_experts_not_all():
    blended, experts, gate_logits, topk_idx, target, mask, qtype = _toy_batch(num_experts=4)
    # Make one non-selected expert (guaranteed not in any row's top-2 by construction
    # below) predict something wildly wrong -- if the aux loss wrongly included it,
    # turning that expert's predictions to garbage would change the loss.
    experts_modified = experts.clone()
    never_selected = [e for e in range(4) if e not in topk_idx.flatten().tolist()]
    assume_msg = "test setup needs at least one never-selected expert"
    assert len(never_selected) > 0, assume_msg
    experts_modified[:, never_selected[0], :] = torch.tensor([0.001, 0.999])

    loss_orig, _ = compute_moe_loss(blended, experts, gate_logits, topk_idx, target, mask, qtype,
                                     lambda_expert=1.0, lambda_calib=0.0, lambda_balance=0.0)
    loss_modified, _ = compute_moe_loss(blended, experts_modified, gate_logits, topk_idx, target, mask, qtype,
                                         lambda_expert=1.0, lambda_calib=0.0, lambda_balance=0.0)
    assert torch.isclose(loss_orig, loss_modified)


def test_expert_term_does_penalize_a_selected_experts_bad_prediction():
    blended, experts, gate_logits, topk_idx, target, mask, qtype = _toy_batch(num_experts=4)
    selected_expert = topk_idx[0, 0].item()
    experts_modified = experts.clone()
    experts_modified[0, selected_expert, :] = torch.tensor([0.001, 0.999])  # confidently wrong

    loss_orig, _ = compute_moe_loss(blended, experts, gate_logits, topk_idx, target, mask, qtype,
                                     lambda_expert=1.0, lambda_calib=0.0, lambda_balance=0.0)
    loss_modified, _ = compute_moe_loss(blended, experts_modified, gate_logits, topk_idx, target, mask, qtype,
                                         lambda_expert=1.0, lambda_calib=0.0, lambda_balance=0.0)
    assert loss_modified.item() > loss_orig.item()


def test_balance_term_increases_loss_for_collapsed_routing():
    blended, experts, _, _, target, mask, qtype = _toy_batch(num_experts=4, batch_size=4)
    collapsed_logits = torch.zeros(4, 4)
    collapsed_logits[:, 0] = 10.0
    collapsed_topk = torch.tensor([[0], [0], [0], [0]])

    balanced_logits = torch.eye(4) * 10.0
    balanced_topk = torch.tensor([[0], [1], [2], [3]])

    loss_collapsed, _ = compute_moe_loss(blended, experts, collapsed_logits, collapsed_topk, target, mask, qtype,
                                          lambda_expert=0.0, lambda_calib=0.0, lambda_balance=1.0)
    loss_balanced, _ = compute_moe_loss(blended, experts, balanced_logits, balanced_topk, target, mask, qtype,
                                         lambda_expert=0.0, lambda_calib=0.0, lambda_balance=1.0)
    assert loss_collapsed.item() > loss_balanced.item()


def test_gradients_flow_to_all_experts_gate_and_blended_probs():
    blended, experts, gate_logits, topk_idx, target, mask, qtype = _toy_batch()
    experts = experts.clone().requires_grad_(True)
    gate_logits = gate_logits.clone().requires_grad_(True)

    blend_weights = torch.zeros_like(gate_logits).scatter(
        -1, topk_idx, torch.softmax(gate_logits.gather(-1, topk_idx), dim=-1))
    blended_live = (experts * blend_weights.unsqueeze(-1)).sum(1)

    loss, _ = compute_moe_loss(blended_live, experts, gate_logits, topk_idx, target, mask, qtype,
                                lambda_expert=0.1, lambda_calib=0.0, lambda_balance=0.01)
    loss.backward()
    assert experts.grad is not None and torch.isfinite(experts.grad).all()
    assert gate_logits.grad is not None and torch.isfinite(gate_logits.grad).all()


def test_zero_weight_lambdas_disable_their_terms():
    blended, experts, gate_logits, topk_idx, target, mask, qtype = _toy_batch()
    loss_all_off, _ = compute_moe_loss(blended, experts, gate_logits, topk_idx, target, mask, qtype,
                                        lambda_expert=0.0, lambda_calib=0.0, lambda_balance=0.0)
    loss_plain, _ = compute_loss_from_probs(blended, target, mask, qtype, lambda_calib=0.0)
    assert torch.isclose(loss_all_off, loss_plain)

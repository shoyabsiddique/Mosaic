import torch

from decision_engine.model.gate import Gate, load_balancing_loss


def _gate(hidden_size=8, num_experts=6, top_k=2):
    torch.manual_seed(0)
    return Gate(hidden_size, num_experts, top_k=top_k)


def test_soft_mode_selects_exactly_top_k_experts():
    gate = _gate(top_k=2)
    pooled = torch.randn(5, 8)
    blend_weights, gate_logits, topk_idx = gate(pooled, hard=False)
    assert topk_idx.shape == (5, 2)
    nonzero_per_row = (blend_weights > 0).sum(-1)
    assert (nonzero_per_row == 2).all()


def test_hard_mode_selects_exactly_one_expert():
    gate = _gate(top_k=2)
    pooled = torch.randn(5, 8)
    blend_weights, gate_logits, topk_idx = gate(pooled, hard=True)
    assert topk_idx.shape == (5, 1)
    nonzero_per_row = (blend_weights > 0).sum(-1)
    assert (nonzero_per_row == 1).all()
    assert torch.allclose(blend_weights.sum(-1), torch.ones(5))


def test_blend_weights_sum_to_one_per_example():
    gate = _gate(top_k=3)
    pooled = torch.randn(4, 8)
    blend_weights, _, _ = gate(pooled, hard=False)
    assert torch.allclose(blend_weights.sum(-1), torch.ones(4), atol=1e-6)


def test_selected_experts_match_the_highest_logits():
    gate = _gate(top_k=2)
    pooled = torch.randn(3, 8)
    blend_weights, gate_logits, topk_idx = gate(pooled, hard=False)
    for i in range(3):
        expected_top2 = set(gate_logits[i].topk(2).indices.tolist())
        assert set(topk_idx[i].tolist()) == expected_top2


def test_gate_gradients_flow_to_projection_weights():
    gate = _gate()
    pooled = torch.randn(3, 8, requires_grad=True)
    blend_weights, gate_logits, topk_idx = gate(pooled, hard=False)
    blend_weights.sum().backward()
    assert gate.proj.weight.grad is not None
    assert torch.isfinite(gate.proj.weight.grad).all()


def test_load_balancing_loss_is_lower_for_balanced_than_collapsed_routing():
    num_experts = 4
    # Balanced: logits and top-1 selection spread evenly across a batch of 4.
    balanced_logits = torch.eye(4) * 5.0  # example i strongly prefers expert i
    balanced_topk = torch.tensor([[0], [1], [2], [3]])

    # Collapsed: every example strongly prefers expert 0.
    collapsed_logits = torch.zeros(4, 4)
    collapsed_logits[:, 0] = 5.0
    collapsed_topk = torch.tensor([[0], [0], [0], [0]])

    balanced_loss = load_balancing_loss(balanced_logits, balanced_topk, num_experts)
    collapsed_loss = load_balancing_loss(collapsed_logits, collapsed_topk, num_experts)
    assert balanced_loss.item() < collapsed_loss.item()


def test_load_balancing_loss_gradients_flow_to_logits():
    logits = torch.randn(4, 4, requires_grad=True)
    topk_idx = logits.detach().topk(1, dim=-1).indices
    loss = load_balancing_loss(logits, topk_idx, num_experts=4)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()

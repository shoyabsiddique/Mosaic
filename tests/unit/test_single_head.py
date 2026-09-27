"""SingleHead tests operate on hand-built tensors -- no encoder or tokenizer needed,
since the head only consumes already-encoded hidden states (docs/expert-head-design.md)."""
import torch

from decision_engine.model.single_head import SingleHead


def _head(hidden_size=16, num_heads=4):
    torch.manual_seed(0)
    return SingleHead(hidden_size, num_heads=num_heads, dropout=0.0)


def test_output_shape_is_batch_by_k():
    head = _head()
    option_queries = torch.randn(3, 5, 16)
    option_mask = torch.ones(3, 5, dtype=torch.bool)
    state_hidden = torch.randn(3, 8, 16)
    state_mask = torch.ones(3, 8, dtype=torch.long)

    logits = head(option_queries, option_mask, state_hidden, state_mask)
    assert logits.shape == (3, 5)


def test_masked_option_positions_are_set_to_large_negative():
    head = _head()
    option_queries = torch.randn(1, 4, 16)
    option_mask = torch.tensor([[True, True, False, False]])
    state_hidden = torch.randn(1, 6, 16)
    state_mask = torch.ones(1, 6, dtype=torch.long)

    logits = head(option_queries, option_mask, state_hidden, state_mask)
    assert (logits[0, 2:] <= -1e4).all()
    assert (logits[0, :2] > -1e4).all()


def test_masked_option_positions_softmax_to_zero_probability():
    head = _head()
    option_queries = torch.randn(1, 4, 16)
    option_mask = torch.tensor([[True, True, True, False]])
    state_hidden = torch.randn(1, 6, 16)
    state_mask = torch.ones(1, 6, dtype=torch.long)

    logits = head(option_queries, option_mask, state_hidden, state_mask)
    probs = torch.softmax(logits, dim=-1)
    assert probs[0, 3].item() < 1e-6
    assert abs(probs[0].sum().item() - 1.0) < 1e-5


def test_padded_state_tokens_do_not_affect_output():
    head = _head()
    torch.manual_seed(1)
    option_queries = torch.randn(1, 2, 16)
    option_mask = torch.ones(1, 2, dtype=torch.bool)
    state_hidden = torch.randn(1, 6, 16)

    full_mask = torch.ones(1, 6, dtype=torch.long)
    out_full = head(option_queries.clone(), option_mask, state_hidden.clone(), full_mask)

    padded_hidden = state_hidden.clone()
    padded_hidden[0, 4:] = 999.0  # garbage in the padded region
    padded_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    out_padded = head(option_queries.clone(), option_mask, padded_hidden, padded_mask)

    # only the first 4 (real) state tokens should influence the result in both cases
    truncated_hidden = state_hidden.clone()[:, :4]
    truncated_mask = torch.ones(1, 4, dtype=torch.long)
    out_truncated = head(option_queries.clone(), option_mask, truncated_hidden, truncated_mask)
    assert not torch.allclose(out_full, out_padded)  # different state lengths, expected to differ
    assert torch.allclose(out_padded, out_truncated, atol=1e-5)  # garbage padding correctly ignored


def test_gradients_flow_to_scorer_and_attention_weights():
    head = _head()
    option_queries = torch.randn(2, 3, 16, requires_grad=True)
    option_mask = torch.ones(2, 3, dtype=torch.bool)
    state_hidden = torch.randn(2, 5, 16, requires_grad=True)
    state_mask = torch.ones(2, 5, dtype=torch.long)

    logits = head(option_queries, option_mask, state_hidden, state_mask)
    logits.sum().backward()

    assert option_queries.grad is not None and torch.isfinite(option_queries.grad).all()
    assert state_hidden.grad is not None and torch.isfinite(state_hidden.grad).all()
    assert head.scorer[1].weight.grad is not None

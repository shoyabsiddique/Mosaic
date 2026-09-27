"""OptionEncoder tests use a fake encoder that returns input_ids as hidden values
(broadcast to hidden_size), so pooling behavior is verifiable by hand -- no real
NeoBERT weights needed."""
import torch
import torch.nn as nn

from decision_engine.model.option_encoder import OptionEncoder
from decision_engine.model.types import QTYPE_TO_IDX


class FakeEncoder(nn.Module):
    """Returns a hidden state per token equal to its own id (broadcast across the
    hidden dimension), so mean-pooling over real tokens has a hand-computable answer."""

    def forward(self, input_ids, attention_mask):
        return input_ids.unsqueeze(-1).float().expand(-1, -1, 4)  # hidden_size=4


def test_pooling_averages_only_real_tokens_not_padding():
    encoder = OptionEncoder(FakeEncoder(), hidden_size=4)
    with torch.no_grad():
        encoder.type_embedding.weight.zero_()  # isolate the pooling behavior

    input_ids = torch.tensor([[10, 20, 0, 0]])  # last two are padding
    attention_mask = torch.tensor([[1, 1, 0, 0]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["choice"]])

    out = encoder(input_ids, attention_mask, qtype_idx)
    assert torch.allclose(out, torch.full((1, 4), 15.0))  # mean(10, 20) == 15, padding excluded


def test_padding_included_would_give_a_different_wrong_answer():
    # Sanity check on the test itself: if padding were wrongly included in the mean,
    # the result would be 7.5, not 15 -- confirms the assertion above is meaningful.
    input_ids = torch.tensor([10, 20, 0, 0])
    assert input_ids.float().mean().item() == 7.5


def test_type_embedding_is_added_and_differs_by_qtype():
    encoder = OptionEncoder(FakeEncoder(), hidden_size=4)
    input_ids = torch.tensor([[5, 5]])
    attention_mask = torch.tensor([[1, 1]])

    out_choice = encoder(input_ids, attention_mask, torch.tensor([QTYPE_TO_IDX["choice"]]))
    out_score = encoder(input_ids, attention_mask, torch.tensor([QTYPE_TO_IDX["score"]]))
    assert not torch.allclose(out_choice, out_score)


def test_same_qtype_gives_same_embedding_offset():
    encoder = OptionEncoder(FakeEncoder(), hidden_size=4)
    input_ids = torch.tensor([[5, 5], [8, 8]])
    attention_mask = torch.tensor([[1, 1], [1, 1]])
    qtype_idx = torch.tensor([QTYPE_TO_IDX["noul"], QTYPE_TO_IDX["noul"]])

    out = encoder(input_ids, attention_mask, qtype_idx)
    diff = out[1] - out[0]  # pooled values differ (5 vs 8) but the type offset should cancel out
    assert torch.allclose(diff, torch.full((4,), 3.0))


def test_output_shape_is_n_by_hidden_size():
    encoder = OptionEncoder(FakeEncoder(), hidden_size=4)
    input_ids = torch.randint(1, 50, (7, 6))
    attention_mask = torch.ones(7, 6, dtype=torch.long)
    qtype_idx = torch.zeros(7, dtype=torch.long)

    out = encoder(input_ids, attention_mask, qtype_idx)
    assert out.shape == (7, 4)


def test_all_padding_row_does_not_produce_nan_or_inf():
    encoder = OptionEncoder(FakeEncoder(), hidden_size=4)
    input_ids = torch.tensor([[0, 0]])
    attention_mask = torch.tensor([[0, 0]])  # clamp(min=1.0) in the encoder guards this
    qtype_idx = torch.tensor([QTYPE_TO_IDX["choice"]])

    out = encoder(input_ids, attention_mask, qtype_idx)
    assert torch.isfinite(out).all()

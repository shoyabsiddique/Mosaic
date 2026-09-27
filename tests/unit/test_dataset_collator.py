"""Collator tests use a minimal stand-in tokenizer (no network, no real model) --
same approach Laya's own test_training.py uses for build_sequence.
"""
import torch

from decision_engine.training.dataset import Collator, _serialize_state


class FakeTokenizer:
    """One id per character; supports both single-text and text-pair batched calls."""

    def __call__(self, text_a, text_b=None, padding=True, truncation=True, max_length=64, return_tensors="pt"):
        if isinstance(text_a, str):
            text_a = [text_a]
        if text_b is not None and isinstance(text_b, str):
            text_b = [text_b]

        sequences = []
        for i, a in enumerate(text_a):
            combined = a if text_b is None else f"{a}|{text_b[i]}"
            ids = [ord(c) % 90 + 10 for c in combined][:max_length]
            sequences.append(ids if ids else [1])  # never fully empty

        max_len = max(len(s) for s in sequences)
        input_ids = torch.zeros(len(sequences), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(sequences), max_len, dtype=torch.long)
        for i, s in enumerate(sequences):
            input_ids[i, :len(s)] = torch.tensor(s)
            attention_mask[i, :len(s)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


CHOICE_EX = {
    "state": "The customer wants a refund.", "domain": "customer-support",
    "question_id": "q1", "question_type": "choice", "instructions": "Which department?",
    "criteria": {"billing": "invoices", "technical": "bugs"},
    "target_distribution": [1.0, 0.0], "target_label_index": 0,
}
SCORE_EX = {
    "state": {"issue": "slow response"}, "domain": "security-ops",
    "question_id": "q2", "question_type": "score", "instructions": "How severe?",
    "criteria": ["low", "medium", "high", "critical"],
    "target_distribution": [0.0, 0.5, 0.5, 0.0], "target_label_index": None,
}
NOUL_EX = {
    "state": "spam message", "domain": "email-communication",
    "question_id": "q3", "question_type": "noul", "instructions": "Is this spam?",
    "criteria": None, "target_distribution": [0.2, 0.8], "target_label_index": 1,
}


def test_serialize_state_passes_through_strings_and_dumps_dicts():
    assert _serialize_state("hello") == "hello"
    assert _serialize_state({"a": 1}) == '{"a": 1}'


def test_batch_pads_options_to_the_widest_question_in_the_batch():
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX])  # 2 options vs 4 options
    assert batch["option_mask"].shape == (2, 4)
    assert batch["option_mask"][0].tolist() == [True, True, False, False]
    assert batch["option_mask"][1].tolist() == [True, True, True, True]


def test_target_is_zero_padded_beyond_real_options():
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX])
    assert batch["target"][0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert batch["target"][1].tolist() == [0.0, 0.5, 0.5, 0.0]


def test_qtype_idx_matches_question_types_in_order():
    from decision_engine.model.types import QTYPE_TO_IDX
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX, NOUL_EX])
    expected = [QTYPE_TO_IDX["choice"], QTYPE_TO_IDX["score"], QTYPE_TO_IDX["noul"]]
    assert batch["qtype_idx"].tolist() == expected


def test_option_shapes_are_batch_by_k_by_seqlen():
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX])
    b, k, _ = batch["option_input_ids"].shape
    assert (b, k) == (2, 4)
    assert batch["option_attention_mask"].shape == batch["option_input_ids"].shape


def test_state_batch_shape_matches_batch_size():
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX, NOUL_EX])
    assert batch["state_input_ids"].shape[0] == 3
    assert batch["state_attention_mask"].shape == batch["state_input_ids"].shape


def test_dict_state_is_serialized_before_tokenization_not_passed_as_object():
    # SCORE_EX has a dict state -- this only works if _serialize_state ran first,
    # since FakeTokenizer assumes string input.
    collate = Collator(FakeTokenizer())
    batch = collate([SCORE_EX])
    assert batch["state_input_ids"].shape[0] == 1


def test_domain_metadata_passes_through_in_order():
    collate = Collator(FakeTokenizer())
    batch = collate([CHOICE_EX, SCORE_EX, NOUL_EX])
    assert batch["domain"] == ["customer-support", "security-ops", "email-communication"]

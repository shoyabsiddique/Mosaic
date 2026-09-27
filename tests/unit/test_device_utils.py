import torch

from decision_engine.training.device_utils import move_batch_to_device, resolve_device


def test_resolve_device_honors_explicit_override():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_detects_when_none():
    # This machine has no CUDA -- auto-detect must fall back to cpu, not error.
    resolved = resolve_device(None)
    assert resolved == torch.device("cuda") if torch.cuda.is_available() else resolved == torch.device("cpu")


def test_move_batch_to_device_moves_tensors():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([[1.0, 2.0]])}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["a"].device == torch.device("cpu")
    assert torch.equal(moved["a"], batch["a"])


def test_move_batch_to_device_leaves_non_tensor_metadata_untouched():
    batch = {"a": torch.tensor([1, 2]), "domain": ["customer-support", "ecommerce"]}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved["domain"] == ["customer-support", "ecommerce"]


def test_move_batch_to_device_does_not_mutate_original_dict():
    batch = {"a": torch.tensor([1, 2])}
    moved = move_batch_to_device(batch, torch.device("cpu"))
    assert moved is not batch

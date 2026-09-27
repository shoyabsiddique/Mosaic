"""Reads the JSONL records written by data/loaders and data/synth, and collates them
into padded batches for BaselineModel. Each JSONL line is already a single (state,
question, target) example per docs/code-structure.md's schema -- Phase 2 batches
across examples independently (re-encoding state per example); the "encode the state
once, reuse across every question in a request" optimization from
docs/expert-head-design.md is a serving-time concern (Phase 6), not a training
correctness requirement.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from decision_engine.model.rendering import render_option_texts
from decision_engine.model.types import QTYPE_TO_IDX


def _serialize_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


class TypedDecisionDataset(Dataset):
    """Reads every `data/<domain>/*.jsonl` file. When `max_examples` truncates the
    total, examples are taken round-robin across files rather than in path order --
    domain folders sort alphabetically, so a naive prefix-truncation silently returned
    only "customer-support" examples for any small `max_examples` cap (caught when a
    Phase 3 MoE smoke test's routing diagnostic showed every example from the same
    domain -- not a model bug, a sampling one). Round-robining guarantees a small cap
    still samples from every available domain.
    """

    def __init__(self, data_dir: Path, max_examples: Optional[int] = None):
        per_file: list[list[dict]] = []
        for path in sorted(Path(data_dir).glob("*/*.jsonl")):
            with open(path, encoding="utf-8") as f:
                per_file.append([json.loads(line) for line in f])

        if max_examples is None:
            self.examples = [ex for file_examples in per_file for ex in file_examples]
            return

        self.examples = []
        row = 0
        while len(self.examples) < max_examples and any(row < len(fe) for fe in per_file):
            for file_examples in per_file:
                if row < len(file_examples):
                    self.examples.append(file_examples[row])
                    if len(self.examples) >= max_examples:
                        break
            row += 1

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class ListDataset(Dataset):
    """Thin wrapper so a plain list of example dicts (e.g. one half of a train/val
    split) can be used with a DataLoader + Collator the same way TypedDecisionDataset is."""

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def train_val_split(dataset: TypedDecisionDataset, val_fraction: float = 0.2,
                     seed: int = 0) -> tuple[ListDataset, ListDataset]:
    """Random split, not stratified by domain -- fine for a smoke-level evaluation
    check; a real training run would want to stratify by domain and question type."""
    examples = list(dataset.examples)
    random.Random(seed).shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    return ListDataset(examples[n_val:]), ListDataset(examples[:n_val])


class Collator:
    def __init__(self, tokenizer, max_state_len: int = 256, max_option_len: int = 64):
        self.tokenizer = tokenizer
        self.max_state_len = max_state_len
        self.max_option_len = max_option_len

    def __call__(self, batch: list[dict]) -> dict:
        state_texts = [_serialize_state(ex["state"]) for ex in batch]
        state_enc = self.tokenizer(state_texts, padding=True, truncation=True,
                                    max_length=self.max_state_len, return_tensors="pt")

        option_texts_per_ex = [render_option_texts(ex["question_type"], ex["criteria"]) for ex in batch]
        k_max = max(len(o) for o in option_texts_per_ex)

        flat_instructions: list[str] = []
        flat_options: list[str] = []
        option_mask = torch.zeros(len(batch), k_max, dtype=torch.bool)
        for i, (ex, opts) in enumerate(zip(batch, option_texts_per_ex)):
            for j in range(k_max):
                if j < len(opts):
                    flat_instructions.append(ex["instructions"])
                    flat_options.append(opts[j])
                    option_mask[i, j] = True
                else:
                    flat_instructions.append("")
                    flat_options.append("")

        option_enc = self.tokenizer(flat_instructions, flat_options, padding=True, truncation=True,
                                     max_length=self.max_option_len, return_tensors="pt")
        l_opt = option_enc["input_ids"].shape[1]
        option_input_ids = option_enc["input_ids"].view(len(batch), k_max, l_opt)
        option_attention_mask = option_enc["attention_mask"].view(len(batch), k_max, l_opt)

        target = torch.zeros(len(batch), k_max)
        for i, ex in enumerate(batch):
            dist = ex["target_distribution"]
            target[i, :len(dist)] = torch.tensor(dist)

        qtype_idx = torch.tensor([QTYPE_TO_IDX[ex["question_type"]] for ex in batch])

        return {
            "state_input_ids": state_enc["input_ids"],
            "state_attention_mask": state_enc["attention_mask"],
            "option_input_ids": option_input_ids,
            "option_attention_mask": option_attention_mask,
            "option_mask": option_mask,
            "target": target,
            "qtype_idx": qtype_idx,
            "domain": [ex["domain"] for ex in batch],  # metadata only, not model input --
                                                        # used to check gate/domain alignment
        }

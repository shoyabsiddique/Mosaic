"""Unified record schema every domain loader must produce.

One Record = one typed question asked about one state. A single source row (e.g. one
support ticket with a department AND a priority label) becomes multiple Records, one
per question -- this keeps the training pipeline agnostic to how many questions a given
state happens to have, matching how the model is actually queried at inference time
(see docs/expert-head-design.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Union


class QuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True)
class Question:
    """Mirrors the Jev/Laya-style typed-question request shape.

    `criteria` is a dict of {option_label: description} for `choice`, an ordered list
    of level descriptions for `score`, or None for `noul` (always a forced yes/no).
    """
    type: QuestionType
    instructions: str
    criteria: Union[dict[str, str], list[str], None] = None

    def option_labels(self) -> list[str]:
        if self.type is QuestionType.CHOICE:
            assert isinstance(self.criteria, dict), "choice question needs dict criteria"
            return list(self.criteria.keys())
        if self.type is QuestionType.SCORE:
            assert isinstance(self.criteria, list), "score question needs list criteria"
            return [str(i) for i in range(len(self.criteria))]
        return ["false", "true"]


@dataclass(frozen=True)
class TypedTarget:
    """Target distribution over the question's options, aligned to `option_labels()` order.

    Soft (non-one-hot) distributions are supported directly -- several verified
    datasets ship calibrated targets natively (e.g. Civil Comments' continuous 0-1
    toxicity scores, Davidson's annotator-count fractions) and those should flow
    through as soft targets, not be collapsed to a hard label and lose that signal.
    """
    distribution: tuple[float, ...]
    label_index: int | None = None

    def __post_init__(self) -> None:
        total = sum(self.distribution)
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"target distribution must sum to ~1.0, got {total}")


@dataclass(frozen=True)
class Record:
    """One training example: a state, one typed question about it, and its target."""
    state: Union[str, dict]
    domain: str
    question_id: str
    question: Question
    target: TypedTarget
    source_dataset: str
    license: str
    split: str = "train"

    def __post_init__(self) -> None:
        if self.split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {self.split!r}")
        n_options = len(self.question.option_labels())
        if len(self.target.distribution) != n_options:
            raise ValueError(
                f"target has {len(self.target.distribution)} entries but question "
                f"{self.question_id!r} has {n_options} options"
            )

"""Turns a Question's criteria into plain option text -- see docs/expert-head-design.md.

Each option's text is combined with the question's instructions at query-construction
time (option_encoder.py encodes them as a sentence pair), so a bare option label like
"low" or "yes" is still question-aware once paired with its instructions. This module
only produces the option-side text, not the combined query.
"""
from __future__ import annotations

from typing import Optional, Union


def render_option_texts(qtype: str, criteria: Optional[Union[dict, list]]) -> list[str]:
    if qtype == "choice":
        assert isinstance(criteria, dict), "choice question needs dict criteria"
        return [f"{label}: {desc}" if desc else label for label, desc in criteria.items()]
    if qtype == "score":
        assert isinstance(criteria, list), "score question needs list criteria"
        return list(criteria)
    if qtype == "noul":
        return ["no", "yes"]
    raise ValueError(f"unknown question type: {qtype!r}")

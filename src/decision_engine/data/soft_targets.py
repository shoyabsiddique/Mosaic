"""Shared helper for building soft ordinal targets from a continuous value.

Used wherever a level is derived from an underlying continuous signal (real, as in
trust_safety.py's toxicity score, or synthetic, as in the generators in data/synth/):
linear interpolation between the two nearest levels gives a genuinely soft target
instead of collapsing to a one-hot label at an arbitrary bin boundary.
"""
from __future__ import annotations


def interpolated_ordinal_target(value: float, n_levels: int, value_range: tuple[float, float] = (0.0, 1.0)) -> tuple[float, ...]:
    """`value` in `value_range` -> a soft distribution over `n_levels` ordinal levels."""
    lo_range, hi_range = value_range
    normalized = max(0.0, min(1.0, (value - lo_range) / (hi_range - lo_range)))
    pos = normalized * (n_levels - 1)
    lo, hi = int(pos), min(int(pos) + 1, n_levels - 1)
    frac = pos - lo
    dist = [0.0] * n_levels
    if lo == hi:
        dist[lo] = 1.0
    else:
        dist[lo] = 1.0 - frac
        dist[hi] = frac
    return tuple(dist)

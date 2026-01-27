"""Typed evaluator for CrossCondition (Phase 7)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vibe_trade_shared.models.ir import CrossCondition

from .context import EvalContext


def _cross_key(condition: CrossCondition) -> str:
    """Stable key for cross_state so same condition reuses previous (left, right)."""
    payload = condition.model_dump(mode="json")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def evaluate_cross(condition: CrossCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate crossover: left crosses above or below right."""
    left_val = ctx.resolve_value(condition.left, bar)
    right_val = ctx.resolve_value(condition.right, bar)

    key = _cross_key(condition)
    prev = ctx.cross_state.get(key)
    ctx.cross_state[key] = (left_val, right_val)

    if prev is None:
        return False

    prev_left, prev_right = prev
    if condition.direction == "above":
        return prev_left <= prev_right and left_val > right_val
    else:  # "below"
        return prev_left >= prev_right and left_val < right_val

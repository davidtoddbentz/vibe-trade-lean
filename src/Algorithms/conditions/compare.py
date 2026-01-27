"""Typed evaluator for CompareCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import CompareCondition

from .context import EvalContext


def evaluate_compare(condition: CompareCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate left op right using typed condition and context."""
    left_val = ctx.resolve_value(condition.left, bar)
    right_val = ctx.resolve_value(condition.right, bar)
    return condition.op.apply(left_val, right_val)

"""Typed evaluators for basic logic conditions: allOf, anyOf, not (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import AllOfCondition, AnyOfCondition, NotCondition

from .context import EvalContext


def evaluate_allof(condition: AllOfCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate allOf (AND): all sub-conditions must be true."""
    for sub in condition.conditions:
        if not ctx.evaluate_condition(sub, bar):
            return False
    return True


def evaluate_anyof(condition: AnyOfCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate anyOf (OR): at least one sub-condition must be true."""
    for sub in condition.conditions:
        if ctx.evaluate_condition(sub, bar):
            return True
    return False


def evaluate_not(condition: NotCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate not: negate the inner condition."""
    return not ctx.evaluate_condition(condition.condition, bar)

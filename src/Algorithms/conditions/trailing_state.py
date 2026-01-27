"""Typed evaluator for TrailingStateCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import TrailingStateCondition

from .context import EvalContext


def _price_val(bar: Any, field: str) -> float:
    if field == "high":
        return bar.High
    if field == "low":
        return bar.Low
    if field == "open":
        return bar.Open
    return bar.Close


def evaluate_trailing_state(
    condition: TrailingStateCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate trailing state: trail price, trigger when price crosses level ± ATR."""
    update_val = _price_val(bar, condition.update_price)
    trigger_val = _price_val(bar, condition.trigger_price)

    atr_ind = (
        ctx.indicators.get(f"atr_{condition.atr_period}")
        or ctx.indicators.get("atr")
    )
    atr_value = atr_ind.Current.Value if atr_ind else 0.0

    trailing_val = ctx.state.get(condition.state_id)
    if trailing_val is None:
        trailing_val = update_val
        ctx.state[condition.state_id] = trailing_val
    else:
        if condition.update_rule == "max":
            trailing_val = max(trailing_val, update_val)
        else:
            trailing_val = min(trailing_val, update_val)
        ctx.state[condition.state_id] = trailing_val

    if condition.trigger_op == "below":
        trigger_level = trailing_val - (condition.atr_mult * atr_value)
        return trigger_val < trigger_level
    trigger_level = trailing_val + (condition.atr_mult * atr_value)
    return trigger_val > trigger_level

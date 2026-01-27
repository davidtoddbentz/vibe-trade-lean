"""Typed evaluator for BreakoutCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import BreakoutCondition

from .context import EvalContext


def evaluate_breakout(
    condition: BreakoutCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate N-bar high/low breakout.

    A breakout occurs when close exceeds the PREVIOUS N-bar high/low.
    Uses ctx.indicators and ctx.breakout_prev_max/Min; updates those dicts.
    """
    max_id = condition.max_indicator or f"max_{condition.lookback_bars}"
    min_id = condition.min_indicator or f"min_{condition.lookback_bars}"

    max_ind = ctx.indicators.get(max_id) or ctx.indicators.get("max_50")
    min_ind = ctx.indicators.get(min_id) or ctx.indicators.get("min_50")

    if not max_ind or not min_ind:
        return False

    current_max = max_ind.Current.Value
    current_min = min_ind.Current.Value

    prev_max = ctx.breakout_prev_max.get(max_id, current_max)
    prev_min = ctx.breakout_prev_min.get(min_id, current_min)

    ctx.breakout_prev_max[max_id] = current_max
    ctx.breakout_prev_min[min_id] = current_min

    buffer_mult = 1 + (condition.buffer_bps / 10000)
    high_level = prev_max * buffer_mult
    low_level = prev_min / buffer_mult

    is_high_breakout = bar.Close > high_level
    is_low_breakout = bar.Close < low_level

    return is_high_breakout or is_low_breakout

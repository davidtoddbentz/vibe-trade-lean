"""Typed evaluator for TrailingBreakoutCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import TrailingBreakoutCondition

from .context import EvalContext


def evaluate_trailing_breakout(
    condition: TrailingBreakoutCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate trailing breakout: trail band edge, trigger on price cross."""
    band_prefix_map = {"bollinger": "bb", "keltner": "kc", "donchian": "dc"}
    band_prefix = band_prefix_map.get(condition.band_type, condition.band_type)
    band_id = f"{band_prefix}_{condition.band_length}"
    band_ind = (
        ctx.indicators.get(band_id)
        or ctx.indicators.get(band_prefix)
        or ctx.indicators.get("bb")
    )

    if not band_ind:
        return False

    if condition.band_edge == "upper":
        band_value = band_ind.UpperBand.Current.Value
    elif condition.band_edge == "lower":
        band_value = band_ind.LowerBand.Current.Value
    else:
        band_value = band_ind.MiddleBand.Current.Value

    state_key = f"trailing_breakout_{condition.band_type}_{condition.band_length}"
    trailing_level = ctx.state.get(state_key)

    if trailing_level is None:
        trailing_level = band_value
        ctx.state[state_key] = trailing_level

    if condition.update_rule == "min":
        new_level = min(trailing_level, band_value)
    else:
        new_level = max(trailing_level, band_value)
    ctx.state[state_key] = new_level

    if condition.trigger_direction == "above":
        return bar.Close > new_level
    return bar.Close < new_level

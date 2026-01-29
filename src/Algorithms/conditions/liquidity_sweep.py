"""Typed evaluator for LiquiditySweepCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import LiquiditySweepCondition

from .context import EvalContext


def evaluate_liquidity_sweep(
    condition: LiquiditySweepCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate liquidity sweep: break below level then reclaim within lookback_bars."""
    if condition.level_type == "rolling_min":
        min_ind = ctx.indicators.get(f"min_{condition.level_period}")
        if not min_ind or not min_ind.IsReady:
            return False
        level_value = min_ind.Current.Value
    elif condition.level_type == "rolling_max":
        max_ind = ctx.indicators.get(f"max_{condition.level_period}")
        if not max_ind or not max_ind.IsReady:
            return False
        level_value = max_ind.Current.Value
    else:
        level_value = getattr(condition, "fixed_level", None)
        if level_value is None:
            return False

    state_key = f"liquidity_sweep_{condition.level_period}"
    sweep_triggered = ctx.state.get(f"{state_key}_triggered", False)
    sweep_bar_count = ctx.state.get(f"{state_key}_bars", 0)

    if not sweep_triggered:
        if bar.Low < level_value:
            ctx.state[f"{state_key}_triggered"] = True
            ctx.state[f"{state_key}_bars"] = 0
            ctx.state[f"{state_key}_level"] = level_value
        return False

    sweep_bar_count += 1
    ctx.state[f"{state_key}_bars"] = sweep_bar_count

    if sweep_bar_count > condition.lookback_bars:
        ctx.state[f"{state_key}_triggered"] = False
        return False

    sweep_level = ctx.state.get(f"{state_key}_level", level_value)
    if bar.Close > sweep_level:
        ctx.state[f"{state_key}_triggered"] = False
        return True

    return False

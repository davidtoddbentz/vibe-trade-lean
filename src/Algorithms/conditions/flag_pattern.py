"""Typed evaluator for FlagPatternCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import FlagPatternCondition

from .context import EvalContext


def evaluate_flag_pattern(
    condition: FlagPatternCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate flag pattern: momentum pole -> consolidation -> breakout."""
    state_key = "flag_pattern"
    phase = ctx.state.get(f"{state_key}_phase", "scanning")
    consol_count = ctx.state.get(f"{state_key}_consol_count", 0)

    roc_id = f"roc_{condition.momentum_period}"
    roc_ind = ctx.indicators.get(roc_id) or ctx.indicators.get("momentum_roc")
    if not roc_ind:
        return False

    current_roc = roc_ind.Current.Value * 100

    if phase == "scanning":
        if abs(current_roc) >= condition.momentum_threshold:
            ctx.state[f"{state_key}_phase"] = "consolidating"
            ctx.state[f"{state_key}_direction"] = 1 if current_roc > 0 else -1
            ctx.state[f"{state_key}_consol_count"] = 0
            ctx.state[f"{state_key}_high"] = bar.High
            ctx.state[f"{state_key}_low"] = bar.Low
        return False

    if phase == "consolidating":
        prev_high = ctx.state.get(f"{state_key}_high", bar.High)
        prev_low = ctx.state.get(f"{state_key}_low", bar.Low)
        ctx.state[f"{state_key}_high"] = max(prev_high, bar.High)
        ctx.state[f"{state_key}_low"] = min(prev_low, bar.Low)
        consol_count += 1
        ctx.state[f"{state_key}_consol_count"] = consol_count

        if consol_count < condition.consolidation_bars:
            return False

        pole_dir = ctx.state.get(f"{state_key}_direction", 1)
        expected_dir = pole_dir if condition.breakout_direction == "same" else -pole_dir

        if expected_dir > 0 and bar.Close > prev_high:
            ctx.state[f"{state_key}_phase"] = "scanning"
            return True
        if expected_dir < 0 and bar.Close < prev_low:
            ctx.state[f"{state_key}_phase"] = "scanning"
            return True

        if abs(current_roc) >= condition.momentum_threshold * 0.5:
            if (current_roc > 0 and pole_dir < 0) or (current_roc < 0 and pole_dir > 0):
                ctx.state[f"{state_key}_phase"] = "scanning"

    return False

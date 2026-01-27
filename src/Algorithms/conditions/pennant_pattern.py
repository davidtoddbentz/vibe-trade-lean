"""Typed evaluator for PennantPatternCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import PennantPatternCondition

from .context import EvalContext


def evaluate_pennant_pattern(
    condition: PennantPatternCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate pennant pattern: momentum -> converging consolidation -> breakout."""
    state_key = "pennant_pattern"
    phase = ctx.state.get(f"{state_key}_phase", "scanning")

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
            ctx.state[f"{state_key}_highs"] = [bar.High]
            ctx.state[f"{state_key}_lows"] = [bar.Low]
        return False

    if phase == "consolidating":
        highs = ctx.state.get(f"{state_key}_highs", [])
        lows = ctx.state.get(f"{state_key}_lows", [])
        highs.append(bar.High)
        lows.append(bar.Low)
        ctx.state[f"{state_key}_highs"] = highs
        ctx.state[f"{state_key}_lows"] = lows
        ctx.state[f"{state_key}_consol_count"] = len(highs)

        consol_count = len(highs)
        if consol_count < condition.consolidation_bars:
            return False

        if len(highs) >= 3:
            is_converging = (highs[-1] < highs[0]) and (lows[-1] > lows[0])
            if is_converging:
                pole_dir = ctx.state.get(f"{state_key}_direction", 1)
                expected_dir = pole_dir if condition.breakout_direction == "same" else -pole_dir
                recent_high = max(highs[-3:])
                recent_low = min(lows[-3:])
                if expected_dir > 0 and bar.Close > recent_high:
                    ctx.state[f"{state_key}_phase"] = "scanning"
                    return True
                if expected_dir < 0 and bar.Close < recent_low:
                    ctx.state[f"{state_key}_phase"] = "scanning"
                    return True

        if consol_count > condition.consolidation_bars * 3:
            ctx.state[f"{state_key}_phase"] = "scanning"

    return False

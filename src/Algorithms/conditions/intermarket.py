"""Typed evaluator for IntermarketCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import IntermarketCondition

from .context import EvalContext


def evaluate_intermarket(
    condition: IntermarketCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate intermarket condition (leader ROC vs threshold)."""
    if condition.trigger_feature != "ret_pct":
        return False

    roc_ind = (
        ctx.indicators.get("roc")
        or ctx.indicators.get(f"roc_{condition.window_bars}")
    )

    if roc_ind:
        roc_value = roc_ind.Current.Value * 100
    else:
        close_rw = ctx.rolling_windows.get("close") or ctx.rolling_windows.get("prev_close")
        if close_rw and getattr(close_rw.get("window"), "IsReady", False):
            window = close_rw["window"]
            closes = list(window) if hasattr(window, "__iter__") else []
            if len(closes) > condition.window_bars:
                old_close = closes[condition.window_bars]
                if old_close != 0:
                    roc_value = ((bar.Close - old_close) / old_close) * 100
                else:
                    return False
            else:
                return False
        else:
            state_key = f"intermarket_closes_{condition.leader_symbol}"
            closes_list = ctx.state.get(state_key, [])
            closes_list.append(bar.Close)
            if len(closes_list) > condition.window_bars + 1:
                closes_list = closes_list[-(condition.window_bars + 1) :]
            ctx.state[state_key] = closes_list
            if len(closes_list) > condition.window_bars:
                old_close = closes_list[0]
                if old_close != 0:
                    roc_value = ((bar.Close - old_close) / old_close) * 100
                else:
                    return False
            else:
                return False

    # Condition detects significant leader movement.
    # "same"/"opposite" direction is an action-level concern (which way to trade),
    # not a condition-level concern (whether the signal fires).
    return abs(roc_value) > condition.trigger_threshold

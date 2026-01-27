"""Typed evaluator for GapCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import GapCondition

from .context import EvalContext


def evaluate_gap(condition: GapCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate gap condition using prev_close from rolling_windows."""
    prev_close_rw = ctx.rolling_windows.get("prev_close")
    if not prev_close_rw or not getattr(
        prev_close_rw.get("window"), "IsReady", False
    ):
        return False

    window = prev_close_rw["window"]
    prev_close = window[1] if hasattr(window, "__getitem__") else None
    if prev_close is None or prev_close == 0:
        return False

    gap_pct = (bar.Open - prev_close) / prev_close * 100

    if abs(gap_pct) < condition.min_gap_pct:
        return False

    gap_is_up = gap_pct > 0

    if condition.mode == "gap_fade":
        if condition.direction == "long":
            return not gap_is_up
        if condition.direction == "short":
            return gap_is_up
        return True
    # gap_go
    if condition.direction == "long":
        return gap_is_up
    if condition.direction == "short":
        return not gap_is_up
    return True

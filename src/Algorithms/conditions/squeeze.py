"""Typed evaluator for SqueezeCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import SqueezeCondition

from .context import EvalContext


def evaluate_squeeze(condition: SqueezeCondition, bar: Any, ctx: EvalContext) -> bool:
    """Evaluate squeeze condition (BB inside KC, width percentile)."""
    bb_ind = ctx.indicators.get("bb") or ctx.indicators.get("bb_20")
    kc_ind = ctx.indicators.get("kc") or ctx.indicators.get("kc_20")

    if not bb_ind:
        return False

    if kc_ind:
        bb_upper = bb_ind.UpperBand.Current.Value
        bb_lower = bb_ind.LowerBand.Current.Value
        kc_upper = kc_ind.UpperBand.Current.Value
        kc_lower = kc_ind.LowerBand.Current.Value
        in_squeeze = bb_lower > kc_lower and bb_upper < kc_upper
        if not in_squeeze:
            return False

    width_rw = ctx.rolling_windows.get("bb_width")
    if width_rw and getattr(width_rw.get("window"), "IsReady", False):
        upper = bb_ind.UpperBand.Current.Value
        lower = bb_ind.LowerBand.Current.Value
        middle = bb_ind.MiddleBand.Current.Value
        if middle != 0:
            width = (upper - lower) / middle
            window = width_rw["window"]
            widths = list(window) if hasattr(window, "__iter__") else []
            if widths:
                pctile = sum(1 for w in widths if w < width) / len(widths) * 100
                if pctile > condition.pctile_threshold:
                    return False

    if condition.with_trend:
        ema_fast = ctx.indicators.get("ema_fast") or ctx.indicators.get("ema_20")
        ema_slow = ctx.indicators.get("ema_slow") or ctx.indicators.get("ema_50")
        if ema_fast and ema_slow and ema_fast.Current.Value <= ema_slow.Current.Value:
            return False

    return True

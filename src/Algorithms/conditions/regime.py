"""Typed evaluator for RegimeCondition (Phase 7)."""
from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import RegimeCondition
from vibe_trade_shared.models.ir.enums import CompareOp, RegimeMetric

from .context import EvalContext


def _apply_op(op: CompareOp, left: float | None, right: float | str | int) -> bool:
    """Apply comparison operator; coerce right to float for numeric comparison."""
    if left is None or right is None:
        return False
    r = float(right)
    return op.apply(left, r)


def evaluate_regime(
    condition: RegimeCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate a regime condition using EvalContext."""
    # Handle both enum and string metric types
    metric = condition.metric.value if isinstance(condition.metric, RegimeMetric) else str(condition.metric)
    op = (
        condition.op
        if isinstance(condition.op, CompareOp)
        else CompareOp(condition.op)
        if isinstance(condition.op, str)
        else CompareOp.EQ
    )
    value = condition.value
    if metric == "trend_ma_relation":
        fast_id = "ema_" + str(condition.ma_fast or 20)
        slow_id = "ema_" + str(condition.ma_slow or 50)
        fast_ind = ctx.indicators.get(fast_id) or ctx.indicators.get("ema_fast")
        slow_ind = ctx.indicators.get(slow_id) or ctx.indicators.get("ema_slow")
        if fast_ind and slow_ind:
            return _apply_op(op, fast_ind.Current.Value - slow_ind.Current.Value, value)
        return False
    if metric == "trend_adx":
        adx_ind = ctx.indicators.get("adx") or ctx.indicators.get("adx_14")
        if adx_ind:
            return _apply_op(op, adx_ind.Current.Value, value)
        return False
    if metric in ("vol_bb_width_pctile", "bb_width_pctile"):
        bb_period = condition.lookback_bars or 20
        bb_ind = ctx.indicators.get(f"bb_{bb_period}")
        if not bb_ind:
            return False
        from .helpers import compute_bb_width_percentile
        pctile = compute_bb_width_percentile(bb_ind, ctx.rolling_windows.get("bb_width"))
        if pctile is None:
            return False
        return _apply_op(op, pctile, value)
    if metric == "vol_atr_pct":
        atr_ind = ctx.indicators.get("atr") or ctx.indicators.get("atr_14")
        if atr_ind and bar.Close != 0:
            return _apply_op(op, (atr_ind.Current.Value / bar.Close) * 100, value)
        return False
    if metric == "volume_pctile":
        # Use PercentileRank indicator (works on volume data)
        # IR translator creates indicator with ID: volume_pctile_{pctile_period} (defaults to 100)
        pctile_id = "volume_pctile_100"  # IR translator defaults to 100 if not specified
        pctile_ind = ctx.indicators.get(pctile_id)
        
        if not pctile_ind or not pctile_ind.IsReady:
            return False
        
        pctile = pctile_ind.Current.Value
        return _apply_op(op, pctile, value)
    if metric == "dist_from_vwap_pct":
        vwap_ind = ctx.indicators.get("vwap")
        if vwap_ind and getattr(vwap_ind.Current, "Value", 0) != 0:
            return _apply_op(op, (bar.Close - vwap_ind.Current.Value) / vwap_ind.Current.Value * 100, value)
        return False
    if metric == "ret_pct":
        roc_ind = ctx.indicators.get("roc")
        if roc_ind:
            return _apply_op(op, roc_ind.Current.Value * 100, value)
        return False
    if metric == "gap_pct":
        prev_rw = ctx.rolling_windows.get("prev_close")
        if prev_rw and prev_rw.get("window") and prev_rw["window"].IsReady:
            w = prev_rw["window"]
            prev = w[1] if w.Count > 1 else None
            if prev and prev != 0:
                return _apply_op(op, (bar.Open - prev) / prev * 100, value)
        return False
    if metric == "price_level_touch":
        # Check if price touches a level (bar.Low <= level <= bar.High)
        lookback = condition.lookback_bars or 20
        level_reference = condition.level_reference or "previous_low"
        
        # Get level from MAX/MIN indicators
        if level_reference == "previous_low":
            min_ind = ctx.indicators.get(f"min_{lookback}")
            if not min_ind or not min_ind.IsReady:
                return False
            level_value = min_ind.Current.Value
        elif level_reference == "previous_high":
            max_ind = ctx.indicators.get(f"max_{lookback}")
            if not max_ind or not max_ind.IsReady:
                return False
            level_value = max_ind.Current.Value
        else:
            # Other level_reference types not yet supported
            return False
        
        # Check if bar touches the level: bar.Low <= level <= bar.High
        touches = bar.Low <= level_value <= bar.High
        return _apply_op(op, 1 if touches else 0, value)
    if metric == "liquidity_sweep":
        from vibe_trade_shared.models.ir import LiquiditySweepCondition
        from .liquidity_sweep import evaluate_liquidity_sweep
        lookback = condition.lookback_bars or 20
        sweep_cond = LiquiditySweepCondition(
            type="liquidity_sweep",
            level_type="rolling_min",
            level_period=lookback,
            lookback_bars=lookback,
        )
        result = evaluate_liquidity_sweep(sweep_cond, bar, ctx)
        return _apply_op(op, 1 if result else 0, value)
    return False

"""Typed evaluator for RegimeCondition (Phase 7)."""
from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import RegimeCondition
from vibe_trade_shared.models.ir.enums import CompareOp

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
    metric = (
        condition.metric.value
        if hasattr(condition.metric, "value")
        else str(condition.metric)
    )
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
        bb_ind = ctx.indicators.get("bb") or ctx.indicators.get("bb_20")
        if not bb_ind:
            return False
        u = bb_ind.UpperBand.Current.Value
        lo = bb_ind.LowerBand.Current.Value
        m = bb_ind.MiddleBand.Current.Value
        if m == 0:
            return False
        width = (u - lo) / m
        width_rw = ctx.rolling_windows.get("bb_width")
        if width_rw and getattr(width_rw.get("window"), "IsReady", False):
            w = width_rw["window"]
            widths = list(w) if hasattr(w, "__iter__") else []
            pctile = (sum(1 for x in widths if x < width) / len(widths) * 100) if widths else 0
            return _apply_op(op, pctile, value)
        return _apply_op(op, width * 100, value)
    if metric == "vol_atr_pct":
        atr_ind = ctx.indicators.get("atr") or ctx.indicators.get("atr_14")
        if atr_ind and bar.Close != 0:
            return _apply_op(op, (atr_ind.Current.Value / bar.Close) * 100, value)
        return False
    if metric == "volume_pctile":
        vol_rw = ctx.rolling_windows.get("volume")
        if vol_rw and getattr(vol_rw.get("window"), "IsReady", False):
            w = vol_rw["window"]
            vols = list(w) if hasattr(w, "__iter__") else []
            pctile = (sum(1 for v in vols if v < float(bar.Volume)) / len(vols) * 100) if vols else 0
            return _apply_op(op, pctile, value)
        return False
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
        if prev_rw and getattr(prev_rw.get("window"), "IsReady", False):
            w = prev_rw["window"]
            prev = w[1] if hasattr(w, "__getitem__") else None
            if prev and prev != 0:
                return _apply_op(op, (bar.Open - prev) / prev * 100, value)
        return False
    if metric == "price_level_touch":
        # Check if price touches a level (bar.Low <= level <= bar.High)
        lookback = condition.lookback_bars or 20
        level_reference = getattr(condition, "level_reference", None) or "previous_low"
        
        # Get level from rolling min/max based on level_reference
        if level_reference == "previous_low":
            level_ind = ctx.rolling_minmax.get(f"min_{lookback}") or ctx.rolling_minmax.get(f"rmm_low_{lookback}")
            if level_ind and getattr(level_ind.get("window"), "IsReady", False):
                window = level_ind["window"]
                level_value = min(list(window)) if hasattr(window, "__iter__") else None
            else:
                return False
        elif level_reference == "previous_high":
            level_ind = ctx.rolling_minmax.get(f"max_{lookback}") or ctx.rolling_minmax.get(f"rmm_high_{lookback}")
            if level_ind and getattr(level_ind.get("window"), "IsReady", False):
                window = level_ind["window"]
                level_value = max(list(window)) if hasattr(window, "__iter__") else None
            else:
                return False
        else:
            # Other level_reference types not yet supported
            return False
        
        if level_value is None:
            return False
        
        # Check if bar touches the level: bar.Low <= level <= bar.High
        touches = bar.Low <= level_value <= bar.High
        return _apply_op(op, 1 if touches else 0, value)
    if metric == "liquidity_sweep":
        # Delegate to liquidity sweep evaluator logic
        lookback = condition.lookback_bars or 20
        state_key = f"liquidity_sweep_{lookback}"
        sweep_triggered = ctx.state.get(f"{state_key}_triggered", False)
        sweep_bar_count = ctx.state.get(f"{state_key}_bars", 0)
        
        # Get level from rolling min (default for liquidity sweep)
        level_ind = ctx.rolling_minmax.get(f"min_{lookback}") or ctx.rolling_minmax.get(f"rmm_low_{lookback}")
        if level_ind and getattr(level_ind.get("window"), "IsReady", False):
            window = level_ind["window"]
            level_value = min(list(window)) if hasattr(window, "__iter__") else None
        else:
            return False
        
        if level_value is None:
            return False
        
        if not sweep_triggered:
            if bar.Low < level_value:
                ctx.state[f"{state_key}_triggered"] = True
                ctx.state[f"{state_key}_bars"] = 0
                ctx.state[f"{state_key}_level"] = level_value
            return False
        
        sweep_bar_count += 1
        ctx.state[f"{state_key}_bars"] = sweep_bar_count
        
        if sweep_bar_count > lookback:
            ctx.state[f"{state_key}_triggered"] = False
            return False
        
        sweep_level = ctx.state.get(f"{state_key}_level", level_value)
        if bar.Close > sweep_level:
            ctx.state[f"{state_key}_triggered"] = False
            return _apply_op(op, 1, value)
        
        return False
    return False

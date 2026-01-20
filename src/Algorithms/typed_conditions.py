"""Typed condition models for StrategyRuntime.

These models mirror the IR schema and provide type-safe access
to condition fields, replacing .get() calls with typed attributes.
"""
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TypedRegimeCondition:
    """Typed wrapper for RegimeCondition."""
    type: Literal["regime"] = "regime"
    metric: str = ""
    op: str = "=="
    value: float | str = 0.0
    ma_fast: int | None = None
    ma_slow: int | None = None
    lookback_bars: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TypedRegimeCondition":
        return cls(
            metric=d.get("metric", ""),
            op=d.get("op", "=="),
            value=d.get("value", 0.0),
            ma_fast=d.get("ma_fast"),
            ma_slow=d.get("ma_slow"),
            lookback_bars=d.get("lookback_bars"),
        )


@dataclass
class TypedBreakoutCondition:
    """Typed wrapper for BreakoutCondition."""
    type: Literal["breakout"] = "breakout"
    lookback_bars: int = 50
    buffer_bps: int = 0
    max_indicator: str | None = None
    min_indicator: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TypedBreakoutCondition":
        return cls(
            lookback_bars=d.get("lookback_bars", 50),
            buffer_bps=d.get("buffer_bps", 0),
            max_indicator=d.get("max_indicator"),
            min_indicator=d.get("min_indicator"),
        )


@dataclass
class TypedSqueezeCondition:
    """Typed wrapper for SqueezeCondition."""
    type: Literal["squeeze"] = "squeeze"
    squeeze_metric: str = "bb_width_pctile"
    pctile_threshold: float = 10.0
    break_rule: str = "donchian"
    with_trend: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "TypedSqueezeCondition":
        return cls(
            squeeze_metric=d.get("squeeze_metric", "bb_width_pctile"),
            pctile_threshold=d.get("pctile_threshold", 10.0),
            break_rule=d.get("break_rule", "donchian"),
            with_trend=d.get("with_trend", False),
        )


@dataclass
class TypedSpreadCondition:
    """Typed wrapper for SpreadCondition."""
    type: Literal["spread"] = "spread"
    symbol_a: str = ""
    symbol_b: str = ""
    calc_type: str = "zscore"
    window_bars: int = 100
    trigger_op: str = "above"
    threshold: float = 2.0
    hedge_ratio: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TypedSpreadCondition":
        return cls(
            symbol_a=d.get("symbol_a", ""),
            symbol_b=d.get("symbol_b", ""),
            calc_type=d.get("calc_type", "zscore"),
            window_bars=d.get("window_bars", 100),
            trigger_op=d.get("trigger_op", "above"),
            threshold=d.get("threshold", 2.0),
            hedge_ratio=d.get("hedge_ratio"),
        )


@dataclass
class TypedGapCondition:
    """Typed wrapper for GapCondition."""
    type: Literal["gap"] = "gap"
    session: str = "us"
    mode: str = "gap_fade"
    min_gap_pct: float = 0.0
    direction: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> "TypedGapCondition":
        return cls(
            session=d.get("session", "us"),
            mode=d.get("mode", "gap_fade"),
            min_gap_pct=d.get("min_gap_pct", 0.0),
            direction=d.get("direction", "auto"),
        )


def parse_condition(condition: dict) -> Any:
    """Parse a condition dict into a typed wrapper.

    Returns the typed wrapper or the original dict if no wrapper exists.
    """
    cond_type = condition.get("type")
    parsers = {
        "regime": TypedRegimeCondition.from_dict,
        "breakout": TypedBreakoutCondition.from_dict,
        "squeeze": TypedSqueezeCondition.from_dict,
        "spread": TypedSpreadCondition.from_dict,
        "gap": TypedGapCondition.from_dict,
    }
    parser = parsers.get(cond_type)
    if parser:
        return parser(condition)
    return condition

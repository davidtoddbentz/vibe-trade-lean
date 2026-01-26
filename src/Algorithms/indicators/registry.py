"""Indicator registry - factory pattern for indicator creation.

This module centralizes indicator creation, update, and value resolution,
eliminating the large switch statements in StrategyRuntime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:
    from QuantConnect import Symbol
    from QuantConnect.Algorithm import QCAlgorithm
    from QuantConnect.Data.Market import TradeBar


class IndicatorCategory(Enum):
    """Category of indicator - determines storage and update behavior."""

    LEAN = "lean"  # LEAN built-in, auto-updated
    ROLLING_WINDOW = "rolling_window"  # RollingWindow for historical values
    VOL_SMA = "vol_sma"  # Volume SMA, manual update
    ROLLING_MINMAX = "rolling_minmax"  # Rolling min/max tracker
    AVWAP = "avwap"  # Anchored VWAP tracker


@dataclass
class IndicatorResult:
    """Result of indicator creation.

    Attributes:
        category: How this indicator should be stored/updated
        indicator: The LEAN indicator object (for LEAN category)
        data: Dict data for custom indicators (RW, VOL_SMA, RMM, AVWAP)
    """

    category: IndicatorCategory
    indicator: Any | None = None
    data: dict[str, Any] | None = None


class IndicatorCreator(Protocol):
    """Protocol for indicator creator functions."""

    def __call__(
        self,
        ind_def: dict[str, Any],
        algo: QCAlgorithm,
        symbol: Symbol,
        resolution: Any,
    ) -> IndicatorResult: ...


# =============================================================================
# LEAN Built-in Indicators
# =============================================================================


def create_ema(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create EMA indicator."""
    period = ind_def.get("period", 20)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.EMA(symbol, period, resolution=resolution),
    )


def create_sma(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create SMA indicator."""
    period = ind_def.get("period", 20)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.SMA(symbol, period, resolution=resolution),
    )


def create_bb(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Bollinger Bands indicator."""
    period = ind_def.get("period", 20)
    mult = ind_def.get("multiplier", 2.0)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.BB(symbol, period, mult, resolution=resolution),
    )


def create_kc(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Keltner Channel indicator."""
    period = ind_def.get("period", 20)
    mult = ind_def.get("multiplier", 2.0)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.KCH(symbol, period, mult, resolution=resolution),
    )


def create_atr(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create ATR indicator."""
    period = ind_def.get("period", 14)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.ATR(symbol, period, resolution=resolution),
    )


def create_max(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create MAX indicator."""
    period = ind_def.get("period", 50)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.MAX(symbol, period, resolution=resolution),
    )


def create_min(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create MIN indicator."""
    period = ind_def.get("period", 50)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.MIN(symbol, period, resolution=resolution),
    )


def create_roc(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create ROC indicator."""
    period = ind_def.get("period", 1)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.ROC(symbol, period, resolution=resolution),
    )


def create_adx(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create ADX indicator."""
    period = ind_def.get("period", 14)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.ADX(symbol, period, resolution=resolution),
    )


def create_rsi(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create RSI indicator."""
    period = ind_def.get("period", 14)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.RSI(symbol, period, resolution=resolution),
    )


def create_macd(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create MACD indicator."""
    fast = ind_def.get("fast_period", 12)
    slow = ind_def.get("slow_period", 26)
    signal = ind_def.get("signal_period", 9)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.MACD(symbol, fast, slow, signal, resolution=resolution),
    )


def create_dc(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Donchian Channel indicator."""
    period = ind_def.get("period", 20)
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.DCH(symbol, period, resolution=resolution),
    )


def create_vwap(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create VWAP indicator."""
    period = ind_def.get("period", 0)
    if period == 0:
        # Intraday VWAP (resets daily)
        indicator = algo.VWAP(symbol)
    else:
        # Rolling VWAP with period
        indicator = algo.VWAP(symbol, period)
    return IndicatorResult(category=IndicatorCategory.LEAN, indicator=indicator)


def create_supertrend(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create SuperTrend indicator.

    SuperTrend is a trend-following indicator that uses ATR to set trailing stops.
    Returns a value representing the stop level (above price for downtrend, below for uptrend).
    """
    from QuantConnect.Indicators import MovingAverageType

    period = ind_def.get("period", 10)
    multiplier = ind_def.get("multiplier", 3.0)
    ma_type = ind_def.get("ma_type", "wilders")

    # Map string to MovingAverageType enum
    ma_type_map = {
        "wilders": MovingAverageType.Wilders,
        "simple": MovingAverageType.Simple,
        "exponential": MovingAverageType.Exponential,
    }
    ma_enum = ma_type_map.get(ma_type.lower(), MovingAverageType.Wilders)

    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.STR(symbol, period, multiplier, ma_enum, resolution=resolution),
    )


def create_stochastic(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Stochastic indicator.

    Returns %K and %D oscillators (0-100 range).
    Properties: .StochK (fast), .StochD (slow/signal)
    """
    period = ind_def.get("period", 14)
    k_period = ind_def.get("k_period", 3)  # Smoothing for %K
    d_period = ind_def.get("d_period", 3)  # Smoothing for %D

    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.STO(symbol, period, k_period, d_period, resolution=resolution),
    )


def create_cci(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Commodity Channel Index indicator.

    CCI measures the deviation of price from its statistical mean.
    Typical range: -100 to +100, with values outside indicating strong trends.
    """
    period = ind_def.get("period", 20)

    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.CCI(symbol, period, resolution=resolution),
    )


def create_obv(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create On-Balance Volume indicator.

    OBV is a cumulative volume indicator that adds/subtracts volume
    based on whether price closed up or down.
    """
    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.OBV(symbol, resolution=resolution),
    )


def create_aroon(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Aroon Oscillator indicator.

    The Aroon Oscillator is the difference between AroonUp and AroonDown.
    Range: -100 to +100. Positive = uptrend bias, Negative = downtrend bias.
    Properties: .AroonUp, .AroonDown (both 0-100)
    """
    period = ind_def.get("period", 25)

    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=algo.AROON(symbol, period, resolution=resolution),
    )


# =============================================================================
# Candlestick Patterns
# =============================================================================

# Map pattern names to CandlestickPatterns methods
# Values: 0 = no pattern, +1 = bullish, -1 = bearish
CANDLESTICK_PATTERNS = {
    "doji": "Doji",
    "hammer": "Hammer",
    "hanging_man": "HangingMan",
    "engulfing": "Engulfing",
    "harami": "Harami",
    "harami_cross": "HaramiCross",
    "morning_star": "MorningStar",
    "evening_star": "EveningStar",
    "morning_doji_star": "MorningDojiStar",
    "evening_doji_star": "EveningDojiStar",
    "three_white_soldiers": "ThreeWhiteSoldiers",
    "three_black_crows": "ThreeBlackCrows",
    "piercing_line": "PiercingLine",
    "dark_cloud_cover": "DarkCloudCover",
    "abandoned_baby": "AbandonedBaby",
    "belt_hold": "BeltHold",
    "breakaway": "Breakaway",
    "closing_marubozu": "ClosingMarubozu",
    "concealed_baby_swallow": "ConcealedBabySwallow",
    "counterattack": "Counterattack",
    "doji_star": "DojiStar",
    "dragonfly_doji": "DragonflyDoji",
    "gravestone_doji": "GravestoneDoji",
    "high_wave_candle": "HighWaveCandle",
    "hikkake": "Hikkake",
    "hikkake_modified": "HikkakeModified",
    "homing_pigeon": "HomingPigeon",
    "identical_three_crows": "IdenticalThreeCrows",
    "in_neck": "InNeck",
    "inverted_hammer": "InvertedHammer",
    "kicking": "Kicking",
    "kicking_by_length": "KickingByLength",
    "ladder_bottom": "LadderBottom",
    "long_legged_doji": "LongLeggedDoji",
    "long_line_candle": "LongLineCandle",
    "marubozu": "Marubozu",
    "matching_low": "MatchingLow",
    "mat_hold": "MatHold",
    "on_neck": "OnNeck",
    "rickshaw_man": "RickshawMan",
    "rising_falling_three_methods": "RisingFallingThreeMethods",
    "separating_lines": "SeparatingLines",
    "shooting_star": "ShootingStar",
    "short_line_candle": "ShortLineCandle",
    "spinning_top": "SpinningTop",
    "stalled_pattern": "StalledPattern",
    "stick_sandwich": "StickSandwich",
    "takuri": "Takuri",
    "tasuki_gap": "TasukiGap",
    "three_inside": "ThreeInside",
    "three_line_strike": "ThreeLineStrike",
    "three_outside": "ThreeOutside",
    "three_stars_in_south": "ThreeStarsInSouth",
    "tristar": "Tristar",
    "two_crows": "TwoCrows",
    "unique_three_river": "UniqueThreeRiver",
    "upside_gap_two_crows": "UpsideGapTwoCrows",
    "upside_downside_gap_three_methods": "UpsideDownsideGapThreeMethods",
}


def create_candlestick_pattern(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create a candlestick pattern indicator.

    Candlestick patterns return: 0 = no pattern, +1 = bullish, -1 = bearish.

    Args:
        ind_def: Must contain 'pattern' field with pattern name (e.g., 'doji', 'hammer')
    """
    pattern_name = ind_def.get("pattern", "doji").lower().replace("-", "_")

    if pattern_name not in CANDLESTICK_PATTERNS:
        raise ValueError(f"Unknown candlestick pattern: {pattern_name}. "
                        f"Available: {list(CANDLESTICK_PATTERNS.keys())[:10]}...")

    method_name = CANDLESTICK_PATTERNS[pattern_name]
    pattern_method = getattr(algo.CandlestickPatterns, method_name)

    return IndicatorResult(
        category=IndicatorCategory.LEAN,
        indicator=pattern_method(symbol, resolution=resolution),
    )


# =============================================================================
# Custom Indicators
# =============================================================================


def create_avwap(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Anchored VWAP tracker."""
    anchor = ind_def.get("anchor", "session_open")
    if not anchor:
        anchor = ind_def.get("params", {}).get("anchor", "session_open")

    return IndicatorResult(
        category=IndicatorCategory.AVWAP,
        data={
            "symbol": symbol,
            "anchor": anchor,
            "cum_volume": 0.0,
            "cum_pv": 0.0,  # price * volume
            "cum_pv2": 0.0,  # price^2 * volume for std dev
            "last_reset": None,
            "current_vwap": 0.0,
            "current_std_dev": 0.0,
        },
    )


def create_rw(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Rolling Window for historical values."""
    from QuantConnect.Indicators import RollingWindow

    period = ind_def.get("period", 2)
    field = ind_def.get("field", "close")

    return IndicatorResult(
        category=IndicatorCategory.ROLLING_WINDOW,
        data={
            "window": RollingWindow[float](period),
            "field": field,
            "symbol": symbol,
        },
    )


def create_vol_sma(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Volume SMA indicator."""
    from QuantConnect.Indicators import SimpleMovingAverage

    period = ind_def.get("period", 20)

    return IndicatorResult(
        category=IndicatorCategory.VOL_SMA,
        data={
            "sma": SimpleMovingAverage(period),
            "period": period,
            "symbol": symbol,
        },
    )


def create_rmm(
    ind_def: dict[str, Any], algo: QCAlgorithm, symbol: Symbol, resolution: Any
) -> IndicatorResult:
    """Create Rolling Min/Max tracker."""
    from QuantConnect.Indicators import RollingWindow

    period = ind_def.get("period", 20)
    mode = ind_def.get("mode", "min")
    field = ind_def.get("field", "close")

    return IndicatorResult(
        category=IndicatorCategory.ROLLING_MINMAX,
        data={
            "window": RollingWindow[float](period),
            "mode": mode,
            "field": field,
            "symbol": symbol,
        },
    )


# =============================================================================
# Registry
# =============================================================================

INDICATOR_CREATORS: dict[str, IndicatorCreator] = {
    # LEAN built-ins
    "EMA": create_ema,
    "SMA": create_sma,
    "BB": create_bb,
    "KC": create_kc,
    "ATR": create_atr,
    "MAX": create_max,
    "MIN": create_min,
    "ROC": create_roc,
    "ADX": create_adx,
    "RSI": create_rsi,
    "MACD": create_macd,
    "DC": create_dc,
    "VWAP": create_vwap,
    # New LEAN built-ins (Phase 4)
    "STR": create_supertrend,  # SuperTrend
    "SUPERTREND": create_supertrend,  # Alias
    "STO": create_stochastic,  # Stochastic
    "STOCHASTIC": create_stochastic,  # Alias
    "CCI": create_cci,  # Commodity Channel Index
    "OBV": create_obv,  # On-Balance Volume
    "AROON": create_aroon,  # Aroon Oscillator
    # Candlestick Patterns (Phase 5)
    "CANDLESTICK": create_candlestick_pattern,
    "CANDLE": create_candlestick_pattern,  # Alias
    # Custom
    "AVWAP": create_avwap,
    "RW": create_rw,
    "VOL_SMA": create_vol_sma,
    "RMM": create_rmm,
}

INDICATOR_TYPES = list(INDICATOR_CREATORS.keys())


def create_indicator(
    ind_def: dict[str, Any],
    algo: QCAlgorithm,
    symbol: Symbol,
    resolution: Any,
) -> IndicatorResult:
    """Create an indicator from its definition.

    Args:
        ind_def: Indicator definition dict with 'type', 'id', and params
        algo: QCAlgorithm instance
        symbol: Symbol for the indicator
        resolution: Resolution for the indicator

    Returns:
        IndicatorResult with category and indicator/data

    Raises:
        ValueError: If indicator type is unknown
    """
    ind_type = ind_def.get("type")
    creator = INDICATOR_CREATORS.get(ind_type)

    if not creator:
        raise ValueError(f"Unknown indicator type: {ind_type}")

    return creator(ind_def, algo, symbol, resolution)


def update_indicator(
    category: IndicatorCategory,
    data: dict[str, Any],
    bar: TradeBar,
    time: Any,
) -> None:
    """Update a custom indicator with new bar data.

    LEAN indicators are auto-updated, so this only handles custom categories.

    Args:
        category: The indicator category
        data: The indicator data dict
        bar: Current bar
        time: Current algorithm time
    """
    if category == IndicatorCategory.ROLLING_WINDOW:
        _update_rolling_window(data, bar)
    elif category == IndicatorCategory.VOL_SMA:
        _update_vol_sma(data, bar, time)
    elif category == IndicatorCategory.ROLLING_MINMAX:
        _update_rolling_minmax(data, bar)
    elif category == IndicatorCategory.AVWAP:
        _update_avwap(data, bar, time)
    # LEAN indicators are auto-updated, no action needed


def _update_rolling_window(data: dict[str, Any], bar: TradeBar) -> None:
    """Update rolling window with price field."""
    field = data["field"]
    window = data["window"]

    if field == "close":
        window.Add(bar.Close)
    elif field == "open":
        window.Add(bar.Open)
    elif field == "high":
        window.Add(bar.High)
    elif field == "low":
        window.Add(bar.Low)


def _update_vol_sma(data: dict[str, Any], bar: TradeBar, time: Any) -> None:
    """Update volume SMA with current volume."""
    data["sma"].Update(time, float(bar.Volume))


def _update_rolling_minmax(data: dict[str, Any], bar: TradeBar) -> None:
    """Update rolling min/max with price field."""
    field = data["field"]
    window = data["window"]

    if field == "close":
        window.Add(bar.Close)
    elif field == "open":
        window.Add(bar.Open)
    elif field == "high":
        window.Add(bar.High)
    elif field == "low":
        window.Add(bar.Low)


def _update_avwap(data: dict[str, Any], bar: TradeBar, time: Any) -> None:
    """Update AVWAP tracker with new bar data."""
    anchor = data.get("anchor", "session_open")
    typical_price = (bar.High + bar.Low + bar.Close) / 3
    volume = float(bar.Volume) if bar.Volume > 0 else 1.0

    # Check if we need to reset based on anchor
    should_reset = False
    last_reset = data.get("last_reset")

    if anchor == "session_open":
        # Reset at start of each day
        if last_reset is None or time.date() != last_reset.date():
            should_reset = True
    elif anchor == "week_open":
        # Reset at start of each week (Monday)
        if last_reset is None or (
            time.weekday() == 0
            and (last_reset is None or time.date() != last_reset.date())
        ):
            should_reset = True
    elif anchor == "month_open":
        # Reset at start of each month
        if last_reset is None or time.month != last_reset.month:
            should_reset = True

    if should_reset:
        data["cum_volume"] = 0.0
        data["cum_pv"] = 0.0
        data["cum_pv2"] = 0.0
        data["last_reset"] = time

    # Update cumulative values
    data["cum_volume"] = data.get("cum_volume", 0) + volume
    data["cum_pv"] = data.get("cum_pv", 0) + (typical_price * volume)
    data["cum_pv2"] = data.get("cum_pv2", 0) + (typical_price**2 * volume)

    # Calculate and cache current VWAP and std_dev values
    if data["cum_volume"] > 0:
        vwap = data["cum_pv"] / data["cum_volume"]
        data["current_vwap"] = vwap
        # Calculate standard deviation
        variance = (data["cum_pv2"] / data["cum_volume"]) - (vwap**2)
        if variance > 0:
            data["current_std_dev"] = variance**0.5
        else:
            data["current_std_dev"] = 0.0


def is_indicator_ready(category: IndicatorCategory, indicator_or_data: Any) -> bool:
    """Check if an indicator is ready.

    Args:
        category: The indicator category
        indicator_or_data: LEAN indicator object or data dict

    Returns:
        True if the indicator is ready
    """
    if category == IndicatorCategory.LEAN:
        return indicator_or_data.IsReady

    elif category == IndicatorCategory.ROLLING_WINDOW:
        return indicator_or_data["window"].IsReady

    elif category == IndicatorCategory.VOL_SMA:
        return indicator_or_data["sma"].IsReady

    elif category == IndicatorCategory.ROLLING_MINMAX:
        return indicator_or_data["window"].IsReady

    elif category == IndicatorCategory.AVWAP:
        return indicator_or_data["cum_volume"] > 0

    return False


def resolve_indicator_value(
    category: IndicatorCategory,
    indicator_or_data: Any,
    field: str | None = None,
) -> float:
    """Resolve an indicator's current value.

    Args:
        category: The indicator category
        indicator_or_data: LEAN indicator object or data dict
        field: Optional field name for multi-value indicators

    Returns:
        The current value as a float
    """
    if category == IndicatorCategory.LEAN:
        ind = indicator_or_data
        # Handle MACD fields
        if field == "signal" and hasattr(ind, "Signal"):
            return ind.Signal.Current.Value
        elif field == "histogram" and hasattr(ind, "Histogram"):
            return ind.Histogram.Current.Value
        elif field == "macd" and hasattr(ind, "Fast"):
            return ind.Current.Value
        # Handle Stochastic fields
        elif field == "k" and hasattr(ind, "StochK"):
            return ind.StochK.Current.Value
        elif field == "d" and hasattr(ind, "StochD"):
            return ind.StochD.Current.Value
        # Handle Aroon fields
        elif field == "up" and hasattr(ind, "AroonUp"):
            return ind.AroonUp.Current.Value
        elif field == "down" and hasattr(ind, "AroonDown"):
            return ind.AroonDown.Current.Value
        # Handle SuperTrend direction (1 = uptrend, -1 = downtrend)
        elif field == "direction" and hasattr(ind, "Direction"):
            return float(ind.Direction)
        # Default: return main value
        return ind.Current.Value

    elif category == IndicatorCategory.VOL_SMA:
        return indicator_or_data["sma"].Current.Value

    elif category == IndicatorCategory.ROLLING_MINMAX:
        window = indicator_or_data["window"]
        if window.IsReady:
            if indicator_or_data["mode"] == "min":
                return min(list(window))
            else:  # max
                return max(list(window))
        return 0.0

    elif category == IndicatorCategory.AVWAP:
        if field == "std_dev":
            return indicator_or_data.get("current_std_dev", 0.0)
        return indicator_or_data.get("current_vwap", 0.0)

    elif category == IndicatorCategory.ROLLING_WINDOW:
        window = indicator_or_data["window"]
        if window.IsReady and window.Count > 0:
            return float(window[0])
        return 0.0

    return 0.0

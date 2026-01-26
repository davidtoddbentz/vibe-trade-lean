"""Indicator registry for StrategyRuntime.

This module provides a registry pattern for creating and managing indicators,
replacing the large switch statements in StrategyRuntime._create_indicators().
"""

from .registry import (
    IndicatorCategory,
    IndicatorResult,
    create_indicator,
    update_indicator,
    is_indicator_ready,
    resolve_indicator_value,
    INDICATOR_TYPES,
    CANDLESTICK_PATTERNS,
)

__all__ = [
    "IndicatorCategory",
    "IndicatorResult",
    "create_indicator",
    "update_indicator",
    "is_indicator_ready",
    "resolve_indicator_value",
    "INDICATOR_TYPES",
    "CANDLESTICK_PATTERNS",
]

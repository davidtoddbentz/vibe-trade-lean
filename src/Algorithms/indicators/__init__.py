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
from .resolvers import resolve_value
from .creation import (
    create_all_indicators,
    initialize_state_variables,
    check_indicators_ready,
    update_all_indicators,
)

__all__ = [
    "IndicatorCategory",
    "IndicatorResult",
    "create_indicator",
    "update_indicator",
    "is_indicator_ready",
    "resolve_indicator_value",
    "resolve_value",
    "create_all_indicators",
    "initialize_state_variables",
    "check_indicators_ready",
    "update_all_indicators",
    "INDICATOR_TYPES",
    "CANDLESTICK_PATTERNS",
]

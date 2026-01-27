"""Indicator creation and management for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime to reduce line count.
"""

from __future__ import annotations

from typing import Any

from indicators import (
    IndicatorCategory,
    IndicatorResult,
    create_indicator,
    is_indicator_ready,
    update_indicator,
)


def create_all_indicators(
    ir_indicators: list[Any],
    symbol: Any,
    resolution: Any,
    indicator_registry: dict[str, tuple[IndicatorCategory, Any]],
    indicators: dict[str, Any],
    rolling_windows: dict[str, Any],
    vol_sma_indicators: dict[str, Any],
    rolling_minmax: dict[str, Any],
    avwap_trackers: dict[str, Any],
    log: Any,
    runtime: Any,  # StrategyRuntime instance for create_indicator callback
) -> None:
    """Create all indicators defined in the IR.

    Uses the indicator registry pattern to create indicators, eliminating
    the large switch statement.

    Args:
        ir_indicators: List of IndicatorSpec from StrategyIR
        symbol: Primary trading symbol
        resolution: Data resolution
        indicator_registry: Unified registry to populate
        indicators: Legacy LEAN indicators dict
        rolling_windows: Legacy rolling windows dict
        vol_sma_indicators: Legacy volume SMA dict
        rolling_minmax: Legacy rolling minmax dict
        avwap_trackers: Legacy AVWAP dict
        log: Logging function
        runtime: StrategyRuntime instance (for create_indicator callbacks)
    """
    for ind_def in ir_indicators:
        # ind_def is typed IndicatorSpec from IR
        ind_id = ind_def.id

        # Get symbol for this indicator (defaults to primary symbol)
        # IndicatorSpec doesn't have a symbol field, but it may be passed as extra
        # Check both the typed model and the dict for symbol field
        ind_symbol = symbol
        try:
            # create_indicator expects a dict, so convert typed model to dict
            ind_dict = ind_def.model_dump(mode="json")
            # Check if symbol is specified in the dict (for multi-symbol indicators)
            if "symbol" in ind_dict and ind_dict["symbol"]:
                # Get the symbol object from runtime's symbols dict
                symbol_str = ind_dict["symbol"]
                symbols_dict = getattr(runtime, "symbols", {})
                normalized = runtime._normalize_symbol(symbol_str)
                ind_symbol = symbols_dict.get(normalized)
                if ind_symbol is None:
                    # Symbol not found - try to add it
                    ind_symbol = runtime._add_symbol(symbol_str)
            result = create_indicator(ind_dict, runtime, ind_symbol, resolution)

            # Store in unified registry
            if result.indicator is not None:
                indicator_registry[ind_id] = (result.category, result.indicator)
            else:
                indicator_registry[ind_id] = (result.category, result.data)

            # Also populate legacy dicts for backward compatibility
            if result.category == IndicatorCategory.LEAN:
                indicators[ind_id] = result.indicator
            elif result.category == IndicatorCategory.ROLLING_WINDOW:
                rolling_windows[ind_id] = result.data
            elif result.category == IndicatorCategory.VOL_SMA:
                vol_sma_indicators[ind_id] = result.data
            elif result.category == IndicatorCategory.ROLLING_MINMAX:
                rolling_minmax[ind_id] = result.data
            elif result.category == IndicatorCategory.AVWAP:
                avwap_trackers[ind_id] = result.data

        except ValueError as e:
            log(f"⚠️ {e}")


def initialize_state_variables(ir_state: list[Any], state: dict[str, Any]) -> None:
    """Initialize state variables from IR.

    Args:
        ir_state: List of StateVarSpec from StrategyIR
        state: Runtime state dict to populate
    """
    for state_var in ir_state:
        # state_var is typed StateVarSpec from IR
        state_id = state_var.id
        default = state_var.default if state_var.default is not None else 0.0
        state[state_id] = default


def check_indicators_ready(
    indicator_registry: dict[str, tuple[IndicatorCategory, Any]],
    is_indicator_ready_func: Any,
) -> bool:
    """Check if all indicators are ready (have enough data).

    Args:
        indicator_registry: Unified indicator registry
        is_indicator_ready_func: Function to check if indicator is ready

    Returns:
        True if all indicators are ready
    """
    for ind_id, (category, indicator_or_data) in indicator_registry.items():
        if not is_indicator_ready_func(category, indicator_or_data):
            return False
    return True


def update_all_indicators(
    indicator_registry: dict[str, tuple[IndicatorCategory, Any]],
    bar: Any,
    current_time: Any,
) -> None:
    """Update all custom indicators with new bar data.

    Args:
        indicator_registry: Unified indicator registry
        bar: Current bar data
        current_time: Current time for indicators that need it
    """
    for ind_id, (category, indicator_or_data) in indicator_registry.items():
        # LEAN indicators are auto-updated by the framework
        if category != IndicatorCategory.LEAN:
            update_indicator(category, indicator_or_data, bar, current_time)

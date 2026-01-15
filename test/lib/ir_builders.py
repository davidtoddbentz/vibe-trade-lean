"""IR builder helpers for constructing strategy IR fragments.

This module provides reusable building blocks for creating strategy IR
in tests. Each builder returns a dict fragment that can be composed
into a complete strategy IR.

Design Philosophy:
- Each function builds ONE logical piece (indicator, condition, exit, etc.)
- Functions are composable - combine them to build complex strategies
- Explicit over implicit - no hidden defaults that affect behavior
- Type hints and docstrings for clarity

Usage:
    from lib.ir_builders import (
        build_strategy_ir,
        ind_ema, ind_bb, ind_max,
        cond_compare, cond_allof, cond_regime,
        exit_profit_target, exit_trailing_stop,
        gate_regime, gate_time_filter,
    )

    ir = build_strategy_ir(
        name="MyStrategy",
        symbol="TESTUSD",
        indicators=[ind_ema("ema_20", 20), ind_bb("bb", 20, 2.0)],
        entry_condition=cond_allof([
            cond_compare(val_ind("ema_20"), ">", val_ind("ema_50")),
            cond_compare(val_price("close"), "<", val_ind_band("bb", "lower")),
        ]),
        exits=[exit_profit_target(0.02), exit_band_upper("bb")],
        gates=[gate_regime_trend("bullish")],
    )
"""

from typing import Any, Literal


# =============================================================================
# Value References - Building blocks for conditions
# =============================================================================

def val_price(field: Literal["open", "high", "low", "close"] = "close") -> dict:
    """Reference to a price field."""
    return {"type": "price", "field": field}


def val_volume() -> dict:
    """Reference to volume."""
    return {"type": "volume"}


def val_indicator(indicator_id: str) -> dict:
    """Reference to an indicator's primary value."""
    return {"type": "indicator", "indicator_id": indicator_id}


def val_ind(indicator_id: str) -> dict:
    """Shorthand for val_indicator."""
    return val_indicator(indicator_id)


def val_indicator_band(indicator_id: str, band: Literal["upper", "middle", "lower"]) -> dict:
    """Reference to a band indicator's band value (BB, Keltner, Donchian)."""
    return {"type": "indicator_band", "indicator_id": indicator_id, "band": band}


def val_ind_band(indicator_id: str, band: Literal["upper", "middle", "lower"]) -> dict:
    """Shorthand for val_indicator_band."""
    return val_indicator_band(indicator_id, band)


def val_indicator_property(indicator_id: str, prop: str) -> dict:
    """Reference to an indicator's property (e.g., MACD.Signal)."""
    return {"type": "indicator_property", "indicator_id": indicator_id, "property": prop}


def val_state(state_id: str) -> dict:
    """Reference to a state variable."""
    return {"type": "state", "state_id": state_id}


def val_literal(value: float | int | bool | str) -> dict:
    """A literal/constant value."""
    return {"type": "literal", "value": value}


def val_expr(op: str, left: dict, right: dict) -> dict:
    """An expression combining two values."""
    return {"type": "expr", "op": op, "left": left, "right": right}


# =============================================================================
# Indicators - Define what to calculate
# =============================================================================

def ind_ema(id: str, period: int, field: str = "close") -> dict:
    """Exponential Moving Average indicator."""
    return {"type": "EMA", "id": id, "period": period, "field": field}


def ind_sma(id: str, period: int, field: str = "close") -> dict:
    """Simple Moving Average indicator."""
    return {"type": "SMA", "id": id, "period": period, "field": field}


def ind_bb(id: str, period: int = 20, multiplier: float = 2.0) -> dict:
    """Bollinger Bands indicator."""
    return {"type": "BB", "id": id, "period": period, "multiplier": multiplier}


def ind_keltner(id: str, period: int = 20, multiplier: float = 1.5) -> dict:
    """Keltner Channel indicator."""
    return {"type": "KC", "id": id, "period": period, "multiplier": multiplier}


def ind_donchian(id: str, period: int = 20) -> dict:
    """Donchian Channel indicator."""
    return {"type": "DC", "id": id, "period": period}


def ind_max(id: str, period: int) -> dict:
    """Maximum value over period."""
    return {"type": "MAX", "id": id, "period": period}


def ind_min(id: str, period: int) -> dict:
    """Minimum value over period."""
    return {"type": "MIN", "id": id, "period": period}


def ind_rsi(id: str, period: int = 14) -> dict:
    """Relative Strength Index."""
    return {"type": "RSI", "id": id, "period": period}


def ind_macd(id: str, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """MACD indicator."""
    return {
        "type": "MACD", "id": id,
        "fast_period": fast, "slow_period": slow, "signal_period": signal
    }


def ind_atr(id: str, period: int = 14) -> dict:
    """Average True Range."""
    return {"type": "ATR", "id": id, "period": period}


def ind_adx(id: str, period: int = 14) -> dict:
    """Average Directional Index."""
    return {"type": "ADX", "id": id, "period": period}


# =============================================================================
# Conditions - Define when to act
# =============================================================================

def cond_compare(
    left: dict,
    op: Literal[">", "<", ">=", "<=", "==", "!="],
    right: dict,
) -> dict:
    """Compare two values."""
    return {"type": "compare", "left": left, "op": op, "right": right}


def cond_allof(conditions: list[dict]) -> dict:
    """All conditions must be true (AND)."""
    return {"type": "allOf", "conditions": conditions}


def cond_anyof(conditions: list[dict]) -> dict:
    """Any condition must be true (OR)."""
    return {"type": "anyOf", "conditions": conditions}


def cond_not(condition: dict) -> dict:
    """Negate a condition."""
    return {"type": "not", "condition": condition}


def cond_cross(
    left: dict,
    right: dict,
    direction: Literal["above", "below"] = "above",
) -> dict:
    """Detect when left crosses right."""
    return {"type": "cross", "left": left, "right": right, "direction": direction}


def cond_regime(
    metric: str,
    op: Literal[">", "<", ">=", "<=", "==", "!="],
    threshold: float,
) -> dict:
    """Regime-based condition.

    Common metrics: 'trend_strength', 'volatility', 'ret_pct', 'adx'
    """
    return {"type": "regime", "metric": metric, "op": op, "threshold": threshold}


def cond_time_filter(
    start_hour: int,
    end_hour: int,
    start_minute: int = 0,
    end_minute: int = 0,
    days_of_week: list[int] | None = None,
) -> dict:
    """Time-based filter condition.

    Args:
        start_hour: Start hour (0-23)
        end_hour: End hour (0-23)
        start_minute: Start minute (0-59)
        end_minute: End minute (0-59)
        days_of_week: Optional list of allowed days (0=Mon, 6=Sun)
    """
    # Format time window as "HH:MM-HH:MM"
    time_window = f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"
    result = {
        "type": "time_filter",
        "time_window": time_window,
    }
    if days_of_week:
        result["days_of_week"] = days_of_week
    return result


def cond_true() -> dict:
    """Always true condition (useful for testing)."""
    return {"type": "compare", "left": val_literal(1), "op": "==", "right": val_literal(1)}


def cond_false() -> dict:
    """Always false condition (useful for testing)."""
    return {"type": "compare", "left": val_literal(1), "op": "==", "right": val_literal(0)}


# =============================================================================
# State Variables - Track information across bars
# =============================================================================

def state_var(id: str, var_type: str = "float", default: Any = 0) -> dict:
    """Define a state variable."""
    return {"id": id, "var_type": var_type, "default": default}


def state_int(id: str, default: int = 0) -> dict:
    """Define an integer state variable."""
    return state_var(id, "int", default)


def state_float(id: str, default: float = 0.0) -> dict:
    """Define a float state variable."""
    return state_var(id, "float", default)


# =============================================================================
# State Operations - Modify state on events
# =============================================================================

def op_set_state(state_id: str, value: dict) -> dict:
    """Set a state variable to a value."""
    return {"type": "set_state", "state_id": state_id, "value": value}


def op_increment(state_id: str, amount: int | float | dict = 1) -> dict:
    """Increment a state variable."""
    return {"type": "increment", "state_id": state_id, "amount": amount}


def op_max_state(state_id: str, value: dict) -> dict:
    """Set state to max of current and new value."""
    return {"type": "max_state", "state_id": state_id, "value": value}


# =============================================================================
# Actions - What to do on entry/exit
# =============================================================================

def action_set_holdings(allocation: float = 0.95) -> dict:
    """Set portfolio allocation."""
    return {"type": "set_holdings", "allocation": allocation}


def action_liquidate() -> dict:
    """Liquidate position."""
    return {"type": "liquidate"}


# =============================================================================
# Entry Rule - Define entry conditions and actions
# =============================================================================

def entry_rule(
    condition: dict,
    allocation: float = 0.95,
    on_fill: list[dict] | None = None,
) -> dict:
    """Build an entry rule."""
    return {
        "condition": condition,
        "action": action_set_holdings(allocation),
        "on_fill": on_fill or [],
    }


# =============================================================================
# Exit Rules - Define exit conditions and actions
# =============================================================================

def exit_rule(
    id: str,
    condition: dict,
    priority: int = 1,
    action: dict | None = None,
) -> dict:
    """Build a generic exit rule."""
    return {
        "id": id,
        "condition": condition,
        "action": action or action_liquidate(),
        "priority": priority,
    }


def exit_profit_target(target_pct: float, priority: int = 2) -> dict:
    """Exit when profit reaches target percentage.

    Requires 'entry_price' state variable to be set on entry.
    """
    profit_expr = val_expr(
        "/",
        val_expr("-", val_price("close"), val_state("entry_price")),
        val_state("entry_price"),
    )
    return exit_rule(
        id="profit_target",
        condition=cond_compare(profit_expr, ">=", val_literal(target_pct)),
        priority=priority,
    )


def exit_stop_loss(stop_pct: float, priority: int = 1) -> dict:
    """Exit when loss reaches stop percentage.

    Requires 'entry_price' state variable to be set on entry.
    """
    loss_expr = val_expr(
        "/",
        val_expr("-", val_state("entry_price"), val_price("close")),
        val_state("entry_price"),
    )
    return exit_rule(
        id="stop_loss",
        condition=cond_compare(loss_expr, ">=", val_literal(stop_pct)),
        priority=priority,
    )


def exit_band_upper(indicator_id: str, priority: int = 2) -> dict:
    """Exit when price reaches upper band."""
    return exit_rule(
        id="band_exit_upper",
        condition=cond_compare(
            val_price("close"), ">=", val_ind_band(indicator_id, "upper")
        ),
        priority=priority,
    )


def exit_band_lower(indicator_id: str, priority: int = 2) -> dict:
    """Exit when price reaches lower band."""
    return exit_rule(
        id="band_exit_lower",
        condition=cond_compare(
            val_price("close"), "<=", val_ind_band(indicator_id, "lower")
        ),
        priority=priority,
    )


def exit_trailing_stop(
    trail_pct: float,
    priority: int = 1,
) -> dict:
    """Trailing stop exit.

    Requires:
    - 'entry_price' state variable set on entry
    - 'max_price' state variable updated on_bar_invested
    """
    # Calculate trailing stop level as max_price * (1 - trail_pct)
    stop_level = val_expr(
        "*",
        val_state("max_price"),
        val_literal(1 - trail_pct),
    )
    return exit_rule(
        id="trailing_stop",
        condition=cond_compare(val_price("close"), "<=", stop_level),
        priority=priority,
    )


def exit_time_stop(bars: int, priority: int = 3) -> dict:
    """Exit after N bars in position.

    Requires 'bars_since_entry' state variable.
    """
    return exit_rule(
        id="time_stop",
        condition=cond_compare(val_state("bars_since_entry"), ">=", val_literal(bars)),
        priority=priority,
    )


def exit_indicator_cross(
    indicator_id: str,
    threshold: float,
    op: str = "<",
    priority: int = 2,
) -> dict:
    """Exit when indicator crosses threshold."""
    return exit_rule(
        id=f"{indicator_id}_exit",
        condition=cond_compare(val_ind(indicator_id), op, val_literal(threshold)),
        priority=priority,
    )


# =============================================================================
# Gates - Filter when entries are allowed
# =============================================================================

def gate_rule(id: str, condition: dict, mode: str = "block") -> dict:
    """Build a gate rule.

    Args:
        id: Gate identifier
        condition: When this is TRUE, the gate allows entry (if mode='allow')
                   or blocks entry (if mode='block')
        mode: 'allow' = condition must be true to enter
              'block' = if condition is true, entry is blocked
    """
    return {
        "id": id,
        "condition": condition,
        "mode": mode,
    }


def gate_regime_trend(
    direction: Literal["bullish", "bearish"],
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> dict:
    """Gate that only allows entries in specified trend direction.

    Uses EMA crossover as trend proxy.
    """
    if direction == "bullish":
        condition = cond_compare(
            val_ind(f"gate_ema_{ema_fast}"), ">", val_ind(f"gate_ema_{ema_slow}")
        )
    else:
        condition = cond_compare(
            val_ind(f"gate_ema_{ema_fast}"), "<", val_ind(f"gate_ema_{ema_slow}")
        )
    return gate_rule(id=f"trend_gate_{direction}", condition=condition, mode="allow")


def gate_regime_volatility(
    mode: Literal["high", "low"],
    atr_threshold: float,
    atr_period: int = 14,
) -> dict:
    """Gate based on volatility regime.

    'high' mode: only enter when ATR > threshold (high vol environment)
    'low' mode: only enter when ATR < threshold (low vol environment)
    """
    if mode == "high":
        condition = cond_compare(val_ind("gate_atr"), ">", val_literal(atr_threshold))
    else:
        condition = cond_compare(val_ind("gate_atr"), "<", val_literal(atr_threshold))
    return gate_rule(id=f"volatility_gate_{mode}", condition=condition, mode="allow")


def gate_time_filter_rule(
    start_hour: int,
    end_hour: int,
    start_minute: int = 0,
    end_minute: int = 0,
    days_of_week: list[int] | None = None,
) -> dict:
    """Gate that only allows entries during specific time window."""
    return gate_rule(
        id="time_filter_gate",
        condition=cond_time_filter(start_hour, end_hour, start_minute, end_minute, days_of_week),
        mode="allow",
    )


def gate_adx_strength(min_adx: float = 25.0) -> dict:
    """Gate that requires minimum ADX (trend strength)."""
    return gate_rule(
        id="adx_strength_gate",
        condition=cond_compare(val_ind("gate_adx"), ">", val_literal(min_adx)),
        mode="allow",
    )


# =============================================================================
# Complete Strategy IR Builder
# =============================================================================

def build_strategy_ir(
    name: str,
    symbol: str = "TESTUSD",
    resolution: str = "Minute",
    indicators: list[dict] | None = None,
    state: list[dict] | None = None,
    entry_condition: dict | None = None,
    entry_allocation: float = 0.95,
    entry_on_fill: list[dict] | None = None,
    exits: list[dict] | None = None,
    gates: list[dict] | None = None,
    overlays: list[dict] | None = None,
    on_bar: list[dict] | None = None,
    on_bar_invested: list[dict] | None = None,
) -> dict:
    """Build a complete strategy IR.

    This is the main function for constructing test strategies.
    """
    strategy_id = name.lower().replace(" ", "-")

    return {
        "strategy_id": strategy_id,
        "strategy_name": name,
        "symbol": symbol,
        "resolution": resolution,
        "indicators": indicators or [],
        "state": state or [],
        "gates": gates or [],
        "overlays": overlays or [],
        "entry": entry_rule(
            condition=entry_condition or cond_true(),
            allocation=entry_allocation,
            on_fill=entry_on_fill,
        ),
        "exits": exits or [],
        "on_bar": on_bar or [],
        "on_bar_invested": on_bar_invested or [],
    }


# =============================================================================
# Pre-built Strategy Templates
# =============================================================================

def strategy_trend_pullback(
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    profit_target: float = 0.02,
    stop_loss: float = 0.01,
) -> dict:
    """Pre-built TrendPullback strategy.

    Entry: EMA fast > EMA slow AND price < BB lower (only once)
    Exits: Profit target OR stop loss OR upper band
    """
    return build_strategy_ir(
        name="TrendPullback",
        indicators=[
            ind_ema("ema_fast", ema_fast),
            ind_ema("ema_slow", ema_slow),
            ind_bb("bb", bb_period, bb_mult),
        ],
        state=[
            state_float("entry_price"),
            state_int("bars_since_entry"),
            state_int("has_entered", default=0),  # Prevent re-entry
        ],
        entry_condition=cond_allof([
            cond_compare(val_state("has_entered"), "==", val_literal(0)),
            cond_compare(val_ind("ema_fast"), ">", val_ind("ema_slow")),
            cond_compare(val_price("close"), "<", val_ind_band("bb", "lower")),
        ]),
        entry_on_fill=[
            op_set_state("entry_price", val_price("close")),
            op_set_state("bars_since_entry", val_literal(0)),
            op_set_state("has_entered", val_literal(1)),
        ],
        exits=[
            exit_profit_target(profit_target, priority=2),
            exit_stop_loss(stop_loss, priority=1),
            exit_band_upper("bb", priority=3),
        ],
        on_bar_invested=[
            op_increment("bars_since_entry"),
        ],
    )


def strategy_trailing_stop(
    entry_condition: dict,
    trail_pct: float = 0.02,
    initial_stop_pct: float = 0.01,
    indicators: list[dict] | None = None,
) -> dict:
    """Strategy with trailing stop exit.

    Requires managing 'max_price' state for trailing calculation.
    """
    return build_strategy_ir(
        name="TrailingStop",
        indicators=indicators or [],
        state=[
            state_float("entry_price"),
            state_float("max_price"),
            state_int("bars_since_entry"),
        ],
        entry_condition=entry_condition,
        entry_on_fill=[
            op_set_state("entry_price", val_price("close")),
            op_set_state("max_price", val_price("close")),
        ],
        exits=[
            exit_trailing_stop(trail_pct, priority=1),
            exit_stop_loss(initial_stop_pct, priority=2),
        ],
        on_bar_invested=[
            op_increment("bars_since_entry"),
            op_max_state("max_price", val_price("high")),
        ],
    )


def strategy_with_gate(
    entry_condition: dict,
    gate: dict,
    indicators: list[dict] | None = None,
    exits: list[dict] | None = None,
) -> dict:
    """Strategy with a gate condition."""
    return build_strategy_ir(
        name="GatedStrategy",
        indicators=indicators or [],
        state=[state_float("entry_price")],
        entry_condition=entry_condition,
        entry_on_fill=[op_set_state("entry_price", val_price("close"))],
        gates=[gate],
        exits=exits or [],
    )

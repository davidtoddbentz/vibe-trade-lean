"""Condition evaluator registry for StrategyRuntime.

This module provides a registry pattern for evaluating conditions,
replacing the large if/elif chain in StrategyRuntime._evaluate_condition().

Design:
- Evaluators are functions that take (condition, bar, runtime) and return bool
- `runtime` is the StrategyRuntime instance, providing access to:
  - runtime._resolve_value(ref, bar) - resolve ValueRefs
  - runtime.indicators - LEAN indicator instances
  - runtime.rolling_windows - historical value storage
  - Other runtime state needed for condition evaluation
- Basic conditions (compare, allOf, anyOf, not) use recursion via evaluate_condition
- Complex conditions delegate to runtime methods (preserving existing behavior)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Any

if TYPE_CHECKING:
    # Avoid circular import - StrategyRuntime imports this module
    pass


class CompareOp(Enum):
    """Comparison operators for conditions."""

    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    EQ = "=="
    NEQ = "!="

    def apply(self, left: float, right: float) -> bool:
        """Apply the comparison operator."""
        if left is None or right is None:
            return False
        ops = {
            CompareOp.LT: lambda l, r: l < r,
            CompareOp.LTE: lambda l, r: l <= r,
            CompareOp.GT: lambda l, r: l > r,
            CompareOp.GTE: lambda l, r: l >= r,
            CompareOp.EQ: lambda l, r: l == r,
            CompareOp.NEQ: lambda l, r: l != r,
        }
        return ops[self](left, right)


# Type alias for condition evaluator functions
# Signature: (condition: dict, bar: Any, runtime: "StrategyRuntime") -> bool
ConditionEvaluator = Callable[[dict, Any, Any], bool]


def _evaluate_compare(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a compare condition."""
    left_val = runtime._resolve_value(condition.get("left"), bar)
    right_val = runtime._resolve_value(condition.get("right"), bar)
    op_str = condition.get("op")
    op = CompareOp(op_str)
    return op.apply(left_val, right_val)


def _evaluate_allof(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate an allOf condition (logical AND)."""
    for sub in condition.get("conditions", []):
        if not evaluate_condition(sub, bar, runtime):
            return False
    return True


def _evaluate_anyof(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate an anyOf condition (logical OR)."""
    for sub in condition.get("conditions", []):
        if evaluate_condition(sub, bar, runtime):
            return True
    return False


def _evaluate_not(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a not condition (logical NOT)."""
    inner = condition.get("condition")
    return not evaluate_condition(inner, bar, runtime)


def _evaluate_regime(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a regime condition - delegates to runtime."""
    return runtime._evaluate_regime(condition, bar)


def _evaluate_cross(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a cross condition - delegates to runtime."""
    return runtime._evaluate_cross(condition, bar)


def _evaluate_squeeze(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a squeeze condition - delegates to runtime."""
    return runtime._evaluate_squeeze(condition, bar)


def _evaluate_breakout(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a breakout condition - delegates to runtime."""
    return runtime._evaluate_breakout(condition, bar)


def _evaluate_spread(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a spread condition - delegates to runtime."""
    return runtime._evaluate_spread(condition, bar)


def _evaluate_intermarket(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate an intermarket condition - delegates to runtime."""
    return runtime._evaluate_intermarket(condition, bar)


def _evaluate_time_filter(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a time filter condition - delegates to runtime."""
    return runtime._evaluate_time_filter(condition, bar)


def _evaluate_state_condition(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a state condition - delegates to runtime."""
    return runtime._evaluate_state_condition(condition, bar)


def _evaluate_gap(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a gap condition - delegates to runtime."""
    return runtime._evaluate_gap(condition, bar)


def _evaluate_trailing_breakout(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a trailing breakout condition - delegates to runtime."""
    return runtime._evaluate_trailing_breakout(condition, bar)


def _evaluate_trailing_state(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a trailing state condition - delegates to runtime."""
    return runtime._evaluate_trailing_state(condition, bar)


def _evaluate_sequence(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a sequence condition - delegates to runtime."""
    return runtime._evaluate_sequence(condition, bar)


def _evaluate_event_window(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate an event window condition - delegates to runtime."""
    return runtime._evaluate_event_window(condition, bar)


def _evaluate_multi_leader_intermarket(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a multi-leader intermarket condition - delegates to runtime."""
    return runtime._evaluate_multi_leader_intermarket(condition, bar)


def _evaluate_liquidity_sweep(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a liquidity sweep condition - delegates to runtime."""
    return runtime._evaluate_liquidity_sweep_typed(condition, bar)


def _evaluate_flag_pattern(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a flag pattern condition - delegates to runtime."""
    return runtime._evaluate_flag_pattern_typed(condition, bar)


def _evaluate_pennant_pattern(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a pennant pattern condition - delegates to runtime."""
    return runtime._evaluate_pennant_pattern_typed(condition, bar)


def _evaluate_candlestick(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a candlestick pattern condition.

    Checks if a specified candlestick pattern is currently detected.

    Condition schema:
    {
        "type": "candlestick_pattern",
        "pattern": "doji",           # Pattern name (required)
        "direction": "bullish",      # "bullish", "bearish", or "any" (default: "any")
        "indicator_id": "candle_doji" # Optional: specific indicator ID
    }

    The condition is true if:
    - direction="bullish": pattern value == 1
    - direction="bearish": pattern value == -1
    - direction="any": pattern value != 0 (either bullish or bearish)
    """
    pattern = condition.get("pattern", "doji")
    direction = condition.get("direction", "any").lower()
    indicator_id = condition.get("indicator_id", f"candle_{pattern}")

    # Get the candlestick pattern indicator value
    indicator = runtime.indicators.get(indicator_id)
    if indicator is None:
        # Try to find by pattern name
        for ind_id, ind in runtime.indicators.items():
            if pattern.lower() in ind_id.lower() and "candle" in ind_id.lower():
                indicator = ind
                break

    if indicator is None:
        # Pattern indicator not found - return False
        return False

    # Get current value: 0 = no pattern, +1 = bullish, -1 = bearish
    value = indicator.Current.Value

    if direction == "bullish":
        return value == 1
    elif direction == "bearish":
        return value == -1
    else:  # "any"
        return value != 0


# Registry mapping condition type -> evaluator function
CONDITION_EVALUATORS: dict[str, ConditionEvaluator] = {
    # Basic logic conditions (self-contained)
    "compare": _evaluate_compare,
    "allOf": _evaluate_allof,
    "anyOf": _evaluate_anyof,
    "not": _evaluate_not,
    # Market conditions (delegate to runtime)
    "regime": _evaluate_regime,
    "cross": _evaluate_cross,
    "squeeze": _evaluate_squeeze,
    "breakout": _evaluate_breakout,
    "spread": _evaluate_spread,
    "intermarket": _evaluate_intermarket,
    "time_filter": _evaluate_time_filter,
    "state_condition": _evaluate_state_condition,
    "gap": _evaluate_gap,
    "trailing_breakout": _evaluate_trailing_breakout,
    "trailing_state": _evaluate_trailing_state,
    "sequence": _evaluate_sequence,
    "event_window": _evaluate_event_window,
    "multi_leader_intermarket": _evaluate_multi_leader_intermarket,
    # Pattern conditions
    "liquidity_sweep": _evaluate_liquidity_sweep,
    "flag_pattern": _evaluate_flag_pattern,
    "pennant_pattern": _evaluate_pennant_pattern,
    # Candlestick patterns (Phase 5)
    "candlestick_pattern": _evaluate_candlestick,
    "candlestick": _evaluate_candlestick,  # Alias
}


def evaluate_condition(condition: dict, bar: Any, runtime: Any) -> bool:
    """Evaluate a condition from IR using the registry.

    Args:
        condition: IR condition dict with 'type' field
        bar: Current market data bar
        runtime: StrategyRuntime instance for state access

    Returns:
        Boolean result of condition evaluation

    Raises:
        RuntimeError: If condition type is not registered
    """
    if not condition:
        return True

    cond_type = condition.get("type")
    evaluator = CONDITION_EVALUATORS.get(cond_type)

    if evaluator is None:
        raise RuntimeError(f"Unimplemented condition type: {cond_type}")

    return evaluator(condition, bar, runtime)

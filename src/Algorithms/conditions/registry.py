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
- All condition types use typed evaluators in conditions/*.py (EvalContext + Pydantic models)

Phase 7: "compare" uses typed CompareCondition and EvalContext (conditions/compare.py).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, cast

from pydantic import TypeAdapter

from indicators import resolve_value as _resolve_value_impl
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    CrossCondition,
    EventWindowCondition,
    FlagPatternCondition,
    GapCondition,
    IntermarketCondition,
    LiquiditySweepCondition,
    MultiLeaderIntermarketCondition,
    NotCondition,
    PennantPatternCondition,
    SequenceCondition,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    TimeFilterCondition,
    TrailingBreakoutCondition,
    TrailingStateCondition,
    RegimeCondition,
)

from .basic import (
    evaluate_allof as _evaluate_allof_typed,
    evaluate_anyof as _evaluate_anyof_typed,
    evaluate_not as _evaluate_not_typed,
)
from .context import EvalContext
from .compare import evaluate_compare as _evaluate_compare_typed
from .cross import evaluate_cross as _evaluate_cross_typed
from .time_filter import evaluate_time_filter as _evaluate_time_filter_typed
from .state_condition import evaluate_state_condition as _evaluate_state_condition_typed
from .sequence import evaluate_sequence as _evaluate_sequence_typed
from .event_window import evaluate_event_window as _evaluate_event_window_typed
from .gap import evaluate_gap as _evaluate_gap_typed
from .breakout import evaluate_breakout as _evaluate_breakout_typed
from .squeeze import evaluate_squeeze as _evaluate_squeeze_typed
from .trailing_breakout import evaluate_trailing_breakout as _evaluate_trailing_breakout_typed
from .trailing_state import evaluate_trailing_state as _evaluate_trailing_state_typed
from .spread import evaluate_spread as _evaluate_spread_typed
from .intermarket import evaluate_intermarket as _evaluate_intermarket_typed
from .multi_leader_intermarket import (
    evaluate_multi_leader_intermarket as _evaluate_multi_leader_intermarket_typed,
)
from .liquidity_sweep import evaluate_liquidity_sweep as _evaluate_liquidity_sweep_typed
from .flag_pattern import evaluate_flag_pattern as _evaluate_flag_pattern_typed
from .pennant_pattern import evaluate_pennant_pattern as _evaluate_pennant_pattern_typed
from .regime import evaluate_regime as _evaluate_regime_typed

if TYPE_CHECKING:
    # Avoid circular import - StrategyRuntime imports this module
    pass

CompareConditionAdapter = TypeAdapter(CompareCondition)
AllOfConditionAdapter = TypeAdapter(AllOfCondition)
AnyOfConditionAdapter = TypeAdapter(AnyOfCondition)
NotConditionAdapter = TypeAdapter(NotCondition)
CrossConditionAdapter = TypeAdapter(CrossCondition)
TimeFilterConditionAdapter = TypeAdapter(TimeFilterCondition)
StateConditionAdapter = TypeAdapter(StateCondition)
SequenceConditionAdapter = TypeAdapter(SequenceCondition)
EventWindowConditionAdapter = TypeAdapter(EventWindowCondition)
GapConditionAdapter = TypeAdapter(GapCondition)
BreakoutConditionAdapter = TypeAdapter(BreakoutCondition)
SqueezeConditionAdapter = TypeAdapter(SqueezeCondition)
TrailingBreakoutConditionAdapter = TypeAdapter(TrailingBreakoutCondition)
TrailingStateConditionAdapter = TypeAdapter(TrailingStateCondition)
SpreadConditionAdapter = TypeAdapter(SpreadCondition)
IntermarketConditionAdapter = TypeAdapter(IntermarketCondition)
MultiLeaderIntermarketConditionAdapter = TypeAdapter(MultiLeaderIntermarketCondition)
LiquiditySweepConditionAdapter = TypeAdapter(LiquiditySweepCondition)
FlagPatternConditionAdapter = TypeAdapter(FlagPatternCondition)
PennantPatternConditionAdapter = TypeAdapter(PennantPatternCondition)
RegimeConditionAdapter = TypeAdapter(RegimeCondition)

_DAY_NAME_TO_INT = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _build_eval_context(runtime: Any) -> EvalContext:
    """Build EvalContext from StrategyRuntime for typed evaluators."""
    def resolve_value(ref: Any, bar: Any) -> float:
        return _resolve_value_impl(
            ref,
            bar,
            indicator_registry=runtime.indicator_registry,
            state=runtime.state,
            current_time=runtime.Time,
            rolling_windows=runtime.rolling_windows,
        )
    def recurse(cond: Any, bar: Any) -> bool:
        return evaluate_condition(cond, bar, runtime)

    return EvalContext(
        resolve_value=resolve_value,
        evaluate_condition=recurse,
        state=runtime.state,
        current_time=runtime.Time,
        cross_state=getattr(runtime, "_cross_prev", {}),
        rolling_windows=getattr(runtime, "rolling_windows", {}),
        rolling_minmax=getattr(runtime, "rolling_minmax", {}),
        indicators=getattr(runtime, "indicators", {}),
        breakout_prev_max=getattr(runtime, "_breakout_prev_max", {}),
        breakout_prev_min=getattr(runtime, "_breakout_prev_min", {}),
    )


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
            CompareOp.LT: lambda left_val, right_val: left_val < right_val,
            CompareOp.LTE: lambda left_val, right_val: left_val <= right_val,
            CompareOp.GT: lambda left_val, right_val: left_val > right_val,
            CompareOp.GTE: lambda left_val, right_val: left_val >= right_val,
            CompareOp.EQ: lambda left_val, right_val: left_val == right_val,
            CompareOp.NEQ: lambda left_val, right_val: left_val != right_val,
        }
        return ops[self](left, right)


# Type alias: condition is Condition (typed) from StrategyIR or dict (legacy/candlestick)
ConditionEvaluator = Callable[[Any, Any, Any], bool]


def _evaluate_compare(condition: Any, bar: Any, runtime: Any) -> bool:
    """Evaluate a compare condition. Condition is typed from StrategyIR or dict (legacy)."""
    ctx = _build_eval_context(runtime)
    typed = cast(CompareCondition, CompareConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(CompareCondition, condition)
    return _evaluate_compare_typed(typed, bar, ctx)


def _evaluate_allof(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(AllOfCondition, AllOfConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(AllOfCondition, condition)
    return _evaluate_allof_typed(typed, bar, ctx)


def _evaluate_anyof(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(AnyOfCondition, AnyOfConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(AnyOfCondition, condition)
    return _evaluate_anyof_typed(typed, bar, ctx)


def _evaluate_not(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(NotCondition, NotConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(NotCondition, condition)
    return _evaluate_not_typed(typed, bar, ctx)


def _evaluate_regime(condition: Any, bar: Any, runtime: Any) -> bool:
    """Evaluate a regime condition via typed evaluator (conditions/regime.py)."""
    ctx = _build_eval_context(runtime)
    typed = (
        cast(RegimeCondition, RegimeConditionAdapter.validate_python(condition))
        if isinstance(condition, dict)
        else cast(RegimeCondition, condition)
    )
    return _evaluate_regime_typed(typed, bar, ctx)


def _evaluate_cross(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    if isinstance(condition, dict):
        cond = dict(condition)
        d = cond.get("direction", "above")
        if d in ("cross_above", "above"):
            cond["direction"] = "above"
        elif d in ("cross_below", "below"):
            cond["direction"] = "below"
        typed = CrossConditionAdapter.validate_python(cond)
    else:
        typed = cast(CrossCondition, condition)
    return _evaluate_cross_typed(typed, bar, ctx)


def _evaluate_squeeze(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(SqueezeCondition, SqueezeConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(SqueezeCondition, condition)
    return _evaluate_squeeze_typed(typed, bar, ctx)


def _evaluate_breakout(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(BreakoutCondition, BreakoutConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(BreakoutCondition, condition)
    return _evaluate_breakout_typed(typed, bar, ctx)


def _evaluate_spread(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(SpreadCondition, SpreadConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(SpreadCondition, condition)
    return _evaluate_spread_typed(typed, bar, ctx)


def _evaluate_intermarket(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(IntermarketCondition, IntermarketConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(IntermarketCondition, condition)
    return _evaluate_intermarket_typed(typed, bar, ctx)


def _evaluate_time_filter(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    if isinstance(condition, dict):
        cond = dict(condition)
        dow = cond.get("days_of_week", [])
        if dow:
            allowed = []
            for d in dow:
                if isinstance(d, int):
                    allowed.append(d)
                elif isinstance(d, str):
                    allowed.append(_DAY_NAME_TO_INT.get(d.lower(), -1))
            cond["days_of_week"] = allowed
        typed = TimeFilterConditionAdapter.validate_python(cond)
    else:
        typed = cast(TimeFilterCondition, condition)
    return _evaluate_time_filter_typed(typed, bar, ctx)


def _evaluate_state_condition(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(StateCondition, StateConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(StateCondition, condition)
    return _evaluate_state_condition_typed(typed, bar, ctx)


def _evaluate_gap(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(GapCondition, GapConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(GapCondition, condition)
    return _evaluate_gap_typed(typed, bar, ctx)


def _evaluate_trailing_breakout(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(TrailingBreakoutCondition, TrailingBreakoutConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(TrailingBreakoutCondition, condition)
    return _evaluate_trailing_breakout_typed(typed, bar, ctx)


def _evaluate_trailing_state(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(TrailingStateCondition, TrailingStateConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(TrailingStateCondition, condition)
    return _evaluate_trailing_state_typed(typed, bar, ctx)


def _evaluate_sequence(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(SequenceCondition, SequenceConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(SequenceCondition, condition)
    return _evaluate_sequence_typed(typed, bar, ctx)


def _evaluate_event_window(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(EventWindowCondition, EventWindowConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(EventWindowCondition, condition)
    return _evaluate_event_window_typed(typed, bar, ctx)


def _evaluate_multi_leader_intermarket(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(MultiLeaderIntermarketCondition, MultiLeaderIntermarketConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(MultiLeaderIntermarketCondition, condition)
    return _evaluate_multi_leader_intermarket_typed(typed, bar, ctx)


def _evaluate_liquidity_sweep(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(LiquiditySweepCondition, LiquiditySweepConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(LiquiditySweepCondition, condition)
    return _evaluate_liquidity_sweep_typed(typed, bar, ctx)


def _evaluate_flag_pattern(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(FlagPatternCondition, FlagPatternConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(FlagPatternCondition, condition)
    return _evaluate_flag_pattern_typed(typed, bar, ctx)


def _evaluate_pennant_pattern(condition: Any, bar: Any, runtime: Any) -> bool:
    ctx = _build_eval_context(runtime)
    typed = cast(PennantPatternCondition, PennantPatternConditionAdapter.validate_python(condition)) if isinstance(condition, dict) else cast(PennantPatternCondition, condition)
    return _evaluate_pennant_pattern_typed(typed, bar, ctx)


def _evaluate_candlestick(condition: Any, bar: Any, runtime: Any) -> bool:
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
    pattern = getattr(condition, "pattern", None) or (condition.get("pattern", "doji") if isinstance(condition, dict) else "doji")
    direction = (getattr(condition, "direction", None) or (condition.get("direction", "any") if isinstance(condition, dict) else "any")).lower()
    indicator_id = getattr(condition, "indicator_id", None) or (condition.get("indicator_id", f"candle_{pattern}") if isinstance(condition, dict) else f"candle_{pattern}")

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


def evaluate_condition(condition: Any, bar: Any, runtime: Any) -> bool:
    """Evaluate a condition from IR using the registry.

    Condition is usually a typed Condition (Pydantic model) from StrategyIR;
    dict is supported for legacy/candlestick paths. No per-eval validation
    when condition is already typed (validated once at request load).
    """
    if condition is None or (isinstance(condition, dict) and not condition):
        return True

    cond_type = getattr(condition, "type", None) or (
        condition.get("type") if isinstance(condition, dict) else None
    )
    evaluator = CONDITION_EVALUATORS.get(cond_type)

    if evaluator is None:
        raise RuntimeError(f"Unimplemented condition type: {cond_type}")

    return evaluator(condition, bar, runtime)

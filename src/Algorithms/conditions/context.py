"""Minimal context for typed condition evaluators.

Phase 7: Replaces passing the full StrategyRuntime to evaluators.
Each evaluator receives only what it needs via EvalContext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class EvalContext:
    """Minimal context needed by condition evaluators."""

    resolve_value: Callable[[Any, Any], float]  # (ValueRef, bar) -> float
    evaluate_condition: Callable[[Any, Any], bool]  # (condition dict, bar) -> bool; for recursion
    state: dict[str, Any]
    current_time: Any
    cross_state: dict[str, tuple[float, float]]  # key -> (left_val, right_val) for cross detection
    rolling_windows: dict[str, Any]  # id -> { "window": RollingWindow-like } for gap, etc.
    indicators: dict[str, Any]  # id -> LEAN indicator (includes MAX/MIN for level detection)
    breakout_prev_max: dict[str, float]  # max_ind_id -> previous N-bar max
    breakout_prev_min: dict[str, float]  # min_ind_id -> previous N-bar min

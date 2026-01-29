"""State operation execution logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any, Callable

from vibe_trade_shared.models.ir import (
    SetStateAction,
    IncrementStateAction,
    MaxStateAction,
    MinStateAction,
    SetStateFromConditionAction,
    StateOp,
)


def execute_state_op(
    op: StateOp,
    bar: Any,
    state: dict[str, float],
    resolve_value_func: Callable,
    evaluate_condition_func: Callable,
    log_func: Callable[[str], None],
) -> None:
    """Execute a state operation.

    Args:
        op: Typed StateOp from IR
        bar: Current bar data
        state: Runtime state dict (will be modified)
        resolve_value_func: Function to resolve ValueRef to float
        evaluate_condition_func: Function to evaluate conditions
        log_func: Logging function
    """
    if not op:
        return

    if isinstance(op, SetStateAction):
        value = resolve_value_func(op.value, bar)
        state[op.state_id] = value

    elif isinstance(op, IncrementStateAction):
        # State vars are initialized, so direct access is safe
        current = state.get(op.state_id, 0.0)
        # amount can be ValueRef | int | float
        if isinstance(op.amount, (int, float)):
            increment = op.amount
        else:
            increment = resolve_value_func(op.amount, bar)
        state[op.state_id] = current + increment

    elif isinstance(op, MaxStateAction):
        new_value = resolve_value_func(op.value, bar)
        # State vars are initialized, so direct access is safe
        current = state.get(op.state_id, float("-inf"))
        if new_value > current:
            state[op.state_id] = new_value

    elif isinstance(op, MinStateAction):
        new_value = resolve_value_func(op.value, bar)
        # State vars are initialized, so direct access is safe
        current = state.get(op.state_id, float("inf"))
        if new_value < current:
            state[op.state_id] = new_value

    elif isinstance(op, SetStateFromConditionAction):
        result = evaluate_condition_func(op.condition, bar)
        state[op.state_id] = 1.0 if result else 0.0

    else:
        log_func(f"⚠️ Unknown state op type: {type(op)}")

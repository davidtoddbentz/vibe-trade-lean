"""Typed evaluator for StateCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import StateCondition

from .context import EvalContext


def evaluate_state_condition(
    condition: StateCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate state-based condition with transition tracking.

    Mode 1: current_condition only -> evaluate it (no state).
    Mode 2: outside_condition + inside_condition + state_var -> track
    outside/inside and return True on transition from outside to inside
    when trigger_on_transition is True.
    """
    # Mode 1: Simple current condition check
    if condition.current_condition is not None and condition.outside_condition is None:
        return ctx.evaluate_condition(condition.current_condition, bar)

    # Mode 2: Transition tracking
    if (
        condition.outside_condition is not None
        and condition.inside_condition is not None
        and condition.state_var
    ):
        state_var = condition.state_var
        was_outside = ctx.state.get(state_var, False)

        is_outside = ctx.evaluate_condition(condition.outside_condition, bar)
        is_inside = ctx.evaluate_condition(condition.inside_condition, bar)

        if is_outside:
            ctx.state[state_var] = True
            return False
        if was_outside and is_inside:
            ctx.state[state_var] = False
            return condition.trigger_on_transition
        if is_inside:
            return False

    return False

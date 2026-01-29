"""Typed evaluator for SequenceCondition (Phase 7)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vibe_trade_shared.models.ir import SequenceCondition

from .context import EvalContext


def _sequence_key(condition: SequenceCondition) -> str:
    """Stable key for sequence progress in ctx.state."""
    payload = condition.model_dump(mode="json")
    return "sequence_" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def evaluate_sequence(
    condition: SequenceCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate sequence: steps must occur in order with optional hold_bars/within_bars.

    Returns True on the same bar that the final step completes (not one bar later).
    """
    steps = condition.steps
    if not steps:
        return True

    seq_key = _sequence_key(condition)
    current_step = ctx.state.get(seq_key, 0)
    step_bar_count = ctx.state.get(f"{seq_key}_bars", 0)

    if current_step >= len(steps):
        ctx.state[seq_key] = 0
        ctx.state[f"{seq_key}_bars"] = 0
        return True

    step = steps[current_step]
    step_condition = step.condition
    hold_bars = step.hold_bars
    within_bars = step.within_bars

    # within_bars timeout (steps after the first)
    if current_step > 0 and within_bars is not None:
        if step_bar_count > within_bars:
            ctx.state[seq_key] = 0
            ctx.state[f"{seq_key}_bars"] = 0
            return False

    ctx.state[f"{seq_key}_bars"] = step_bar_count + 1

    step_met = ctx.evaluate_condition(step_condition, bar)
    if step_met:
        if hold_bars is not None:
            hold_count = ctx.state.get(f"{seq_key}_hold", 0) + 1
            ctx.state[f"{seq_key}_hold"] = hold_count
            if hold_count >= hold_bars:
                next_step = current_step + 1
                ctx.state[seq_key] = next_step
                ctx.state[f"{seq_key}_bars"] = 0
                ctx.state[f"{seq_key}_hold"] = 0
                if next_step >= len(steps):
                    ctx.state[seq_key] = 0
                    return True
        else:
            next_step = current_step + 1
            ctx.state[seq_key] = next_step
            ctx.state[f"{seq_key}_bars"] = 0
            if next_step >= len(steps):
                ctx.state[seq_key] = 0
                return True
    else:
        ctx.state[f"{seq_key}_hold"] = 0

    return False

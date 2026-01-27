"""Typed evaluator for EventWindowCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import EventWindowCondition

from .context import EvalContext


def evaluate_event_window(
    condition: EventWindowCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate event window condition.

    Uses ctx.state["_event_calendar"] and ctx.state["_bar_count"].
    within = True when current bar is in any matching event window;
    outside = True when no event data or not in any window.
    """
    events = ctx.state.get("_event_calendar", {})

    if not events:
        return condition.mode == "outside"

    current_bar = ctx.state.get("_bar_count", 0)

    for event_type in condition.event_types:
        event_list = events.get(event_type, [])
        for event in event_list:
            event_bar = event.get("bar_index", -1)
            bars_before = current_bar - event_bar
            if -condition.pre_window_bars <= bars_before <= condition.post_window_bars:
                return condition.mode == "within"

    return condition.mode == "outside"

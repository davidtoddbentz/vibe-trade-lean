"""Typed evaluator for TimeFilterCondition (Phase 7)."""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import TimeFilterCondition

from .context import EvalContext


def evaluate_time_filter(
    condition: TimeFilterCondition, bar: Any, ctx: EvalContext
) -> bool:
    """Evaluate time-based filter: days of week, day of month, time window.

    Uses bar.Time for calendar checks when available (so daily bars evaluate
    as the bar's date); uses ctx.current_time for intra-bar time window.
    """
    # Reference time for calendar: bar period start when available
    t = getattr(bar, "Time", None) if bar is not None else None
    if t is None:
        t = ctx.current_time

    # Day of week (0=Monday, 6=Sunday)
    if condition.days_of_week:
        current_day = t.weekday()
        if current_day not in condition.days_of_week:
            return False

    # Day of month
    if condition.days_of_month:
        current_dom = t.day
        if current_dom not in condition.days_of_month:
            return False

    # Time window: "HH:MM-HH:MM" or "HHMM-HHMM" uses algorithm/current time
    time_window = (condition.time_window or "").strip()
    if time_window:
        try:
            start_str, end_str = (s.strip() for s in time_window.split("-", 1))
            # Parse HH:MM or HHMM
            if ":" in start_str:
                start_hour, start_min = map(int, start_str.split(":"))
                end_hour, end_min = map(int, end_str.split(":"))
            else:
                start_hour, start_min = int(start_str[:2]), int(start_str[2:4])
                end_hour, end_min = int(end_str[:2]), int(end_str[2:4])

            current_minutes = ctx.current_time.hour * 60 + ctx.current_time.minute
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min

            if start_minutes <= end_minutes:
                if not (start_minutes <= current_minutes <= end_minutes):
                    return False
            else:
                # Overnight window (e.g. 22:00-06:00)
                if not (
                    current_minutes >= start_minutes or current_minutes <= end_minutes
                ):
                    return False
        except (ValueError, AttributeError):
            pass  # Invalid format, skip filter

    return True

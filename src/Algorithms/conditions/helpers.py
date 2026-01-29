"""Shared helper functions for condition evaluators."""

from __future__ import annotations

from typing import Any


def compute_bb_width_percentile(
    bb_ind: Any,
    rolling_window: Any | None,
) -> float | None:
    """Compute Bollinger Band width percentile.

    Args:
        bb_ind: Bollinger Bands indicator with UpperBand, MiddleBand, LowerBand
        rolling_window: Dict with "window" key containing RollingWindow of historical widths

    Returns:
        Percentile (0-100) or None if not computable
    """
    upper = bb_ind.UpperBand.Current.Value
    lower = bb_ind.LowerBand.Current.Value
    middle = bb_ind.MiddleBand.Current.Value
    if middle == 0:
        return None

    width = (upper - lower) / middle

    if rolling_window and getattr(rolling_window.get("window"), "IsReady", False):
        window = rolling_window["window"]
        widths = list(window) if hasattr(window, "__iter__") else []
        if widths:
            return sum(1 for w in widths if w < width) / len(widths) * 100

    # Fallback: return width * 100 (raw, not percentile)
    return width * 100

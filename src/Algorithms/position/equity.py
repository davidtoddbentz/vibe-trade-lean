"""Equity curve tracking for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any
from AlgorithmImports import Resolution


def track_equity(
    equity: float,
    cash: float,
    holdings: float,
    drawdown: float,
    current_time: Any,
    bar_count: int,
    resolution: Any,
    equity_curve: list[dict[str, Any]],
    peak_equity: float,
    max_drawdown: float,
) -> tuple[float, float]:
    """Track portfolio equity for equity curve and update peak/drawdown.

    Args:
        equity: Current total portfolio value
        cash: Current cash
        holdings: Current holdings value
        drawdown: Current drawdown percentage
        current_time: Current timestamp
        bar_count: Current bar index
        resolution: Data resolution
        equity_curve: List to append equity samples to
        peak_equity: Current peak equity
        max_drawdown: Current max drawdown

    Returns:
        Tuple of (updated_peak_equity, updated_max_drawdown)
    """
    # Update peak and drawdown
    new_peak = equity if equity > peak_equity else peak_equity
    new_max_drawdown = drawdown if drawdown > max_drawdown else max_drawdown

    # Sample equity curve based on resolution
    # For minute data: every 60 bars (~hourly)
    # For hourly data: every bar
    # For daily data: every bar
    # Target: ~100-200 points for a typical backtest
    if resolution == Resolution.Minute:
        sample_interval = 60  # Every 60 minutes = hourly
    elif resolution == Resolution.Hour:
        sample_interval = 1   # Every hour
    else:
        sample_interval = 1   # Daily or other: every bar

    if bar_count == 0 or bar_count % sample_interval == 0:
        equity_curve.append({
            "time": str(current_time),
            "equity": float(equity),
            "cash": float(cash),
            "holdings": float(holdings),
            "drawdown": float(drawdown),
        })

    return new_peak, new_max_drawdown

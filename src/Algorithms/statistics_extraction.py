"""Utilities for extracting LEAN portfolio statistics in JSON-friendly form."""

from __future__ import annotations

from typing import Any


def _safe_get_float(obj: Any, attr: str) -> float | None:
    """Safely read an attribute from an object and convert it to float.

    Args:
        obj: Object expected to expose the attribute.
        attr: Attribute name to retrieve.

    Returns:
        Float value when available and convertible, otherwise None.
    """
    try:
        value = getattr(obj, attr)
    except AttributeError:
        return None

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_lean_statistics(statistics: Any) -> dict[str, float | None]:
    """Extract portfolio statistics from LEAN's Statistics object.

    LEAN exposes the PortfolioStatistics object under
    `statistics.TotalPerformance.PortfolioStatistics`. This helper converts the
    24 known properties into a JSON-serializable dictionary with snake_case keys
    and rate conversions expressed as percentages.

    Args:
        statistics: LEAN's self.Statistics object from a completed backtest.

    Returns:
        Dictionary mapping snake_case statistic names to float values or None
        when unavailable.
    """
    fields = (
        ("AverageWinRate", "average_win_rate", True),
        ("AverageLossRate", "average_loss_rate", True),
        ("ProfitLossRatio", "profit_loss_ratio", False),
        ("WinRate", "win_rate", True),
        ("LossRate", "loss_rate", True),
        ("Expectancy", "expectancy", False),
        ("StartEquity", "start_equity", False),
        ("EndEquity", "end_equity", False),
        ("CompoundingAnnualReturn", "compounding_annual_return", True),
        ("Drawdown", "drawdown", True),
        ("TotalNetProfit", "total_net_profit", False),
        ("SharpeRatio", "sharpe_ratio", False),
        ("ProbabilisticSharpeRatio", "probabilistic_sharpe_ratio", False),
        ("SortinoRatio", "sortino_ratio", False),
        ("Alpha", "alpha", False),
        ("Beta", "beta", False),
        ("AnnualStandardDeviation", "annual_standard_deviation", True),
        ("AnnualVariance", "annual_variance", False),
        ("InformationRatio", "information_ratio", False),
        ("TrackingError", "tracking_error", False),
        ("TreynorRatio", "treynor_ratio", False),
        ("PortfolioTurnover", "portfolio_turnover", False),
        ("ValueAtRisk99", "value_at_risk_99", False),
        ("ValueAtRisk95", "value_at_risk_95", False),
    )

    result: dict[str, float | None] = {snake: None for _, snake, _ in fields}

    try:
        portfolio_stats = statistics.TotalPerformance.PortfolioStatistics
    except AttributeError:
        return result

    for pascal, snake, is_rate in fields:
        value = _safe_get_float(portfolio_stats, pascal)
        if value is None:
            continue
        if is_rate:
            value *= 100.0
        result[snake] = value

    return result

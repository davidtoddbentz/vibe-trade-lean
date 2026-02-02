"""Unit tests for LEAN statistics extraction helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from src.Algorithms.statistics_extraction import _safe_get_float, extract_lean_statistics


FIELDS = (
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


def _build_full_mock_stats() -> tuple[dict[str, float], MagicMock]:
    values = {
        "AverageWinRate": 0.12,
        "AverageLossRate": 0.08,
        "ProfitLossRatio": 1.7,
        "WinRate": 0.65,
        "LossRate": 0.35,
        "Expectancy": 0.45,
        "StartEquity": 100000.0,
        "EndEquity": 112000.0,
        "CompoundingAnnualReturn": 0.18,
        "Drawdown": 0.23,
        "TotalNetProfit": 12000.0,
        "SharpeRatio": 1.85,
        "ProbabilisticSharpeRatio": 1.2,
        "SortinoRatio": 2.1,
        "Alpha": 0.04,
        "Beta": 1.1,
        "AnnualStandardDeviation": 0.15,
        "AnnualVariance": 0.0225,
        "InformationRatio": 0.6,
        "TrackingError": 0.03,
        "TreynorRatio": 0.5,
        "PortfolioTurnover": 0.8,
        "ValueAtRisk99": -0.02,
        "ValueAtRisk95": -0.015,
    }

    portfolio_stats = Mock()
    for attr, value in values.items():
        setattr(portfolio_stats, attr, value)

    stats = MagicMock()
    stats.TotalPerformance.PortfolioStatistics = portfolio_stats
    return values, stats


class TestExtractLeanStatistics:
    def test_extract_all_statistics(self):
        values, stats = _build_full_mock_stats()
        result = extract_lean_statistics(stats)

        assert len(result) == 24
        for pascal, snake, is_rate in FIELDS:
            expected = values[pascal] * 100.0 if is_rate else values[pascal]
            assert result[snake] == pytest.approx(expected)

    def test_rate_conversion(self):
        portfolio_stats = SimpleNamespace(
            AverageWinRate=0.5,
            AverageLossRate=0.4,
            WinRate=0.65,
            LossRate=0.35,
            CompoundingAnnualReturn=0.1,
            Drawdown=0.2,
            AnnualStandardDeviation=0.25,
        )
        stats = SimpleNamespace(TotalPerformance=SimpleNamespace(PortfolioStatistics=portfolio_stats))

        result = extract_lean_statistics(stats)

        assert result["average_win_rate"] == pytest.approx(50.0)
        assert result["average_loss_rate"] == pytest.approx(40.0)
        assert result["win_rate"] == pytest.approx(65.0)
        assert result["loss_rate"] == pytest.approx(35.0)
        assert result["compounding_annual_return"] == pytest.approx(10.0)
        assert result["drawdown"] == pytest.approx(20.0)
        assert result["annual_standard_deviation"] == pytest.approx(25.0)

    def test_missing_fields_handled_gracefully(self):
        portfolio_stats = SimpleNamespace(WinRate=0.5, SharpeRatio=1.2, LossRate=None)
        stats = SimpleNamespace(TotalPerformance=SimpleNamespace(PortfolioStatistics=portfolio_stats))

        result = extract_lean_statistics(stats)

        assert result["win_rate"] == pytest.approx(50.0)
        assert result["sharpe_ratio"] == pytest.approx(1.2)
        assert result["loss_rate"] is None
        assert sum(value is None for value in result.values()) >= 21

    def test_missing_statistics_structure(self):
        stats = SimpleNamespace()
        result = extract_lean_statistics(stats)

        assert len(result) == 24
        assert all(value is None for value in result.values())

    def test_json_serializable(self):
        _, stats = _build_full_mock_stats()
        result = extract_lean_statistics(stats)

        payload = json.dumps(result)
        loaded = json.loads(payload)

        assert set(loaded.keys()) == {snake for _, snake, _ in FIELDS}


class TestSafeGetFloat:
    def test_safe_get_float_handles_none(self):
        obj = SimpleNamespace(value=None)
        assert _safe_get_float(obj, "value") is None

    def test_safe_get_float_handles_invalid_types(self):
        obj = SimpleNamespace(bad="not-a-number", worse=["still-bad"])
        assert _safe_get_float(obj, "bad") is None
        assert _safe_get_float(obj, "worse") is None

"""Smoke tests for LEAN HTTP backtest endpoint.

These tests verify basic code functionality:
- HTTP endpoint accepts requests and returns responses
- Basic strategy execution works
- Error handling is correct

Uses Pydantic StrategyIR models serialized to JSON.
For comprehensive strategy testing, see vibe-trade-execution/tests/e2e/.
"""

import pytest
from fastapi.testclient import TestClient

from src.serve_backtest import app
from vibe_trade_shared.models.ir import (
    StrategyIR,
    EntryRule,
    CompareCondition,
    PriceRef,
    LiteralRef,
    SetHoldingsAction,
    CompareOp,
    PriceField,
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_returns_ok(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestBacktestEndpoint:
    """Basic backtest endpoint smoke tests."""

    def test_backtest_executes_successfully(self, client):
        """Backtest with valid input returns success."""
        strategy_ir = StrategyIR(
            strategy_id="smoke-test",
            strategy_name="Smoke Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = [
            {"t": i * 60000, "o": 100, "h": 101, "l": 99, "c": 100 + i, "v": 1000}
            for i in range(5)
        ]

        response = client.post(
            "/backtest",
            json={
                "strategy_ir": strategy_ir.model_dump(),
                "data": {"symbol": "TESTUSD", "resolution": "1m", "bars": bars},
                "config": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-01",
                    "initial_cash": 100000,
                },
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "trades" in result
        assert "summary" in result

    def test_backtest_no_data_returns_validation_error(self, client):
        """Backtest without data returns 422 validation error."""
        strategy_ir = StrategyIR(
            strategy_id="test",
            strategy_name="Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=None,
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        response = client.post(
            "/backtest",
            json={
                "strategy_ir": strategy_ir.model_dump(),
                "data": {"symbol": "TESTUSD", "resolution": "1m"},
                "config": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-01",
                    "initial_cash": 100000,
                },
            },
        )

        assert response.status_code == 422

    def test_backtest_gcs_uri_without_deps_returns_error(self, client):
        """GCS URI returns error when dependencies not installed."""
        strategy_ir = StrategyIR(
            strategy_id="test",
            strategy_name="Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=None,
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        response = client.post(
            "/backtest",
            json={
                "strategy_ir": strategy_ir.model_dump(),
                "data": {
                    "symbol": "TESTUSD",
                    "resolution": "1m",
                    "gcs_uri": "gs://test-bucket/data.parquet",
                },
                "config": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-01",
                    "initial_cash": 100000,
                },
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "error"


class TestResponseStructure:
    """Response structure validation."""

    def test_successful_response_has_required_fields(self, client):
        """Successful backtest has all required response fields."""
        strategy_ir = StrategyIR(
            strategy_id="test",
            strategy_name="Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=None,
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = [{"t": i * 60000, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000} for i in range(5)]

        response = client.post(
            "/backtest",
            json={
                "strategy_ir": strategy_ir.model_dump(),
                "data": {"symbol": "TESTUSD", "resolution": "1m", "bars": bars},
                "config": {"start_date": "2024-01-01", "end_date": "2024-01-01", "initial_cash": 100000},
            },
        )

        result = response.json()
        assert result["status"] == "success"
        assert isinstance(result["trades"], list)
        assert "summary" in result

        summary = result["summary"]
        assert "total_trades" in summary
        assert "winning_trades" in summary
        assert "losing_trades" in summary
        assert "total_pnl" in summary
        assert "total_pnl_pct" in summary

"""Tests for HTTP endpoint functionality.

These tests verify HTTP layer behavior (routing, validation, error handling)
using FastAPI TestClient. They do NOT test actual backtest execution.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.serve_backtest import app, LEANBacktestResponse, LEANBacktestSummary


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client):
        """Health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.json()["status"] == "healthy"


class TestBacktestEndpointValidation:
    """Request validation tests (no LEAN execution)."""

    def test_rejects_missing_strategy_ir(self, client):
        """Reject request without strategy_ir."""
        response = client.post(
            "/backtest",
            json={
                "data": {
                    "symbol": "BTC-USD",
                    "resolution": "1m",
                    "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
                },
                "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            },
        )
        assert response.status_code == 422

    def test_rejects_missing_data(self, client):
        """Reject request without data."""
        response = client.post(
            "/backtest",
            json={
                "strategy_ir": {"strategy_id": "test"},
                "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            },
        )
        assert response.status_code == 422

    def test_rejects_missing_config(self, client):
        """Reject request without config."""
        response = client.post(
            "/backtest",
            json={
                "strategy_ir": {"strategy_id": "test"},
                "data": {
                    "symbol": "BTC-USD",
                    "resolution": "1m",
                    "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
                },
            },
        )
        assert response.status_code == 422

    def test_rejects_data_without_bars_or_gcs(self, client):
        """Reject data input with neither bars nor gcs_uri."""
        response = client.post(
            "/backtest",
            json={
                "strategy_ir": {"strategy_id": "test"},
                "data": {"symbol": "BTC-USD", "resolution": "1m"},
                "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            },
        )
        assert response.status_code == 422


class TestBacktestEndpointMocked:
    """Tests with mocked LEAN execution."""

    def test_returns_success_response_structure(self, client):
        """Successful execution returns proper response structure."""
        mock_response = LEANBacktestResponse(
            status="success",
            trades=[],
            summary=LEANBacktestSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
            ),
            equity_curve=[100000.0],
        )

        with patch(
            "src.serve_backtest._run_lean_backtest",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/backtest",
                json={
                    "strategy_ir": {"strategy_id": "test"},
                    "data": {
                        "symbol": "BTC-USD",
                        "resolution": "1m",
                        "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
                    },
                    "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
                },
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            assert "trades" in result
            assert "summary" in result
            assert "equity_curve" in result

    def test_returns_error_response_on_failure(self, client):
        """Failed execution returns error response."""
        mock_response = LEANBacktestResponse(
            status="error",
            error="LEAN execution failed",
        )

        with patch(
            "src.serve_backtest._run_lean_backtest",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            response = client.post(
                "/backtest",
                json={
                    "strategy_ir": {"strategy_id": "test"},
                    "data": {
                        "symbol": "BTC-USD",
                        "resolution": "1m",
                        "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
                    },
                    "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
                },
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "error"
            assert "error" in result

    def test_handles_exception_gracefully(self, client):
        """Exception during execution returns error response."""
        with patch(
            "src.serve_backtest._run_lean_backtest",
            new_callable=AsyncMock,
            side_effect=Exception("Unexpected error"),
        ):
            response = client.post(
                "/backtest",
                json={
                    "strategy_ir": {"strategy_id": "test"},
                    "data": {
                        "symbol": "BTC-USD",
                        "resolution": "1m",
                        "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
                    },
                    "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
                },
            )

            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "error"
            assert "Unexpected error" in result["error"]

"""Tests for request/response model parsing and validation.

These tests verify that models correctly parse and serialize data
without requiring the LEAN runtime.
"""

import pytest
from pydantic import ValidationError

from src.serve_backtest import (
    OHLCVBar,
    BacktestDataInput,
    BacktestConfig,
    LEANBacktestRequest,
    Trade,
    LEANBacktestSummary,
    LEANBacktestResponse,
)


class TestOHLCVBarParsing:
    """Test OHLCVBar model parsing with aliased fields."""

    def test_parses_abbreviated_fields(self):
        """Parse bar with t,o,h,l,c,v field names."""
        data = {"t": 1704067200000, "o": 100.0, "h": 105.0, "l": 95.0, "c": 102.0, "v": 1000.0}
        bar = OHLCVBar.model_validate(data)

        assert bar.t == 1704067200000
        assert bar.o == 100.0
        assert bar.h == 105.0
        assert bar.low == 95.0  # 'l' aliased to 'low'
        assert bar.c == 102.0
        assert bar.v == 1000.0

    def test_rejects_missing_required_fields(self):
        """Reject bar missing required fields."""
        data = {"t": 1704067200000, "o": 100.0}  # Missing h, l, c, v
        with pytest.raises(ValidationError):
            OHLCVBar.model_validate(data)

    def test_handles_zero_volume(self):
        """Accept bar with zero volume."""
        data = {"t": 1704067200000, "o": 100.0, "h": 100.0, "l": 100.0, "c": 100.0, "v": 0.0}
        bar = OHLCVBar.model_validate(data)
        assert bar.v == 0.0


class TestBacktestDataInput:
    """Test BacktestDataInput validation."""

    def test_accepts_inline_bars(self):
        """Accept data input with inline bars."""
        data = {
            "symbol": "BTC-USD",
            "resolution": "1m",
            "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
        }
        input_data = BacktestDataInput.model_validate(data)
        assert input_data.symbol == "BTC-USD"
        assert len(input_data.bars) == 1

    def test_accepts_gcs_uri(self):
        """Accept data input with GCS URI."""
        data = {
            "symbol": "BTC-USD",
            "resolution": "1h",
            "gcs_uri": "gs://bucket/path/data.parquet",
        }
        input_data = BacktestDataInput.model_validate(data)
        assert input_data.gcs_uri == "gs://bucket/path/data.parquet"
        assert input_data.bars is None

    def test_rejects_neither_bars_nor_gcs(self):
        """Reject data input with neither bars nor GCS URI."""
        data = {"symbol": "BTC-USD", "resolution": "1m"}
        with pytest.raises(ValidationError):
            BacktestDataInput.model_validate(data)


class TestBacktestConfig:
    """Test BacktestConfig validation."""

    def test_accepts_valid_config(self):
        """Accept valid backtest config."""
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31", "initial_cash": 100000.0}
        config = BacktestConfig.model_validate(data)
        assert config.start_date == "2024-01-01"
        assert config.initial_cash == 100000.0

    def test_uses_default_initial_cash(self):
        """Use default initial cash when not specified."""
        data = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
        config = BacktestConfig.model_validate(data)
        assert config.initial_cash == 100000.0


class TestLEANBacktestRequest:
    """Test full request parsing."""

    def test_parses_minimal_request(self):
        """Parse minimal valid request."""
        data = {
            "strategy_ir": {"strategy_id": "test", "entry": None},
            "data": {
                "symbol": "BTC-USD",
                "resolution": "1m",
                "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
            },
            "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
        }
        request = LEANBacktestRequest.model_validate(data)
        assert request.strategy_ir["strategy_id"] == "test"
        assert request.data.symbol == "BTC-USD"

    def test_parses_request_with_additional_data(self):
        """Parse request with multiple symbols."""
        data = {
            "strategy_ir": {"strategy_id": "multi-symbol"},
            "data": {
                "symbol": "BTC-USD",
                "resolution": "1h",
                "bars": [{"t": 0, "o": 100, "h": 101, "l": 99, "c": 100, "v": 1000}],
            },
            "config": {"start_date": "2024-01-01", "end_date": "2024-01-01"},
            "additional_data": [
                {
                    "symbol": "ETH-USD",
                    "resolution": "1h",
                    "bars": [{"t": 0, "o": 50, "h": 51, "l": 49, "c": 50, "v": 500}],
                }
            ],
        }
        request = LEANBacktestRequest.model_validate(data)
        assert len(request.additional_data) == 1
        assert request.additional_data[0].symbol == "ETH-USD"


class TestTradeModel:
    """Test Trade response model."""

    def test_creates_open_trade(self):
        """Create trade without exit (still open)."""
        trade = Trade(
            entry_bar=5,
            entry_price=100.0,
            entry_time="2024-01-01T10:00:00",
            direction="long",
            quantity=1.0,
        )
        assert trade.exit_bar is None
        assert trade.exit_price is None
        assert trade.pnl is None

    def test_creates_closed_trade(self):
        """Create fully closed trade with P&L."""
        trade = Trade(
            entry_bar=5,
            entry_price=100.0,
            entry_time="2024-01-01T10:00:00",
            exit_bar=10,
            exit_price=110.0,
            exit_time="2024-01-01T10:05:00",
            exit_reason="take_profit",
            direction="long",
            quantity=1.0,
            pnl=10.0,
            pnl_pct=10.0,
        )
        assert trade.exit_reason == "take_profit"
        assert trade.pnl == 10.0

    def test_accepts_short_direction(self):
        """Accept short trade direction."""
        trade = Trade(
            entry_bar=0,
            entry_price=100.0,
            entry_time="2024-01-01T10:00:00",
            direction="short",
            quantity=1.0,
        )
        assert trade.direction == "short"


class TestLEANBacktestResponse:
    """Test response model construction."""

    def test_creates_success_response(self):
        """Create successful backtest response."""
        response = LEANBacktestResponse(
            status="success",
            trades=[],
            summary=LEANBacktestSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
            ),
        )
        assert response.status == "success"
        assert response.error is None

    def test_creates_error_response(self):
        """Create error backtest response."""
        response = LEANBacktestResponse(
            status="error",
            error="LEAN execution failed: timeout",
        )
        assert response.status == "error"
        assert "timeout" in response.error

    def test_includes_equity_curve(self):
        """Include equity curve in response."""
        response = LEANBacktestResponse(
            status="success",
            trades=[],
            summary=LEANBacktestSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                total_pnl_pct=0.0,
            ),
            equity_curve=[100000.0, 100500.0, 101000.0],
        )
        assert len(response.equity_curve) == 3

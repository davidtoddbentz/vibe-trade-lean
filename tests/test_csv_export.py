"""Tests for CSV export functionality.

These tests verify that market data is correctly converted to LEAN CSV format
without requiring the LEAN runtime.
"""

import csv
import tempfile
from pathlib import Path

from src.serve_backtest import OHLCVBar, _write_ohlcv_bars_to_csv


class TestWriteOHLCVBarsToCSV:
    """Test CSV export for LEAN consumption."""

    def test_writes_csv_with_correct_headers(self):
        """CSV has datetime,open,high,low,close,volume headers."""
        bars = [OHLCVBar(t=1704067200000, o=100.0, h=105.0, l=95.0, c=102.0, v=1000.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_ohlcv_bars_to_csv(bars, "BTC-USD", output_dir)

            csv_file = output_dir / "btc_usd_data.csv"
            assert csv_file.exists()

            with open(csv_file) as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == ["datetime", "open", "high", "low", "close", "volume"]

    def test_converts_timestamp_to_datetime_string(self):
        """Converts millisecond timestamp to YYYY-MM-DD HH:MM:SS format."""
        # 2024-01-01 00:00:00 UTC in milliseconds
        bars = [OHLCVBar(t=1704067200000, o=100.0, h=105.0, l=95.0, c=102.0, v=1000.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_ohlcv_bars_to_csv(bars, "BTC-USD", output_dir)

            csv_file = output_dir / "btc_usd_data.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip headers
                row = next(reader)
                assert row[0] == "2024-01-01 00:00:00"

    def test_normalizes_symbol_to_filename(self):
        """Converts BTC-USD to btc_usd_data.csv."""
        bars = [OHLCVBar(t=0, o=100.0, h=100.0, l=100.0, c=100.0, v=100.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_ohlcv_bars_to_csv(bars, "ETH-USD", output_dir)

            assert (output_dir / "eth_usd_data.csv").exists()

    def test_returns_bar_count(self):
        """Returns number of bars written."""
        bars = [
            OHLCVBar(t=i * 60000, o=100.0, h=100.0, l=100.0, c=100.0, v=100.0)
            for i in range(10)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            count = _write_ohlcv_bars_to_csv(bars, "BTC-USD", output_dir)
            assert count == 10

    def test_returns_zero_for_empty_bars(self):
        """Returns 0 when no bars provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            count = _write_ohlcv_bars_to_csv([], "BTC-USD", output_dir)
            assert count == 0

    def test_formats_prices_with_two_decimals(self):
        """Formats OHLCV values with 2 decimal places."""
        bars = [OHLCVBar(t=0, o=100.123456, h=105.999, l=95.001, c=102.5, v=1000.0)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_ohlcv_bars_to_csv(bars, "BTC-USD", output_dir)

            csv_file = output_dir / "btc_usd_data.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip headers
                row = next(reader)
                # open, high, low, close, volume
                assert row[1] == "100.12"
                assert row[2] == "106.00"
                assert row[3] == "95.00"
                assert row[4] == "102.50"
                assert row[5] == "1000.00"

    def test_writes_multiple_bars_in_order(self):
        """Writes multiple bars preserving order."""
        bars = [
            OHLCVBar(t=0, o=100.0, h=101.0, l=99.0, c=100.5, v=100.0),
            OHLCVBar(t=60000, o=100.5, h=102.0, l=100.0, c=101.5, v=200.0),
            OHLCVBar(t=120000, o=101.5, h=103.0, l=101.0, c=102.5, v=300.0),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _write_ohlcv_bars_to_csv(bars, "BTC-USD", output_dir)

            csv_file = output_dir / "btc_usd_data.csv"
            with open(csv_file) as f:
                reader = csv.reader(f)
                next(reader)  # Skip headers
                rows = list(reader)

                assert len(rows) == 3
                # Verify close prices in order
                assert rows[0][4] == "100.50"
                assert rows[1][4] == "101.50"
                assert rows[2][4] == "102.50"

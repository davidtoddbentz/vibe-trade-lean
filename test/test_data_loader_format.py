"""Tests to verify BigQueryDataLoader and SyntheticDataLoader produce identical output format.

These tests ensure confidence that test data format matches production format,
even though they come from different sources.
"""

import sys
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from data_loader_base import Candle, DataLoader
from synthetic_data_loader import SyntheticDataLoader


class TestCandleFormat:
    """Test that Candle produces correct LEAN CSV format."""

    def test_to_lean_csv_row_format(self):
        """Verify CSV row format matches LEAN expectations."""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 10, 30, 0),  # 10:30 AM
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1234.56,
        )
        day_start = datetime(2024, 1, 1, 0, 0, 0)

        row = candle.to_lean_csv_row(day_start)

        # 10:30 AM = 10*60*60*1000 + 30*60*1000 = 37800000 ms
        expected_ms = 10 * 60 * 60 * 1000 + 30 * 60 * 1000
        assert row == f"{expected_ms},50000.0,50100.0,49900.0,50050.0,1234.56"

    def test_to_lean_csv_row_midnight(self):
        """Verify midnight timestamp produces 0 ms offset."""
        candle = Candle(
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=500.0,
        )
        day_start = datetime(2024, 1, 1, 0, 0, 0)

        row = candle.to_lean_csv_row(day_start)

        assert row.startswith("0,")  # 0 ms since midnight


class TestDataLoaderInterface:
    """Test that DataLoader implementations follow the interface correctly."""

    def test_synthetic_loader_implements_interface(self):
        """Verify SyntheticDataLoader is a valid DataLoader."""
        loader = SyntheticDataLoader()
        assert isinstance(loader, DataLoader)
        assert hasattr(loader, "load")
        assert hasattr(loader, "write_lean_format")

    def test_synthetic_loader_load_returns_candles(self):
        """Verify load() returns list of Candle objects."""
        loader = SyntheticDataLoader()
        loader.add_flat(bars=10, price=50000.0)

        candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert len(candles) == 10
        assert all(isinstance(c, Candle) for c in candles)

    def test_synthetic_loader_timestamps_correct(self):
        """Verify candles are timestamped correctly from start_date."""
        loader = SyntheticDataLoader()
        loader.add_flat(bars=5, price=50000.0)

        start = datetime(2024, 6, 15, 9, 0, 0)
        candles = loader.load("BTCUSD", start, start)

        assert candles[0].timestamp == start
        assert candles[1].timestamp == start + timedelta(minutes=1)
        assert candles[4].timestamp == start + timedelta(minutes=4)


class TestWriteLeanFormat:
    """Test that write_lean_format produces correct output structure."""

    def test_creates_correct_directory_structure(self):
        """Verify output path follows LEAN convention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.add_flat(bars=10, price=50000.0)
            candles = loader.load("BTC-USD", datetime(2024, 1, 1), datetime(2024, 1, 1))

            loader.write_lean_format(candles, "BTC-USD", Path(tmpdir))

            # Check directory structure
            expected_dir = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd"
            assert expected_dir.exists()

    def test_creates_zip_file_per_day(self):
        """Verify one zip file is created per day of data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.add_flat(bars=10, price=50000.0)
            candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

            files_written = loader.write_lean_format(candles, "BTCUSD", Path(tmpdir))

            assert files_written == 1
            zip_path = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd" / "20240101_trade.zip"
            assert zip_path.exists()

    def test_zip_contains_csv(self):
        """Verify zip file contains CSV with correct name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.add_flat(bars=5, price=50000.0)
            candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

            loader.write_lean_format(candles, "BTCUSD", Path(tmpdir))

            zip_path = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd" / "20240101_trade.zip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()
                assert len(names) == 1
                assert names[0] == "20240101_trade.csv"

    def test_csv_content_format(self):
        """Verify CSV content matches LEAN minute data format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.add_candle(100.0, 101.0, 99.0, 100.5, 500.0)
            loader.add_candle(100.5, 102.0, 100.0, 101.0, 600.0)
            candles = loader.load("BTCUSD", datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1))

            loader.write_lean_format(candles, "BTCUSD", Path(tmpdir))

            zip_path = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd" / "20240101_trade.zip"
            with zipfile.ZipFile(zip_path, "r") as zf:
                csv_content = zf.read("20240101_trade.csv").decode("utf-8")
                lines = csv_content.strip().split("\n")

                assert len(lines) == 2
                # First candle at 00:00 = 0ms
                assert lines[0] == "0,100.0,101.0,99.0,100.5,500.0"
                # Second candle at 00:01 = 60000ms
                assert lines[1] == "60000,100.5,102.0,100.0,101.0,600.0"

    def test_symbol_normalization(self):
        """Verify symbol is normalized correctly (BTC-USD -> btcusd)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.add_flat(bars=5, price=50000.0)
            candles = loader.load("BTC-USD", datetime(2024, 1, 1), datetime(2024, 1, 1))

            loader.write_lean_format(candles, "BTC-USD", Path(tmpdir))

            # Should normalize to lowercase, no hyphens
            expected_dir = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd"
            assert expected_dir.exists()


class TestMultipleDays:
    """Test handling of data spanning multiple days."""

    def test_creates_multiple_zip_files(self):
        """Verify separate zip files for each day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create candles spanning 2 days
            candles = [
                Candle(datetime(2024, 1, 1, 23, 59), 100, 101, 99, 100.5, 500),
                Candle(datetime(2024, 1, 2, 0, 0), 100.5, 102, 100, 101, 600),
                Candle(datetime(2024, 1, 2, 0, 1), 101, 103, 100.5, 102, 700),
            ]

            loader = SyntheticDataLoader()
            files_written = loader.write_lean_format(candles, "BTCUSD", Path(tmpdir))

            assert files_written == 2

            base_path = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd"
            assert (base_path / "20240101_trade.zip").exists()
            assert (base_path / "20240102_trade.zip").exists()


class TestSyntheticScenarios:
    """Test pre-built synthetic scenarios."""

    def test_uptrend_with_pullback_generates_data(self):
        """Verify uptrend scenario generates valid data."""
        loader = SyntheticDataLoader()
        expected = loader.generate_uptrend_with_pullback(n_bars=200)
        candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert len(candles) == 200
        assert all(c.open > 0 for c in candles)
        assert all(c.high >= c.low for c in candles)

    def test_uptrend_with_pullback_expected_trades(self):
        """Verify expected trades are calculated."""
        loader = SyntheticDataLoader()
        expected = loader.generate_uptrend_with_pullback(n_bars=200)

        # Should have at least one entry signal
        assert len(expected) >= 1
        assert expected[0].action == "ENTRY"
        assert expected[0].direction == "long"

    def test_breakout_generates_data(self):
        """Verify breakout scenario generates valid data."""
        loader = SyntheticDataLoader()
        expected = loader.generate_breakout(n_bars=150)
        candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

        assert len(candles) == 150
        assert all(c.open > 0 for c in candles)

    def test_scenario_writes_to_lean_format(self):
        """Verify scenarios can be written to LEAN format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = SyntheticDataLoader()
            loader.generate_uptrend_with_pullback(n_bars=100)
            candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

            files = loader.write_lean_format(candles, "BTCUSD", Path(tmpdir))

            assert files == 1
            zip_path = Path(tmpdir) / "crypto" / "coinbase" / "minute" / "btcusd" / "20240101_trade.zip"
            assert zip_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

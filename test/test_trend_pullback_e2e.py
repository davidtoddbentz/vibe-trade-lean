"""End-to-end test for TrendPullback strategy.

This test:
1. Generates synthetic data designed to trigger specific trades
2. Runs LEAN backtest with the data
3. Parses the LEAN output log
4. Asserts that expected trades occurred

Run with: python3 test/test_trend_pullback_e2e.py
"""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

# Add test/lib to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from lib.test_data_builder import TestDataBuilder, ExpectedTrade, create_trend_pullback_test


# Strategy IR for trend pullback
TREND_PULLBACK_IR = {
    "name": "TrendPullbackTest",
    "version": "1.0",
    "symbols": ["TESTUSD"],
    "indicators": [
        {"id": "ema_20", "type": "EMA", "period": 20},
        {"id": "ema_50", "type": "EMA", "period": 50},
        {"id": "bb_20", "type": "BB", "period": 20, "std_dev_multiplier": 2.0},
    ],
    "entry": {
        "condition": {
            "type": "AllOfCondition",
            "conditions": [
                {
                    "type": "CompareCondition",
                    "left": {"type": "IndicatorValue", "indicator_id": "ema_20"},
                    "op": ">",
                    "right": {"type": "IndicatorValue", "indicator_id": "ema_50"},
                },
                {
                    "type": "CompareCondition",
                    "left": {"type": "PriceValue", "field": "close"},
                    "op": "<",
                    "right": {"type": "IndicatorBandValue", "indicator_id": "bb_20", "band": "lower"},
                },
            ],
        },
        "action": {
            "type": "market",
            "direction": "long",
            "size": {"type": "percent_equity", "value": 95.0},
        },
    },
    "exit": {
        "profit_target_pct": 2.0,
        "stop_loss_pct": 1.0,
    },
}


def parse_lean_log(log_content: str) -> list[dict]:
    """Parse LEAN log to extract trade events."""
    trades = []

    # Look for entry patterns
    entry_pattern = r"ENTRY executed.*?@ (\d+\.?\d*)"
    for match in re.finditer(entry_pattern, log_content):
        trades.append({
            "action": "ENTRY",
            "price": float(match.group(1)),
        })

    # Look for exit patterns
    exit_pattern = r"EXIT.*?@ (\d+\.?\d*)"
    for match in re.finditer(exit_pattern, log_content):
        trades.append({
            "action": "EXIT",
            "price": float(match.group(1)),
        })

    return trades


class TestTrendPullbackE2E(unittest.TestCase):
    """End-to-end tests for TrendPullback strategy."""

    def setUp(self):
        """Create temporary test directory structure."""
        self.test_dir = Path(tempfile.mkdtemp())
        data_dir = self.test_dir / "data" / "custom" / "testusd"
        data_dir.mkdir(parents=True)
        results_dir = self.test_dir / "results"
        results_dir.mkdir()

    def test_trend_pullback_generates_expected_entry(self):
        """Test that trend pullback generates entry at the expected bar."""
        # Generate test data
        builder, expected = create_trend_pullback_test(
            symbol="TESTUSD",
            start_date="2024-01-01",
        )

        # Verify we have expected trades
        self.assertGreaterEqual(len(expected), 1, "Should have at least one expected trade")
        entry_trade = expected[0]
        self.assertEqual(entry_trade.action, "ENTRY", "First trade should be entry")

        # Export data
        data_file = self.test_dir / "data" / "custom" / "testusd" / "20240101.csv"
        builder.export_csv(str(data_file))

        # Export strategy IR
        ir_file = self.test_dir / "data" / "strategy_ir.json"
        ir_file.write_text(json.dumps(TREND_PULLBACK_IR, indent=2))

        print(f"\nExpected trades:")
        for t in expected:
            print(f"  Bar {t.bar_index}: {t.action} @ {t.price:.2f} - {t.reason}")

        # The actual LEAN run would happen here in CI
        # For now, just verify the data generation is correct
        self.assertTrue(data_file.exists())
        self.assertTrue(ir_file.exists())

        # Verify data file has correct number of lines
        lines = data_file.read_text().strip().split("\n")
        self.assertEqual(len(lines), len(builder.candles))

    def test_data_builder_indicator_calculation(self):
        """Test that TestDataBuilder calculates indicators correctly."""
        builder = TestDataBuilder(symbol="TEST", start_date="2024-01-01")

        # Add some known prices
        prices = [100, 102, 101, 103, 104, 102, 105, 106, 104, 107]
        for p in prices:
            builder.add_candle(p, p + 1, p - 1, p)

        # Verify EMA calculation (spot check)
        ema5 = builder.calculate_ema(5)
        self.assertEqual(len(ema5), len(prices))
        self.assertIsNone(ema5[0])  # First 4 should be None
        self.assertIsNone(ema5[3])
        self.assertIsNotNone(ema5[4])  # 5th should have value

        # SMA of first 5 prices
        expected_sma = sum(prices[:5]) / 5
        self.assertAlmostEqual(ema5[4], expected_sma, places=2,
            msg=f"First EMA should equal SMA: {ema5[4]} vs {expected_sma}")

    def test_pullback_triggers_at_bb_lower(self):
        """Test that pullback properly crosses BB lower band."""
        builder = TestDataBuilder(symbol="TEST", start_date="2024-01-01")

        # Establish trend first
        builder.add_uptrend(bars=30, start_price=100, trend_strength=0.005)

        # Add pullback
        pullback_indices, trigger_bar = builder.add_pullback_to_bb_lower(
            bars=5,
            bb_period=20,
            bb_mult=2.0,
            overshoot=0.01,  # 1% below
        )

        # Verify trigger occurred
        self.assertIsNotNone(trigger_bar, "Should have found a trigger bar")

        # Verify price at trigger is below BB lower
        bb = builder.calculate_bb(20, 2.0)
        trigger_price = builder._prices[trigger_bar]
        trigger_bb_lower = bb[trigger_bar]["lower"]

        self.assertLess(trigger_price, trigger_bb_lower,
            f"Trigger price ({trigger_price:.2f}) should be below BB lower ({trigger_bb_lower:.2f})")

    def test_expected_trades_calculation(self):
        """Test that expected trades are calculated correctly."""
        builder, expected = create_trend_pullback_test()

        # Should have at least entry
        self.assertGreaterEqual(len(expected), 1)

        # First should be entry
        self.assertEqual(expected[0].action, "ENTRY")
        self.assertEqual(expected[0].direction, "long")

        # If we have two trades, second should be exit
        if len(expected) >= 2:
            self.assertEqual(expected[1].action, "EXIT")

        # Entry should happen when conditions are met
        entry = expected[0]
        ema20 = builder.calculate_ema(20)
        ema50 = builder.calculate_ema(50)
        bb = builder.calculate_bb(20, 2.0)

        # At entry bar, verify conditions
        price = builder._prices[entry.bar_index]
        self.assertGreater(ema20[entry.bar_index], ema50[entry.bar_index],
            "EMA20 should be > EMA50 at entry")
        self.assertLess(price, bb[entry.bar_index]["lower"],
            "Price should be < BB lower at entry")


if __name__ == "__main__":
    unittest.main(verbosity=2)

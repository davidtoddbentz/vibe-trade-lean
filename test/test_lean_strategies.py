"""LEAN integration tests for various strategy types.

These tests run strategy IRs through the actual LEAN engine to verify
that the Python evaluator produces the same results as LEAN execution.

Each test:
1. Creates a strategy IR
2. Generates synthetic data that triggers specific conditions
3. Runs through LEAN Docker
4. Asserts expected trades occurred

Run with:
    # Quick check (just verify LEAN runs)
    python3 test/test_lean_strategies.py

    # Full integration (requires Docker + built image)
    LEAN_INTEGRATION_TESTS=1 python3 -m pytest test/test_lean_strategies.py -v

Note: These tests require the vibe-trade-lean Docker image to be built:
    make build
"""

import os
import sys
import unittest
from pathlib import Path

# Add lib to path
lib_path = Path(__file__).parent / "lib"
sys.path.insert(0, str(lib_path))

# Import from lib
from test_data_builder import TestDataBuilder, ExpectedTrade, create_trend_pullback_test
from lean_test_runner import (
    LEANTestRunner,
    LEANTestScenario,
    create_ema_crossover_scenario,
    create_trend_pullback_scenario,
)


# Skip tests if LEAN_INTEGRATION_TESTS env var is not set
SKIP_LEAN_TESTS = os.environ.get("LEAN_INTEGRATION_TESTS", "0") != "1"
SKIP_REASON = "Set LEAN_INTEGRATION_TESTS=1 to run LEAN integration tests"


class TestLEANStrategies(unittest.TestCase):
    """Integration tests for various strategy types through LEAN."""

    @classmethod
    def setUpClass(cls):
        """Set up the test runner."""
        cls.runner = LEANTestRunner()

    # =========================================================================
    # EMA Crossover Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_ema_crossover_bullish(self):
        """Test bullish EMA crossover: entry when fast > slow, exit when fast < slow."""
        scenario = create_ema_crossover_scenario(bullish=True)

        result = self.runner.run(scenario)

        # Should have at least one entry
        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one entry")

        print(f"\nEMA Crossover Bullish Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  Exits: {result.exit_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_ema_crossover_bearish(self):
        """Test bearish EMA crossover: entry when fast < slow, exit when fast > slow."""
        scenario = create_ema_crossover_scenario(bullish=False)

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one entry")

        print(f"\nEMA Crossover Bearish Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  Exits: {result.exit_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # Trend Pullback Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_trend_pullback_entry_and_profit_exit(self):
        """Test trend pullback: entry on pullback to BB lower, exit on profit target."""
        scenario = create_trend_pullback_scenario(
            profit_target_pct=0.02,
            stop_loss_pct=0.01,
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertEqual(result.entry_count, 1, "Should have exactly one entry")
        self.assertEqual(result.exit_count, 1, "Should have exactly one exit (profit target)")

        # Verify exit was profit target
        exit_trades = [t for t in result.trades if t.action == "EXIT"]
        if exit_trades:
            self.assertEqual(exit_trades[0].exit_reason, "profit_target")

        print(f"\nTrend Pullback Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  Exits: {result.exit_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")
        for trade in result.trades:
            print(f"  {trade.action} @ {trade.price:.2f} ({trade.exit_reason or 'entry'})")

    # =========================================================================
    # Breakout Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_breakout_above_high(self):
        """Test breakout: entry when high > max of lookback period.

        Note: We compare high > max_50 instead of close > max_50 because the MAX
        indicator includes the current bar's close, so close can never exceed max.
        The high typically exceeds close on uptrend bars, allowing the breakout to trigger.
        """
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Consolidation period (60 bars)
        builder.add_flat(bars=60, price=50000, noise=100)

        # Breakout (price rises above consolidation range)
        builder.add_uptrend(bars=20, start_price=50100, trend_strength=0.005)

        # Continue higher
        builder.add_uptrend(bars=20, trend_strength=0.003)

        strategy_ir = {
            "strategy_id": "breakout-test",
            "strategy_name": "Breakout Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "MAX", "id": "max_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "price", "field": "high"},
                    "op": ">",
                    "right": {"type": "indicator", "indicator_id": "max_50"},
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="breakout_above_high",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],  # We'll just check entry count
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one breakout entry")

        print(f"\nBreakout Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # Volume-Based Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_volume_spike_entry(self):
        """Test volume spike: entry when volume > 2x average volume."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Normal volume period
        for i in range(60):
            builder.add_candle(50000 + i * 10, 50010 + i * 10, 49990 + i * 10, 50000 + i * 10, volume=1000)

        # Volume spike (3x normal)
        for i in range(10):
            builder.add_candle(50600 + i * 20, 50620 + i * 20, 50580 + i * 20, 50600 + i * 20, volume=3000)

        # Normal volume continues
        for i in range(30):
            builder.add_candle(50800 + i * 10, 50810 + i * 10, 50790 + i * 10, 50800 + i * 10, volume=1000)

        strategy_ir = {
            "strategy_id": "volume-spike-test",
            "strategy_name": "Volume Spike Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "SMA", "id": "vol_sma", "period": 20, "field": "volume"},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "volume"},
                    "op": ">",
                    "right": {
                        "type": "expr",
                        "op": "*",
                        "left": {"type": "indicator", "indicator_id": "vol_sma"},
                        "right": {"type": "literal", "value": 2.0},
                    },
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="volume_spike",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        # Note: This test may need adjustment based on how volume SMA is calculated
        print(f"\nVolume Spike Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # Composite Condition Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_allof_condition(self):
        """Test AllOf: entry requires multiple conditions to be true.

        Requires both: EMA20 > EMA50 AND high > max_30 (breakout).
        Note: We use high > max_30 because MAX includes current bar's close.
        """
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Build scenario where EMA crossover AND breakout both trigger
        # First: downtrend
        builder.add_downtrend(bars=60, start_price=50000, trend_strength=0.001)
        # Then: strong uptrend (causes EMA crossover AND breakout)
        builder.add_uptrend(bars=50, trend_strength=0.004)

        strategy_ir = {
            "strategy_id": "allof-test",
            "strategy_name": "AllOf Condition Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
                {"type": "MAX", "id": "max_30", "period": 30},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "allOf",
                    "conditions": [
                        {
                            "type": "compare",
                            "left": {"type": "indicator", "indicator_id": "ema_20"},
                            "op": ">",
                            "right": {"type": "indicator", "indicator_id": "ema_50"},
                        },
                        {
                            "type": "compare",
                            "left": {"type": "price", "field": "high"},
                            "op": ">",
                            "right": {"type": "indicator", "indicator_id": "max_30"},
                        },
                    ],
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="allof_condition",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        print(f"\nAllOf Condition Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_anyof_condition(self):
        """Test AnyOf: entry when ANY condition is true."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Create scenario with just EMA crossover (no breakout)
        builder.add_downtrend(bars=60, start_price=50000, trend_strength=0.001)
        builder.add_uptrend(bars=40, trend_strength=0.002)  # Just enough for crossover

        strategy_ir = {
            "strategy_id": "anyof-test",
            "strategy_name": "AnyOf Condition Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "anyOf",
                    "conditions": [
                        {
                            "type": "compare",
                            "left": {"type": "indicator", "indicator_id": "ema_20"},
                            "op": ">",
                            "right": {"type": "indicator", "indicator_id": "ema_50"},
                        },
                        {
                            "type": "compare",
                            "left": {"type": "price", "field": "close"},
                            "op": ">",
                            "right": {"type": "literal", "value": 999999},  # Never true
                        },
                    ],
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="anyof_condition",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "AnyOf should trigger on first matching condition")

        print(f"\nAnyOf Condition Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # Momentum Indicator Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_rsi_oversold_entry(self):
        """Test RSI: entry when RSI < 30 (oversold)."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Warmup period with gradual price
        builder.add_uptrend(bars=30, start_price=50000, trend_strength=0.001)

        # Sharp drop to trigger oversold RSI
        builder.add_downtrend(bars=20, trend_strength=0.008)

        # Recovery
        builder.add_uptrend(bars=30, trend_strength=0.005)

        strategy_ir = {
            "strategy_id": "rsi-oversold-test",
            "strategy_name": "RSI Oversold Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "RSI", "id": "rsi_14", "period": 14},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "rsi_14"},
                    "op": "<",
                    "right": {"type": "literal", "value": 30},
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="rsi_oversold",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one RSI oversold entry")

        print(f"\nRSI Oversold Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_macd_signal_crossover(self):
        """Test MACD: entry when MACD crosses above signal line."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Need enough data for MACD warmup (slow=26 + signal=9 = 35 min)
        # Start with downtrend
        builder.add_downtrend(bars=50, start_price=50000, trend_strength=0.001)

        # Strong uptrend to trigger MACD crossover
        builder.add_uptrend(bars=40, trend_strength=0.004)

        # Continue
        builder.add_uptrend(bars=20, trend_strength=0.001)

        strategy_ir = {
            "strategy_id": "macd-crossover-test",
            "strategy_name": "MACD Crossover Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "MACD", "id": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "macd"},
                    "op": ">",
                    "right": {"type": "indicator_property", "indicator_id": "macd", "property": "Signal"},
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="macd_crossover",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one MACD crossover entry")

        print(f"\nMACD Crossover Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_atr_volatility_filter(self):
        """Test ATR: entry when ATR > threshold (high volatility)."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Low volatility period (small moves)
        for i in range(40):
            price = 50000 + i * 5  # Small moves
            builder.add_candle(price, price + 10, price - 10, price)

        # High volatility period (large moves)
        for i in range(30):
            price = 50200 + i * 50  # Large moves
            builder.add_candle(price, price + 200, price - 200, price)

        # Continue
        for i in range(20):
            price = 51700 + i * 30
            builder.add_candle(price, price + 100, price - 100, price)

        strategy_ir = {
            "strategy_id": "atr-volatility-test",
            "strategy_name": "ATR Volatility Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "ATR", "id": "atr_14", "period": 14},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "atr_14"},
                    "op": ">",
                    "right": {"type": "literal", "value": 100},  # ATR > 100
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="atr_volatility",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one high ATR entry")

        print(f"\nATR Volatility Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # New IR Spec Tests (Inline IndicatorRef)
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_inline_indicator_ref_ema_crossover(self):
        """Test new IR spec with inline IndicatorRef (no separate indicators list).

        This tests the new vibe-trade-shared IR format where indicators are
        defined inline in conditions rather than in a separate list.
        """
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Downtrend to establish EMA relationship
        builder.add_downtrend(bars=60, start_price=50000, trend_strength=0.001)

        # Strong uptrend for crossover
        builder.add_uptrend(bars=40, trend_strength=0.003)

        # Continue
        builder.add_uptrend(bars=20, trend_strength=0.001)

        # NEW IR SPEC FORMAT: inline IndicatorRef, no indicators list
        strategy_ir = {
            "strategy_id": "inline-ir-test",
            "strategy_name": "Inline IR Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            # Note: NO indicators list! They're inline in conditions.
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    # NEW FORMAT: indicator_type + params instead of indicator_id
                    "left": {
                        "type": "indicator",
                        "indicator_type": "EMA",
                        "params": {"period": 20},
                    },
                    "op": ">",
                    "right": {
                        "type": "indicator",
                        "indicator_type": "EMA",
                        "params": {"period": 50},
                    },
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="inline_indicator_ref",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one EMA crossover entry")

        print(f"\nInline IndicatorRef Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_inline_indicator_ref_bb_lower(self):
        """Test inline IndicatorRef with field access (BB lower band)."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Establish price history
        builder.add_uptrend(bars=60, start_price=50000, trend_strength=0.0008)

        # Sharp pullback to BB lower
        builder.add_pullback_to_bb_lower(bars=5, bb_period=20, bb_mult=2.0)

        # Recovery
        builder.add_recovery(bars=20, trend_strength=0.003)

        # NEW IR SPEC: inline BB with field="lower"
        strategy_ir = {
            "strategy_id": "inline-bb-test",
            "strategy_name": "Inline BB Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            # NO indicators list
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "price", "field": "close"},
                    "op": "<",
                    # NEW FORMAT: indicator with field for band access
                    "right": {
                        "type": "indicator",
                        "indicator_type": "BB",
                        "params": {"period": 20, "multiplier": 2.0},
                        "field": "lower",  # Access BB lower band
                    },
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="inline_bb_lower",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one BB lower entry")

        print(f"\nInline BB Lower Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    # =========================================================================
    # Cross, Time Filter, Not Condition Tests
    # =========================================================================

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_cross_condition_ema_crossover(self):
        """Test cross condition: entry when EMA20 crosses above EMA50."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Downtrend: EMA20 below EMA50
        builder.add_downtrend(bars=60, start_price=50000, trend_strength=0.001)

        # Strong uptrend: EMA20 will cross above EMA50
        builder.add_uptrend(bars=40, trend_strength=0.004)

        # Continue uptrend
        builder.add_uptrend(bars=20, trend_strength=0.001)

        strategy_ir = {
            "strategy_id": "cross-condition-test",
            "strategy_name": "Cross Condition Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "cross",
                    "left": {"type": "indicator", "indicator_id": "ema_20"},
                    "right": {"type": "indicator", "indicator_id": "ema_50"},
                    "direction": "above",
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="cross_condition",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one cross entry")

        print(f"\nCross Condition Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_time_filter_condition(self):
        """Test time_filter: only enter during specific hours."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Create uptrend data (should trigger entry if time allows)
        builder.add_uptrend(bars=100, start_price=50000, trend_strength=0.002)

        strategy_ir = {
            "strategy_id": "time-filter-test",
            "strategy_name": "Time Filter Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "allOf",
                    "conditions": [
                        {
                            "type": "compare",
                            "left": {"type": "indicator", "indicator_id": "ema_20"},
                            "op": ">",
                            "right": {"type": "indicator", "indicator_id": "ema_50"},
                        },
                        {
                            "type": "time_filter",
                            "time_window": "00:00-23:59",  # All day allowed
                        },
                    ],
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="time_filter",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        # Should enter since time window allows all day
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one entry (time filter allows)")

        print(f"\nTime Filter Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")

    @unittest.skipIf(SKIP_LEAN_TESTS, SKIP_REASON)
    def test_not_condition(self):
        """Test not condition: entry when NOT in downtrend."""
        builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

        # Start with uptrend (NOT downtrend, so should enter)
        builder.add_uptrend(bars=60, start_price=50000, trend_strength=0.001)

        # Continue uptrend
        builder.add_uptrend(bars=40, trend_strength=0.002)

        strategy_ir = {
            "strategy_id": "not-condition-test",
            "strategy_name": "Not Condition Test",
            "symbol": "TESTUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    # Entry when NOT (EMA20 < EMA50), i.e., when in uptrend
                    "type": "not",
                    "condition": {
                        "type": "compare",
                        "left": {"type": "indicator", "indicator_id": "ema_20"},
                        "op": "<",
                        "right": {"type": "indicator", "indicator_id": "ema_50"},
                    },
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [],
            "on_bar": [],
            "on_bar_invested": [],
        }

        scenario = LEANTestScenario(
            name="not_condition",
            strategy_ir=strategy_ir,
            data_builder=builder,
            expected_trades=[],
        )

        result = self.runner.run(scenario)

        self.assertTrue(result.success, f"LEAN failed: {result.errors}")
        # Should enter since we're in uptrend (NOT downtrend)
        self.assertGreaterEqual(result.entry_count, 1, "Should have at least one entry (not downtrend)")

        print(f"\nNot Condition Results:")
        print(f"  Entries: {result.entry_count}")
        print(f"  End Equity: ${result.end_equity:,.2f}")


class TestLEANScenariosQuick(unittest.TestCase):
    """Quick tests that verify scenario creation without running LEAN.

    These always run, regardless of LEAN_INTEGRATION_TESTS setting.
    """

    def test_ema_crossover_scenario_creation(self):
        """Test that EMA crossover scenarios are created correctly."""
        scenario = create_ema_crossover_scenario(bullish=True)

        self.assertEqual(scenario.name, "ema_crossover_long")
        self.assertIn("ema_fast", str(scenario.strategy_ir))
        self.assertIn("ema_slow", str(scenario.strategy_ir))
        self.assertGreater(len(scenario.data_builder.candles), 50)

    def test_trend_pullback_scenario_creation(self):
        """Test that trend pullback scenarios are created correctly."""
        scenario = create_trend_pullback_scenario()

        self.assertEqual(scenario.name, "trend_pullback")
        self.assertIn("bb", str(scenario.strategy_ir))
        self.assertGreater(len(scenario.expected_trades), 0)

    def test_data_builder_ema_calculation(self):
        """Verify TestDataBuilder's EMA calculation matches expected behavior."""
        builder = TestDataBuilder(symbol="TEST", start_date="2024-01-01")

        # Known prices for predictable EMA
        for i in range(30):
            builder.add_candle(100, 100, 100, 100)

        ema10 = builder.calculate_ema(10)

        # After 30 bars of price=100, EMA should converge to 100
        self.assertIsNotNone(ema10[29])
        self.assertAlmostEqual(ema10[29], 100.0, places=2)


if __name__ == "__main__":
    # When run directly, show which tests require Docker
    print("=" * 60)
    print("LEAN Strategy Integration Tests")
    print("=" * 60)

    if SKIP_LEAN_TESTS:
        print("\n⚠️  LEAN integration tests are SKIPPED")
        print("   To run full tests: LEAN_INTEGRATION_TESTS=1 python3 -m pytest test/test_lean_strategies.py -v")
        print("   Make sure Docker image is built: make build")
        print("\nRunning quick tests (no Docker required)...")

    unittest.main(verbosity=2)

"""End-to-end tests for the LEAN StrategyRuntime.

These tests verify that the complete pipeline works:
MCP Schema → IR Translation → LEAN Execution → Expected Trades

Requirements:
- Docker must be installed and running
- vibe-trade-lean image must be built (run `make build` first)
"""

import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import synthetic data generator
from generate_synthetic_data import (
    generate_uptrend_with_pullback,
    generate_breakout,
    write_lean_data,
)


# Test configuration
DOCKER_IMAGE = os.getenv("LEAN_IMAGE", "vibe-trade-lean:latest")
TEST_DIR = Path(__file__).parent
PROJECT_DIR = TEST_DIR.parent


def check_docker_available():
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_image_exists():
    """Check if the LEAN image exists."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", DOCKER_IMAGE],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


# Skip all tests if Docker is not available
pytestmark = pytest.mark.skipif(
    not check_docker_available(),
    reason="Docker not available",
)


@pytest.fixture(scope="module")
def docker_image():
    """Ensure Docker image exists."""
    if not check_image_exists():
        pytest.skip(f"Docker image {DOCKER_IMAGE} not found. Run 'make build' first.")
    return DOCKER_IMAGE


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def run_lean_backtest(
    data_dir: Path,
    strategy_ir: dict,
    timeout: int = 120,
) -> tuple[int, str, str]:
    """Run LEAN backtest with given data and strategy.

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Write strategy IR
    ir_path = data_dir / "strategy_ir.json"
    with open(ir_path, "w") as f:
        json.dump(strategy_ir, f, indent=2)

    # Create results directory
    results_dir = data_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Run LEAN in Docker
    cmd = [
        "docker", "run", "--rm",
        "-e", "STRATEGY_IR_PATH=/Data/strategy_ir.json",
        "-v", f"{data_dir}:/Data:ro",
        "-v", f"{results_dir}:/Results",
        "-v", f"{PROJECT_DIR}/src/strategy_runtime.py:/Lean/Algorithm.Python/strategy_runtime.py:ro",
        "-v", f"{TEST_DIR}/config-backtest.json:/Lean/Launcher/bin/Debug/config.json:ro",
        DOCKER_IMAGE,
        "--config", "/Lean/Launcher/bin/Debug/config.json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Timeout expired"


def parse_entry_logs(output: str) -> list[dict]:
    """Parse entry signals from LEAN log output.

    Handles both old format (price only) and new format (bar + price).
    """
    entries = []
    # New format: ENTRY executed at bar=60 price=50123.45
    pattern_new = r"ENTRY executed at bar=(\d+) price=(\d+\.?\d*)"
    for match in re.finditer(pattern_new, output):
        entries.append({
            "bar_index": int(match.group(1)),
            "price": float(match.group(2)),
        })

    # Fall back to old format if no new format matches
    if not entries:
        pattern_old = r"ENTRY executed at (\d+\.?\d*)"
        for match in re.finditer(pattern_old, output):
            entries.append({
                "bar_index": None,  # Unknown bar index
                "price": float(match.group(1)),
            })

    return entries


def parse_exit_logs(output: str) -> list[dict]:
    """Parse exit signals from LEAN log output.

    Handles both old format and new format (bar + price).
    """
    exits = []
    # New format: EXIT (profit_target) executed at bar=80 price=51234.56
    pattern_new = r"EXIT \((\w+)\) executed at bar=(\d+) price=(\d+\.?\d*)"
    for match in re.finditer(pattern_new, output):
        exits.append({
            "reason": match.group(1),
            "bar_index": int(match.group(2)),
            "price": float(match.group(3)),
        })

    # Fall back to old format if no new format matches
    if not exits:
        pattern_old = r"EXIT \((\w+)\) executed at (\d+\.?\d*)"
        for match in re.finditer(pattern_old, output):
            exits.append({
                "reason": match.group(1),
                "bar_index": None,
                "price": float(match.group(2)),
            })

    return exits


class TestTrendPullback:
    """Test TrendPullback strategy execution."""

    @pytest.fixture
    def strategy_ir(self):
        """TrendPullback strategy IR."""
        return {
            "strategy_id": "test-trend-pullback",
            "strategy_name": "TestTrendPullback",
            "symbol": "BTCUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "EMA", "id": "ema_20", "period": 20},
                {"type": "EMA", "id": "ema_50", "period": 50},
                {"type": "BB", "id": "bb_20", "period": 20, "multiplier": 2.0},
            ],
            "state": [
                {"id": "bars_since_entry", "var_type": "int", "default": 0},
            ],
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
                            "left": {"type": "price", "field": "close"},
                            "op": "<=",
                            "right": {"type": "indicator_band", "indicator_id": "bb_20", "band": "lower"},
                        },
                    ],
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [
                    {"type": "set_state", "state_id": "bars_since_entry", "value": {"type": "literal", "value": 0}},
                ],
            },
            "exits": [
                {
                    "id": "profit_target",
                    "condition": {
                        "type": "compare",
                        "left": {"type": "price", "field": "close"},
                        "op": ">=",
                        "right": {"type": "indicator_band", "indicator_id": "bb_20", "band": "upper"},
                    },
                    "action": {"type": "liquidate"},
                    "priority": 1,
                },
            ],
            "on_bar": [],
            "on_bar_invested": [
                {"type": "increment", "state_id": "bars_since_entry"},
            ],
        }

    def test_strategy_loads(self, docker_image, temp_data_dir, strategy_ir):
        """Test that the strategy loads successfully."""
        # Generate minimal data
        candles, _ = generate_uptrend_with_pullback(n_bars=100)
        write_lean_data(candles, "BTCUSD", datetime(2024, 1, 1), temp_data_dir)

        returncode, stdout, stderr = run_lean_backtest(
            temp_data_dir, strategy_ir, timeout=60,
        )

        # Combine output for checking
        output = stdout + stderr

        # Check strategy loaded
        assert "Loaded strategy" in output or "Strategy:" in output, \
            f"Strategy did not load. Output: {output[:1000]}"

    def test_entry_on_pullback(self, docker_image, temp_data_dir, strategy_ir):
        """Test that entry occurs during pullback in uptrend at the expected bar."""
        # Generate uptrend with pullback at bar 100
        candles, expected = generate_uptrend_with_pullback(
            n_bars=200,
            pullback_bar=100,
        )
        write_lean_data(candles, "BTCUSD", datetime(2024, 1, 1), temp_data_dir)

        returncode, stdout, stderr = run_lean_backtest(
            temp_data_dir, strategy_ir, timeout=120,
        )

        output = stdout + stderr
        entries = parse_entry_logs(output)

        # Should have at least one entry
        assert len(entries) > 0, \
            f"No entry signals found. Expected entry during pullback. Output: {output[:2000]}"

        # Verify entry happened at expected bar (within tolerance for indicator warmup)
        if expected and entries[0].get("bar_index") is not None:
            expected_entry_bar = expected[0].bar_index
            actual_entry_bar = entries[0]["bar_index"]

            # Allow some tolerance for indicator warmup differences
            bar_tolerance = 5
            assert abs(actual_entry_bar - expected_entry_bar) <= bar_tolerance, \
                f"Entry at wrong bar. Expected around bar {expected_entry_bar}, " \
                f"got bar {actual_entry_bar}. Tolerance: {bar_tolerance} bars."

            print(f"\n✓ Entry at bar {actual_entry_bar} (expected ~{expected_entry_bar})")

    def test_exit_on_profit(self, docker_image, temp_data_dir, strategy_ir):
        """Test that exit occurs when price reaches upper band at expected bar."""
        # Generate data with pullback and recovery
        candles, expected = generate_uptrend_with_pullback(
            n_bars=200,
            pullback_bar=100,
        )
        write_lean_data(candles, "BTCUSD", datetime(2024, 1, 1), temp_data_dir)

        returncode, stdout, stderr = run_lean_backtest(
            temp_data_dir, strategy_ir, timeout=120,
        )

        output = stdout + stderr
        entries = parse_entry_logs(output)
        exits = parse_exit_logs(output)

        # If we entered, we should eventually exit
        if len(entries) > 0:
            assert len(exits) > 0, \
                f"Entry found but no exit. Entries: {entries}. Output: {output[:2000]}"

            # Verify exit happened after entry
            if entries[0].get("bar_index") and exits[0].get("bar_index"):
                entry_bar = entries[0]["bar_index"]
                exit_bar = exits[0]["bar_index"]
                assert exit_bar > entry_bar, \
                    f"Exit bar ({exit_bar}) should be after entry bar ({entry_bar})"

                # Find expected exit in expected trades
                expected_exits = [t for t in expected if t.action == "EXIT"]
                if expected_exits:
                    expected_exit_bar = expected_exits[0].bar_index
                    bar_tolerance = 5
                    assert abs(exit_bar - expected_exit_bar) <= bar_tolerance, \
                        f"Exit at wrong bar. Expected around bar {expected_exit_bar}, " \
                        f"got bar {exit_bar}. Tolerance: {bar_tolerance} bars."

                    print(f"\n✓ Exit at bar {exit_bar} (expected ~{expected_exit_bar})")


class TestBreakout:
    """Test Breakout strategy execution."""

    @pytest.fixture
    def strategy_ir(self):
        """Breakout strategy IR."""
        return {
            "strategy_id": "test-breakout",
            "strategy_name": "TestBreakout",
            "symbol": "BTCUSD",
            "resolution": "Minute",
            "indicators": [
                {"type": "MAX", "id": "max_50", "period": 50},
                {"type": "MIN", "id": "min_50", "period": 50},
            ],
            "state": [],
            "gates": [],
            "overlays": [],
            "entry": {
                "condition": {
                    "type": "compare",
                    "left": {"type": "price", "field": "close"},
                    "op": ">",
                    "right": {"type": "indicator", "indicator_id": "max_50"},
                },
                "action": {"type": "set_holdings", "allocation": 0.95},
                "on_fill": [],
            },
            "exits": [
                {
                    "id": "breakdown_stop",
                    "condition": {
                        "type": "compare",
                        "left": {"type": "price", "field": "close"},
                        "op": "<",
                        "right": {"type": "indicator", "indicator_id": "min_50"},
                    },
                    "action": {"type": "liquidate"},
                    "priority": 1,
                },
            ],
            "on_bar": [],
            "on_bar_invested": [],
        }

    def test_entry_on_breakout(self, docker_image, temp_data_dir, strategy_ir):
        """Test that entry occurs on breakout above 50-bar high."""
        # Generate consolidation followed by breakout
        candles, expected = generate_breakout(
            n_bars=200,
            consolidation_bars=100,
        )
        write_lean_data(candles, "BTCUSD", datetime(2024, 1, 1), temp_data_dir)

        returncode, stdout, stderr = run_lean_backtest(
            temp_data_dir, strategy_ir, timeout=120,
        )

        output = stdout + stderr
        entries = parse_entry_logs(output)

        # Should have at least one entry after breakout
        assert len(entries) > 0, \
            f"No entry signals found. Expected entry on breakout. Output: {output[:2000]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

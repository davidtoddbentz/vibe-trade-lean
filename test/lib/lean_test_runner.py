"""LEAN test runner for running strategy IRs through actual LEAN backtests.

This module bridges vibe-trade-execution test scenarios with LEAN Docker execution,
enabling the same strategy IRs to be tested against both:
1. The Python evaluator (fast unit tests)
2. The LEAN engine (integration verification)

Usage:
    from test.lib.lean_test_runner import LEANTestRunner, LEANTestScenario

    runner = LEANTestRunner()

    scenario = LEANTestScenario(
        name="ema_crossover_bullish",
        strategy_ir=strategy_ir_dict,
        data_pattern="uptrend_crossover",
        expected_entry_count=1,
        expected_exit_count=1,
    )

    result = runner.run(scenario)
    assert result.success
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    from .test_data_builder import TestDataBuilder, ExpectedTrade
except ImportError:
    from test_data_builder import TestDataBuilder, ExpectedTrade


@dataclass
class LEANTradeResult:
    """A trade parsed from LEAN logs."""
    action: Literal["ENTRY", "EXIT"]
    price: float
    exit_reason: str | None = None
    timestamp: str | None = None


@dataclass
class LEANTestResult:
    """Result of running a LEAN test scenario."""
    success: bool
    trades: list[LEANTradeResult]
    entry_count: int
    exit_count: int
    end_equity: float
    log_file: str
    errors: list[str] = field(default_factory=list)


@dataclass
class LEANTestScenario:
    """A test scenario to run through LEAN."""
    name: str
    strategy_ir: dict
    data_builder: TestDataBuilder
    expected_trades: list[ExpectedTrade]
    symbol: str = "TESTUSD"
    start_date: str = "2024-01-01"
    end_date: str = "2024-01-01"


class LEANTestRunner:
    """Runs strategy IR through LEAN Docker for integration testing."""

    def __init__(
        self,
        image_name: str = "vibe-trade-lean:latest",
        strategy_runtime_path: str | None = None,
        config_path: str | None = None,
    ):
        self.image_name = image_name
        self.project_root = Path(__file__).parent.parent.parent

        self.strategy_runtime_path = strategy_runtime_path or str(
            self.project_root / "src" / "Algorithms" / "StrategyRuntime.py"
        )
        self.config_path = config_path or str(
            self.project_root / "test" / "config-backtest.json"
        )

    def run(self, scenario: LEANTestScenario, timeout: int = 120) -> LEANTestResult:
        """Run a test scenario through LEAN.

        Args:
            scenario: The test scenario to run
            timeout: Timeout in seconds for Docker execution

        Returns:
            LEANTestResult with trades and outcomes
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create data directory structure
            # LEAN expects: /Data/crypto/{market}/{resolution}/{symbol}/
            data_dir = tmp_path / "data"
            crypto_dir = data_dir / "crypto" / "coinbase" / "daily" / scenario.symbol.lower()
            crypto_dir.mkdir(parents=True)

            results_dir = tmp_path / "results"
            results_dir.mkdir()

            # Export test data as ZIP (LEAN format for crypto)
            date_str = scenario.start_date.replace("-", "")
            zip_path = crypto_dir / f"{date_str}_trade.zip"
            scenario.data_builder.export_zip(str(zip_path))

            # Export strategy IR
            ir_path = data_dir / "strategy_ir.json"
            ir_path.write_text(json.dumps(scenario.strategy_ir, indent=2))

            # Create custom config with IR path parameter
            config_path = tmp_path / "config.json"
            with open(self.config_path) as f:
                base_config = json.load(f)
            base_config["parameters"] = {"strategy_ir_path": str(ir_path)}
            config_path.write_text(json.dumps(base_config, indent=2))

            # Run LEAN Docker
            log_file = tmp_path / "lean.log"

            # Path to test symbol database
            test_symbol_db = self.project_root / "test" / "data" / "symbol-properties" / "symbol-properties-database.csv"

            cmd = [
                "docker", "run", "--rm",
                "-e", f"START_DATE={scenario.start_date}",
                "-e", f"END_DATE={scenario.end_date}",
                "-e", "SKIP_DATA_DOWNLOAD=1",
                "-v", f"{crypto_dir}:/Lean/Data/crypto/coinbase/daily/{scenario.symbol.lower()}:ro",
                "-v", f"{ir_path}:{str(ir_path)}:ro",
                "-v", f"{test_symbol_db}:/Lean/Data/symbol-properties/symbol-properties-database.csv:ro",
                "-v", f"{results_dir}:/Results",
                "-v", f"{self.strategy_runtime_path}:/Lean/Algorithm.Python/StrategyRuntime.py:ro",
                "-v", f"{config_path}:/Lean/Launcher/bin/Debug/config.json:ro",
                self.image_name,
                "--config", "/Lean/Launcher/bin/Debug/config.json",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                log_content = result.stdout + result.stderr

                # Save log for debugging
                log_file.write_text(log_content)

                # Parse results
                return self._parse_results(log_content, str(log_file))

            except subprocess.TimeoutExpired:
                return LEANTestResult(
                    success=False,
                    trades=[],
                    entry_count=0,
                    exit_count=0,
                    end_equity=0,
                    log_file=str(log_file),
                    errors=["Timeout expired"],
                )
            except Exception as e:
                return LEANTestResult(
                    success=False,
                    trades=[],
                    entry_count=0,
                    exit_count=0,
                    end_equity=0,
                    log_file=str(log_file),
                    errors=[str(e)],
                )

    def _parse_results(self, log_content: str, log_file: str) -> LEANTestResult:
        """Parse LEAN log output to extract trades and results."""
        trades = []
        errors = []

        # Parse entry trades
        # Log format: "ENTRY executed at bar=64 price=51549.9388143939"
        entry_pattern = r"ENTRY executed at bar=\d+ price=([\d.]+)"
        for match in re.finditer(entry_pattern, log_content):
            trades.append(LEANTradeResult(
                action="ENTRY",
                price=float(match.group(1)),
            ))

        # Parse exit trades
        # Log format: "EXIT (profit_target) executed at bar=74 price=52725.5097161582"
        exit_pattern = r"EXIT \((\w+)\) executed at bar=\d+ price=([\d.]+)"
        for match in re.finditer(exit_pattern, log_content):
            trades.append(LEANTradeResult(
                action="EXIT",
                price=float(match.group(2)),
                exit_reason=match.group(1),
            ))

        # Parse end equity
        equity_pattern = r"End Equity\s+([\d.]+)"
        equity_match = re.search(equity_pattern, log_content)
        end_equity = float(equity_match.group(1)) if equity_match else 0.0

        # Parse errors
        error_pattern = r"ERROR:: (.+)"
        for match in re.finditer(error_pattern, log_content):
            errors.append(match.group(1))

        # Check for strategy load (StrategyRuntime initialization message)
        strategy_loaded = "StrategyRuntime initialized" in log_content

        entry_count = sum(1 for t in trades if t.action == "ENTRY")
        exit_count = sum(1 for t in trades if t.action == "EXIT")

        return LEANTestResult(
            success=strategy_loaded and len(errors) == 0,
            trades=trades,
            entry_count=entry_count,
            exit_count=exit_count,
            end_equity=end_equity,
            log_file=log_file,
            errors=errors,
        )


def create_ema_crossover_scenario(
    bullish: bool = True,
    ema_fast: int = 20,
    ema_slow: int = 50,
) -> LEANTestScenario:
    """Create an EMA crossover test scenario.

    Args:
        bullish: True for bullish crossover (fast crosses above slow)
        ema_fast: Fast EMA period
        ema_slow: Slow EMA period

    Returns:
        LEANTestScenario ready for execution
    """
    builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

    if bullish:
        # Create data where fast EMA crosses above slow EMA
        # Start with downtrend (fast < slow), then uptrend (fast > slow)
        builder.add_downtrend(bars=ema_slow + 10, start_price=50000, trend_strength=0.001)
        builder.add_uptrend(bars=40, trend_strength=0.003)  # Strong uptrend for crossover
        builder.add_uptrend(bars=20, trend_strength=0.001)  # Continue for exit
    else:
        # Bearish: start uptrend, then downtrend for crossover
        builder.add_uptrend(bars=ema_slow + 10, start_price=50000, trend_strength=0.001)
        builder.add_downtrend(bars=40, trend_strength=0.003)
        builder.add_downtrend(bars=20, trend_strength=0.001)

    # Strategy IR for EMA crossover
    direction = "long" if bullish else "short"
    entry_op = ">" if bullish else "<"
    exit_op = "<" if bullish else ">"

    strategy_ir = {
        "strategy_id": f"ema-crossover-{direction}",
        "strategy_name": f"EMA Crossover {direction.title()}",
        "symbol": "TESTUSD",
        "resolution": "Minute",
        "indicators": [
            {"type": "EMA", "id": "ema_fast", "period": ema_fast},
            {"type": "EMA", "id": "ema_slow", "period": ema_slow},
        ],
        "state": [],
        "gates": [],
        "overlays": [],
        "entry": {
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "ema_fast"},
                "op": entry_op,
                "right": {"type": "indicator", "indicator_id": "ema_slow"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "crossover_exit",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "ema_fast"},
                    "op": exit_op,
                    "right": {"type": "indicator", "indicator_id": "ema_slow"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "on_bar": [],
        "on_bar_invested": [],
    }

    # Calculate expected trades
    # For bullish: entry when fast crosses above slow, exit when crosses below
    expected_trades = []
    ema_fast_values = builder.calculate_ema(ema_fast)
    ema_slow_values = builder.calculate_ema(ema_slow)

    in_position = False
    warmup = max(ema_fast, ema_slow)

    for i in range(warmup, len(builder._prices)):
        fast = ema_fast_values[i]
        slow = ema_slow_values[i]

        if fast is None or slow is None:
            continue

        if bullish:
            entry_cond = fast > slow
            exit_cond = fast < slow
        else:
            entry_cond = fast < slow
            exit_cond = fast > slow

        if not in_position and entry_cond:
            expected_trades.append(ExpectedTrade(
                bar_index=i,
                action="ENTRY",
                direction=direction,
                price=builder._prices[i],
                reason=f"EMA{ema_fast} {entry_op} EMA{ema_slow}"
            ))
            in_position = True
        elif in_position and exit_cond:
            expected_trades.append(ExpectedTrade(
                bar_index=i,
                action="EXIT",
                direction=direction,
                price=builder._prices[i],
                reason=f"EMA{ema_fast} {exit_op} EMA{ema_slow}"
            ))
            in_position = False

    return LEANTestScenario(
        name=f"ema_crossover_{direction}",
        strategy_ir=strategy_ir,
        data_builder=builder,
        expected_trades=expected_trades,
    )


def create_trend_pullback_scenario(
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    profit_target_pct: float = 0.02,
    stop_loss_pct: float = 0.01,
) -> LEANTestScenario:
    """Create a trend pullback test scenario.

    Entry: EMA fast > EMA slow AND close < BB lower
    Exit: profit target or stop loss

    Returns:
        LEANTestScenario ready for execution
    """
    try:
        from .test_data_builder import create_trend_pullback_test
    except ImportError:
        from test_data_builder import create_trend_pullback_test

    builder, expected_trades = create_trend_pullback_test(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        bb_period=bb_period,
        bb_mult=bb_mult,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
    )

    # Strategy IR matching the test data expectations
    strategy_ir = {
        "strategy_id": "trend-pullback-test",
        "strategy_name": "TrendPullback",
        "symbol": "TESTUSD",
        "resolution": "Minute",
        "indicators": [
            {"type": "EMA", "id": "ema_fast", "period": ema_fast},
            {"type": "EMA", "id": "ema_slow", "period": ema_slow},
            {"type": "BB", "id": "bb", "period": bb_period, "multiplier": bb_mult},
        ],
        "state": [
            {"id": "entry_price", "var_type": "float", "default": 0.0},
        ],
        "gates": [],
        "overlays": [],
        "entry": {
            "condition": {
                "type": "allOf",
                "conditions": [
                    {
                        "type": "compare",
                        "left": {"type": "indicator", "indicator_id": "ema_fast"},
                        "op": ">",
                        "right": {"type": "indicator", "indicator_id": "ema_slow"},
                    },
                    {
                        "type": "compare",
                        "left": {"type": "price", "field": "close"},
                        "op": "<",
                        "right": {"type": "indicator_band", "indicator_id": "bb", "band": "lower"},
                    },
                ],
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [
                {"type": "set_state", "state_id": "entry_price", "value": {"type": "price", "field": "close"}},
            ],
        },
        "exits": [
            {
                "id": "profit_target",
                "priority": 2,
                "condition": {
                    "type": "compare",
                    "left": {
                        "type": "expr",
                        "op": "/",
                        "left": {
                            "type": "expr",
                            "op": "-",
                            "left": {"type": "price", "field": "close"},
                            "right": {"type": "state", "state_id": "entry_price"},
                        },
                        "right": {"type": "state", "state_id": "entry_price"},
                    },
                    "op": ">=",
                    "right": {"type": "literal", "value": profit_target_pct},
                },
                "action": {"type": "liquidate"},
            },
            {
                "id": "stop_loss",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {
                        "type": "expr",
                        "op": "/",
                        "left": {
                            "type": "expr",
                            "op": "-",
                            "left": {"type": "state", "state_id": "entry_price"},
                            "right": {"type": "price", "field": "close"},
                        },
                        "right": {"type": "state", "state_id": "entry_price"},
                    },
                    "op": ">=",
                    "right": {"type": "literal", "value": stop_loss_pct},
                },
                "action": {"type": "liquidate"},
            },
        ],
        "on_bar": [],
        "on_bar_invested": [],
    }

    return LEANTestScenario(
        name="trend_pullback",
        strategy_ir=strategy_ir,
        data_builder=builder,
        expected_trades=expected_trades,
    )


# Pre-defined scenario creators for common test cases
SCENARIO_CREATORS = {
    "ema_crossover_bullish": lambda: create_ema_crossover_scenario(bullish=True),
    "ema_crossover_bearish": lambda: create_ema_crossover_scenario(bullish=False),
    "trend_pullback": create_trend_pullback_scenario,
}

"""Test utilities for vibe-trade-lean."""

from .test_data_builder import (
    ExpectedTrade,
    OHLCV,
    TestDataBuilder,
    create_trend_pullback_test,
)

from .lean_test_runner import (
    LEANTestRunner,
    LEANTestScenario,
    LEANTestResult,
    LEANTradeResult,
    create_ema_crossover_scenario,
    create_trend_pullback_scenario,
    SCENARIO_CREATORS,
)

__all__ = [
    # Data builder
    "ExpectedTrade",
    "OHLCV",
    "TestDataBuilder",
    "create_trend_pullback_test",
    # LEAN runner
    "LEANTestRunner",
    "LEANTestScenario",
    "LEANTestResult",
    "LEANTradeResult",
    "create_ema_crossover_scenario",
    "create_trend_pullback_scenario",
    "SCENARIO_CREATORS",
]

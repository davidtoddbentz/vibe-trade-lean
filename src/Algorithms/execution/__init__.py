"""Execution layer types and context for StrategyRuntime.

Note: orchestration and actions are imported directly by their consumers
to avoid circular imports (orchestration → trades → execution.types).
"""

from .context import ExecutionContext
from .types import (
    ClosedLot,
    EquityPoint,
    FillInfo,
    Lot,
    TrackingState,
    TradeStats,
)

__all__ = [
    "ClosedLot",
    "EquityPoint",
    "ExecutionContext",
    "FillInfo",
    "Lot",
    "TrackingState",
    "TradeStats",
]

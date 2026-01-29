"""Entry/exit execution orchestration for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime.
"""

from .actions import execute_action
from .context import ExecutionContext
from .orchestration import execute_entry, execute_exit
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
    "execute_action",
    "execute_entry",
    "execute_exit",
]

"""Trade and lot tracking for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime to reduce line count.
"""

from .tracking import (
    create_lot,
    close_lots,
    close_lots_at_end,
    calculate_trade_stats,
    generate_report,
    split_lot,
)

__all__ = [
    "create_lot",
    "close_lots",
    "close_lots_at_end",
    "calculate_trade_stats",
    "generate_report",
    "split_lot",
]

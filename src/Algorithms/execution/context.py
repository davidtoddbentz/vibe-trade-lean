"""ExecutionContext — bundles LEAN primitives for the execution layer.

Eliminates passing 8+ separate Any parameters to every function.
LEAN types are imported under TYPE_CHECKING (runtime-only in Docker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from execution.types import FillInfo


@dataclass
class ExecutionContext:
    """LEAN execution primitives bundled for clean function signatures.

    Created once in StrategyRuntime.Initialize(), passed to execution functions.
    """
    symbol: Any  # Symbol at runtime
    portfolio: Any  # SecurityPortfolioManager at runtime
    securities: Any  # SecurityManager at runtime
    set_holdings: Callable  # self.SetHoldings
    market_order: Callable  # self.MarketOrder
    liquidate: Callable  # self.Liquidate
    log: Callable[[str], None]  # self.Log
    get_last_fill: Callable[[], FillInfo | None]  # self._get_and_clear_last_fill

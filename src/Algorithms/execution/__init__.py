"""Entry/exit execution orchestration for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime.
"""

from .actions import execute_action
from .orchestration import execute_entry, execute_exit

__all__ = ["execute_action", "execute_entry", "execute_exit"]

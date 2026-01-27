"""Trading costs (fees and slippage) for StrategyRuntime.

Phase 12: Extracted from StrategyRuntime.
"""

from .calculation import apply_slippage, calculate_fee

__all__ = ["apply_slippage", "calculate_fee"]

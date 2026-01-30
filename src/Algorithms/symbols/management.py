"""Symbol management utilities.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any
from AlgorithmImports import Symbol


def normalize_symbol(symbol_str: str) -> str:
    """Normalize symbol string for dictionary keys (lowercase, no dashes)."""
    return symbol_str.lower().replace("-", "")


def get_symbol_obj(
    symbol_str: str | None,
    symbols_dict: dict[str, Symbol],
    primary_symbol: Symbol,
    normalize_func: Any,
) -> Symbol:
    """Get symbol object from string, defaulting to primary if None.

    Args:
        symbol_str: Symbol string or None
        symbols_dict: Dict of normalized symbol strings -> Symbol objects
        primary_symbol: Primary trading symbol
        normalize_func: Function to normalize symbol strings

    Returns:
        Symbol object
    """
    if symbol_str is None:
        return primary_symbol
    normalized = normalize_func(symbol_str)
    return symbols_dict.get(normalized, primary_symbol)

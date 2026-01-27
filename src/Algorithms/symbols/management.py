"""Symbol management utilities.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any
from AlgorithmImports import Symbol, SymbolProperties

# CustomCryptoData is in /Lean/src/data/ - import will be resolved at runtime
# We pass the class as a parameter to avoid import issues


def normalize_symbol(symbol_str: str) -> str:
    """Normalize symbol string for dictionary keys (lowercase, no dashes)."""
    return symbol_str.lower().replace("-", "")


def add_symbol(
    symbol_str: str,
    resolution: Any,
    add_data_func: Any,
    log_func: Any,
    custom_data_class: Any,  # CustomCryptoData class
) -> Symbol:
    """Add symbol using custom data reader for CSV files.

    After adding the symbol, we configure its lot size to allow
    fractional orders (like 0.1 BTC for a $10 order).

    Args:
        symbol_str: Symbol string (e.g., "BTC-USD")
        resolution: Data resolution
        add_data_func: Function to add data (typically runtime.AddData)
        log_func: Logging function
        custom_data_class: CustomCryptoData class

    Returns:
        Symbol object
    """
    # Use AddData with CustomCryptoData for CSV files
    security = add_data_func(custom_data_class, symbol_str, resolution)
    symbol = security.Symbol

    # CRITICAL: Configure lot size to allow fractional orders
    # By default, custom data uses lot size 1, which prevents orders < 1 unit
    # For crypto-like assets, we need very small lot sizes (0.00000001)
    security.SymbolProperties = SymbolProperties(
        description=symbol_str,
        quoteCurrency="USD",
        contractMultiplier=1,
        minimumPriceVariation=0.01,
        lotSize=0.00000001,  # Allow tiny fractional orders
        marketTicker=symbol_str,
        minimumOrderSize=0.00000001  # Allow tiny orders
    )
    log_func(f"   Added symbol: {symbol_str} (lot_size=0.00000001)")

    return symbol


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

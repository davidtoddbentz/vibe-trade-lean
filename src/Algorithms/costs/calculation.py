"""Fee and slippage calculation.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations


def apply_slippage(price: float, is_buy: bool, slippage_bps: int) -> float:
    """Apply slippage to a price.

    Args:
        price: Base price
        is_buy: True for buy orders, False for sell orders
        slippage_bps: Slippage in basis points (e.g., 10 = 0.1%)

    Returns:
        Price with slippage applied
    """
    slippage_pct = slippage_bps / 10000.0
    if is_buy:
        # Buying: pay more (slippage up)
        return price * (1 + slippage_pct)
    else:
        # Selling: receive less (slippage down)
        return price * (1 - slippage_pct)


def calculate_fee(trade_value: float, fee_percentage: float) -> float:
    """Calculate fee as percentage of trade value.

    Args:
        trade_value: Total value of the trade
        fee_percentage: Fee as percentage (e.g., 0.1 for 0.1%)

    Returns:
        Fee amount
    """
    return trade_value * (fee_percentage / 100.0)

"""Action execution logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import (
    SetHoldingsAction,
    LiquidateAction,
    MarketOrderAction,
    EntryAction,
    ExitAction,
)


def execute_action(
    action: EntryAction | ExitAction,
    symbol: Any,
    portfolio: Any,
    securities: Any,
    calculate_order_quantity_func: Any,
    market_order_func: Any,
    liquidate_func: Any,
    log_func: Any,
    bar: Any = None,
) -> None:
    """Execute an action from IR.

    Supports multiple sizing modes for set_holdings:
    - pct_equity: uses allocation field directly (fraction of portfolio)
    - fixed_usd: fixed USD amount, converted to quantity at current price
    - fixed_units: fixed number of units/shares to trade

    Note: For all sizing modes, we use CalculateOrderQuantity + MarketOrder
    instead of SetHoldings to ensure small orders are properly executed.
    See: https://www.quantconnect.com/forum/discussion/2978/minimum-order-clip-size/

    Args:
        action: Typed action from IR (SetHoldingsAction, LiquidateAction, MarketOrderAction)
        symbol: Trading symbol
        portfolio: LEAN Portfolio object
        securities: LEAN Securities object
        calculate_order_quantity_func: Function to calculate order quantity
        market_order_func: Function to place market order
        liquidate_func: Function to liquidate position
        log_func: Logging function
        bar: Current bar data (used for fixed_usd/fixed_units pricing)
    """
    if not action:
        return

    if isinstance(action, SetHoldingsAction):
        sizing_mode = action.sizing_mode

        if sizing_mode == "pct_equity":
            # Use CalculateOrderQuantity + MarketOrder instead of SetHoldings
            # This ensures small orders are executed rather than silently skipped
            allocation = action.allocation
            quantity = calculate_order_quantity_func(symbol, allocation)
            if quantity != 0:
                market_order_func(symbol, quantity)
            else:
                log_func(f"⚠️ Order quantity is zero for allocation={allocation}, skipping order")

        elif sizing_mode == "fixed_usd":
            # Fixed USD amount - convert to portfolio allocation
            # We use CalculateOrderQuantity to ensure LEAN handles the order properly
            if action.fixed_usd is None:
                log_func("⚠️ Cannot execute fixed_usd: fixed_usd is None")
                return
            fixed_usd = action.fixed_usd
            portfolio_value = portfolio.TotalPortfolioValue
            if portfolio_value > 0:
                allocation = fixed_usd / portfolio_value
                quantity = calculate_order_quantity_func(symbol, allocation)
                if quantity != 0:
                    market_order_func(symbol, quantity)
                    price = securities[symbol].Price
                    log_func(f"   Fixed USD: ${abs(fixed_usd):.2f} -> {abs(quantity):.6f} units @ ${price:.2f}")
                else:
                    log_func(f"⚠️ Order quantity is zero for fixed_usd=${fixed_usd}, skipping order")
            else:
                log_func(f"⚠️ Cannot execute fixed_usd: portfolio value is {portfolio_value}")

        elif sizing_mode == "fixed_units":
            # Fixed number of units - use CalculateOrderQuantity for proper handling
            if action.fixed_units is None:
                log_func("⚠️ Cannot execute fixed_units: fixed_units is None")
                return
            fixed_units = action.fixed_units
            price = float(bar.Close) if bar else securities[symbol].Price
            if price > 0:
                portfolio_value = portfolio.TotalPortfolioValue
                if portfolio_value > 0:
                    allocation = (fixed_units * price) / portfolio_value
                    quantity = calculate_order_quantity_func(symbol, allocation)
                    if quantity != 0:
                        market_order_func(symbol, quantity)
                        log_func(f"   Fixed units: {abs(quantity):.6f}")
                    else:
                        log_func("⚠️ Order quantity is zero for fixed_units, skipping order")
                else:
                    log_func(f"⚠️ Cannot execute fixed_units: portfolio value is {portfolio_value}")
            else:
                log_func(f"⚠️ Cannot execute fixed_units: price is {price}")

        else:
            log_func(f"⚠️ Unknown sizing_mode: {sizing_mode}")

    elif isinstance(action, LiquidateAction):
        liquidate_func(symbol)

    elif isinstance(action, MarketOrderAction):
        quantity = action.quantity
        if quantity != 0:
            market_order_func(symbol, quantity)
        else:
            log_func("⚠️ Order quantity is zero for market_order, skipping order")

    else:
        log_func(f"⚠️ Unknown action type: {type(action)}")

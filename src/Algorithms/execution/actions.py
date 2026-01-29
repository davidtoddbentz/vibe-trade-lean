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
from execution.context import ExecutionContext


def execute_action(
    action: EntryAction | ExitAction,
    ctx: ExecutionContext,
    bar: Any = None,
) -> None:
    """Execute an action from IR.

    Supports multiple sizing modes for set_holdings:
    - pct_equity: uses LEAN's SetHoldings directly (MinimumOrderMarginPortfolioPercentage=0)
    - fixed_usd: fixed USD amount, converted to quantity at current price
    - fixed_units: fixed number of units/shares to trade

    Args:
        action: Typed action from IR (SetHoldingsAction, LiquidateAction, MarketOrderAction)
        ctx: ExecutionContext bundle for LEAN primitives
        bar: Current bar data (used for fixed_usd/fixed_units pricing)
    """
    if not action:
        return

    if isinstance(action, SetHoldingsAction):
        sizing_mode = action.sizing_mode

        if sizing_mode == "pct_equity":
            # MinimumOrderMarginPortfolioPercentage=0 is set in Initialize(),
            # so SetHoldings works correctly for small orders
            ctx.set_holdings(ctx.symbol, action.allocation)

        elif sizing_mode == "fixed_usd":
            # Fixed USD amount - compute quantity directly from price
            # We bypass CalculateOrderQuantity because it computes target allocation
            # (not incremental), which gives wrong results for accumulate mode.
            if action.fixed_usd is None:
                ctx.log("⚠️ Cannot execute fixed_usd: fixed_usd is None")
                return
            fixed_usd = action.fixed_usd
            price = float(bar.Close) if bar else ctx.securities[ctx.symbol].Price
            if price > 0:
                quantity = fixed_usd / price
                if quantity > 0:
                    ctx.market_order(ctx.symbol, quantity)
                    ctx.log(f"   Fixed USD: ${abs(fixed_usd):.2f} -> {abs(quantity):.6f} units @ ${price:.2f}")
                else:
                    ctx.log(f"⚠️ Order quantity is zero for fixed_usd=${fixed_usd}, skipping order")
            else:
                ctx.log(f"⚠️ Cannot execute fixed_usd: price is {price}")

        elif sizing_mode == "fixed_units":
            # Fixed number of units - use MarketOrder directly
            if action.fixed_units is None:
                ctx.log("⚠️ Cannot execute fixed_units: fixed_units is None")
                return
            fixed_units = action.fixed_units
            if fixed_units > 0:
                ctx.market_order(ctx.symbol, fixed_units)
                ctx.log(f"   Fixed units: {abs(fixed_units):.6f}")
            else:
                ctx.log("⚠️ Order quantity is zero for fixed_units, skipping order")

        else:
            ctx.log(f"⚠️ Unknown sizing_mode: {sizing_mode}")

    elif isinstance(action, LiquidateAction):
        ctx.liquidate(ctx.symbol)

    elif isinstance(action, MarketOrderAction):
        quantity = action.quantity
        if quantity != 0:
            ctx.market_order(ctx.symbol, quantity)
        else:
            ctx.log("⚠️ Order quantity is zero for market_order, skipping order")

    else:
        ctx.log(f"⚠️ Unknown action type: {type(action)}")

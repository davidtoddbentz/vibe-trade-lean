"""Action execution logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import (
    SetHoldingsAction,
    LiquidateAction,
    ReducePositionAction,
    MarketOrderAction,
    EntryAction,
    ExitAction,
)
from execution.context import ExecutionContext


def _clamp_quantity_by_notional(
    quantity: float,
    price: float,
    min_usd: float | None,
    max_usd: float | None,
    ctx: ExecutionContext,
) -> float | None:
    """Clamp a quantity to min/max notional USD constraints.

    Returns the clamped quantity, or None if the order should be skipped
    (notional below min_usd floor).
    """
    if price <= 0:
        return quantity
    notional = abs(quantity) * price
    sign = 1.0 if quantity >= 0 else -1.0

    if min_usd is not None and notional < min_usd:
        ctx.log(f"   Skipping order: notional ${notional:.2f} < min_usd ${min_usd:.2f}")
        return None

    if max_usd is not None and notional > max_usd:
        clamped_qty = sign * (max_usd / price)
        ctx.log(f"   Clamping: ${notional:.2f} -> max_usd ${max_usd:.2f} ({abs(clamped_qty):.6f} units)")
        return clamped_qty

    return quantity


def _compute_quantity(
    action: SetHoldingsAction,
    ctx: ExecutionContext,
    bar: Any,
) -> float | None:
    """Compute order quantity from a SetHoldingsAction's sizing params.

    Returns the signed quantity, or None if the order should be skipped.
    Used by typed orders (limit/stop/stop_limit) which need explicit quantity
    rather than LEAN's SetHoldings percentage API.
    """
    price = float(bar.Close) if bar else ctx.securities[ctx.symbol].Price
    min_usd = action.min_usd
    max_usd = action.max_usd

    if action.sizing_mode == "pct_equity":
        equity = float(ctx.portfolio.TotalPortfolioValue)
        if equity <= 0 or price <= 0:
            ctx.log(f"   Cannot compute quantity: equity={equity}, price={price}")
            return None
        target_notional = abs(action.allocation) * equity
        sign = 1.0 if action.allocation >= 0 else -1.0
        quantity = sign * (target_notional / price)

    elif action.sizing_mode == "fixed_usd":
        if action.fixed_usd is None:
            ctx.log("   Cannot compute quantity: fixed_usd is None")
            return None
        if price <= 0:
            ctx.log(f"   Cannot compute quantity: price is {price}")
            return None
        quantity = action.fixed_usd / price

    elif action.sizing_mode == "fixed_units":
        if action.fixed_units is None:
            ctx.log("   Cannot compute quantity: fixed_units is None")
            return None
        quantity = action.fixed_units

    else:
        ctx.log(f"   Unknown sizing_mode: {action.sizing_mode}")
        return None

    # Apply notional clamps
    quantity = _clamp_quantity_by_notional(quantity, price, min_usd, max_usd, ctx)
    if quantity is None:
        return None

    if quantity == 0:
        ctx.log("   Skipping order: computed quantity is zero")
        return None

    return quantity


def _execute_typed_order(
    action: SetHoldingsAction,
    ctx: ExecutionContext,
    bar: Any,
) -> int | None:
    """Execute a non-market order via LEAN's native order APIs.

    CustomDataFillModel (C#) overrides GetPrices() to read OHLC from
    PythonData's DynamicData storage, so LEAN's fill model handles
    fill conditions and prices correctly.

    Returns the LEAN OrderId if an order was placed, or None if skipped.
    """
    order_type = action.order_type
    quantity = _compute_quantity(action, ctx, bar)
    if quantity is None:
        return None

    # Resolve price refs via ctx.resolve_value(value_ref, bar) -> float
    limit_price = None
    stop_price = None

    if action.limit_price_ref is not None:
        limit_price = ctx.resolve_value(action.limit_price_ref, bar)
        if limit_price is None or limit_price <= 0:
            ctx.log(f"   Invalid limit price: {limit_price}, skipping {order_type} order")
            return None

    if action.stop_price_ref is not None:
        stop_price = ctx.resolve_value(action.stop_price_ref, bar)
        if stop_price is None or stop_price <= 0:
            ctx.log(f"   Invalid stop price: {stop_price}, skipping {order_type} order")
            return None

    # Place native LEAN order — CustomDataFillModel handles fill logic
    ticket = None
    if order_type == "limit":
        ticket = ctx.limit_order(ctx.symbol, quantity, limit_price)
        ctx.log(f"   Limit order placed: {quantity:.6f} units @ limit=${limit_price:.2f} (OrderId={ticket.OrderId})")
    elif order_type == "stop":
        ticket = ctx.stop_market_order(ctx.symbol, quantity, stop_price)
        ctx.log(f"   Stop order placed: {quantity:.6f} units @ stop=${stop_price:.2f} (OrderId={ticket.OrderId})")
    elif order_type == "stop_limit":
        ticket = ctx.stop_limit_order(ctx.symbol, quantity, stop_price, limit_price)
        ctx.log(f"   Stop-limit order placed: {quantity:.6f} units @ stop=${stop_price:.2f}, limit=${limit_price:.2f} (OrderId={ticket.OrderId})")

    if ticket is not None:
        return int(ticket.OrderId)
    return None


def execute_action(
    action: EntryAction | ExitAction,
    ctx: ExecutionContext,
    bar: Any = None,
) -> int | None:
    """Execute an action from IR.

    Supports multiple sizing modes for set_holdings:
    - pct_equity: uses LEAN's SetHoldings directly (MinimumOrderMarginPortfolioPercentage=0)
    - fixed_usd: fixed USD amount, converted to quantity at current price
    - fixed_units: fixed number of units/shares to trade

    For non-market order types (limit, stop, stop_limit), quantity is always
    computed explicitly and dispatched to the appropriate LEAN order API.

    Args:
        action: Typed action from IR (SetHoldingsAction, LiquidateAction, MarketOrderAction)
        ctx: ExecutionContext bundle for LEAN primitives
        bar: Current bar data (used for fixed_usd/fixed_units pricing)

    Returns:
        LEAN OrderId for non-market orders (needed for deferred lot creation),
        or None for market orders and other action types.
    """
    if not action:
        return None

    if isinstance(action, SetHoldingsAction):
        # Non-market orders: compute quantity and dispatch to typed order API
        if action.order_type != "market":
            return _execute_typed_order(action, ctx, bar)

        # Market orders: existing logic (SetHoldings for pct_equity, MarketOrder for others)
        sizing_mode = action.sizing_mode
        min_usd = action.min_usd
        max_usd = action.max_usd

        if sizing_mode == "pct_equity":
            # MinimumOrderMarginPortfolioPercentage=0 is set in Initialize(),
            # so SetHoldings works correctly for small orders
            if min_usd is not None or max_usd is not None:
                # Need to compute notional to clamp
                price = float(bar.Close) if bar else ctx.securities[ctx.symbol].Price
                equity = float(ctx.portfolio.TotalPortfolioValue)
                target_notional = abs(action.allocation) * equity
                sign = 1.0 if action.allocation >= 0 else -1.0
                if min_usd is not None and target_notional < min_usd:
                    ctx.log(f"   Skipping pct_equity order: ${target_notional:.2f} < min_usd ${min_usd:.2f}")
                    return
                if max_usd is not None and target_notional > max_usd:
                    clamped_alloc = sign * (max_usd / equity) if equity > 0 else 0.0
                    ctx.log(f"   Clamping pct_equity: {abs(action.allocation):.2%} -> {abs(clamped_alloc):.2%} (max_usd ${max_usd:.2f})")
                    ctx.set_holdings(ctx.symbol, clamped_alloc)
                    return
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
                quantity = _clamp_quantity_by_notional(quantity, price, min_usd, max_usd, ctx)
                if quantity is None:
                    return
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
            price = float(bar.Close) if bar else ctx.securities[ctx.symbol].Price
            fixed_units = _clamp_quantity_by_notional(fixed_units, price, min_usd, max_usd, ctx)
            if fixed_units is None:
                return
            if fixed_units > 0:
                ctx.market_order(ctx.symbol, fixed_units)
                ctx.log(f"   Fixed units: {abs(fixed_units):.6f}")
            else:
                ctx.log("⚠️ Order quantity is zero for fixed_units, skipping order")

        else:
            ctx.log(f"⚠️ Unknown sizing_mode: {sizing_mode}")

    elif isinstance(action, ReducePositionAction):
        # Reduce position by size_frac of current holdings
        current_qty = float(ctx.portfolio[ctx.symbol].Quantity)
        if current_qty != 0 and action.size_frac > 0:
            reduce_qty = -current_qty * action.size_frac
            ctx.market_order(ctx.symbol, reduce_qty)
            ctx.log(f"   Reduce position: {action.size_frac:.0%} of {abs(current_qty):.6f}")
        elif action.size_frac == 0:
            ctx.log("   Reduce position: size_frac=0, no action")

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

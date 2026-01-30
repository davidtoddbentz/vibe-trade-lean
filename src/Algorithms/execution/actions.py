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


def _bar_ohlc(bar: Any) -> tuple[float, float, float, float]:
    """Extract OHLC from a PythonData bar.

    PythonData stores OHLC as custom properties set via bracket notation
    in Reader(). Access them directly via bar["Key"] rather than
    GetStorageDictionary() which has cross-language interop issues.
    Falls back to bar.Close for any missing field.
    """
    close = float(bar.Close)
    try:
        o = float(bar["Open"])
    except Exception:
        o = close
    try:
        h = float(bar["High"])
    except Exception:
        h = close
    try:
        l = float(bar["Low"])
    except Exception:
        l = close
    try:
        c = float(bar["Close"])
    except Exception:
        c = close
    return o, h, l, c


def _check_fill_condition(
    order_type: str,
    quantity: float,
    limit_price: float | None,
    stop_price: float | None,
    bar_high: float,
    bar_low: float,
) -> bool:
    """Check whether a typed order would fill on this bar's OHLC range.

    For buy orders (quantity > 0):
      - limit: fills if bar Low <= limit_price (price came down to limit)
      - stop:  fills if bar High >= stop_price (price rose to trigger)
      - stop_limit: stop triggered (High >= stop) AND limit fillable (Low <= limit)

    For sell orders (quantity < 0):
      - limit: fills if bar High >= limit_price (price rose to limit)
      - stop:  fills if bar Low <= stop_price (price fell to trigger)
      - stop_limit: stop triggered (Low <= stop) AND limit fillable (High >= limit)
    """
    is_buy = quantity > 0

    if order_type == "limit":
        if is_buy:
            return bar_low <= limit_price
        else:
            return bar_high >= limit_price

    elif order_type == "stop":
        if is_buy:
            return bar_high >= stop_price
        else:
            return bar_low <= stop_price

    elif order_type == "stop_limit":
        if is_buy:
            stop_triggered = bar_high >= stop_price
            limit_fillable = bar_low <= limit_price
            return stop_triggered and limit_fillable
        else:
            stop_triggered = bar_low <= stop_price
            limit_fillable = bar_high >= limit_price
            return stop_triggered and limit_fillable

    return False


def _execute_typed_order(
    action: SetHoldingsAction,
    ctx: ExecutionContext,
    bar: Any,
) -> None:
    """Execute a non-market order (limit, stop, stop_limit) via Python-side fill simulation.

    LEAN's fill model cannot see High/Low from PythonData custom data (security.High/Low
    always equal the close price). Instead of using LEAN's native LimitOrder/StopMarketOrder
    APIs, we check bar OHLC against the limit/stop prices in Python and execute as a
    MarketOrder if the fill condition is met.

    This gives correct fill semantics while keeping all logic in Python.
    """
    order_type = action.order_type
    quantity = _compute_quantity(action, ctx, bar)
    if quantity is None:
        return

    # Resolve price refs via ctx.resolve_value(value_ref, bar) -> float
    limit_price = None
    stop_price = None

    if action.limit_price_ref is not None:
        limit_price = ctx.resolve_value(action.limit_price_ref, bar)
        if limit_price is None or limit_price <= 0:
            ctx.log(f"   Invalid limit price: {limit_price}, skipping {order_type} order")
            return

    if action.stop_price_ref is not None:
        stop_price = ctx.resolve_value(action.stop_price_ref, bar)
        if stop_price is None or stop_price <= 0:
            ctx.log(f"   Invalid stop price: {stop_price}, skipping {order_type} order")
            return

    # Extract OHLC from bar (PythonData stores in storage dictionary)
    _, bar_high, bar_low, _ = _bar_ohlc(bar)

    # Check if fill condition is met on this bar
    if _check_fill_condition(order_type, quantity, limit_price, stop_price, bar_high, bar_low):
        # Fill condition met — execute as MarketOrder
        ctx.market_order(ctx.symbol, quantity)
        price_info = []
        if limit_price is not None:
            price_info.append(f"limit=${limit_price:.2f}")
        if stop_price is not None:
            price_info.append(f"stop=${stop_price:.2f}")
        ctx.log(
            f"   {order_type.replace('_', '-').title()} filled: "
            f"{quantity:.6f} units ({', '.join(price_info)}) | "
            f"bar range [${bar_low:.2f}, ${bar_high:.2f}]"
        )
    else:
        # Fill condition NOT met — no order placed, will re-evaluate next bar
        price_info = []
        if limit_price is not None:
            price_info.append(f"limit=${limit_price:.2f}")
        if stop_price is not None:
            price_info.append(f"stop=${stop_price:.2f}")
        ctx.log(
            f"   {order_type.replace('_', '-').title()} not filled: "
            f"{', '.join(price_info)} | bar range [${bar_low:.2f}, ${bar_high:.2f}]"
        )


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

    For non-market order types (limit, stop, stop_limit), quantity is always
    computed explicitly and dispatched to the appropriate LEAN order API.

    Args:
        action: Typed action from IR (SetHoldingsAction, LiquidateAction, MarketOrderAction)
        ctx: ExecutionContext bundle for LEAN primitives
        bar: Current bar data (used for fixed_usd/fixed_units pricing)
    """
    if not action:
        return

    if isinstance(action, SetHoldingsAction):
        # Non-market orders: compute quantity and dispatch to typed order API
        if action.order_type != "market":
            _execute_typed_order(action, ctx, bar)
            return

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

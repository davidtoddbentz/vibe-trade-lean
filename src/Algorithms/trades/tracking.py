"""Trade and lot tracking logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any


def create_lot(
    lot_id: int,
    symbol: str,
    direction: str,
    entry_time: Any,
    entry_price: float,
    entry_bar: int,
    quantity: float,
    entry_fee: float,
) -> dict[str, Any]:
    """Create a lot dict for tracking.

    Args:
        lot_id: Unique lot identifier
        symbol: Trading symbol
        direction: "long" or "short"
        entry_time: Entry timestamp
        entry_price: Entry price (with slippage)
        entry_bar: Bar index where entry occurred
        quantity: Quantity for this lot
        entry_fee: Fee paid on entry

    Returns:
        Lot dict
    """
    return {
        "lot_id": lot_id,
        "symbol": symbol,
        "direction": direction,
        "entry_time": str(entry_time),
        "entry_price": entry_price,
        "entry_bar": entry_bar,
        "quantity": quantity,
        "entry_fee": entry_fee,
    }


def close_lots(
    lots: list[dict[str, Any]],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    exit_reason: str,
    apply_slippage_func: Any,
    calculate_fee_func: Any,
) -> list[dict[str, Any]]:
    """Close all lots and calculate PnL for each.

    Args:
        lots: List of open lot dicts
        exit_price: Base exit price (before slippage)
        exit_time: Exit timestamp
        exit_bar: Bar index where exit occurred
        exit_reason: Reason for exit
        apply_slippage_func: Function to apply slippage (price, is_buy) -> price
        calculate_fee_func: Function to calculate fee (trade_value) -> fee

    Returns:
        List of closed lot dicts (with exit info and PnL)
    """
    closed_lots = []
    for lot in lots:
        entry_price = lot["entry_price"]
        quantity = lot["quantity"]  # Always set by create_lot
        direction = lot["direction"]  # Always set by create_lot
        entry_fee = lot["entry_fee"]  # Always set by create_lot

        # Apply slippage to exit price
        # For long exit: selling, so receive less (slippage down)
        # For short exit: buying/covering, so pay more (slippage up)
        is_buy_exit = (direction == "short")  # Short covers with a buy
        final_exit_price = apply_slippage_func(exit_price, is_buy_exit)

        # Calculate exit fee
        exit_value = final_exit_price * quantity
        exit_fee = calculate_fee_func(exit_value)
        total_fees = entry_fee + exit_fee

        # PnL calculation: short profits when price drops
        # Subtract total fees from PnL
        if direction == "short":
            gross_pnl = (entry_price - final_exit_price) * quantity
        else:
            gross_pnl = (final_exit_price - entry_price) * quantity

        pnl = gross_pnl - total_fees
        # Calculate pnl_pct based on entry value
        entry_value = entry_price * quantity
        pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

        lot["exit_time"] = str(exit_time)
        lot["exit_price"] = final_exit_price
        lot["exit_bar"] = exit_bar
        lot["pnl"] = pnl
        lot["pnl_percent"] = pnl_pct
        lot["exit_fee"] = exit_fee
        lot["total_fees"] = total_fees
        lot["exit_reason"] = exit_reason

        closed_lots.append(lot)

    return closed_lots


def calculate_trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate trade statistics.

    Args:
        trades: List of completed trade/lot dicts (all should have pnl and pnl_percent from close_lots)

    Returns:
        Dict with win_rate, avg_win, avg_loss, profit_factor, etc.
    """
    # All trades should have pnl set by close_lots, but use .get() for safety
    winning_trades = [t for t in trades if t.get("pnl", 0.0) > 0]
    losing_trades = [t for t in trades if t.get("pnl", 0.0) <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    avg_win = sum(t.get("pnl_percent", 0.0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get("pnl_percent", 0.0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        "total_trades": len(trades),
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "profit_factor": profit_factor,
    }


def generate_report(
    trades: list[dict[str, Any]],
    equity_curve: list[dict[str, Any]],
    initial_cash: float,
    final_equity: float,
    max_drawdown: float,
    strategy_id: str,
    strategy_name: str,
    symbol: str,
    log_func: Any,
) -> dict[str, Any]:
    """Generate backtest report and write to file.

    Args:
        trades: List of completed trades
        equity_curve: Equity curve data points
        initial_cash: Starting capital
        final_equity: Ending equity
        max_drawdown: Maximum drawdown percentage
        strategy_id: Strategy identifier
        strategy_name: Strategy name
        symbol: Trading symbol
        log_func: Logging function

    Returns:
        Report dict
    """
    stats = calculate_trade_stats(trades)

    # Calculate accumulation summary if any lots were used
    lots_with_id = [t for t in trades if t.get("lot_id") is not None]
    accumulation_summary = None
    if lots_with_id:
        total_quantity = sum(t.get("quantity", 0) for t in lots_with_id)
        total_cost = sum(t.get("entry_price", 0) * t.get("quantity", 0) for t in lots_with_id)
        avg_entry_price = total_cost / total_quantity if total_quantity > 0 else 0
        accumulation_summary = {
            "total_lots": len(lots_with_id),
            "total_quantity": total_quantity,
            "total_cost_basis": total_cost,
            "avg_entry_price": avg_entry_price,
        }

    total_return = ((final_equity / initial_cash) - 1) * 100

    output = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "symbol": symbol,
        "initial_cash": initial_cash,
        "final_equity": float(final_equity),
        "total_return_pct": total_return,
        "max_drawdown_pct": max_drawdown,
        "statistics": stats,
        "trades": trades,
        "equity_curve": equity_curve,
    }

    # Add accumulation summary if present
    if accumulation_summary:
        output["accumulation_summary"] = accumulation_summary

    return output


def close_lots_at_end(
    lots: list[dict[str, Any]],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    apply_slippage_func: Any,
    calculate_fee_func: Any,
    log_func: Any,
) -> list[dict[str, Any]]:
    """Close all lots at end of backtest.

    Args:
        lots: List of open lot dicts
        exit_price: Final price
        exit_time: Exit timestamp
        exit_bar: Bar index (typically bar_count - 1)
        apply_slippage_func: Function to apply slippage
        calculate_fee_func: Function to calculate fee
        log_func: Logging function

    Returns:
        List of closed lot dicts
    """
    return close_lots(
        lots=lots,
        exit_price=exit_price,
        exit_time=exit_time,
        exit_bar=exit_bar,
        exit_reason="end_of_backtest",
        apply_slippage_func=apply_slippage_func,
        calculate_fee_func=calculate_fee_func,
    )

"""Trade and lot tracking logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Literal

from execution.types import ClosedLot, EquityPoint, Lot, TradeStats


def create_lot(
    lot_id: int,
    symbol: str,
    direction: Literal["long", "short"],
    entry_time: Any,
    entry_price: float,
    entry_bar: int,
    quantity: float,
    entry_fee: float,
) -> Lot:
    """Create a lot for tracking.

    Args:
        lot_id: Unique lot identifier
        symbol: Trading symbol
        direction: "long" or "short"
        entry_time: Entry timestamp
        entry_price: Entry price at bar close
        entry_bar: Bar index where entry occurred
        quantity: Quantity for this lot
        entry_fee: Fee paid on entry

    Returns:
        Lot instance
    """
    return Lot(
        lot_id=lot_id,
        symbol=symbol,
        direction=direction,
        entry_time=str(entry_time),
        entry_price=entry_price,
        entry_bar=entry_bar,
        quantity=quantity,
        entry_fee=entry_fee,
    )


def split_lot(lot: Lot, keep_frac: float) -> tuple[Lot, Lot]:
    """Split a lot into kept and closed portions.

    Args:
        lot: Original lot
        keep_frac: Fraction to keep (0.0-1.0)

    Returns:
        (kept_lot, closed_lot) with quantities proportioned
    """
    close_frac = 1.0 - keep_frac
    kept = Lot(
        lot_id=lot.lot_id,
        symbol=lot.symbol,
        direction=lot.direction,
        entry_time=lot.entry_time,
        entry_price=lot.entry_price,
        entry_bar=lot.entry_bar,
        quantity=lot.quantity * keep_frac,
        entry_fee=lot.entry_fee * keep_frac,
    )
    closed = Lot(
        lot_id=lot.lot_id,
        symbol=lot.symbol,
        direction=lot.direction,
        entry_time=lot.entry_time,
        entry_price=lot.entry_price,
        entry_bar=lot.entry_bar,
        quantity=lot.quantity * close_frac,
        entry_fee=lot.entry_fee * close_frac,
    )
    return kept, closed


def close_lots(
    lots: list[Lot],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    exit_reason: str,
) -> list[ClosedLot]:
    """Close all lots and calculate PnL for each.

    Args:
        lots: List of open lots
        exit_price: Exit price at bar close
        exit_time: Exit timestamp
        exit_bar: Bar index where exit occurred
        exit_reason: Reason for exit

    Returns:
        List of closed lots (with exit info and PnL)
    """
    closed_lots: list[ClosedLot] = []
    for lot in lots:
        entry_price = lot.entry_price
        quantity = lot.quantity  # Always set by create_lot
        direction = lot.direction  # Always set by create_lot
        entry_fee = lot.entry_fee  # Always set by create_lot

        final_exit_price = exit_price

        # Use exit fee share from LEAN fill if available, else 0
        exit_fee = lot._exit_fee_share
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

        closed_lots.append(
            ClosedLot(
                lot_id=lot.lot_id,
                symbol=lot.symbol,
                direction=lot.direction,
                entry_time=lot.entry_time,
                entry_price=lot.entry_price,
                entry_bar=lot.entry_bar,
                quantity=lot.quantity,
                entry_fee=lot.entry_fee,
                exit_time=str(exit_time),
                exit_price=final_exit_price,
                exit_bar=exit_bar,
                pnl=pnl,
                pnl_percent=pnl_pct,
                exit_fee=exit_fee,
                total_fees=total_fees,
                exit_reason=exit_reason,
            )
        )

    return closed_lots


def calculate_trade_stats(trades: list[ClosedLot]) -> TradeStats:
    """Calculate trade statistics.

    Args:
        trades: List of completed trade lots (all should have pnl and pnl_percent from close_lots)

    Returns:
        TradeStats with win_rate, avg_win, avg_loss, profit_factor, etc.
    """
    winning_trades = [t for t in trades if t.pnl > 0]
    losing_trades = [t for t in trades if t.pnl <= 0]
    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    avg_win = sum(t.pnl_percent for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.pnl_percent for t in losing_trades) / len(losing_trades) if losing_trades else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return TradeStats(
        total_trades=len(trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
    )


def generate_report(
    trades: list[ClosedLot],
    equity_curve: list[EquityPoint],
    initial_cash: float,
    final_equity: float,
    max_drawdown: float,
    strategy_id: str,
    strategy_name: str,
    symbol: str,
    log_func: Callable[[str], None],
    ohlcv_bars: list[dict[str, Any]] | None = None,
    indicator_values: dict[str, list[dict[str, Any]]] | None = None,
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
    lots_with_id = [t for t in trades if t.lot_id is not None]
    accumulation_summary = None
    if lots_with_id:
        total_quantity = sum(t.quantity for t in lots_with_id)
        total_cost = sum(t.entry_price * t.quantity for t in lots_with_id)
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
        "statistics": asdict(stats),
        "trades": [asdict(t) for t in trades],
        "equity_curve": [asdict(e) for e in equity_curve],
    }

    # Add accumulation summary if present
    if accumulation_summary:
        output["accumulation_summary"] = accumulation_summary

    if ohlcv_bars:
        output["ohlcv_bars"] = ohlcv_bars
    if indicator_values:
        output["indicators"] = indicator_values

    return output


def close_lots_at_end(
    lots: list[Lot],
    exit_price: float,
    exit_time: Any,
    exit_bar: int,
    log_func: Callable[[str], None],
) -> list[ClosedLot]:
    """Close all lots at end of backtest.

    Args:
        lots: List of open lots
        exit_price: Final price
        exit_time: Exit timestamp
        exit_bar: Bar index (typically bar_count - 1)
        log_func: Logging function

    Returns:
        List of closed lots
    """
    return close_lots(
        lots=lots,
        exit_price=exit_price,
        exit_time=exit_time,
        exit_bar=exit_bar,
        exit_reason="end_of_backtest",
    )

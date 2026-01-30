"""Entry/exit orchestration logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any, Callable

from vibe_trade_shared.models.ir import (
    EntryRule,
    ExitRule,
    OverlayRule,
    SetHoldingsAction,
    ReducePositionAction,
    MarketOrderAction,
    Condition,
    StateOp,
)
from execution.types import Lot, TrackingState
from execution.context import ExecutionContext
from trades import create_lot, close_lots, split_lot
from position import apply_scale_in, apply_overlay_scale, compute_overlay_scale


def execute_entry(
    entry_rule: EntryRule | None,
    evaluate_condition: Callable[[Condition, Any], bool],
    bar: Any,
    tracking: TrackingState,
    ctx: ExecutionContext,
    current_time: Any,
    execute_action_func: Callable,
    execute_state_op: Callable[[StateOp, Any], None],
    overlays: list[OverlayRule],
) -> int | None:
    """Execute entry if condition is met.

    Args:
        entry_rule: EntryRule from IR
        evaluate_condition: Function to evaluate conditions
        bar: Current bar data
        tracking: Tracking state container (will be modified)
        ctx: ExecutionContext bundle for LEAN primitives
        current_time: Current timestamp
        execute_action_func: Function to execute action
        execute_state_op: Function to execute state op
        overlays: List of OverlayRule from IR

    Returns:
        Updated last_entry_bar or None if no entry fired
    """
    if not entry_rule:
        return None

    result = evaluate_condition(entry_rule.condition, bar)
    if not result:
        return None

    action = entry_rule.action  # Typed EntryAction

    # Get position before executing to calculate lot quantity
    qty_before = float(ctx.portfolio[ctx.symbol].Quantity)

    # Apply scale_in factor if in scale_in mode with existing lots
    if isinstance(action, SetHoldingsAction):
        action = apply_scale_in(action, tracking.current_lots, ctx.log)

    # Apply overlay scaling to position size
    overlay_scale = compute_overlay_scale(
        overlays=overlays,
        evaluate_condition_func=evaluate_condition,
        bar=bar,
        log_func=ctx.log,
    )
    if isinstance(action, SetHoldingsAction):
        action = apply_overlay_scale(action, overlay_scale, ctx.log)

    # Execute action (modifies portfolio, triggers OnOrderEvent synchronously)
    execute_action_func(action, bar)

    # Calculate lot quantity (delta from before)
    qty_after = float(ctx.portfolio[ctx.symbol].Quantity)
    lot_quantity = abs(qty_after - qty_before)

    # For typed orders (limit/stop/stop_limit), Python-side fill simulation
    # either executes a MarketOrder (condition met) or skips (condition not met).
    # In both cases, lot_quantity == 0 means no fill occurred on this bar.
    if lot_quantity == 0:
        return None

    # Detect direction from allocation or quantity
    if isinstance(action, SetHoldingsAction):
        allocation = action.allocation
    elif isinstance(action, MarketOrderAction):
        allocation = 1.0 if action.quantity > 0 else -1.0
    else:
        allocation = 0.95  # Default fallback
    direction = "short" if allocation < 0 or qty_after < 0 else "long"

    # Use LEAN's actual fill price/fee if available, else fall back to bar close
    fill = ctx.get_last_fill()
    entry_price = fill.price if fill else float(bar.Close)
    entry_fee = fill.fee if fill else 0.0

    lot = create_lot(
        lot_id=len(tracking.current_lots),
        symbol=str(ctx.symbol),
        direction=direction,
        entry_time=current_time,
        entry_price=entry_price,
        entry_bar=tracking.bar_count,
        quantity=lot_quantity,
        entry_fee=entry_fee,
    )
    tracking.current_lots.append(lot)
    last_entry_bar = tracking.bar_count
    tracking.entries_today += 1  # Reset daily in StrategyRuntime.OnData

    # Run on_fill hooks (EntryRule is typed; on_fill is list[StateOp])
    on_fill = entry_rule.on_fill or []
    for op in on_fill:
        execute_state_op(op, bar)

    lot_num = len(tracking.current_lots)
    if lot_num > 1:
        ctx.log(f"🟢 ENTRY (lot #{lot_num}) @ ${entry_price:.2f} | qty: {lot_quantity:.6f}")
    else:
        ctx.log(f"🟢 ENTRY @ ${entry_price:.2f}")

    return last_entry_bar


def execute_exit(
    exit_rules: list[ExitRule],
    evaluate_condition: Callable[[Condition, Any], bool],
    bar: Any,
    tracking: TrackingState,
    ctx: ExecutionContext,
    current_time: Any,
    execute_action_func: Callable,
) -> None:
    """Execute exit if condition is met.

    Args:
        exit_rules: List of ExitRule from IR (sorted by priority)
        evaluate_condition: Function to evaluate conditions
        bar: Current bar data
        tracking: Tracking state container (will be modified)
        ctx: ExecutionContext bundle for LEAN primitives
        current_time: Current timestamp
        execute_action_func: Function to execute action

    Returns:
        None
    """
    # Sort by priority (lower priority number = higher priority)
    sorted_exits = sorted(exit_rules, key=lambda x: x.priority or 0)

    for exit_rule in sorted_exits:
        if evaluate_condition(exit_rule.condition, bar):
            num_lots = len(tracking.current_lots) if tracking.current_lots else 0

            # Execute exit action first to get LEAN's actual fill price
            execute_action_func(exit_rule.action)

            # Use LEAN's fill price if available, else bar close
            fill = ctx.get_last_fill()
            exit_price = fill.price if fill else float(bar.Close)
            exit_fee = fill.fee if fill else 0.0

            if tracking.current_lots:
                exit_reason = exit_rule.id or "unknown"

                # Branch: partial exit vs full exit
                is_partial = (
                    isinstance(exit_rule.action, ReducePositionAction)
                    and exit_rule.action.size_frac < 1.0
                )

                if is_partial:
                    # PARTIAL EXIT: split lots, close fraction, keep remainder
                    size_frac = exit_rule.action.size_frac
                    keep_frac = 1.0 - size_frac

                    lots_to_close = []
                    lots_to_keep = []
                    total_qty = sum(lot.quantity for lot in tracking.current_lots)

                    for lot in tracking.current_lots:
                        kept, to_close = split_lot(lot, keep_frac)
                        lots_to_keep.append(kept)
                        to_close._exit_fee_share = (
                            exit_fee * (to_close.quantity / total_qty) if total_qty > 0 else 0.0
                        )
                        lots_to_close.append(to_close)

                    closed_lots = close_lots(
                        lots=lots_to_close,
                        exit_price=exit_price,
                        exit_time=current_time,
                        exit_bar=tracking.bar_count,
                        exit_reason=exit_reason,
                    )
                    tracking.trades.extend(closed_lots)
                    tracking.current_lots = lots_to_keep
                else:
                    # FULL EXIT: close all lots, clear position
                    total_qty = sum(lot.quantity for lot in tracking.current_lots)
                    for lot in tracking.current_lots:
                        lot_qty = lot.quantity
                        lot._exit_fee_share = (
                            exit_fee * (lot_qty / total_qty) if total_qty > 0 else 0.0
                        )

                    closed_lots = close_lots(
                        lots=tracking.current_lots,
                        exit_price=exit_price,
                        exit_time=current_time,
                        exit_bar=tracking.bar_count,
                        exit_reason=exit_reason,
                    )
                    tracking.trades.extend(closed_lots)
                    tracking.current_lots = []

            exit_id = exit_rule.id or "unknown"
            if num_lots > 1:
                ctx.log(f"🔴 EXIT ({exit_id}) @ ${exit_price:.2f} | closed {num_lots} lots")
            else:
                ctx.log(f"🔴 EXIT ({exit_id}) @ ${exit_price:.2f}")
            break  # Only execute first matching exit

    return None

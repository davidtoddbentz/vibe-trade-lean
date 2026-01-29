"""Entry/exit orchestration logic.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

from typing import Any

from vibe_trade_shared.models.ir import SetHoldingsAction, MarketOrderAction
from trades import create_lot
from position import apply_scale_in, apply_overlay_scale, compute_overlay_scale


def execute_entry(
    entry_rule: Any,
    evaluate_condition_func: Any,
    bar: Any,
    current_lots: list[Any],
    bar_count: int,
    symbol: Any,
    portfolio: Any,
    current_time: Any,
    execute_action_func: Any,
    execute_state_op_func: Any,
    overlays: list[Any],
    log_func: Any,
    get_last_fill: Any = None,
) -> tuple[list[Any], int | None]:
    """Execute entry if condition is met.

    Args:
        entry_rule: EntryRule from IR
        evaluate_condition_func: Function to evaluate conditions
        bar: Current bar data
        current_lots: List of open lots (will be modified)
        bar_count: Current bar index
        symbol: Trading symbol
        portfolio: LEAN Portfolio object
        current_time: Current timestamp
        execute_action_func: Function to execute action
        execute_state_op_func: Function to execute state op
        overlays: List of OverlayRule from IR
        log_func: Logging function
        get_last_fill: Optional callable returning last fill dict {price, fee}

    Returns:
        Tuple of (updated_current_lots, updated_last_entry_bar)
    """
    if not entry_rule:
        return current_lots, None

    result = evaluate_condition_func(entry_rule.condition, bar)
    if not result:
        return current_lots, None

    action = entry_rule.action  # Typed EntryAction

    # Get position before executing to calculate lot quantity
    qty_before = float(portfolio[symbol].Quantity)

    # Apply scale_in factor if in scale_in mode with existing lots
    if isinstance(action, SetHoldingsAction):
        action = apply_scale_in(action, current_lots, log_func)

    # Apply overlay scaling to position size
    overlay_scale = compute_overlay_scale(
        overlays=overlays,
        evaluate_condition_func=evaluate_condition_func,
        bar=bar,
        log_func=log_func,
    )
    if isinstance(action, SetHoldingsAction):
        action = apply_overlay_scale(action, overlay_scale, log_func)

    # Execute action (modifies portfolio, triggers OnOrderEvent synchronously)
    execute_action_func(action, bar)

    # Calculate lot quantity (delta from before)
    qty_after = float(portfolio[symbol].Quantity)
    lot_quantity = abs(qty_after - qty_before)

    # Detect direction from allocation or quantity
    if isinstance(action, SetHoldingsAction):
        allocation = action.allocation
    elif isinstance(action, MarketOrderAction):
        allocation = 1.0 if action.quantity > 0 else -1.0
    else:
        allocation = 0.95  # Default fallback
    direction = "short" if allocation < 0 or qty_after < 0 else "long"

    # Use LEAN's actual fill price/fee if available, else fall back to bar close
    fill = get_last_fill() if get_last_fill else None
    entry_price = fill["price"] if fill else float(bar.Close)
    entry_fee = fill["fee"] if fill else 0.0

    lot = create_lot(
        lot_id=len(current_lots),
        symbol=str(symbol),
        direction=direction,
        entry_time=current_time,
        entry_price=entry_price,
        entry_bar=bar_count,
        quantity=lot_quantity,
        entry_fee=entry_fee,
    )
    current_lots.append(lot)
    last_entry_bar = bar_count

    # Run on_fill hooks (EntryRule is typed; on_fill is list[StateOp])
    on_fill = getattr(entry_rule, "on_fill", None) or []
    for op in on_fill:
        execute_state_op_func(op, bar)

    lot_num = len(current_lots)
    if lot_num > 1:
        log_func(f"🟢 ENTRY (lot #{lot_num}) @ ${entry_price:.2f} | qty: {lot_quantity:.6f}")
    else:
        log_func(f"🟢 ENTRY @ ${entry_price:.2f}")

    return current_lots, last_entry_bar


def execute_exit(
    exit_rules: list[Any],
    evaluate_condition_func: Any,
    bar: Any,
    current_lots: list[Any],
    bar_count: int,
    symbol: Any,
    current_time: Any,
    close_lots_func: Any,
    execute_action_func: Any,
    log_func: Any,
    get_last_fill: Any = None,
) -> tuple[list[Any], list[Any]]:
    """Execute exit if condition is met.

    Args:
        exit_rules: List of ExitRule from IR (sorted by priority)
        evaluate_condition_func: Function to evaluate conditions
        bar: Current bar data
        current_lots: List of open lots (will be cleared if exit triggers)
        bar_count: Current bar index
        symbol: Trading symbol
        current_time: Current timestamp
        close_lots_func: Function to close lots
        execute_action_func: Function to execute action
        log_func: Logging function
        get_last_fill: Optional callable returning last fill dict {price, fee}

    Returns:
        Tuple of (updated_current_lots, closed_lots_list)
    """
    closed_lots_list = []

    # Sort by priority (lower priority number = higher priority)
    sorted_exits = sorted(exit_rules, key=lambda x: getattr(x, "priority", 0) or 0)

    for exit_rule in sorted_exits:
        if evaluate_condition_func(exit_rule.condition, bar):
            num_lots = len(current_lots) if current_lots else 0

            # Execute exit action first to get LEAN's actual fill price
            execute_action_func(exit_rule.action)

            # Use LEAN's fill price if available, else bar close
            fill = get_last_fill() if get_last_fill else None
            exit_price = fill["price"] if fill else float(bar.Close)
            exit_fee = fill["fee"] if fill else 0.0

            if current_lots:
                exit_reason = getattr(exit_rule, "id", "unknown") or "unknown"

                # Distribute exit fee across lots proportionally
                total_qty = sum(lot.get("quantity", 0) for lot in current_lots)
                for lot in current_lots:
                    lot_qty = lot.get("quantity", 0)
                    lot["_exit_fee_share"] = exit_fee * (lot_qty / total_qty) if total_qty > 0 else 0.0

                closed_lots = close_lots_func(
                    lots=current_lots,
                    exit_price=exit_price,
                    exit_time=current_time,
                    exit_bar=bar_count,
                    exit_reason=exit_reason,
                )
                closed_lots_list.extend(closed_lots)
                current_lots = []

            exit_id = getattr(exit_rule, "id", "unknown") or "unknown"
            if num_lots > 1:
                log_func(f"🔴 EXIT ({exit_id}) @ ${exit_price:.2f} | closed {num_lots} lots")
            else:
                log_func(f"🔴 EXIT ({exit_id}) @ ${exit_price:.2f}")
            break  # Only execute first matching exit

    return current_lots, closed_lots_list

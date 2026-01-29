"""Initialization setup helpers.

Phase 12: Extracted from StrategyRuntime.

VERSION_MARKER: 2026-01-26-phase12-refactor-v1
This marker confirms the refactored code is being used.
"""

from __future__ import annotations

from typing import Any, Callable
from AlgorithmImports import Resolution
from vibe_trade_shared.models.ir import StrategyIR


def setup_data_folder(
    data_folder_param: str | None,
    custom_data_class: Any,
    log_func: Callable[[str], None],
    debug_func: Callable[[str], None],
) -> None:
    """Set up data folder and debug logging.

    Args:
        data_folder_param: Data folder parameter from GetParameter
        custom_data_class: CustomCryptoData class
        log_func: Logging function
        debug_func: Debug logging function
    """
    import os

    if data_folder_param:
        custom_data_class.DataFolder = data_folder_param
        log_func(f"[INIT] Set data folder to: {data_folder_param}")
    else:
        log_func(f"[INIT] No data_folder parameter, using default: {custom_data_class.DataFolder}")

    # Set up debug log in data folder
    custom_data_class.DebugLogPath = os.path.join(custom_data_class.DataFolder, "debug.log")
    debug_func(f"[INIT] Debug log path: {custom_data_class.DebugLogPath}")
    custom_data_class._log_debug(f"[INIT] Debug logging initialized at {custom_data_class.DataFolder}")

    # Debug: List files in data folder
    try:
        files = os.listdir(custom_data_class.DataFolder)
        debug_func(f"[INIT] Files in data folder: {files}")
        # Check for btc_usd_data.csv specifically
        expected_file = os.path.join(custom_data_class.DataFolder, "btc_usd_data.csv")
        if os.path.exists(expected_file):
            debug_func("[INIT] ✅ Found btc_usd_data.csv")
            # Read first few lines to verify format
            with open(expected_file, 'r') as f:
                lines = f.readlines()[:3]
                for i, line in enumerate(lines):
                    debug_func(f"[INIT] Line {i}: {line.strip()}")
        else:
            debug_func(f"[INIT] ❌ btc_usd_data.csv NOT FOUND at {expected_file}")
    except Exception as e:
        debug_func(f"[INIT] Error listing data folder: {e}")


def setup_dates(
    start_date_str: str | None,
    end_date_str: str | None,
    initial_cash_str: str | None,
    trading_start_str: str | None,
    set_start_date_func: Callable,
    set_end_date_func: Callable,
    set_cash_func: Callable,
    debug_func: Callable[[str], None],
) -> tuple[Any, float]:
    """Set up algorithm dates and initial cash.

    Args:
        start_date_str: Start date parameter (YYYYMMDD)
        end_date_str: End date parameter (YYYYMMDD)
        initial_cash_str: Initial cash parameter
        trading_start_str: Trading start date parameter (YYYYMMDD)
        set_start_date_func: Function to set start date
        set_end_date_func: Function to set end date
        set_cash_func: Function to set cash
        debug_func: Debug logging function

    Returns:
        Tuple of (trading_start_date, initial_cash)
    """
    debug_func(f"start_date parameter: {start_date_str}")
    debug_func(f"end_date parameter: {end_date_str}")
    debug_func(f"initial_cash parameter: {initial_cash_str}")

    # Parse dates or use defaults
    if start_date_str:
        year = int(start_date_str[:4])
        month = int(start_date_str[4:6])
        day = int(start_date_str[6:8])
        set_start_date_func(year, month, day)
    else:
        set_start_date_func(2024, 1, 1)

    if end_date_str:
        year = int(end_date_str[:4])
        month = int(end_date_str[4:6])
        day = int(end_date_str[6:8])
        set_end_date_func(year, month, day)
    else:
        set_end_date_func(2024, 12, 31)

    if initial_cash_str:
        initial_cash = float(initial_cash_str)
        set_cash_func(initial_cash)
    else:
        initial_cash = 100000.0
        set_cash_func(initial_cash)

    # Parse trading_start_date (prevents trades during warmup)
    trading_start_date = None
    if trading_start_str:
        try:
            from datetime import datetime as dt
            year = int(trading_start_str[:4])
            month = int(trading_start_str[4:6])
            day = int(trading_start_str[6:8])
            trading_start_date = dt(year, month, day)
            debug_func(f"trading_start_date: {trading_start_date} (trades blocked before this)")
        except (ValueError, IndexError) as e:
            debug_func(f"Failed to parse trading_start_date '{trading_start_str}': {e}")
    else:
        debug_func("No trading_start_date set (warmup trading allowed)")

    return trading_start_date, initial_cash


def setup_symbols(
    ir: StrategyIR,
    add_symbol_func: Callable,
    normalize_symbol_func: Callable[[str], str],
    log_func: Callable[[str], None],
) -> tuple[Any, dict[str, Any]]:
    """Set up primary and additional symbols.

    Args:
        ir: StrategyIR from IR
        add_symbol_func: Function to add symbol
        normalize_symbol_func: Function to normalize symbol string
        log_func: Logging function

    Returns:
        Tuple of (primary_symbol, symbols_dict)
    """
    symbol_str = ir.symbol or "BTC-USD"
    primary_symbol = add_symbol_func(symbol_str)

    symbols_dict = {normalize_symbol_func(symbol_str): primary_symbol}
    for additional_sym in ir.additional_symbols or []:
        sym_obj = add_symbol_func(additional_sym)
        symbols_dict[normalize_symbol_func(additional_sym)] = sym_obj
        log_func(f"   Added additional symbol: {additional_sym}")

    return primary_symbol, symbols_dict


def setup_rules(ir: StrategyIR) -> dict[str, Any]:
    """Extract rules from IR.

    Args:
        ir: StrategyIR from IR

    Returns:
        Dict with entry_rule, exit_rules, gates, overlays, on_bar_ops, on_bar_invested_ops
    """
    return {
        "entry_rule": ir.entry,
        "exit_rules": ir.exits or [],
        "gates": ir.gates or [],
        "overlays": ir.overlays or [],
        "on_bar_ops": ir.on_bar or [],
        "on_bar_invested_ops": ir.on_bar_invested or [],
    }


def setup_trading_costs(ir: StrategyIR, log_func: Callable[[str], None]) -> dict[str, float]:
    """Configure trading costs from IR.

    Args:
        ir: StrategyIR from IR
        log_func: Logging function

    Returns:
        Dict with fee_pct and slippage_pct
    """
    fee_pct = ir.fee_pct or 0.0
    slippage_pct = ir.slippage_pct or 0.0

    if fee_pct > 0 or slippage_pct > 0:
        log_func(f"   Trading costs: fee={fee_pct}%, slippage={slippage_pct}%")
    else:
        log_func("   No trading costs configured")

    return {"fee_pct": fee_pct, "slippage_pct": slippage_pct}

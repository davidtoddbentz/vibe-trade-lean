#!/usr/bin/env python3
"""
Verify LEAN test results meet expectations.
Checks that strategy executed correctly with Pub/Sub data.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def verify_test_results(results_dir: Path, min_data_points: int = 10, min_orders: int = 1, 
                       expected_orders: int = None, check_orders_filled: bool = True):
    """
    Verify test results meet expectations.
    
    Args:
        results_dir: Directory containing LEAN result files
        min_data_points: Minimum number of data points expected
        min_orders: Minimum number of orders expected
        expected_orders: Exact number of orders expected (if None, just checks >= min_orders)
        check_orders_filled: Whether to verify orders were filled
    
    Returns:
        dict: Verification results with pass/fail status
    """
    results_dir = Path(results_dir)
    
    # Find the main result file
    # LEAN creates multiple files:
    # - AlgorithmName.json (main file - preferred)
    # - AlgorithmName-YYYY-MM-DD_HHMMSS.json (timestamped main file)
    # - AlgorithmName-YYYY-MM-DD_HHMMSS_Strategy Equity.json (chart data - skip)
    # - AlgorithmName-summary.json (summary - different structure)
    
    # First, try to find the simple AlgorithmName.json file
    simple_files = [f for f in results_dir.glob("*.json") 
                   if not any(skip in f.name for skip in [
                       "Strategy Equity", "Benchmark", "Drawdown", 
                       "Exposure", "Portfolio Turnover", "summary"
                   ])]
    
    if simple_files:
        # Prefer files without timestamps (AlgorithmName.json)
        non_timestamped = [f for f in simple_files if f.name.count("-") < 3]
        if non_timestamped:
            latest_file = max(non_timestamped, key=lambda p: p.stat().st_mtime)
        else:
            latest_file = max(simple_files, key=lambda p: p.stat().st_mtime)
    else:
        return {
            "passed": False,
            "error": "No result files found",
            "checks": {}
        }
    
    try:
        with open(latest_file) as f:
            results = json.load(f)
    except Exception as e:
        return {
            "passed": False,
            "error": f"Failed to parse result file: {e}",
            "checks": {}
        }
    
    # Extract key data
    state = results.get("state", {})
    runtime_stats = results.get("runtimeStatistics", {})
    holdings = results.get("holdings", {})
    orders = results.get("orders", {})
    charts = results.get("charts", {})
    
    # Extract order details
    order_count = int(state.get("OrderCount", 0))
    orders_list = list(orders.values()) if isinstance(orders, dict) else []
    
    # Check order types if we have order details
    buy_orders = 0
    sell_orders = 0
    filled_orders = 0
    
    for order in orders_list:
        if isinstance(order, dict):
            quantity = order.get("Quantity", 0)
            if quantity > 0:
                buy_orders += 1
            elif quantity < 0:
                sell_orders += 1
            
            status = order.get("Status", "")
            if status == "Filled":
                filled_orders += 1
    
    # Perform checks
    checks = {
        "result_file_exists": True,
        "result_file_parsable": True,
        "algorithm_ran": state.get("Status") in ["Running", "Completed"],
        "no_runtime_errors": state.get("RuntimeError", "") == "",
        "data_received": int(state.get("LogCount", 0)) >= min_data_points,
        "orders_placed": order_count >= min_orders,
        "expected_order_count": (order_count == expected_orders) if expected_orders is not None else True,
        "has_buy_order": buy_orders > 0,
        "has_sell_order": sell_orders > 0,
        "orders_filled": (filled_orders >= min_orders) if check_orders_filled else True,
        "equity_tracked": "Strategy Equity" in charts,
        "runtime_stats_present": len(runtime_stats) > 0,
        "has_holdings_data": "holdings" in results,
        "has_cash_data": "cash" in results,
    }
    
    # Additional detailed checks
    details = {
        "status": state.get("Status", "Unknown"),
        "log_count": int(state.get("LogCount", 0)),
        "order_count": order_count,
        "buy_orders": buy_orders,
        "sell_orders": sell_orders,
        "filled_orders": filled_orders,
        "equity": runtime_stats.get("Equity", "N/A"),
        "return": runtime_stats.get("Return", "N/A"),
        "net_profit": runtime_stats.get("Net Profit", "N/A"),
        "holdings_count": len(holdings),
        "orders_count": len(orders_list),
        "start_time": state.get("StartTime", "N/A"),
        "end_time": state.get("EndTime", "N/A"),
        "runtime_error": state.get("RuntimeError", ""),
    }
    
    # Check if equity curve has data points
    if "Strategy Equity" in charts:
        equity_series = charts["Strategy Equity"].get("series", {}).get("Equity", {})
        equity_values = equity_series.get("values", [])
        checks["equity_curve_has_data"] = len(equity_values) > 0
        details["equity_points"] = len(equity_values)
    else:
        checks["equity_curve_has_data"] = False
        details["equity_points"] = 0
    
    # Overall pass/fail
    passed = all(checks.values())
    
    return {
        "passed": passed,
        "checks": checks,
        "details": details,
        "result_file": str(latest_file),
    }

def print_verification_results(verification):
    """Print verification results in a readable format."""
    print("=" * 60)
    print("📊 Test Result Verification")
    print("=" * 60)
    print()
    
    if verification.get("error"):
        print(f"❌ ERROR: {verification['error']}")
        return
    
    checks = verification.get("checks", {})
    details = verification.get("details", {})
    
    print("✅ Checks:")
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        check_display = check_name.replace("_", " ").title()
        print(f"  {status} {check_display}")
    
    print()
    print("📈 Details:")
    for key, value in details.items():
        key_display = key.replace("_", " ").title()
        print(f"  {key_display}: {value}")
    
    print()
    print("=" * 60)
    
    if verification["passed"]:
        print("✅ ALL CHECKS PASSED")
    else:
        print("❌ SOME CHECKS FAILED")
    
    print("=" * 60)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: verify_results.py <results_dir> [min_data_points] [min_orders] [expected_orders]")
        sys.exit(1)
    
    results_dir = Path(sys.argv[1])
    min_data_points = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    min_orders = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    expected_orders = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    if not results_dir.exists():
        print(f"❌ Results directory does not exist: {results_dir}")
        sys.exit(1)
    
    verification = verify_test_results(results_dir, min_data_points, min_orders, expected_orders)
    print_verification_results(verification)
    
    sys.exit(0 if verification["passed"] else 1)

if __name__ == "__main__":
    main()


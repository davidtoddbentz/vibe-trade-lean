"""
Generic Strategy Runtime for LEAN.

This algorithm interprets StrategyIR JSON at runtime, allowing a single algorithm
to execute any strategy defined in the IR format.

The IR is passed via:
1. A JSON file in the algorithm's data directory
2. Or as a parameter when launching the algorithm
"""

from AlgorithmImports import *
import json
from indicators import (
    IndicatorCategory,
    IndicatorResult,
    create_indicator,
    resolve_indicator_value,
    resolve_value as _resolve_value_impl,
    update_indicator,
    is_indicator_ready,
    create_all_indicators,
    initialize_state_variables,
    check_indicators_ready,
    update_all_indicators,
)
from pydantic import TypeAdapter
from vibe_trade_shared.models.ir import (
    StrategyIR,
    ValueRef,
    SetHoldingsAction,
    LiquidateAction,
    MarketOrderAction,
    EntryAction,
    ExitAction,
    StateOp,
    PositionPolicy,
)

# ValueRef is a type alias (discriminated Union); use TypeAdapter to validate dicts.
ValueRefAdapter = TypeAdapter(ValueRef)
from conditions import evaluate_condition as registry_evaluate_condition
from trades import create_lot, close_lots, close_lots_at_end, calculate_trade_stats, generate_report
from position import (
    apply_scale_in,
    apply_overlay_scale,
    can_accumulate,
    compute_overlay_scale,
    track_equity,
)
from gates import evaluate_gates
from costs import apply_slippage as _apply_slippage_func, calculate_fee as _calculate_fee_func
from symbols import add_symbol, normalize_symbol, get_symbol_obj
from ir import load_ir_from_file
from execution import execute_action, execute_entry, execute_exit
from initialization import setup_data_folder, setup_dates, setup_symbols, setup_rules, setup_tracking, setup_trading_costs
from state import execute_state_op as _execute_state_op_func

# =============================================================================
# Pydantic Availability Check (experimental)
# =============================================================================
# Test if Pydantic is available in LEAN's Python environment.
# If available, we can reuse IR models from vibe-trade-shared.
# If not, we fall back to dict-based access (current behavior).
PYDANTIC_AVAILABLE = False
PYDANTIC_VERSION = None
try:
    from pydantic import BaseModel, __version__ as pydantic_version
    PYDANTIC_AVAILABLE = True
    PYDANTIC_VERSION = pydantic_version

    # Quick validation test - create a simple model
    class _TestModel(BaseModel):
        value: int
    _test = _TestModel(value=42)
    assert _test.value == 42, "Pydantic model validation failed"
except ImportError:
    pass  # Pydantic not installed, use dict-based access
except Exception:
    PYDANTIC_AVAILABLE = False  # Pydantic import failed


# =============================================================================
# Custom Fee Model for Percentage-Based Fees
# =============================================================================


class PercentageFeeModel(FeeModel):
    """Fee model that charges a percentage of trade value.

    This allows configuring fees as a percentage (e.g., 0.1% per trade)
    rather than a fixed dollar amount.
    """

    def __init__(self, fee_percentage: float):
        """
        Args:
            fee_percentage: Fee as a percentage (e.g., 0.1 for 0.1%)
        """
        super().__init__()
        self.fee_percentage = fee_percentage / 100.0  # Convert to decimal

    def GetOrderFee(self, parameters):
        """Calculate fee as percentage of trade value."""
        price = parameters.Security.Price
        quantity = abs(parameters.Order.Quantity)
        trade_value = price * quantity
        fee = trade_value * self.fee_percentage
        return OrderFee(CashAmount(fee, "USD"))


# =============================================================================
# Custom Data Reader for CSV files
# =============================================================================


class CustomCryptoData(PythonData):
    """Custom data reader for cryptocurrency CSV files."""

    # Class variable to hold the data folder path - set by algorithm
    DataFolder = "/Data"

    # Debug log file path
    DebugLogPath = None

    @staticmethod
    def _log_debug(msg):
        """Write debug message to log file."""
        if CustomCryptoData.DebugLogPath:
            try:
                with open(CustomCryptoData.DebugLogPath, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass

    def GetSource(self, config, date, isLiveMode):
        """Return the data source - CSV file in data directory."""
        import os
        # Convert symbol to filename (e.g., BTC-USD -> btc_usd_data.csv)
        # Symbol.ID.Symbol may include data type suffix (e.g., "BTC-USD.CustomCryptoData")
        # We need to strip the suffix to get the base symbol
        symbol = str(config.Symbol.ID.Symbol).lower()
        # Strip the data type suffix if present (e.g., ".customcryptodata")
        if "." in symbol:
            symbol = symbol.split(".")[0]
        # Normalize: replace dashes with underscores
        symbol_normalized = symbol.replace("-", "_")
        filename = f"{symbol_normalized}_data.csv"
        # Use full path - data folder is set by the algorithm
        full_path = f"{CustomCryptoData.DataFolder}/{filename}"
        # Log for debugging
        exists = os.path.exists(full_path)
        CustomCryptoData._log_debug(f"[GetSource] DataFolder: {CustomCryptoData.DataFolder}, Symbol: {symbol}, Date: {date}, File: {full_path}, Exists: {exists}")
        return SubscriptionDataSource(
            full_path,
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """Parse a line from the CSV file."""
        if not line or line.startswith("datetime"):
            return None

        data = CustomCryptoData()
        try:
            parts = line.split(",")
            data.Time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Symbol = config.Symbol
            data.Value = float(parts[4])  # close price
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except Exception as e:
            CustomCryptoData._log_debug(f"[Reader] Error parsing line: {line[:50]}... Error: {e}")
            return None
        return data


# =============================================================================
# Runtime Algorithm
# =============================================================================


class StrategyRuntime(QCAlgorithm):
    """Generic strategy runtime that interprets IR JSON."""

    def Initialize(self):
        """Initialize the algorithm."""
        # CRITICAL: Allow small orders to be placed
        # By default, LEAN silently skips orders below MinimumOrderMarginPortfolioPercentage
        # This causes small positions (like 1% of equity in BTC at $100k) to be ignored
        # See: https://www.quantconnect.com/forum/discussion/2978/minimum-order-clip-size/
        self.Settings.MinimumOrderMarginPortfolioPercentage = 0

        # Set data folder FIRST - before any subscriptions
        setup_data_folder(
            data_folder_param=self.GetParameter("data_folder"),
            custom_data_class=CustomCryptoData,
            log_func=self.Log,
            debug_func=self.Debug,
        )

        # Get date parameters and set up dates
        self.trading_start_date, initial_cash = setup_dates(
            start_date_str=self.GetParameter("start_date"),
            end_date_str=self.GetParameter("end_date"),
            initial_cash_str=self.GetParameter("initial_cash"),
            trading_start_str=self.GetParameter("trading_start_date"),
            set_start_date_func=self.SetStartDate,
            set_end_date_func=self.SetEndDate,
            set_cash_func=self.SetCash,
            debug_func=self.Debug,
        )

        # Load strategy IR and validate once into Pydantic (typed end-to-end for conditions)
        ir_json = self.GetParameter("strategy_ir")
        self.Debug(f"strategy_ir parameter: {ir_json}")
        if ir_json:
            ir_dict = json.loads(ir_json)
        else:
            ir_path = self.GetParameter("strategy_ir_path")
            self.Debug(f"strategy_ir_path parameter: {ir_path}")
            if not ir_path:
                ir_path = "/Data/strategy_ir.json"
            self.Debug(f"Loading IR from: {ir_path}")
            ir_dict = load_ir_from_file(ir_path, CustomCryptoData.DataFolder)
        self.ir = StrategyIR.model_validate(ir_dict)

        # Set resolution (must be before _add_symbol which uses it)
        resolution_str = getattr(self.ir, "resolution", "Hour") or "Hour"
        self.resolution = getattr(Resolution, resolution_str.capitalize(), Resolution.Hour)

        # Set up symbols
        self.symbol, self.symbols = setup_symbols(
            ir=self.ir,
            add_symbol_func=self._add_symbol,
            normalize_symbol_func=self._normalize_symbol,
            log_func=self.Log,
        )

        # Configure trading costs (fees and slippage)
        costs = setup_trading_costs(self.ir, self.Log)
        self.fee_pct = costs["fee_pct"]
        self.slippage_pct = costs["slippage_pct"]

        # Initialize indicators - unified registry pattern
        # Maps indicator_id -> (category, indicator_or_data)
        self.indicator_registry: dict[str, tuple[IndicatorCategory, Any]] = {}

        # Legacy dicts (for backward compatibility during transition)
        self.indicators = {}
        self.rolling_windows = {}  # For RollingWindow indicators
        self.vol_sma_indicators = {}  # For volume SMA indicators
        self.rolling_minmax = {}  # For rolling min/max trackers
        self.avwap_trackers = {}  # For AVWAP (anchored VWAP) trackers
        self._create_indicators()

        # Initialize state
        self.state = {}
        self._initialize_state()

        # Entry/exit/rules (already typed from StrategyIR)
        rules = setup_rules(self.ir)
        self.entry_rule = rules["entry_rule"]
        self.exit_rules = rules["exit_rules"]
        self.gates = rules["gates"]
        self.overlays = rules["overlays"]
        self.on_bar_invested_ops = rules["on_bar_invested_ops"]
        self.on_bar_ops = rules["on_bar_ops"]

        # Trade tracking for output (lot-based for accumulation support)
        tracking = setup_tracking(initial_cash)
        self.trades = tracking["trades"]
        self.current_lots = tracking["current_lots"]
        self.last_entry_bar = tracking["last_entry_bar"]
        self.equity_curve = tracking["equity_curve"]
        self.peak_equity = tracking["peak_equity"]
        self.max_drawdown = tracking["max_drawdown"]
        self.bar_count = tracking["bar_count"]

        # Crossover detection state
        self._cross_prev = {}  # Stores previous (left, right) values per condition

        # Breakout detection state - tracks previous max/min indicator values
        self._breakout_prev_max = {}  # {ind_id: previous_max_value}
        self._breakout_prev_min = {}  # {ind_id: previous_min_value}

        self.Log("✅ StrategyRuntime initialized")
        self.Log("   🔄 Using refactored Phase 12 code (2026-01-26)")
        self.Log(f"   Strategy: {getattr(self.ir, 'strategy_name', 'Unknown') or 'Unknown'}")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Indicators: {len(self.indicators)}")
        if PYDANTIC_AVAILABLE:
            self.Log(f"   Pydantic: v{PYDANTIC_VERSION} ✓")
        else:
            self.Log("   Pydantic: not available")

    def _add_symbol(self, symbol_str: str) -> Symbol:
        """Add symbol using custom data reader for CSV files."""
        return add_symbol(
            symbol_str=symbol_str,
            resolution=self.resolution,
            add_data_func=self.AddData,
            log_func=self.Log,
            custom_data_class=CustomCryptoData,  # Defined in this file
        )

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to a price."""
        slippage_bps = int(self.slippage_pct * 100) if self.slippage_pct > 0 else 0
        return _apply_slippage_func(price, is_buy, slippage_bps) if slippage_bps > 0 else price

    def _calculate_fee(self, trade_value: float) -> float:
        """Calculate fee for a given trade value."""
        return _calculate_fee_func(trade_value, self.fee_pct) if self.fee_pct > 0 else 0.0

    def _normalize_symbol(self, symbol_str: str) -> str:
        """Normalize symbol string for dictionary keys."""
        return normalize_symbol(symbol_str)

    def _get_symbol_obj(self, symbol_str: str | None) -> Symbol:
        """Get Symbol object by string. Returns primary symbol if None."""
        return get_symbol_obj(symbol_str, self.symbols, self.symbol, normalize_symbol)

    def _create_indicators(self):
        """Create all indicators defined in the IR."""
        create_all_indicators(
            ir_indicators=self.ir.indicators,
            symbol=self.symbol,
            resolution=self.resolution,
            indicator_registry=self.indicator_registry,
            indicators=self.indicators,
            rolling_windows=self.rolling_windows,
            vol_sma_indicators=self.vol_sma_indicators,
            rolling_minmax=self.rolling_minmax,
            avwap_trackers=self.avwap_trackers,
            log=self.Log,
            runtime=self,
        )

    def _initialize_state(self):
        """Initialize state variables from IR."""
        initialize_state_variables(ir_state=self.ir.state, state=self.state)

    def OnData(self, data: Slice):
        """Called when new market data arrives."""
        # Skip if no data for our symbol
        if self.symbol not in data:
            return

        bar = data[self.symbol]

        # Update all custom indicators using the registry
        # (LEAN indicators are auto-updated by the framework)
        update_all_indicators(
            indicator_registry=self.indicator_registry,
            bar=bar,
            current_time=self.Time,
        )

        # Wait for all indicators to be ready
        if not check_indicators_ready(self.indicator_registry, is_indicator_ready):
            return

        # Run on_bar hooks every bar (for state tracking like cross detection)
        self._run_on_bar(bar)

        # Evaluate gates first
        if not self._evaluate_gates(bar):
            return

        # Track equity curve (every bar)
        self._track_equity(bar)

        # Skip trading during warmup period (before user's trading_start_date)
        # This prevents trades from occurring before the user's requested start date
        # while still allowing indicators to warm up and state to be tracked
        if self.trading_start_date and self.Time < self.trading_start_date:
            self.bar_count += 1
            return

        # Check position state
        is_invested = self.Portfolio[self.symbol].Invested

        # If invested, run on_bar_invested hooks and check exits
        if is_invested:
            self._run_on_bar_invested(bar)
            self._evaluate_exits(bar)
            # Check if accumulation is allowed (re-entry while invested)
            if self._can_accumulate():
                self._evaluate_entry(bar)
        else:
            # Check entry
            self._evaluate_entry(bar)

        # Increment bar count AFTER all processing (0-indexed)
        self.bar_count += 1

    def _track_equity(self, bar):
        """Track portfolio equity for equity curve."""
        # Note: bar_count is 0-indexed and incremented AFTER processing
        equity = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        holdings = equity - cash
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0

        self.peak_equity, self.max_drawdown = track_equity(
            equity=equity,
            cash=cash,
            holdings=holdings,
            drawdown=drawdown,
            current_time=self.Time,
            bar_count=self.bar_count,
            resolution=self.resolution,
            equity_curve=self.equity_curve,
            peak_equity=self.peak_equity,
            max_drawdown=self.max_drawdown,
        )


    def _evaluate_gates(self, bar) -> bool:
        """Evaluate gate conditions. Returns True if all gates pass."""
        return evaluate_gates(
            gates=self.gates,
            evaluate_condition_func=self._evaluate_condition,
            bar=bar,
        )

    def _compute_overlay_scale(self, bar) -> float:
        """Compute combined overlay scaling factor for position sizing."""
        return compute_overlay_scale(
            overlays=self.overlays,
            evaluate_condition_func=self._evaluate_condition,
            bar=bar,
            log_func=self.Log,
        )

    def _can_accumulate(self) -> bool:
        """Check if position policy allows another entry while invested."""
        return can_accumulate(
            entry_rule=self.entry_rule,
            current_lots=self.current_lots,
            bar_count=self.bar_count,
            last_entry_bar=self.last_entry_bar,
        )

    def _evaluate_entry(self, bar):
        """Evaluate entry rule and execute if conditions met."""
        self.current_lots, self.last_entry_bar = execute_entry(
            entry_rule=self.entry_rule,
            evaluate_condition_func=self._evaluate_condition,
            bar=bar,
            current_lots=self.current_lots,
            bar_count=self.bar_count,
            symbol=self.symbol,
            portfolio=self.Portfolio,
            current_time=self.Time,
            apply_slippage_func=self._apply_slippage,
            calculate_fee_func=self._calculate_fee,
            execute_action_func=lambda action, b=None: self._execute_action(action, b),
            execute_state_op_func=self._execute_state_op,
            overlays=self.overlays,
            log_func=self.Log,
        )

    def _evaluate_exits(self, bar):
        """Evaluate exit rules in priority order."""
        closed_lots_list = []
        self.current_lots, closed_lots_list = execute_exit(
            exit_rules=self.exit_rules,
            evaluate_condition_func=self._evaluate_condition,
            bar=bar,
            current_lots=self.current_lots,
            bar_count=self.bar_count,
            symbol=self.symbol,
            current_time=self.Time,
            close_lots_func=lambda lots, exit_price, exit_time, exit_bar, exit_reason: close_lots(
                lots=lots,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_bar=exit_bar,
                exit_reason=exit_reason,
                apply_slippage_func=self._apply_slippage,
                calculate_fee_func=self._calculate_fee,
            ),
            execute_action_func=lambda action: self._execute_action(action),
            log_func=self.Log,
        )
        self.trades.extend(closed_lots_list)

    def _run_on_bar_invested(self, bar):
        """Run on_bar_invested state operations."""
        for op in self.on_bar_invested_ops:
            self._execute_state_op(op, bar)

    def _run_on_bar(self, bar):
        """Run on_bar state operations (every bar, for state tracking)."""
        for op in self.on_bar_ops:
            self._execute_state_op(op, bar)

    def _evaluate_condition(self, condition, bar) -> bool:
        """Evaluate a condition from IR using the condition registry (condition is typed Condition)."""
        return registry_evaluate_condition(condition, bar, self)

    def _execute_action(self, action: EntryAction | ExitAction, bar=None):
        """Execute an action from IR."""
        execute_action(
            action=action,
            symbol=self.symbol,
            portfolio=self.Portfolio,
            securities=self.Securities,
            calculate_order_quantity_func=self.CalculateOrderQuantity,
            market_order_func=self.MarketOrder,
            liquidate_func=self.Liquidate,
            log_func=self.Log,
            bar=bar,
        )

    def _resolve_value(self, value_ref, bar):
        """Resolve a ValueRef to a float for state ops and other runtime use."""
        return _resolve_value_impl(
            value_ref,
            bar,
            indicator_registry=self.indicator_registry,
            state=self.state,
            current_time=self.Time,
            rolling_windows=self.rolling_windows,
        )

    def _execute_state_op(self, op: StateOp, bar):
        """Execute a state operation (op is typed StateOp from IR)."""
        _execute_state_op_func(
            op=op,
            bar=bar,
            state=self.state,
            resolve_value_func=self._resolve_value,
            evaluate_condition_func=self._evaluate_condition,
            log_func=self.Log,
        )

    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        # Close any open lots and record as trades
        if self.current_lots and self.Portfolio.Invested:
            raw_price = float(self.Portfolio[self.symbol].Price)
            num_lots = len(self.current_lots)

            closed_lots = close_lots_at_end(
                lots=self.current_lots,
                exit_price=raw_price,
                exit_time=self.Time,
                exit_bar=self.bar_count - 1,  # bar_count has been incremented past the last bar
                apply_slippage_func=self._apply_slippage,
                calculate_fee_func=self._calculate_fee,
                log_func=self.Log,
            )
            self.trades.extend(closed_lots)
            self.current_lots = []

            if num_lots > 1:
                self.Log(f"📊 Closed {num_lots} open lots at end: ${raw_price:.2f}")
            else:
                self.Log(f"📊 Closed open position at end: ${raw_price:.2f}")

        initial_cash = float(self.GetParameter("initial_cash") or 100000)

        # Calculate final equity from trade PnLs (includes manual fee calculations)
        # This is more accurate than LEAN's portfolio value which doesn't include our manual fees
        total_trade_pnl = sum(t.get("pnl", 0.0) for t in self.trades)
        final_equity = initial_cash + total_trade_pnl

        # Also get LEAN's portfolio value for comparison/logging
        lean_portfolio_value = self.Portfolio.TotalPortfolioValue

        # Generate report using trades module
        output = generate_report(
            trades=self.trades,
            equity_curve=self.equity_curve,
            initial_cash=initial_cash,
            final_equity=final_equity,
            max_drawdown=self.max_drawdown,
            strategy_id=getattr(self.ir, "strategy_id", "unknown") or "unknown",
            strategy_name=getattr(self.ir, "strategy_name", "Unknown") or "Unknown",
            symbol=str(self.symbol),
            log_func=self.Log,
        )

        stats = output["statistics"]
        total_return = output["total_return_pct"]

        # Log summary
        self.Log("")
        self.Log(f"{'='*60}")
        self.Log(f"📊 BACKTEST RESULTS: {output['strategy_name']}")
        self.Log(f"{'='*60}")
        self.Log("")
        self.Log("PERFORMANCE")
        self.Log(f"  Initial Capital:    ${initial_cash:,.2f}")
        self.Log(f"  Final Equity:       ${final_equity:,.2f}")
        if abs(final_equity - lean_portfolio_value) > 0.01:
            self.Log(f"  (LEAN Portfolio:    ${lean_portfolio_value:,.2f} - before manual fees)")
        self.Log(f"  Total Return:       {total_return:+.2f}%")
        self.Log(f"  Max Drawdown:       {self.max_drawdown:.2f}%")
        self.Log("")
        self.Log("TRADES")
        self.Log(f"  Total Trades:       {stats['total_trades']}")
        self.Log(f"  Winning Trades:     {stats['winning_trades']}")
        self.Log(f"  Losing Trades:      {stats['losing_trades']}")
        self.Log(f"  Win Rate:           {stats['win_rate']:.1f}%")
        self.Log(f"  Avg Win:            {stats['avg_win_pct']:+.2f}%")
        self.Log(f"  Avg Loss:           {stats['avg_loss_pct']:+.2f}%")
        self.Log(f"  Profit Factor:      {stats['profit_factor']:.2f}")
        self.Log("")

        # Log individual trades
        if self.trades:
            self.Log("TRADE LOG")
            for i, t in enumerate(self.trades):
                exit_price = t.get("exit_price", 0.0)
                pnl_pct = t.get("pnl_percent", 0.0)
                exit_reason = t.get("exit_reason", "N/A")
                self.Log(f"  #{i+1}: {t['direction'].upper()} @ ${t['entry_price']:.2f} -> ${exit_price:.2f} | P&L: {pnl_pct:+.2f}% | Exit: {exit_reason}")
            self.Log("")

        self.Log(f"{'='*60}")

        # Write structured output to JSON file
        import os
        output_path = os.path.join(CustomCryptoData.DataFolder, "strategy_output.json")
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            self.Log(f"📁 Results written to: {output_path}")
        except Exception as e:
            self.Log(f"⚠️ Failed to write output: {e}")

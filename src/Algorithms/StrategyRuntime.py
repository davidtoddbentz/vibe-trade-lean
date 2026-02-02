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
import os
from indicators import (
    IndicatorCategory,
    resolve_value as _resolve_value_impl,
    is_indicator_ready,
    create_all_indicators,
    initialize_state_variables,
    check_indicators_ready,
    update_all_indicators,
)
from vibe_trade_shared.models.ir import (
    StrategyIR,
    EntryAction,
    ExitAction,
    StateOp,
)
from conditions import evaluate_condition as registry_evaluate_condition
from trades import close_lots_at_end, create_lot, generate_report
from position import (
    can_accumulate,
    track_equity,
)
from gates import evaluate_gates
from symbols import normalize_symbol, get_symbol_obj
from ir import load_ir_from_file
from execution.actions import execute_action
from execution.orchestration import execute_entry, execute_exit
from execution.context import ExecutionContext
from execution.types import FillInfo, TrackingState
from initialization import setup_dates, setup_symbols, setup_rules, setup_trading_costs
from state import execute_state_op as _execute_state_op_func
from statistics_extraction import extract_lean_statistics

# Pydantic is required - StrategyIR.model_validate() is used
# If Pydantic is not available, the algorithm will fail at import time
try:
    from pydantic import __version__ as PYDANTIC_VERSION
except ImportError:
    PYDANTIC_VERSION = "unknown"


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

        # Data folder parameter is used for IR IO and outputs
        self.data_folder = self.GetParameter("data_folder") or "/Data"

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
            ir_dict = load_ir_from_file(ir_path, self.data_folder)
        self.ir = StrategyIR.model_validate(ir_dict)

        # Set resolution (must be before _add_symbol which uses it)
        resolution_str = self.ir.resolution or "Hour"
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

        # Apply LEAN-native cost models to securities so portfolio accounting is authoritative
        if self.fee_pct > 0:
            fee_model = PercentageFeeModel(self.fee_pct)
            for sym_key, sym_obj in self.symbols.items():
                self.Securities[sym_obj].SetFeeModel(fee_model)
            self.Log(f"   Applied PercentageFeeModel({self.fee_pct}%) to {len(self.symbols)} securities")

        if self.slippage_pct > 0:
            slippage_decimal = self.slippage_pct / 100.0
            for sym_key, sym_obj in self.symbols.items():
                self.Securities[sym_obj].SetSlippageModel(ConstantSlippageModel(slippage_decimal))
            self.Log(f"   Applied ConstantSlippageModel({self.slippage_pct}%) to {len(self.symbols)} securities")

        # Initialize indicators - unified registry pattern
        # Maps indicator_id -> (category, indicator_or_data)
        self.indicator_registry: dict[str, tuple[IndicatorCategory, Any]] = {}

        # Legacy dicts (for backward compatibility during transition)
        self.indicators = {}
        self.rolling_windows = {}  # For RollingWindow indicators
        self.vol_sma_indicators = {}  # For volume SMA indicators
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
        self.tracking = TrackingState(peak_equity=initial_cash)
        self.tracking.ohlcv_bars = []
        self.tracking.indicator_values = {}

        # Last fill info from LEAN's OnOrderEvent (for accurate lot pricing)
        self._last_fill = None
        # ExecutionContext bundles LEAN primitives for execution functions
        self._exec_ctx = ExecutionContext(
            symbol=self.symbol,
            portfolio=self.Portfolio,
            securities=self.Securities,
            set_holdings=self.SetHoldings,
            market_order=self.MarketOrder,
            liquidate=self.Liquidate,
            log=self.Log,
            get_last_fill=self._get_and_clear_last_fill,
            limit_order=self.LimitOrder,
            stop_market_order=self.StopMarketOrder,
            stop_limit_order=self.StopLimitOrder,
            resolve_value=self._resolve_value,
        )

        # Crossover detection state
        self._cross_prev = {}  # Stores previous (left, right) values per condition

        # Breakout detection state - tracks previous max/min indicator values
        self._breakout_prev_max = {}  # {ind_id: previous_max_value}
        self._breakout_prev_min = {}  # {ind_id: previous_min_value}

        self.Log("✅ StrategyRuntime initialized")
        self.Log("   🔄 Using refactored Phase 12 code (2026-01-26)")
        self.Log(f"   Strategy: {self.ir.strategy_name or 'Unknown'}")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Indicators: {len(self.indicators)}")
        self.Log(f"   Pydantic: v{PYDANTIC_VERSION} ✓")

    def _add_symbol(self, symbol_str: str) -> Symbol:
        """Add LEAN-native crypto subscription via AddCrypto().

        Uses LEAN data layout (/Data/crypto/{market}/{resolution}/{symbol}/...)
        which gives TradeBar with correct OHLC for indicators and fill model.

        fill_forward=False: We provide our own data in all scenarios (ZIP files
        for backtests/tests, PubSub for live). Fill-forward would fabricate
        synthetic bars for gaps in our data, causing OnData to fire on stale
        prices and producing spurious trades.
        """
        lean_ticker = symbol_str.replace("-", "").upper()
        market = os.getenv("LEAN_CRYPTO_MARKET", "coinbase")
        security = self.AddCrypto(lean_ticker, self.resolution, market, False)

        # Ensure fractional orders for crypto
        security.SymbolProperties = SymbolProperties(
            description=symbol_str,
            quoteCurrency="USD",
            contractMultiplier=1,
            minimumPriceVariation=0.01,
            lotSize=0.00000001,
            marketTicker=symbol_str,
            minimumOrderSize=0.00000001,
        )
        self.Log(f"   Added crypto: {symbol_str} as {lean_ticker} (market={market}, lot_size=0.00000001)")
        return security.Symbol

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

        # Cancel unfilled orders from previous bar.
        # Fill evaluation happens BEFORE OnData, so if the order filled,
        # OnOrderEvent(Filled) already fired and the lot was created.
        # If it didn't fill, cancel and re-evaluate conditions.
        open_orders = self.Transactions.GetOpenOrders(self.symbol)
        if open_orders and len(open_orders) > 0:
            self.Transactions.CancelOpenOrders(self.symbol)
            # Clear pending entry — the order was cancelled, not filled
            if self.tracking.pending_entry is not None:
                self.Log(f"⏹ Pending entry cancelled (OrderId={self.tracking.pending_entry.order_id})")
                self.tracking.pending_entry = None

        bar = data[self.symbol]
        self.tracking.ohlcv_bars.append(
            {
                "time": str(bar.EndTime),
                "open": float(bar.Open),
                "high": float(bar.High),
                "low": float(bar.Low),
                "close": float(bar.Close),
                "volume": float(bar.Volume),
            }
        )

        # Execute deferred on_fill state ops from limit/stop fills (queued in OnOrderEvent)
        if self.tracking.deferred_on_fill_ops:
            for op in self.tracking.deferred_on_fill_ops:
                self._execute_state_op(op, bar)
            self.tracking.deferred_on_fill_ops.clear()

        # Reset daily entry counter on new calendar day
        current_date = str(self.Time.date())
        if self.tracking.last_entry_date != current_date:
            self.tracking.entries_today = 0
            self.tracking.last_entry_date = current_date

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

        current_time = str(self.Time)
        for indicator_name, indicator in self.indicators.items():
            if indicator is None or not getattr(indicator, "IsReady", False):
                continue
            if hasattr(indicator, "UpperBand") and hasattr(indicator, "MiddleBand") and hasattr(indicator, "LowerBand"):
                point = {
                    "time": current_time,
                    "upper": float(indicator.UpperBand.Current.Value),
                    "middle": float(indicator.MiddleBand.Current.Value),
                    "lower": float(indicator.LowerBand.Current.Value),
                }
            else:
                point = {"time": current_time, "value": float(indicator.Current.Value)}
            self.tracking.indicator_values.setdefault(indicator_name, []).append(point)

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
            self.tracking.bar_count += 1
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
        self.tracking.bar_count += 1

    def _track_equity(self, bar):
        """Track portfolio equity for equity curve."""
        # Note: bar_count is 0-indexed and incremented AFTER processing
        equity = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        holdings = equity - cash
        drawdown = (self.tracking.peak_equity - equity) / self.tracking.peak_equity * 100 if self.tracking.peak_equity > 0 else 0

        self.tracking.peak_equity, self.tracking.max_drawdown = track_equity(
            equity=equity,
            cash=cash,
            holdings=holdings,
            drawdown=drawdown,
            current_time=self.Time,
            bar_count=self.tracking.bar_count,
            resolution=self.resolution,
            equity_curve=self.tracking.equity_curve,
            peak_equity=self.tracking.peak_equity,
            max_drawdown=self.tracking.max_drawdown,
        )


    def _evaluate_gates(self, bar) -> bool:
        """Evaluate gate conditions. Returns True if all gates pass."""
        return evaluate_gates(
            gates=self.gates,
            evaluate_condition_func=self._evaluate_condition,
            bar=bar,
        )

    def _can_accumulate(self) -> bool:
        """Check if position policy allows another entry while invested."""
        return can_accumulate(
            entry_rule=self.entry_rule,
            current_lots=self.tracking.current_lots,
            bar_count=self.tracking.bar_count,
            last_entry_bar=self.tracking.last_entry_bar,
            entries_today=self.tracking.entries_today,
        )

    def _get_and_clear_last_fill(self) -> FillInfo | None:
        """Return last fill info from OnOrderEvent and clear it."""
        fill = self._last_fill
        self._last_fill = None
        return fill

    def _evaluate_entry(self, bar):
        """Evaluate entry rule and execute if conditions met."""
        new_entry_bar = execute_entry(
            entry_rule=self.entry_rule,
            evaluate_condition=self._evaluate_condition,
            bar=bar,
            tracking=self.tracking,
            ctx=self._exec_ctx,
            current_time=self.Time,
            execute_action_func=self._execute_action,
            execute_state_op=self._execute_state_op,
            overlays=self.overlays,
        )
        # Only update last_entry_bar when an entry actually fired (None = no entry)
        if new_entry_bar is not None:
            self.tracking.last_entry_bar = new_entry_bar

    def _evaluate_exits(self, bar):
        """Evaluate exit rules in priority order."""
        execute_exit(
            exit_rules=self.exit_rules,
            evaluate_condition=self._evaluate_condition,
            bar=bar,
            tracking=self.tracking,
            ctx=self._exec_ctx,
            current_time=self.Time,
            execute_action_func=self._execute_action,
        )

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
        """Execute an action from IR. Returns order_id for non-market orders."""
        return execute_action(action=action, ctx=self._exec_ctx, bar=bar)

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

    def OnOrderEvent(self, orderEvent):
        """Capture LEAN fill prices and fees for lot tracking.

        Handles two fill paths:
        1. Market orders: synchronous fill during execute_entry. FillInfo is
           stored for the lot tracker to pick up immediately.
        2. Limit/stop orders: deferred fill on a subsequent bar. A PendingEntry
           is stored by execute_entry; here we create the lot when the fill arrives.

        Also clears pending_entry on terminal failure statuses (Invalid, Canceled)
        to prevent stale state from blocking future entries.
        """
        # Clear pending_entry on terminal failure statuses
        status = orderEvent.Status
        if status in (OrderStatus.Invalid, OrderStatus.Canceled):
            order_id = int(orderEvent.OrderId)
            pending = self.tracking.pending_entry
            if pending is not None and pending.order_id == order_id:
                self.Log(f"⚠ Pending entry cleared: OrderId={order_id}, Status={status}")
                self.tracking.pending_entry = None
            return

        if status != OrderStatus.Filled:
            return

        fill_price = float(orderEvent.FillPrice)
        fill_qty = float(orderEvent.FillQuantity)
        order_id = int(orderEvent.OrderId)

        # Get the order fee from the event
        order_fee = 0.0
        try:
            order_fee = float(orderEvent.OrderFee.Value.Amount)
        except (AttributeError, Exception):
            pass

        # Check if this fill matches a pending entry (deferred limit/stop order)
        pending = self.tracking.pending_entry
        if pending is not None and pending.order_id == order_id:
            # Deferred fill — create the lot now
            lot = create_lot(
                lot_id=len(self.tracking.current_lots),
                symbol=str(self.symbol),
                direction=pending.direction,
                entry_time=self.Time,
                entry_price=fill_price,
                entry_bar=pending.entry_bar,
                quantity=abs(fill_qty),
                entry_fee=order_fee,
            )
            self.tracking.current_lots.append(lot)
            self.tracking.last_entry_bar = pending.entry_bar
            self.tracking.entries_today += 1

            # Queue on_fill hooks for next OnData (they need bar data for resolve_value)
            if pending.on_fill_ops:
                self.tracking.deferred_on_fill_ops.extend(pending.on_fill_ops)

            lot_num = len(self.tracking.current_lots)
            if lot_num > 1:
                self.Log(f"🟢 ENTRY FILLED (lot #{lot_num}) @ ${fill_price:.2f} | qty: {abs(fill_qty):.6f} (OrderId={order_id})")
            else:
                self.Log(f"🟢 ENTRY FILLED @ ${fill_price:.2f} (OrderId={order_id})")

            self.tracking.pending_entry = None
            return

        # Market order fill — store for synchronous lot creation in execute_entry
        self._last_fill = FillInfo(
            price=fill_price,
            quantity=fill_qty,
            fee=order_fee,
        )

    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        # Append a final equity snapshot BEFORE closing lots, so the curve
        # reflects the state with open positions (important for short backtests
        # where the sample_interval may skip intermediate bars).
        from execution.types import EquityPoint
        pre_close_equity = float(self.Portfolio.TotalPortfolioValue)
        pre_close_cash = float(self.Portfolio.Cash)
        pre_close_holdings = pre_close_equity - pre_close_cash
        pre_close_dd = ((self.tracking.peak_equity - pre_close_equity) / self.tracking.peak_equity * 100
                        if self.tracking.peak_equity > 0 else 0)
        self.tracking.equity_curve.append(
            EquityPoint(
                time=str(self.Time),
                equity=pre_close_equity,
                cash=pre_close_cash,
                holdings=pre_close_holdings,
                drawdown=float(pre_close_dd),
            )
        )

        # Close any open lots and record as trades
        if self.tracking.current_lots and self.Portfolio.Invested:
            raw_price = float(self.Portfolio[self.symbol].Price)
            num_lots = len(self.tracking.current_lots)

            closed_lots = close_lots_at_end(
                lots=self.tracking.current_lots,
                exit_price=raw_price,
                exit_time=self.Time,
                exit_bar=self.tracking.bar_count - 1,  # bar_count has been incremented past the last bar
                log_func=self.Log,
            )
            self.tracking.trades.extend(closed_lots)
            self.tracking.current_lots = []

            if num_lots > 1:
                self.Log(f"📊 Closed {num_lots} open lots at end: ${raw_price:.2f}")
            else:
                self.Log(f"📊 Closed open position at end: ${raw_price:.2f}")

        initial_cash = float(self.GetParameter("initial_cash") or 100000)

        final_equity = float(self.Portfolio.TotalPortfolioValue)

        # Generate report using trades module
        output = generate_report(
            trades=self.tracking.trades,
            equity_curve=self.tracking.equity_curve,
            initial_cash=initial_cash,
            final_equity=final_equity,
            max_drawdown=self.tracking.max_drawdown,
            strategy_id=self.ir.strategy_id or "unknown",
            strategy_name=self.ir.strategy_name or "Unknown",
            symbol=str(self.symbol),
            log_func=self.Log,
            ohlcv_bars=self.tracking.ohlcv_bars,
            indicator_values=self.tracking.indicator_values,
        )

        # Extract LEAN native statistics (if available)
        lean_stats = extract_lean_statistics(self.Statistics)

        # Merge LEAN statistics with custom calculations
        # LEAN stats take precedence for overlapping fields
        if lean_stats:
            # Count non-None statistics
            available_stats = sum(1 for v in lean_stats.values() if v is not None)
            output["statistics"].update(lean_stats)
            self.Log(f"✅ Extracted {available_stats} LEAN statistics")

            # Log key risk-adjusted metrics if available
            sharpe = lean_stats.get("sharpe_ratio")
            sortino = lean_stats.get("sortino_ratio")
            if sharpe is not None:
                self.Log(f"   Sharpe Ratio: {sharpe:.2f}")
            if sortino is not None:
                self.Log(f"   Sortino Ratio: {sortino:.2f}")
        else:
            self.Log("⚠️  LEAN statistics unavailable (using fallback calculations)")

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
        self.Log(f"  Total Return:       {total_return:+.2f}%")
        self.Log(f"  Max Drawdown:       {self.tracking.max_drawdown:.2f}%")
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
        # All trades have exit_price, pnl_percent, and exit_reason set by close_lots()
        if self.tracking.trades:
            self.Log("TRADE LOG")
            for i, t in enumerate(self.tracking.trades):
                self.Log(
                    f"  #{i+1}: {t.direction.upper()} @ ${t.entry_price:.2f} -> "
                    f"${t.exit_price:.2f} | P&L: {t.pnl_percent:+.2f}% | Exit: {t.exit_reason}"
                )
            self.Log("")

        self.Log(f"{'='*60}")

        # Write structured output to JSON file
        import os
        output_path = os.path.join(self.data_folder, "strategy_output.json")
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            self.Log(f"📁 Results written to: {output_path}")
        except Exception as e:
            self.Log(f"⚠️ Failed to write output: {e}")

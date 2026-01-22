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
from enum import Enum
from typed_conditions import (
    TypedRegimeCondition,
    TypedBreakoutCondition,
    TypedSqueezeCondition,
    TypedSpreadCondition,
    TypedGapCondition,
    parse_condition,
)


# =============================================================================
# IR Types (mirrors src/translator/ir.py for LEAN environment)
# =============================================================================


class CompareOp(str, Enum):
    """Comparison operators."""
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    EQ = "=="
    NEQ = "!="

    def apply(self, left: float, right: float) -> bool:
        """Apply the comparison operator."""
        if self == CompareOp.LT:
            return left < right
        elif self == CompareOp.LTE:
            return left <= right
        elif self == CompareOp.GT:
            return left > right
        elif self == CompareOp.GTE:
            return left >= right
        elif self == CompareOp.EQ:
            return left == right
        elif self == CompareOp.NEQ:
            return left != right
        return False


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
        data_folder = self.GetParameter("data_folder")
        if data_folder:
            CustomCryptoData.DataFolder = data_folder
            self.Log(f"[INIT] Set data folder to: {data_folder}")
        else:
            self.Log(f"[INIT] No data_folder parameter, using default: {CustomCryptoData.DataFolder}")

        # Debug: List files in data folder
        import os
        # Set up debug log in data folder
        CustomCryptoData.DebugLogPath = os.path.join(CustomCryptoData.DataFolder, "debug.log")
        self.Debug(f"[INIT] Debug log path: {CustomCryptoData.DebugLogPath}")
        CustomCryptoData._log_debug(f"[INIT] Debug logging initialized at {CustomCryptoData.DataFolder}")

        try:
            files = os.listdir(CustomCryptoData.DataFolder)
            self.Debug(f"[INIT] Files in data folder: {files}")
            # Check for btc_usd_data.csv specifically
            expected_file = os.path.join(CustomCryptoData.DataFolder, "btc_usd_data.csv")
            if os.path.exists(expected_file):
                self.Debug("[INIT] ✅ Found btc_usd_data.csv")
                # Read first few lines to verify format
                with open(expected_file, 'r') as f:
                    lines = f.readlines()[:3]
                    for i, line in enumerate(lines):
                        self.Debug(f"[INIT] Line {i}: {line.strip()}")
            else:
                self.Debug(f"[INIT] ❌ btc_usd_data.csv NOT FOUND at {expected_file}")
        except Exception as e:
            self.Debug(f"[INIT] Error listing data folder: {e}")

        # Get date parameters (format: YYYYMMDD)
        start_date_str = self.GetParameter("start_date")
        end_date_str = self.GetParameter("end_date")
        initial_cash_str = self.GetParameter("initial_cash")

        self.Debug(f"start_date parameter: {start_date_str}")
        self.Debug(f"end_date parameter: {end_date_str}")
        self.Debug(f"initial_cash parameter: {initial_cash_str}")

        # Parse dates or use defaults
        if start_date_str:
            year = int(start_date_str[:4])
            month = int(start_date_str[4:6])
            day = int(start_date_str[6:8])
            self.SetStartDate(year, month, day)
        else:
            self.SetStartDate(2024, 1, 1)

        if end_date_str:
            year = int(end_date_str[:4])
            month = int(end_date_str[4:6])
            day = int(end_date_str[6:8])
            self.SetEndDate(year, month, day)
        else:
            self.SetEndDate(2024, 12, 31)

        if initial_cash_str:
            self.SetCash(float(initial_cash_str))
        else:
            self.SetCash(100000)

        # Load strategy IR
        ir_json = self.GetParameter("strategy_ir")
        self.Debug(f"strategy_ir parameter: {ir_json}")
        if ir_json:
            self.ir = json.loads(ir_json)
        else:
            # Try loading from file (parameter or default path)
            ir_path = self.GetParameter("strategy_ir_path")
            self.Debug(f"strategy_ir_path parameter: {ir_path}")
            if not ir_path:
                # Default path for backtest service
                ir_path = "/Data/strategy_ir.json"
            self.Debug(f"Loading IR from: {ir_path}")
            self.ir = self._load_ir_from_file(ir_path)

        # Set resolution (must be before _add_symbol which uses it)
        resolution_str = self.ir.get("resolution", "Hour")
        self.resolution = getattr(Resolution, resolution_str.capitalize(), Resolution.Hour)

        # Data folder was set at the beginning of Initialize

        # Set up primary symbol
        symbol_str = self.ir.get("symbol", "BTC-USD")
        self.symbol = self._add_symbol(symbol_str)

        # Set up additional symbols for multi-symbol strategies
        self.symbols = {self._normalize_symbol(symbol_str): self.symbol}
        for additional_sym in self.ir.get("additional_symbols", []):
            sym_obj = self._add_symbol(additional_sym)
            self.symbols[self._normalize_symbol(additional_sym)] = sym_obj
            self.Log(f"   Added additional symbol: {additional_sym}")

        # Configure trading costs (fees and slippage)
        self._configure_trading_costs()

        # Initialize indicators
        self.indicators = {}
        self.rolling_windows = {}  # For RollingWindow indicators
        self.vol_sma_indicators = {}  # For volume SMA indicators
        self.rolling_minmax = {}  # For rolling min/max trackers
        self._create_indicators()

        # Initialize state
        self.state = {}
        self._initialize_state()

        # Parse entry and exit rules
        self.entry_rule = self.ir.get("entry")
        self.exit_rules = self.ir.get("exits", [])
        self.gates = self.ir.get("gates", [])
        self.overlays = self.ir.get("overlays", [])
        self.on_bar_invested_ops = self.ir.get("on_bar_invested", [])
        self.on_bar_ops = self.ir.get("on_bar", [])

        # Trade tracking for output
        self.trades = []  # List of completed trades
        self.current_trade = None  # Active trade
        self.equity_curve = []  # Portfolio value over time
        self.peak_equity = float(initial_cash_str) if initial_cash_str else 100000
        self.max_drawdown = 0.0
        self.bar_count = 0  # Count bars for equity sampling

        # Crossover detection state
        self._cross_prev = {}  # Stores previous (left, right) values per condition

        self.Log("✅ StrategyRuntime initialized")
        self.Log(f"   Strategy: {self.ir.get('strategy_name', 'Unknown')}")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Indicators: {len(self.indicators)}")

    def _add_symbol(self, symbol_str: str) -> Symbol:
        """Add symbol using custom data reader for CSV files.
        
        After adding the symbol, we configure its lot size to allow
        fractional orders (like 0.1 BTC for a $10 order).
        """
        # Use AddData with CustomCryptoData for CSV files
        security = self.AddData(CustomCryptoData, symbol_str, self.resolution)
        symbol = security.Symbol
        
        # CRITICAL: Configure lot size to allow fractional orders
        # By default, custom data uses lot size 1, which prevents orders < 1 unit
        # For crypto-like assets, we need very small lot sizes (0.00000001)
        security.SymbolProperties = SymbolProperties(
            description=symbol_str,
            quoteCurrency="USD",
            contractMultiplier=1,
            minimumPriceVariation=0.01,
            lotSize=0.00000001,  # Allow tiny fractional orders
            marketTicker=symbol_str,
            minimumOrderSize=0.00000001  # Allow tiny orders
        )
        self.Log(f"   Added symbol: {symbol_str} (lot_size=0.00000001)")
        
        return symbol

    def _configure_trading_costs(self):
        """Configure trading costs for manual PnL calculations.

        Reads fee_pct and slippage_pct from IR and stores them for use in
        trade tracking. Costs are applied manually in _evaluate_entry() and
        _evaluate_exits() rather than via LEAN's built-in models to ensure
        our custom trade records accurately reflect the costs.

        fee_pct: Trading fee as percentage of trade value (e.g., 0.1 = 0.1%)
        slippage_pct: Slippage as percentage of price (e.g., 0.05 = 0.05%)
        """
        fee_pct = self.ir.get("fee_pct", 0.0)
        slippage_pct = self.ir.get("slippage_pct", 0.0)

        # Store for manual PnL calculations in trade tracking
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

        if fee_pct > 0 or slippage_pct > 0:
            self.Log(f"   Trading costs: fee={fee_pct}%, slippage={slippage_pct}%")
        else:
            self.Log(f"   No trading costs configured")

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to a price.

        For buys: price increases (pay more)
        For sells: price decreases (receive less)

        Args:
            price: The base price
            is_buy: True for buy orders, False for sell orders

        Returns:
            Price adjusted for slippage
        """
        if self.slippage_pct <= 0:
            return price
        slip_mult = self.slippage_pct / 100.0
        if is_buy:
            return price * (1 + slip_mult)  # Pay more
        else:
            return price * (1 - slip_mult)  # Receive less

    def _calculate_fee(self, trade_value: float) -> float:
        """Calculate fee for a given trade value.

        Args:
            trade_value: Absolute value of the trade (price × quantity)

        Returns:
            Fee amount in the same currency
        """
        if self.fee_pct <= 0:
            return 0.0
        return abs(trade_value) * (self.fee_pct / 100.0)

    def _normalize_symbol(self, symbol_str: str) -> str:
        """Normalize symbol string for dictionary keys (lowercase, no dashes)."""
        return symbol_str.lower().replace("-", "")

    def _get_symbol_obj(self, symbol_str: str | None) -> Symbol:
        """Get Symbol object by string. Returns primary symbol if None."""
        if symbol_str is None:
            return self.symbol
        normalized = self._normalize_symbol(symbol_str)
        if normalized in self.symbols:
            return self.symbols[normalized]
        # Fallback to primary
        return self.symbol

    def _load_ir_from_file(self, path: str) -> dict:
        """Load IR JSON from a file path."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Strategy IR file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in strategy IR file: {e}")

    def _create_indicators(self):
        """Create all indicators defined in the IR.

        Indicators can specify a 'symbol' field to use a different symbol than the primary.
        This enables multi-symbol strategies (e.g., BTC/ETH correlation).
        """
        for ind_def in self.ir.get("indicators", []):
            ind_type = ind_def.get("type")
            ind_id = ind_def.get("id")

            # Get symbol for this indicator (defaults to primary symbol)
            ind_symbol = self._get_symbol_obj(ind_def.get("symbol"))

            # All indicators use named resolution parameter to avoid signature issues
            if ind_type == "EMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.EMA(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "SMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.SMA(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "BB":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.BB(ind_symbol, period, mult, resolution=self.resolution)
            elif ind_type == "KC":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.KCH(ind_symbol, period, mult, resolution=self.resolution)
            elif ind_type == "ATR":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ATR(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "MAX":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MAX(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "MIN":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MIN(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "ROC":
                period = ind_def.get("period", 1)
                self.indicators[ind_id] = self.ROC(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "ADX":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ADX(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "RSI":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.RSI(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "MACD":
                fast = ind_def.get("fast_period", 12)
                slow = ind_def.get("slow_period", 26)
                signal = ind_def.get("signal_period", 9)
                self.indicators[ind_id] = self.MACD(ind_symbol, fast, slow, signal, resolution=self.resolution)
            elif ind_type == "DC":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.DCH(ind_symbol, period, resolution=self.resolution)
            elif ind_type == "VWAP":
                period = ind_def.get("period", 0)
                if period == 0:
                    # Intraday VWAP (resets daily)
                    self.indicators[ind_id] = self.VWAP(ind_symbol)
                else:
                    # Rolling VWAP with period
                    self.indicators[ind_id] = self.VWAP(ind_symbol, period)
            elif ind_type == "RW":
                # Rolling window for historical values (e.g., previous close)
                period = ind_def.get("period", 2)
                field = ind_def.get("field", "close")
                # Store as a RollingWindow - handled specially in OnData
                # Note: symbol stored for multi-symbol support
                self.rolling_windows[ind_id] = {
                    "window": RollingWindow[float](period),
                    "field": field,
                    "symbol": ind_symbol,
                }
            elif ind_type == "VOL_SMA":
                # Simple Moving Average of volume
                period = ind_def.get("period", 20)
                # Use SMA indicator on volume data
                # Note: symbol stored for multi-symbol support
                self.vol_sma_indicators[ind_id] = {
                    "sma": SimpleMovingAverage(period),
                    "period": period,
                    "symbol": ind_symbol,
                }
            elif ind_type == "RMM":
                # Rolling Min/Max tracker
                period = ind_def.get("period", 20)
                mode = ind_def.get("mode", "min")
                field = ind_def.get("field", "close")
                self.rolling_minmax[ind_id] = {
                    "window": RollingWindow[float](period),
                    "mode": mode,
                    "field": field,
                    "symbol": ind_symbol,
                }
            else:
                self.Log(f"⚠️ Unknown indicator type: {ind_type}")

    def _initialize_state(self):
        """Initialize state variables from IR."""
        for state_var in self.ir.get("state", []):
            state_id = state_var.get("id")
            default = state_var.get("default")
            self.state[state_id] = default

    def OnData(self, data: Slice):
        """Called when new market data arrives."""
        # Skip if no data for our symbol
        if self.symbol not in data:
            return

        bar = data[self.symbol]

        # Update rolling windows before checking indicators
        for rw_id, rw_data in self.rolling_windows.items():
            field = rw_data["field"]
            if field == "close":
                rw_data["window"].Add(bar.Close)
            elif field == "open":
                rw_data["window"].Add(bar.Open)
            elif field == "high":
                rw_data["window"].Add(bar.High)
            elif field == "low":
                rw_data["window"].Add(bar.Low)

        # Update volume SMA indicators
        for vol_id, vol_data in self.vol_sma_indicators.items():
            vol_data["sma"].Update(self.Time, float(bar.Volume))

        # Update rolling min/max trackers
        for rmm_id, rmm_data in self.rolling_minmax.items():
            field = rmm_data["field"]
            if field == "close":
                rmm_data["window"].Add(bar.Close)
            elif field == "open":
                rmm_data["window"].Add(bar.Open)
            elif field == "high":
                rmm_data["window"].Add(bar.High)
            elif field == "low":
                rmm_data["window"].Add(bar.Low)

        # Wait for all indicators to be ready
        if not self._indicators_ready():
            return

        # Run on_bar hooks every bar (for state tracking like cross detection)
        self._run_on_bar(bar)

        # Evaluate gates first
        if not self._evaluate_gates(bar):
            return

        # Track equity curve (every bar)
        self._track_equity(bar)

        # Check position state
        is_invested = self.Portfolio[self.symbol].Invested

        # If invested, run on_bar_invested hooks and check exits
        if is_invested:
            self._run_on_bar_invested(bar)
            self._evaluate_exits(bar)
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

        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Sample equity curve (every 60 bars ~= hourly for minute data)
        # Note: bar_count is 0-indexed at this point
        if self.bar_count == 0 or self.bar_count % 60 == 0:
            self.equity_curve.append({
                "time": str(self.Time),
                "equity": float(equity),
                "cash": float(cash),
                "holdings": float(holdings),
                "drawdown": float(drawdown),
            })

    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        for ind in self.indicators.values():
            if not ind.IsReady:
                return False
        # Also check rolling windows
        for rw_data in self.rolling_windows.values():
            if not rw_data["window"].IsReady:
                return False
        # Check volume SMA indicators
        for vol_data in self.vol_sma_indicators.values():
            if not vol_data["sma"].IsReady:
                return False
        # Check rolling min/max
        for rmm_data in self.rolling_minmax.values():
            if not rmm_data["window"].IsReady:
                return False
        return True

    def _evaluate_gates(self, bar) -> bool:
        """Evaluate gate conditions. Returns True if all gates pass."""
        for gate in self.gates:
            condition = gate.get("condition")
            mode = gate.get("mode", "allow")

            result = self._evaluate_condition(condition, bar)

            if mode == "allow" and not result:
                return False
            elif mode == "block" and result:
                return False

        return True

    def _compute_overlay_scale(self, bar) -> float:
        """Compute combined overlay scaling factor for position sizing.

        Evaluates all overlays that target "entry" and multiplies their
        scale_size_frac values when conditions are true.
        """
        scale = 1.0
        for overlay in self.overlays:
            # Check if overlay targets entry
            target_roles = overlay.get("target_roles", ["entry", "exit"])
            if "entry" not in target_roles:
                continue

            condition = overlay.get("condition")
            if self._evaluate_condition(condition, bar):
                # Apply scale factor when condition is true
                scale_size = overlay.get("scale_size_frac", 1.0)
                scale *= scale_size
                self.Log(f"   Overlay '{overlay.get('id', 'unknown')}' active: scale={scale_size}")

        return scale

    def _evaluate_entry(self, bar):
        """Evaluate entry rule and execute if conditions met."""
        if not self.entry_rule:
            return

        condition = self.entry_rule.get("condition")
        result = self._evaluate_condition(condition, bar)
        if result:
            action = self.entry_rule.get("action", {}).copy()  # Copy to modify

            # Apply overlay scaling to position size
            overlay_scale = self._compute_overlay_scale(bar)
            if overlay_scale != 1.0:
                original_allocation = action.get("allocation", 0.95)
                action["allocation"] = original_allocation * overlay_scale
                self.Log(f"   Position scaled: {original_allocation} -> {action['allocation']}")

            self._execute_action(action, bar)

            # Detect direction from allocation or quantity
            allocation = action.get("allocation", 0.95)
            quantity = float(self.Portfolio[self.symbol].Quantity)
            direction = "short" if allocation < 0 or quantity < 0 else "long"

            # Track trade with slippage-adjusted entry price
            # For long: buying, so pay more (slippage up)
            # For short: selling, so receive less (slippage down)
            is_buy_entry = (direction == "long")
            entry_price = self._apply_slippage(float(bar.Close), is_buy_entry)
            entry_value = entry_price * abs(quantity)
            entry_fee = self._calculate_fee(entry_value)

            self.current_trade = {
                "symbol": str(self.symbol),
                "direction": direction,
                "entry_time": str(self.Time),
                "entry_price": entry_price,
                "entry_bar": self.bar_count,  # Bar index where entry occurred
                "quantity": abs(quantity),  # Store absolute quantity
                "entry_fee": entry_fee,  # Fee paid on entry
            }

            # Run on_fill hooks
            for op in self.entry_rule.get("on_fill", []):
                self._execute_state_op(op, bar)

            self.Log(f"🟢 ENTRY @ ${bar.Close:.2f}")

    def _evaluate_exits(self, bar):
        """Evaluate exit rules in priority order."""
        # Sort by priority (lower priority number = higher priority)
        sorted_exits = sorted(self.exit_rules, key=lambda x: x.get("priority", 0))

        for exit_rule in sorted_exits:
            condition = exit_rule.get("condition")
            if self._evaluate_condition(condition, bar):
                # Complete trade tracking before executing action
                if self.current_trade:
                    entry_price = self.current_trade["entry_price"]
                    quantity = self.current_trade.get("quantity", 0)
                    direction = self.current_trade.get("direction", "long")
                    entry_fee = self.current_trade.get("entry_fee", 0)

                    # Apply slippage to exit price
                    # For long exit: selling, so receive less (slippage down)
                    # For short exit: buying/covering, so pay more (slippage up)
                    is_buy_exit = (direction == "short")  # Short covers with a buy
                    exit_price = self._apply_slippage(float(bar.Close), is_buy_exit)

                    # Calculate exit fee
                    exit_value = exit_price * quantity
                    exit_fee = self._calculate_fee(exit_value)
                    total_fees = entry_fee + exit_fee

                    # PnL calculation: short profits when price drops
                    # Subtract total fees from PnL
                    if direction == "short":
                        gross_pnl = (entry_price - exit_price) * quantity
                    else:
                        gross_pnl = (exit_price - entry_price) * quantity

                    pnl = gross_pnl - total_fees
                    # Calculate pnl_pct based on entry value
                    entry_value = entry_price * quantity
                    pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                    self.current_trade["exit_time"] = str(self.Time)
                    self.current_trade["exit_price"] = exit_price
                    self.current_trade["exit_bar"] = self.bar_count
                    self.current_trade["pnl"] = pnl
                    self.current_trade["pnl_percent"] = pnl_pct
                    self.current_trade["exit_fee"] = exit_fee
                    self.current_trade["total_fees"] = total_fees
                    self.current_trade["exit_reason"] = exit_rule.get("id", "unknown")

                    self.trades.append(self.current_trade)
                    self.current_trade = None

                action = exit_rule.get("action", {})
                self._execute_action(action)
                self.Log(f"🔴 EXIT ({exit_rule.get('id', 'unknown')}) @ ${bar.Close:.2f}")
                break  # Only execute first matching exit

    def _run_on_bar_invested(self, bar):
        """Run on_bar_invested state operations."""
        for op in self.on_bar_invested_ops:
            self._execute_state_op(op, bar)

    def _run_on_bar(self, bar):
        """Run on_bar state operations (every bar, for state tracking)."""
        for op in self.on_bar_ops:
            self._execute_state_op(op, bar)

    def _evaluate_condition(self, condition: dict, bar) -> bool:
        """Evaluate a condition from IR."""
        if not condition:
            return True

        cond_type = condition.get("type")

        if cond_type == "compare":
            left_val = self._resolve_value(condition.get("left"), bar)
            right_val = self._resolve_value(condition.get("right"), bar)
            op_str = condition.get("op")
            op = CompareOp(op_str)
            return op.apply(left_val, right_val)

        elif cond_type == "allOf":
            for sub in condition.get("conditions", []):
                if not self._evaluate_condition(sub, bar):
                    return False
            return True

        elif cond_type == "anyOf":
            for sub in condition.get("conditions", []):
                if self._evaluate_condition(sub, bar):
                    return True
            return False

        elif cond_type == "not":
            inner = condition.get("condition")
            return not self._evaluate_condition(inner, bar)

        elif cond_type == "regime":
            # Handle regime conditions by mapping to indicator comparisons
            return self._evaluate_regime(condition, bar)

        elif cond_type == "cross":
            return self._evaluate_cross(condition, bar)

        elif cond_type == "squeeze":
            return self._evaluate_squeeze(condition, bar)

        elif cond_type == "breakout":
            # Typed BreakoutCondition - delegate to breakout evaluator
            return self._evaluate_breakout(condition, bar)

        elif cond_type == "spread":
            # Typed SpreadCondition - multi-symbol spread/ratio comparison
            return self._evaluate_spread(condition, bar)

        elif cond_type == "intermarket":
            # Typed IntermarketCondition - leader/follower trigger
            return self._evaluate_intermarket(condition, bar)

        elif cond_type == "time_filter":
            return self._evaluate_time_filter(condition, bar)

        elif cond_type == "state_condition":
            return self._evaluate_state_condition(condition, bar)

        elif cond_type == "gap":
            return self._evaluate_gap(condition, bar)

        elif cond_type == "trailing_breakout":
            return self._evaluate_trailing_breakout(condition, bar)

        elif cond_type == "trailing_state":
            return self._evaluate_trailing_state(condition, bar)

        elif cond_type == "sequence":
            return self._evaluate_sequence(condition, bar)

        elif cond_type == "event_window":
            return self._evaluate_event_window(condition, bar)

        elif cond_type == "multi_leader_intermarket":
            return self._evaluate_multi_leader_intermarket(condition, bar)

        elif cond_type == "liquidity_sweep":
            return self._evaluate_liquidity_sweep_typed(condition, bar)

        elif cond_type == "flag_pattern":
            return self._evaluate_flag_pattern_typed(condition, bar)

        elif cond_type == "pennant_pattern":
            return self._evaluate_pennant_pattern_typed(condition, bar)

        else:
            raise RuntimeError(f"Unimplemented condition type: {cond_type}")

    def _evaluate_regime(self, regime: dict, bar) -> bool:
        """Evaluate a regime condition."""
        metric = regime.get("metric")
        op_str = regime.get("op", "==")
        value = regime.get("value")
        op = CompareOp(op_str)

        if metric == "trend_ma_relation":
            fast_id = f"ema_{regime.get('ma_fast', 20)}"
            slow_id = f"ema_{regime.get('ma_slow', 50)}"

            # Fall back to named indicators
            fast_ind = self.indicators.get(fast_id) or self.indicators.get("ema_fast")
            slow_ind = self.indicators.get(slow_id) or self.indicators.get("ema_slow")

            if fast_ind and slow_ind:
                diff = fast_ind.Current.Value - slow_ind.Current.Value
                return op.apply(diff, float(value))

        elif metric == "trend_adx":
            # ADX trend strength
            adx_ind = self.indicators.get("adx") or self.indicators.get("adx_14")
            if adx_ind:
                return op.apply(adx_ind.Current.Value, float(value))

        elif metric in ("vol_bb_width_pctile", "bb_width_pctile"):
            # Bollinger Band width percentile
            bb_ind = self.indicators.get("bb") or self.indicators.get("bb_20")
            lookback = regime.get("lookback_bars", 100)
            if bb_ind:
                # Calculate BB width: (upper - lower) / middle
                upper = bb_ind.UpperBand.Current.Value
                lower = bb_ind.LowerBand.Current.Value
                middle = bb_ind.MiddleBand.Current.Value
                if middle != 0:
                    width = (upper - lower) / middle
                    # Get rolling window for width percentile
                    width_rw = self.rolling_windows.get("bb_width")
                    if width_rw and width_rw["window"].IsReady:
                        # Calculate percentile
                        widths = list(width_rw["window"])
                        pctile = sum(1 for w in widths if w < width) / len(widths) * 100
                        return op.apply(pctile, float(value))
                    # Fallback: just compare width directly
                    return op.apply(width * 100, float(value))  # scale for comparison

        elif metric == "vol_atr_pct":
            # ATR as percentage of price
            atr_ind = self.indicators.get("atr") or self.indicators.get("atr_14")
            if atr_ind and bar.Close != 0:
                atr_pct = (atr_ind.Current.Value / bar.Close) * 100
                return op.apply(atr_pct, float(value))

        elif metric == "dist_from_vwap_pct":
            # Distance from VWAP as percentage
            vwap_ind = self.indicators.get("vwap")
            if vwap_ind and vwap_ind.Current.Value != 0:
                dist_pct = ((bar.Close - vwap_ind.Current.Value) / vwap_ind.Current.Value) * 100
                return op.apply(dist_pct, float(value))

        elif metric == "volume_pctile":
            # Volume percentile
            vol_rw = self.rolling_windows.get("volume")
            if vol_rw and vol_rw["window"].IsReady:
                volumes = list(vol_rw["window"])
                current_vol = float(bar.Volume)
                pctile = sum(1 for v in volumes if v < current_vol) / len(volumes) * 100
                return op.apply(pctile, float(value))

        elif metric == "ret_pct":
            roc_ind = self.indicators.get("roc")
            if roc_ind:
                return op.apply(roc_ind.Current.Value * 100, float(value))

        elif metric == "gap_pct":
            # Gap percentage: (Open - PrevClose) / PrevClose * 100
            prev_close_rw = self.rolling_windows.get("prev_close")
            if prev_close_rw and prev_close_rw["window"].IsReady:
                # Rolling window: index 0 = most recent (current bar's close after update)
                # index 1 = previous bar's close
                prev_close = prev_close_rw["window"][1]
                if prev_close != 0:
                    gap = (bar.Open - prev_close) / prev_close * 100
                    return op.apply(gap, float(value))

        elif metric == "liquidity_sweep":
            # Liquidity sweep: break below level then reclaim
            # Requires state tracking (implemented via state vars in IR)
            return self._evaluate_liquidity_sweep(regime, bar)

        elif metric == "flag_pattern":
            # Flag pattern: momentum + consolidation + breakout
            return self._evaluate_flag_pattern(regime, bar)

        elif metric == "pennant_pattern":
            # Pennant pattern: similar to flag with triangular consolidation
            return self._evaluate_pennant_pattern(regime, bar)

        elif metric in ("price_level_touch", "price_level_cross"):
            # These should be lowered by translator, but handle fallback
            # For dynamic levels that need runtime resolution
            return self._evaluate_price_level(metric, regime, bar)

        # Unknown metric - return True to not block
        self.Log(f"⚠️ Unknown regime metric: {metric}")
        return True

    def _evaluate_cross(self, condition: dict, bar) -> bool:
        """Evaluate a crossover condition.

        Detects when left crosses above or below right by comparing
        current and previous values.
        """
        left_val = self._resolve_value(condition.get("left"), bar)
        right_val = self._resolve_value(condition.get("right"), bar)

        # Create a unique key for this condition
        key = str(condition)
        prev = self._cross_prev.get(key)

        # Store current values for next bar
        self._cross_prev[key] = (left_val, right_val)

        # First bar after indicator warmup - no previous values yet
        if prev is None:
            return False

        prev_left, prev_right = prev
        direction = condition.get("direction", "above")

        if direction in ("above", "cross_above"):
            # Crossed above: was below or equal, now above
            return prev_left <= prev_right and left_val > right_val
        elif direction in ("below", "cross_below"):
            # Crossed below: was above or equal, now below
            return prev_left >= prev_right and left_val < right_val
        else:
            self.Log(f"⚠️ Unknown cross direction: {direction}")
            return False

    def _evaluate_breakout(self, condition: dict, bar) -> bool:
        """Evaluate breakout condition (N-bar high/low breakout)."""
        typed = TypedBreakoutCondition.from_dict(condition)

        # Get rolling max/min indicators
        max_ind = self.indicators.get(f"max_{typed.lookback_bars}") or self.indicators.get("max_50")
        min_ind = self.indicators.get(f"min_{typed.lookback_bars}") or self.indicators.get("min_50")

        if not max_ind or not min_ind:
            return False

        high_level = max_ind.Current.Value
        low_level = min_ind.Current.Value

        # Apply buffer
        buffer_mult = 1 + (typed.buffer_bps / 10000)
        high_level *= buffer_mult
        low_level /= buffer_mult

        # Check for breakout
        return bar.Close > high_level or bar.Close < low_level

    def _evaluate_spread(self, condition: dict, bar) -> bool:
        """Evaluate spread condition (multi-symbol ratio/difference).

        Calculates spread between two symbols and compares to threshold.
        Requires multi-symbol data subscription.
        """
        symbol_a = condition.get("symbol_a")
        symbol_b = condition.get("symbol_b")
        calc_type = condition.get("calc_type", "zscore")
        trigger_op = condition.get("trigger_op", "above")
        threshold = condition.get("threshold", 2.0)

        # Get prices for both symbols
        # Note: This is a placeholder - actual multi-symbol implementation
        # requires LEAN multi-symbol data handling
        price_a = bar.Close  # Placeholder: would need symbol_a's price
        price_b = bar.Close  # Placeholder: would need symbol_b's price

        if price_b == 0:
            return False

        # Calculate spread value based on calc_type
        if calc_type == "ratio":
            spread_val = price_a / price_b
        elif calc_type == "difference":
            spread_val = price_a - price_b
        elif calc_type == "log_ratio":
            import math
            spread_val = math.log(price_a / price_b) if price_a > 0 and price_b > 0 else 0
        else:  # zscore - would need rolling mean/std
            spread_val = 0  # Placeholder for z-score calculation

        # Apply trigger operation
        if trigger_op == "above":
            return spread_val > threshold
        elif trigger_op == "below":
            return spread_val < threshold
        elif trigger_op == "crosses_above":
            # Would need previous spread value tracking
            return spread_val > threshold
        elif trigger_op == "crosses_below":
            return spread_val < threshold

        return False

    def _evaluate_intermarket(self, condition: dict, bar) -> bool:
        """Evaluate intermarket condition (leader/follower trigger).

        Monitors leader symbol and triggers based on its movement.
        Requires multi-symbol data subscription.
        """
        leader_symbol = condition.get("leader_symbol")
        trigger_feature = condition.get("trigger_feature", "ret_pct")
        trigger_threshold = condition.get("trigger_threshold", 1.0)
        window_bars = condition.get("window_bars", 20)

        # Note: This is a placeholder - actual multi-symbol implementation
        # requires LEAN multi-symbol data handling
        # Would need leader symbol's price data

        # Placeholder: return False until multi-symbol support is implemented
        self.Log(f"⚠️ IntermarketCondition not fully implemented: {leader_symbol}")
        return False

    def _evaluate_squeeze(self, condition: dict, bar) -> bool:
        """Evaluate squeeze condition (low volatility compression).

        Squeeze detects when Bollinger Bands narrow inside Keltner Channels,
        indicating low volatility that often precedes large moves.
        """
        squeeze_metric = condition.get("squeeze_metric", "bb_width_pctile")
        pctile_threshold = condition.get("pctile_threshold", 20)
        break_rule = condition.get("break_rule", "any")
        with_trend = condition.get("with_trend", False)

        # Get BB and KC indicators
        bb_ind = self.indicators.get("bb") or self.indicators.get("bb_20")
        kc_ind = self.indicators.get("kc") or self.indicators.get("kc_20")

        if not bb_ind:
            return False

        # Check if BB is inside KC (classic squeeze)
        if kc_ind:
            bb_upper = bb_ind.UpperBand.Current.Value
            bb_lower = bb_ind.LowerBand.Current.Value
            kc_upper = kc_ind.UpperBand.Current.Value
            kc_lower = kc_ind.LowerBand.Current.Value

            in_squeeze = bb_lower > kc_lower and bb_upper < kc_upper

            if not in_squeeze:
                return False

        # Check width percentile
        width_rw = self.rolling_windows.get("bb_width")
        if width_rw and width_rw["window"].IsReady:
            upper = bb_ind.UpperBand.Current.Value
            lower = bb_ind.LowerBand.Current.Value
            middle = bb_ind.MiddleBand.Current.Value
            if middle != 0:
                width = (upper - lower) / middle
                widths = list(width_rw["window"])
                pctile = sum(1 for w in widths if w < width) / len(widths) * 100
                if pctile > pctile_threshold:
                    return False

        # Optionally check trend alignment
        if with_trend:
            ema_fast = self.indicators.get("ema_fast") or self.indicators.get("ema_20")
            ema_slow = self.indicators.get("ema_slow") or self.indicators.get("ema_50")
            if ema_fast and ema_slow:
                if ema_fast.Current.Value <= ema_slow.Current.Value:
                    return False  # Not in uptrend

        return True

    def _evaluate_time_filter(self, condition: dict, bar) -> bool:
        """Evaluate time-based filter condition."""
        days_of_week = condition.get("days_of_week", [])
        time_window = condition.get("time_window", "")
        days_of_month = condition.get("days_of_month", [])

        # Check day of week (0=Monday, 6=Sunday)
        if days_of_week:
            current_day = self.Time.weekday()
            if current_day not in days_of_week:
                return False

        # Check day of month
        if days_of_month:
            current_dom = self.Time.day
            if current_dom not in days_of_month:
                return False

        # Check time window (format: "HH:MM-HH:MM")
        if time_window:
            try:
                start_str, end_str = time_window.split("-")
                start_hour, start_min = map(int, start_str.split(":"))
                end_hour, end_min = map(int, end_str.split(":"))

                current_minutes = self.Time.hour * 60 + self.Time.minute
                start_minutes = start_hour * 60 + start_min
                end_minutes = end_hour * 60 + end_min

                if start_minutes <= end_minutes:
                    if not (start_minutes <= current_minutes <= end_minutes):
                        return False
                else:
                    # Overnight window (e.g., 22:00-06:00)
                    if not (current_minutes >= start_minutes or current_minutes <= end_minutes):
                        return False
            except (ValueError, AttributeError):
                pass  # Invalid format, ignore filter

        return True

    def _evaluate_state_condition(self, condition: dict, bar) -> bool:
        """Evaluate a state-based condition with transition tracking.

        StateCondition supports two modes:
        1. Transition tracking: triggers when transitioning from outside_condition to inside_condition
        2. Current condition: simple check of current_condition (no state tracking)

        For transition tracking:
        - Evaluates outside_condition and inside_condition each bar
        - Tracks whether we're "outside" in state[state_var]
        - Returns True only when transitioning from outside to inside
        """
        state_var = condition.get("state_var")
        trigger_on_transition = condition.get("trigger_on_transition", True)
        outside_condition = condition.get("outside_condition")
        inside_condition = condition.get("inside_condition")
        current_condition = condition.get("current_condition")

        # Mode 1: Simple current condition check (no state tracking)
        if current_condition and not outside_condition:
            return self._evaluate_condition(current_condition, bar)

        # Mode 2: Transition tracking
        if outside_condition and inside_condition and state_var:
            was_outside = self.state.get(state_var, False)

            # Evaluate both conditions
            is_outside = self._evaluate_condition(outside_condition, bar)
            is_inside = self._evaluate_condition(inside_condition, bar)

            if is_outside:
                # Track that we're now outside
                self.state[state_var] = True
                return False
            elif was_outside and is_inside:
                # Transition from outside to inside!
                self.state[state_var] = False
                return trigger_on_transition
            elif is_inside:
                # Still inside (but never was outside)
                return False

        # Fallback: simple state value comparison (legacy mode)
        op_str = condition.get("op", "==")
        value = condition.get("value", 0)
        current = self.state.get(state_var)
        if current is None:
            return False
        op = CompareOp(op_str)
        return op.apply(float(current), float(value))

    def _evaluate_gap(self, condition: dict, bar) -> bool:
        """Evaluate a gap condition."""
        typed = TypedGapCondition.from_dict(condition)

        # Get previous close from rolling window
        prev_close_rw = self.rolling_windows.get("prev_close")
        if not prev_close_rw or not prev_close_rw["window"].IsReady:
            return False

        prev_close = prev_close_rw["window"][1]
        if prev_close == 0:
            return False

        # Calculate gap percentage
        gap_pct = (bar.Open - prev_close) / prev_close * 100

        # Check minimum gap threshold
        if abs(gap_pct) < typed.min_gap_pct:
            return False

        # Determine if we should trigger based on mode and direction
        gap_is_up = gap_pct > 0

        if typed.mode == "gap_fade":
            if typed.direction == "long":
                return not gap_is_up
            elif typed.direction == "short":
                return gap_is_up
            else:  # auto
                return True
        else:  # gap_go
            if typed.direction == "long":
                return gap_is_up
            elif typed.direction == "short":
                return not gap_is_up
            else:
                return True

    def _evaluate_trailing_breakout(self, condition: dict, bar) -> bool:
        """Evaluate a trailing breakout condition.

        TrailingBreakoutCondition:
        1. Tracks a state variable that trails the band's edge
        2. For long entries (update_rule="min"): trails DOWN
        3. For short entries (update_rule="max"): trails UP
        4. Triggers when price crosses the trailing level
        """
        band_type = condition.get("band_type", "bollinger")
        band_length = condition.get("band_length", 20)
        band_mult = condition.get("band_mult", 2.0)
        update_rule = condition.get("update_rule", "min")
        band_edge = condition.get("band_edge", "upper")
        trigger_direction = condition.get("trigger_direction", "above")

        # Get the band indicator
        band_id = f"{band_type}_{band_length}"
        band_ind = self.indicators.get(band_id) or self.indicators.get("bb")

        if not band_ind:
            self.Log(f"⚠️ No band indicator found for trailing_breakout: {band_id}")
            return False

        # Get current band edge value
        if band_edge == "upper":
            band_value = band_ind.UpperBand.Current.Value
        elif band_edge == "lower":
            band_value = band_ind.LowerBand.Current.Value
        else:
            band_value = band_ind.MiddleBand.Current.Value

        # State key for trailing level
        state_key = f"trailing_breakout_{band_type}_{band_length}"
        trailing_level = self.state.get(state_key)

        # Initialize trailing level if not set
        if trailing_level is None:
            trailing_level = band_value
            self.state[state_key] = trailing_level

        # Update trailing level based on rule
        if update_rule == "min":
            new_level = min(trailing_level, band_value)
        else:  # max
            new_level = max(trailing_level, band_value)

        self.state[state_key] = new_level

        # Check trigger condition
        if trigger_direction == "above":
            return bar.Close > new_level
        else:  # below
            return bar.Close < new_level

    def _evaluate_trailing_state(self, condition: dict, bar) -> bool:
        """Evaluate a trailing state condition.

        TrailingStateCondition:
        1. Tracks a state variable that trails price
        2. Computes trigger level = state_value ± (atr_mult * ATR)
        3. Triggers when trigger_price crosses the trigger level
        """
        state_id = condition.get("state_id")
        update_rule = condition.get("update_rule", "max")
        update_price = condition.get("update_price", "high")
        trigger_op = condition.get("trigger_op", "below")
        trigger_price = condition.get("trigger_price", "close")
        atr_period = condition.get("atr_period", 20)
        atr_mult = condition.get("atr_mult", 2.0)

        if not state_id:
            return False

        # Get update price value
        if update_price == "high":
            update_val = bar.High
        elif update_price == "low":
            update_val = bar.Low
        elif update_price == "open":
            update_val = bar.Open
        else:
            update_val = bar.Close

        # Get trigger price value
        if trigger_price == "high":
            trigger_val = bar.High
        elif trigger_price == "low":
            trigger_val = bar.Low
        elif trigger_price == "open":
            trigger_val = bar.Open
        else:
            trigger_val = bar.Close

        # Get ATR for offset
        atr_ind = self.indicators.get(f"atr_{atr_period}") or self.indicators.get("atr")
        atr_value = atr_ind.Current.Value if atr_ind else 0.0

        # Get/update trailing state
        trailing_val = self.state.get(state_id)
        if trailing_val is None:
            trailing_val = update_val
            self.state[state_id] = trailing_val
        else:
            if update_rule == "max":
                trailing_val = max(trailing_val, update_val)
            else:  # min
                trailing_val = min(trailing_val, update_val)
            self.state[state_id] = trailing_val

        # Compute trigger level with ATR offset
        if trigger_op == "below":
            trigger_level = trailing_val - (atr_mult * atr_value)
            return trigger_val < trigger_level
        else:  # above
            trigger_level = trailing_val + (atr_mult * atr_value)
            return trigger_val > trigger_level

    def _evaluate_sequence(self, condition: dict, bar) -> bool:
        """Evaluate a sequence condition.

        SequenceCondition: steps must occur in order with optional timing constraints.
        Tracks which step we're waiting for in state.
        """
        steps = condition.get("steps", [])
        if not steps:
            return True

        # State key for sequence progress
        seq_key = f"sequence_{id(condition)}"
        current_step = self.state.get(seq_key, 0)
        step_bar_count = self.state.get(f"{seq_key}_bars", 0)

        if current_step >= len(steps):
            # All steps completed - reset and return True
            self.state[seq_key] = 0
            self.state[f"{seq_key}_bars"] = 0
            return True

        step = steps[current_step]
        step_condition = step.get("condition")
        hold_bars = step.get("hold_bars")
        within_bars = step.get("within_bars")

        # Check within_bars timeout (only for steps after the first)
        if current_step > 0 and within_bars is not None:
            if step_bar_count > within_bars:
                # Timeout - reset sequence
                self.state[seq_key] = 0
                self.state[f"{seq_key}_bars"] = 0
                return False

        # Increment bar count for timing constraints
        self.state[f"{seq_key}_bars"] = step_bar_count + 1

        # Evaluate current step's condition
        if self._evaluate_condition(step_condition, bar):
            if hold_bars is not None:
                # Need to hold for N bars
                hold_count = self.state.get(f"{seq_key}_hold", 0) + 1
                self.state[f"{seq_key}_hold"] = hold_count
                if hold_count >= hold_bars:
                    # Step complete - advance
                    self.state[seq_key] = current_step + 1
                    self.state[f"{seq_key}_bars"] = 0
                    self.state[f"{seq_key}_hold"] = 0
            else:
                # No hold required - advance immediately
                self.state[seq_key] = current_step + 1
                self.state[f"{seq_key}_bars"] = 0
        else:
            # Reset hold counter if condition not met
            self.state[f"{seq_key}_hold"] = 0

        return False  # Only return True when all steps complete

    def _evaluate_event_window(self, condition: dict, bar) -> bool:
        """Evaluate an event window condition.

        EventWindowCondition checks if current time is within a window around events.
        Note: Requires event calendar data which may not be available in all contexts.
        """
        event_types = condition.get("event_types", [])
        pre_window_bars = condition.get("pre_window_bars", 0)
        post_window_bars = condition.get("post_window_bars", 0)
        mode = condition.get("mode", "within")

        # Check if we have event data
        events = self.state.get("_event_calendar", {})

        if not events:
            # No event data - return based on mode
            # "within" with no events = False, "outside" with no events = True
            return mode == "outside"

        # Check each event type
        for event_type in event_types:
            event_list = events.get(event_type, [])
            for event in event_list:
                event_bar = event.get("bar_index", -1)
                current_bar = self.state.get("_bar_count", 0)

                # Check if we're in the window
                bars_before = current_bar - event_bar
                if -pre_window_bars <= bars_before <= post_window_bars:
                    # We're in the window
                    return mode == "within"

        # Not in any event window
        return mode == "outside"

    def _evaluate_multi_leader_intermarket(self, condition: dict, bar) -> bool:
        """Evaluate a multi-leader intermarket condition.

        Aggregates signals from multiple leader symbols and triggers based on
        the aggregated value exceeding threshold.
        """
        leader_symbols = condition.get("leader_symbols", [])
        follower_symbol = condition.get("follower_symbol", "")
        aggregate_feature = condition.get("aggregate_feature", "ret_pct")
        aggregate_op = condition.get("aggregate_op", "avg")
        trigger_threshold = condition.get("trigger_threshold", 1.0)
        window_bars = condition.get("window_bars", 20)
        direction = condition.get("direction", "same")

        if not leader_symbols:
            return False

        # Collect feature values from each leader
        feature_values = []
        for leader in leader_symbols:
            # Get ROC indicator for this leader (set up by translator)
            roc_id = f"roc_{leader}_{window_bars}"
            roc_ind = self.indicators.get(roc_id)

            if roc_ind:
                if aggregate_feature == "ret_pct":
                    feature_values.append(roc_ind.Current.Value * 100)
                else:
                    feature_values.append(roc_ind.Current.Value)

        if not feature_values:
            return False

        # Aggregate the values
        if aggregate_op == "max":
            aggregated = max(feature_values)
        elif aggregate_op == "min":
            aggregated = min(feature_values)
        else:  # avg
            aggregated = sum(feature_values) / len(feature_values)

        # Check threshold
        triggered = aggregated > trigger_threshold

        return triggered

    def _evaluate_liquidity_sweep_typed(self, condition: dict, bar) -> bool:
        """Evaluate typed LiquiditySweepCondition.

        Uses typed parameters from the condition rather than regime.metric dispatch.
        """
        level_type = condition.get("level_type", "rolling_min")
        level_period = condition.get("level_period", 20)
        lookback_bars = condition.get("lookback_bars", 3)

        # Get the level indicator based on type
        if level_type == "rolling_min":
            level_ind = self.rolling_minmax.get(f"min_{level_period}") or self.rolling_minmax.get("level_min")
            if level_ind and level_ind["window"].IsReady:
                level_value = min(list(level_ind["window"]))
            else:
                return False
        elif level_type == "rolling_max":
            level_ind = self.rolling_minmax.get(f"max_{level_period}") or self.rolling_minmax.get("level_max")
            if level_ind and level_ind["window"].IsReady:
                level_value = max(list(level_ind["window"]))
            else:
                return False
        else:
            # Fixed level - should be provided
            level_value = condition.get("fixed_level")
            if level_value is None:
                return False

        # State tracking
        state_key = f"liquidity_sweep_{level_period}"
        sweep_triggered = self.state.get(f"{state_key}_triggered", False)
        sweep_bar_count = self.state.get(f"{state_key}_bars", 0)

        if not sweep_triggered:
            # Check if price broke below level (sweep)
            if bar.Low < level_value:
                self.state[f"{state_key}_triggered"] = True
                self.state[f"{state_key}_bars"] = 0
                self.state[f"{state_key}_level"] = level_value
        else:
            # Increment bar count
            sweep_bar_count += 1
            self.state[f"{state_key}_bars"] = sweep_bar_count

            if sweep_bar_count > lookback_bars:
                # Timeout - reset
                self.state[f"{state_key}_triggered"] = False
                return False

            # Check if price reclaimed above level
            sweep_level = self.state.get(f"{state_key}_level", level_value)
            if bar.Close > sweep_level:
                # Sweep complete - reset and signal
                self.state[f"{state_key}_triggered"] = False
                return True

        return False

    def _evaluate_flag_pattern_typed(self, condition: dict, bar) -> bool:
        """Evaluate typed FlagPatternCondition.

        Uses typed parameters from the condition.
        """
        momentum_threshold = condition.get("momentum_threshold", 5.0)
        momentum_period = condition.get("momentum_period", 10)
        consolidation_bars = condition.get("consolidation_bars", 5)
        breakout_direction = condition.get("breakout_direction", "same")

        # State tracking
        state_key = "flag_pattern"
        phase = self.state.get(f"{state_key}_phase", "scanning")
        pole_direction = self.state.get(f"{state_key}_direction", 0)
        consol_count = self.state.get(f"{state_key}_consol_count", 0)

        # Get momentum ROC indicator
        roc_id = f"roc_{momentum_period}"
        roc_ind = self.indicators.get(roc_id) or self.indicators.get("momentum_roc")

        if not roc_ind:
            return False

        current_roc = roc_ind.Current.Value * 100

        if phase == "scanning":
            # Look for strong momentum
            if abs(current_roc) >= momentum_threshold:
                self.state[f"{state_key}_phase"] = "consolidating"
                self.state[f"{state_key}_direction"] = 1 if current_roc > 0 else -1
                self.state[f"{state_key}_consol_count"] = 0
                self.state[f"{state_key}_high"] = bar.High
                self.state[f"{state_key}_low"] = bar.Low
            return False

        elif phase == "consolidating":
            # Track consolidation (narrowing range)
            prev_high = self.state.get(f"{state_key}_high", bar.High)
            prev_low = self.state.get(f"{state_key}_low", bar.Low)

            # Update range
            self.state[f"{state_key}_high"] = max(prev_high, bar.High)
            self.state[f"{state_key}_low"] = min(prev_low, bar.Low)

            consol_count += 1
            self.state[f"{state_key}_consol_count"] = consol_count

            if consol_count < consolidation_bars:
                return False

            # Check for breakout
            pole_dir = self.state.get(f"{state_key}_direction", 1)
            expected_dir = pole_dir if breakout_direction == "same" else -pole_dir

            if expected_dir > 0 and bar.Close > prev_high:
                # Bullish breakout
                self.state[f"{state_key}_phase"] = "scanning"
                return True
            elif expected_dir < 0 and bar.Close < prev_low:
                # Bearish breakout
                self.state[f"{state_key}_phase"] = "scanning"
                return True

            # Check for pattern failure (ROC reverses strongly)
            if abs(current_roc) >= momentum_threshold * 0.5:
                if (current_roc > 0 and pole_dir < 0) or (current_roc < 0 and pole_dir > 0):
                    # Pattern failed - reset
                    self.state[f"{state_key}_phase"] = "scanning"

        return False

    def _evaluate_pennant_pattern_typed(self, condition: dict, bar) -> bool:
        """Evaluate typed PennantPatternCondition.

        Similar to flag but with converging trendlines.
        """
        momentum_threshold = condition.get("momentum_threshold", 5.0)
        momentum_period = condition.get("momentum_period", 10)
        consolidation_bars = condition.get("consolidation_bars", 5)
        breakout_direction = condition.get("breakout_direction", "same")

        # State tracking
        state_key = "pennant_pattern"
        phase = self.state.get(f"{state_key}_phase", "scanning")

        # Get momentum ROC indicator
        roc_id = f"roc_{momentum_period}"
        roc_ind = self.indicators.get(roc_id) or self.indicators.get("momentum_roc")

        if not roc_ind:
            return False

        current_roc = roc_ind.Current.Value * 100

        if phase == "scanning":
            # Look for strong momentum
            if abs(current_roc) >= momentum_threshold:
                self.state[f"{state_key}_phase"] = "consolidating"
                self.state[f"{state_key}_direction"] = 1 if current_roc > 0 else -1
                self.state[f"{state_key}_consol_count"] = 0
                self.state[f"{state_key}_highs"] = [bar.High]
                self.state[f"{state_key}_lows"] = [bar.Low]
            return False

        elif phase == "consolidating":
            # Track consolidation with converging highs/lows
            highs = self.state.get(f"{state_key}_highs", [])
            lows = self.state.get(f"{state_key}_lows", [])

            highs.append(bar.High)
            lows.append(bar.Low)
            self.state[f"{state_key}_highs"] = highs
            self.state[f"{state_key}_lows"] = lows

            consol_count = len(highs)
            self.state[f"{state_key}_consol_count"] = consol_count

            if consol_count < consolidation_bars:
                return False

            # Check for converging pattern (lower highs and higher lows)
            if len(highs) >= 3:
                # Simple convergence check
                is_converging = (highs[-1] < highs[0]) and (lows[-1] > lows[0])

                if is_converging:
                    # Check for breakout
                    pole_dir = self.state.get(f"{state_key}_direction", 1)
                    expected_dir = pole_dir if breakout_direction == "same" else -pole_dir

                    recent_high = max(highs[-3:])
                    recent_low = min(lows[-3:])

                    if expected_dir > 0 and bar.Close > recent_high:
                        # Bullish breakout
                        self.state[f"{state_key}_phase"] = "scanning"
                        return True
                    elif expected_dir < 0 and bar.Close < recent_low:
                        # Bearish breakout
                        self.state[f"{state_key}_phase"] = "scanning"
                        return True

            # Timeout check
            if consol_count > consolidation_bars * 3:
                self.state[f"{state_key}_phase"] = "scanning"

        return False

    def _resolve_value(self, value_ref: dict, bar) -> float:
        """Resolve a value reference to a float."""
        if not value_ref:
            return 0.0

        val_type = value_ref.get("type")

        if val_type == "literal":
            return float(value_ref.get("value", 0))

        elif val_type == "indicator":
            ind_id = value_ref.get("indicator_id")
            field = value_ref.get("field")  # Optional field for multi-value indicators
            # First check regular indicators
            ind = self.indicators.get(ind_id)
            if ind:
                # Handle MACD fields
                if field == "signal" and hasattr(ind, 'Signal'):
                    return ind.Signal.Current.Value
                elif field == "histogram" and hasattr(ind, 'Histogram'):
                    return ind.Histogram.Current.Value
                elif field == "macd" and hasattr(ind, 'Fast'):
                    # MACD line = Fast - Slow internally, but .Current.Value gives us the MACD line
                    return ind.Current.Value
                # Default: return the main value
                return ind.Current.Value
            # Check volume SMA indicators
            vol_data = self.vol_sma_indicators.get(ind_id)
            if vol_data:
                return vol_data["sma"].Current.Value
            # Check rolling min/max indicators
            rmm_data = self.rolling_minmax.get(ind_id)
            if rmm_data:
                window = rmm_data["window"]
                if window.IsReady:
                    if rmm_data["mode"] == "min":
                        return min(list(window))
                    else:  # max
                        return max(list(window))
                return 0.0
            self.Log(f"⚠️ Unknown indicator: {ind_id}")
            return 0.0

        elif val_type == "indicator_band":
            ind_id = value_ref.get("indicator_id")
            band = value_ref.get("band")
            ind = self.indicators.get(ind_id)
            if ind:
                if band == "upper":
                    return ind.UpperBand.Current.Value
                elif band == "middle":
                    return ind.MiddleBand.Current.Value
                elif band == "lower":
                    return ind.LowerBand.Current.Value
            return 0.0

        elif val_type == "indicator_property":
            ind_id = value_ref.get("indicator_id")
            prop = value_ref.get("property")
            ind = self.indicators.get(ind_id)
            if ind:
                if prop == "StandardDeviation":
                    # Bollinger Bands have StandardDeviation property
                    return ind.StandardDeviation.Current.Value
                elif prop == "BandWidth":
                    # Band width calculation
                    if hasattr(ind, 'BandWidth'):
                        return ind.BandWidth.Current.Value
                    # Fallback: calculate manually
                    upper = ind.UpperBand.Current.Value
                    lower = ind.LowerBand.Current.Value
                    middle = ind.MiddleBand.Current.Value
                    if middle != 0:
                        return (upper - lower) / middle
            return 0.0

        elif val_type == "price":
            field = value_ref.get("field", "close")
            if field == "open":
                return bar.Open
            elif field == "high":
                return bar.High
            elif field == "low":
                return bar.Low
            else:  # close
                return bar.Close

        elif val_type == "volume":
            # Current bar's volume
            return float(bar.Volume)

        elif val_type == "time":
            # Time component from current bar
            component = value_ref.get("component", "hour")
            if component == "hour":
                return float(self.Time.hour)
            elif component == "minute":
                return float(self.Time.minute)
            elif component == "day_of_week":
                return float(self.Time.weekday())  # 0=Monday, 6=Sunday
            return 0.0

        elif val_type == "state":
            state_id = value_ref.get("state_id")
            val = self.state.get(state_id)
            if val is None:
                return 0.0
            return float(val)

        elif val_type == "expr":
            op = value_ref.get("op")
            left = self._resolve_value(value_ref.get("left"), bar)
            right = self._resolve_value(value_ref.get("right"), bar)

            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                if right == 0:
                    return 0.0
                return left / right

        self.Log(f"⚠️ Unknown value type: {val_type}")
        return 0.0

    def _evaluate_liquidity_sweep(self, regime: dict, bar) -> bool:
        """Evaluate liquidity sweep pattern.

        Liquidity sweep: price breaks below a level (taking out stops),
        then reclaims above it within N bars.

        State tracking: Uses state vars 'sweep_triggered' and 'sweep_bar_count'
        """
        lookback_bars = regime.get("lookback_bars", 3)

        # Get the level indicator (set up by translator)
        level_min = self.rolling_minmax.get("level_min")
        level_max = self.rolling_minmax.get("level_max")

        level_value = None
        if level_min and level_min["window"].IsReady:
            level_value = min(list(level_min["window"]))
        elif level_max and level_max["window"].IsReady:
            level_value = max(list(level_max["window"]))

        if level_value is None:
            return False

        # Check state
        sweep_triggered = self.state.get("sweep_triggered", False)
        sweep_bar_count = self.state.get("sweep_bar_count", 0)

        if not sweep_triggered:
            # Check if price broke below level (sweep)
            if bar.Low < level_value:
                self.state["sweep_triggered"] = True
                self.state["sweep_bar_count"] = 0
                self.state["sweep_level"] = level_value
        else:
            # Increment bar count
            sweep_bar_count += 1
            self.state["sweep_bar_count"] = sweep_bar_count

            if sweep_bar_count > lookback_bars:
                # Timeout - reset
                self.state["sweep_triggered"] = False
                return False

            # Check if price reclaimed above level
            sweep_level = self.state.get("sweep_level", level_value)
            if bar.Close > sweep_level:
                # Sweep complete - reset and signal
                self.state["sweep_triggered"] = False
                return True

        return False

    def _evaluate_flag_pattern(self, regime: dict, bar) -> bool:
        """Evaluate flag pattern.

        Flag pattern: Initial strong momentum move + consolidation with
        narrowing range + breakout in direction of initial move.
        """
        breakout_dir = regime.get("value", "same")  # "same" or "opposite"

        # Get indicators (set up by translator)
        momentum_roc = self.indicators.get("momentum_roc")
        pattern_atr = self.indicators.get("pattern_atr")
        pattern_max = self.indicators.get("pattern_max")
        pattern_min = self.indicators.get("pattern_min")

        if not all([momentum_roc, pattern_atr, pattern_max, pattern_min]):
            return False

        roc_value = momentum_roc.Current.Value
        atr_value = pattern_atr.Current.Value
        range_high = pattern_max.Current.Value
        range_low = pattern_min.Current.Value

        # Check for consolidation (narrowing range relative to ATR)
        current_range = range_high - range_low
        if atr_value == 0 or current_range / atr_value > 2.0:
            # Not consolidating enough
            return False

        # Check for momentum direction
        initial_momentum_up = roc_value > 0.02  # 2% momentum
        initial_momentum_down = roc_value < -0.02

        if not (initial_momentum_up or initial_momentum_down):
            # No clear momentum
            return False

        # Check breakout
        if breakout_dir == "same":
            if initial_momentum_up and bar.Close > range_high:
                return True
            if initial_momentum_down and bar.Close < range_low:
                return True
        else:  # opposite
            if initial_momentum_up and bar.Close < range_low:
                return True
            if initial_momentum_down and bar.Close > range_high:
                return True

        return False

    def _evaluate_pennant_pattern(self, regime: dict, bar) -> bool:
        """Evaluate pennant pattern.

        Similar to flag but with triangular (converging) consolidation.
        """
        # For now, use same logic as flag - true triangular detection
        # would require tracking converging highs/lows over time
        return self._evaluate_flag_pattern(regime, bar)

    def _evaluate_price_level(self, metric: str, regime: dict, bar) -> bool:
        """Evaluate price level touch/cross for dynamic levels.

        This handles cases where the translator couldn't fully lower the condition.
        """
        level_ref = regime.get("value", "")
        direction = "up" if "_up" in level_ref else "down"

        # Get dynamic level from indicators
        level_min = self.rolling_minmax.get("level_min")
        level_max = self.rolling_minmax.get("level_max")

        level_value = None
        if level_min and level_min["window"].IsReady:
            level_value = min(list(level_min["window"]))
        elif level_max and level_max["window"].IsReady:
            level_value = max(list(level_max["window"]))

        if level_value is None:
            return False

        if metric == "price_level_touch":
            # Check if bar touches the level
            return bar.Low <= level_value <= bar.High
        else:  # price_level_cross
            if direction == "up":
                return bar.Close > level_value
            else:
                return bar.Close < level_value

    def _execute_action(self, action: dict, bar=None):
        """Execute an action from IR.

        Supports multiple sizing modes for set_holdings:
        - pct_equity: uses allocation field directly (fraction of portfolio)
        - fixed_usd: fixed USD amount, converted to quantity at current price
        - fixed_units: fixed number of units/shares to trade

        Note: For all sizing modes, we use CalculateOrderQuantity + MarketOrder
        instead of SetHoldings to ensure small orders are properly executed.
        See: https://www.quantconnect.com/forum/discussion/2978/minimum-order-clip-size/

        Args:
            action: The action dict from IR (type, sizing_mode, allocation, etc.)
            bar: The current bar data (used for fixed_usd/fixed_units pricing)
        """
        if not action:
            return

        action_type = action.get("type")

        if action_type == "set_holdings":
            sizing_mode = action.get("sizing_mode", "pct_equity")

            if sizing_mode == "pct_equity":
                # Use CalculateOrderQuantity + MarketOrder instead of SetHoldings
                # This ensures small orders are executed rather than silently skipped
                allocation = action.get("allocation", 0.95)
                quantity = self.CalculateOrderQuantity(self.symbol, allocation)
                if quantity != 0:
                    self.MarketOrder(self.symbol, quantity)
                else:
                    self.Log(f"⚠️ Order quantity is zero for allocation={allocation}, skipping order")

            elif sizing_mode == "fixed_usd":
                # Fixed USD amount - convert to portfolio allocation
                # We use CalculateOrderQuantity to ensure LEAN handles the order properly
                fixed_usd = action.get("fixed_usd", 1000.0)
                portfolio_value = self.Portfolio.TotalPortfolioValue
                if portfolio_value > 0:
                    allocation = fixed_usd / portfolio_value
                    quantity = self.CalculateOrderQuantity(self.symbol, allocation)
                    if quantity != 0:
                        self.MarketOrder(self.symbol, quantity)
                        price = self.Securities[self.symbol].Price
                        self.Log(f"   Fixed USD: ${abs(fixed_usd):.2f} -> {abs(quantity):.6f} units @ ${price:.2f}")
                    else:
                        self.Log(f"⚠️ Order quantity is zero for fixed_usd=${fixed_usd}, skipping order")
                else:
                    self.Log(f"⚠️ Cannot execute fixed_usd: portfolio value is {portfolio_value}")

            elif sizing_mode == "fixed_units":
                # Fixed number of units - use CalculateOrderQuantity for proper handling
                fixed_units = action.get("fixed_units", 1.0)
                price = float(bar.Close) if bar else self.Securities[self.symbol].Price
                if price > 0:
                    portfolio_value = self.Portfolio.TotalPortfolioValue
                    if portfolio_value > 0:
                        allocation = (fixed_units * price) / portfolio_value
                        quantity = self.CalculateOrderQuantity(self.symbol, allocation)
                        if quantity != 0:
                            self.MarketOrder(self.symbol, quantity)
                            self.Log(f"   Fixed units: {abs(quantity):.6f}")
                        else:
                            self.Log(f"⚠️ Order quantity is zero for fixed_units, skipping order")
                    else:
                        self.Log(f"⚠️ Cannot execute fixed_units: portfolio value is {portfolio_value}")
                else:
                    self.Log(f"⚠️ Cannot execute fixed_units: price is {price}")

            else:
                self.Log(f"⚠️ Unknown sizing_mode: {sizing_mode}")

        elif action_type == "liquidate":
            self.Liquidate(self.symbol)

        elif action_type == "market_order":
            quantity = action.get("quantity", 0)
            if quantity != 0:
                self.MarketOrder(self.symbol, quantity)
            else:
                self.Log(f"⚠️ Order quantity is zero for market_order, skipping order")

        else:
            self.Log(f"⚠️ Unknown action type: {action_type}")

    def _execute_state_op(self, op: dict, bar):
        """Execute a state operation."""
        if not op:
            return

        op_type = op.get("type")
        state_id = op.get("state_id")

        if op_type == "set_state":
            value_ref = op.get("value")
            value = self._resolve_value(value_ref, bar)
            self.state[state_id] = value

        elif op_type == "increment":
            current = self.state.get(state_id, 0) or 0
            self.state[state_id] = current + 1

        elif op_type == "max_state":
            value_ref = op.get("value")
            new_value = self._resolve_value(value_ref, bar)
            current = self.state.get(state_id)
            if current is None or new_value > current:
                self.state[state_id] = new_value

        elif op_type == "set_state_from_condition":
            condition = op.get("condition")
            result = self._evaluate_condition(condition, bar)
            # Store as 1.0 or 0.0 for float compatibility
            self.state[state_id] = 1.0 if result else 0.0

        else:
            self.Log(f"⚠️ Unknown state op type: {op_type}")

    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        # Close any open position and record as trade
        if self.current_trade and self.Portfolio.Invested:
            # Get last known price and apply slippage
            raw_price = float(self.Portfolio[self.symbol].Price)
            entry_price = self.current_trade["entry_price"]
            quantity = self.current_trade.get("quantity", 0)
            direction = self.current_trade.get("direction", "long")
            entry_fee = self.current_trade.get("entry_fee", 0)

            # Apply slippage to exit price
            # For long exit: selling, so receive less (slippage down)
            # For short exit: buying/covering, so pay more (slippage up)
            is_buy_exit = (direction == "short")  # Short covers with a buy
            last_price = self._apply_slippage(raw_price, is_buy_exit)

            # Calculate exit fee
            exit_value = last_price * quantity
            exit_fee = self._calculate_fee(exit_value)
            total_fees = entry_fee + exit_fee

            # PnL calculation: short profits when price drops
            # Subtract total fees from PnL
            if direction == "short":
                gross_pnl = (entry_price - last_price) * quantity
            else:
                gross_pnl = (last_price - entry_price) * quantity

            pnl = gross_pnl - total_fees
            # Calculate pnl_pct based on entry value
            entry_value = entry_price * quantity
            pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

            self.current_trade["exit_time"] = str(self.Time)
            self.current_trade["exit_price"] = last_price
            # bar_count has been incremented past the last bar, so use -1
            self.current_trade["exit_bar"] = self.bar_count - 1
            self.current_trade["pnl"] = pnl
            self.current_trade["pnl_percent"] = pnl_pct
            self.current_trade["exit_fee"] = exit_fee
            self.current_trade["total_fees"] = total_fees
            self.current_trade["exit_reason"] = "end_of_backtest"

            self.trades.append(self.current_trade)
            self.current_trade = None
            self.Log(f"📊 Closed open position at end: ${last_price:.2f}")

        initial_cash = float(self.GetParameter("initial_cash") or 100000)

        # Calculate final equity from trade PnLs (includes manual fee calculations)
        # This is more accurate than LEAN's portfolio value which doesn't include our manual fees
        total_trade_pnl = sum(t.get("pnl", 0) for t in self.trades)
        final_equity = initial_cash + total_trade_pnl

        # Also get LEAN's portfolio value for comparison/logging
        lean_portfolio_value = self.Portfolio.TotalPortfolioValue

        # Calculate statistics using fee-adjusted final equity
        total_return = ((final_equity / initial_cash) - 1) * 100
        winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("pnl", 0) <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        avg_win = sum(t.get("pnl_percent", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get("pnl_percent", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Log summary
        self.Log("")
        self.Log(f"{'='*60}")
        self.Log(f"📊 BACKTEST RESULTS: {self.ir.get('strategy_name', 'Unknown')}")
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
        self.Log(f"  Total Trades:       {len(self.trades)}")
        self.Log(f"  Winning Trades:     {len(winning_trades)}")
        self.Log(f"  Losing Trades:      {len(losing_trades)}")
        self.Log(f"  Win Rate:           {win_rate:.1f}%")
        self.Log(f"  Avg Win:            {avg_win:+.2f}%")
        self.Log(f"  Avg Loss:           {avg_loss:+.2f}%")
        self.Log(f"  Profit Factor:      {profit_factor:.2f}")
        self.Log("")

        # Log individual trades
        if self.trades:
            self.Log("TRADE LOG")
            for i, t in enumerate(self.trades):
                self.Log(f"  #{i+1}: {t['direction'].upper()} @ ${t['entry_price']:.2f} -> ${t.get('exit_price', 0):.2f} | P&L: {t.get('pnl_percent', 0):+.2f}% | Exit: {t.get('exit_reason', 'N/A')}")
            self.Log("")

        self.Log(f"{'='*60}")

        # Write structured output to JSON file
        import os
        output = {
            "strategy_id": self.ir.get("strategy_id", "unknown"),
            "strategy_name": self.ir.get("strategy_name", "Unknown"),
            "symbol": str(self.symbol),
            "initial_cash": initial_cash,
            "final_equity": float(final_equity),
            "total_return_pct": total_return,
            "max_drawdown_pct": self.max_drawdown,
            "statistics": {
                "total_trades": len(self.trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "profit_factor": profit_factor,
            },
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

        # Write to data folder (same location as strategy_ir.json and debug.log)
        output_path = os.path.join(CustomCryptoData.DataFolder, "strategy_output.json")
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            self.Log(f"📁 Results written to: {output_path}")
        except Exception as e:
            self.Log(f"⚠️ Failed to write output: {e}")

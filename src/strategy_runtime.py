"""Strategy Runtime for executing StrategyIR in LEAN.

This algorithm loads a StrategyIR from JSON and executes it:
1. Creates LEAN indicators based on IR requirements
2. Evaluates conditions each bar using the evaluator
3. Executes entry/exit actions

For live trading, this uses the same evaluator as backtesting, ensuring
consistent behavior between backtest and production.

Data Architecture:
- All data comes from Vibe Trade sources (BigQuery for backtest, Pub/Sub for live)
- VibeTradeCryptoData custom class reads our CSV format directly
- No dependency on LEAN's built-in crypto data or currency conversion
"""

from AlgorithmImports import *
import os
import json
from dataclasses import dataclass
from typing import Any
from datetime import datetime, timedelta


# =============================================================================
# Custom Data Class for Vibe Trade Data
# =============================================================================


class VibeTradeCryptoData(PythonData):
    """Custom data class for loading Vibe Trade crypto data."""

    def default_resolution(self):
        """Return the default resolution for this data type."""
        return Resolution.Minute

    def is_sparse_data(self):
        """Data is not sparse - we have data for every minute."""
        return False

    def data_time_zone(self):
        """Return the timezone of the data source (UTC for crypto)."""
        return TimeZones.Utc

    def get_source(self, config, date, is_live_mode):
        """Return the data source for the given date."""
        symbol = config.symbol.value.lower()
        date_str = date.strftime("%Y%m%d")
        source = f"/Data/custom/{symbol}/{date_str}.csv"
        return SubscriptionDataSource(source, SubscriptionTransportMedium.LocalFile, FileFormat.Csv)

    def reader(self, config, line, date, is_live_mode):
        """Parse a line of CSV data into a data point."""
        if not line or not line.strip():
            return None
        
        if not line[0].isdigit():
            return None

        try:
            parts = line.split(',')
            if len(parts) < 6:
                return None

            ms_since_midnight = int(parts[0])
            
            coin = VibeTradeCryptoData()
            coin.symbol = config.symbol
            # Set time for this bar
            bar_time = date + timedelta(milliseconds=ms_since_midnight)
            coin.time = bar_time
            # EndTime is when the bar is complete and available to the algorithm
            coin.end_time = bar_time + timedelta(minutes=1)
            coin.value = float(parts[4])
            coin["Open"] = float(parts[1])
            coin["High"] = float(parts[2])
            coin["Low"] = float(parts[3])
            coin["Close"] = float(parts[4])
            coin["Volume"] = float(parts[5])

            return coin

        except Exception:
            return None

# =============================================================================
# Indicator Registry - Declarative Indicator Definitions
# =============================================================================
# This registry centralizes all indicator metadata, eliminating if/else sprawl
# in creation, update, and value resolution methods.

# Update modes for indicators
UPDATE_SCALAR = "scalar"      # Update with single value (close, volume, etc.)
UPDATE_TRADEBAR = "tradebar"  # Update with TradeBar (OHLCV)

INDICATOR_REGISTRY = {
    # Moving Averages - scalar update with close
    "EMA": {
        "class_name": "ExponentialMovingAverage",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },
    "SMA": {
        "class_name": "SimpleMovingAverage",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },

    # Bands - tradebar update, have upper/middle/lower
    "BB": {
        "class_name": "BollingerBands",
        "params": ["period", "multiplier"],
        "param_defaults": {"multiplier": 2.0},
        "update_mode": UPDATE_TRADEBAR,
        "bands": ["upper", "middle", "lower"],
        "properties": ["StandardDeviation"],
    },
    "KC": {
        "class_name": "KeltnerChannels",
        "params": ["period", "multiplier"],
        "param_defaults": {"multiplier": 2.0},
        "update_mode": UPDATE_TRADEBAR,
        "bands": ["upper", "middle", "lower"],
    },
    "DC": {
        "class_name": "DonchianChannel",
        "params": ["period"],
        "update_mode": UPDATE_TRADEBAR,
        "bands": ["upper", "lower"],  # No middle band
    },

    # Volatility - tradebar update
    "ATR": {
        "class_name": "AverageTrueRange",
        "params": ["period"],
        "update_mode": UPDATE_TRADEBAR,
    },
    "ADX": {
        "class_name": "AverageDirectionalIndex",
        "params": ["period"],
        "update_mode": UPDATE_TRADEBAR,
    },

    # Min/Max - scalar update with close
    "MAX": {
        "class_name": "Maximum",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },
    "MIN": {
        "class_name": "Minimum",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },

    # Momentum - scalar update with close
    "ROC": {
        "class_name": "RateOfChange",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },

    # Volume-based - tradebar update
    "VWAP": {
        "class_name": "VolumeWeightedAveragePriceIndicator",
        "params": [],
        "update_mode": UPDATE_TRADEBAR,
    },

    # Volume SMA - scalar update with volume
    "VOL_SMA": {
        "class_name": "SimpleMovingAverage",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "volume",  # Uses volume, not close
    },

    # Rolling Window - special handling
    "RW": {
        "class_name": "RollingWindow",
        "params": ["period"],
        "update_mode": "rolling_window",  # Special mode
        "default_field": "close",
    },

    # RSI - scalar update with close
    "RSI": {
        "class_name": "RelativeStrengthIndex",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },

    # MACD - scalar update with close, has signal and histogram
    "MACD": {
        "class_name": "MovingAverageConvergenceDivergence",
        "params": ["fast_period", "slow_period", "signal_period"],
        "param_defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
        "properties": ["Signal", "Histogram", "Fast", "Slow"],
    },

    # Stochastic - tradebar update, has K and D lines
    "STOCH": {
        "class_name": "Stochastic",
        "params": ["period", "k_period", "d_period"],
        "param_defaults": {"period": 14, "k_period": 3, "d_period": 3},
        "update_mode": UPDATE_TRADEBAR,
        "properties": ["StochK", "StochD"],
    },

    # Momentum - scalar update with close
    "MOM": {
        "class_name": "Momentum",
        "params": ["period"],
        "update_mode": UPDATE_SCALAR,
        "default_field": "close",
    },

    # CCI - tradebar update
    "CCI": {
        "class_name": "CommodityChannelIndex",
        "params": ["period"],
        "update_mode": UPDATE_TRADEBAR,
    },

    # Williams %R - tradebar update
    "WILLR": {
        "class_name": "WilliamsPercentR",
        "params": ["period"],
        "update_mode": UPDATE_TRADEBAR,
    },
}

# Map class names to actual LEAN classes (populated at runtime)
INDICATOR_CLASSES = {}


def _init_indicator_classes():
    """Initialize the indicator class mapping from LEAN imports."""
    global INDICATOR_CLASSES
    INDICATOR_CLASSES = {
        # Moving Averages
        "ExponentialMovingAverage": ExponentialMovingAverage,
        "SimpleMovingAverage": SimpleMovingAverage,
        # Bands
        "BollingerBands": BollingerBands,
        "KeltnerChannels": KeltnerChannels,
        "DonchianChannel": DonchianChannel,
        # Volatility
        "AverageTrueRange": AverageTrueRange,
        "AverageDirectionalIndex": AverageDirectionalIndex,
        # Min/Max
        "Maximum": Maximum,
        "Minimum": Minimum,
        # Momentum
        "RateOfChange": RateOfChange,
        "RelativeStrengthIndex": RelativeStrengthIndex,
        "MovingAverageConvergenceDivergence": MovingAverageConvergenceDivergence,
        "Stochastic": Stochastic,
        "Momentum": Momentum,
        "CommodityChannelIndex": CommodityChannelIndex,
        "WilliamsPercentR": WilliamsPercentR,
        # Volume
        "VolumeWeightedAveragePriceIndicator": VolumeWeightedAveragePriceIndicator,
        # Rolling Window
        "RollingWindow": RollingWindow,
    }


# Initialize on import
_init_indicator_classes()


# Import Pydantic IR models from vibe-trade-shared
# These are REQUIRED - the runtime uses typed models, not dicts
from vibe_trade_shared.models.ir import (
    # Value references
    IndicatorRef,
    IndicatorBandRef,
    PriceRef,
    VolumeRef,
    LiteralRef,
    StateRef,
    TimeRef,
    IRExpression,
    ValueRef,
    # Conditions
    IRCompare,
    IRAllOf,
    IRAnyOf,
    IRNot,
    IRCross,
    IRCondition,
    # Strategy structure
    StrategyIR,
    IndicatorSpec,
    EntryRule,
    ExitRule,
    SetHoldingsAction,
    LiquidateAction,
    SetStateAction,
    IncrementStateAction,
    MaxStateAction,
    # Strategy condition types (used by StrategyIR entry/exit)
    CompareCondition,
    AllOfCondition,
    AnyOfCondition,
    NotCondition,
    TimeFilterCondition,
    CrossCondition,
)
from vibe_trade_shared.models.ir.strategy_ir import OnFillAction, OnBarAction
from vibe_trade_shared.models.ir.enums import CompareOp, PriceField, CrossDirection

# Check if vibe-trade-execution package is installed (for logging purposes)
# The inline evaluator is always used since it handles dict-based IR from JSON
try:
    import vibe_trade_execution  # noqa: F401
    USE_PACKAGE_EVALUATOR = True
except ImportError:
    USE_PACKAGE_EVALUATOR = False


# =============================================================================
# Protocols and EvalContext (inline fallback)
# =============================================================================


@dataclass
class PriceBar:
    """Simple price bar for eval context."""

    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


@dataclass
class EvalContext:
    """Context for evaluating conditions."""

    indicators: dict[str, Any]  # LEAN indicator objects
    state: dict[str, float | int | bool | None]
    price_bar: PriceBar
    hour: int = 12
    minute: int = 0
    day_of_week: int = 2

    def get_indicator(self, indicator_id: str) -> Any:
        ind = self.indicators.get(indicator_id)
        if ind is None:
            raise KeyError(f"Unknown indicator: {indicator_id}")
        return ind

    def get_state(self, state_id: str) -> float | int | bool | None:
        if state_id not in self.state:
            raise KeyError(f"Unknown state variable: {state_id}")
        return self.state[state_id]

    def set_state(self, state_id: str, value: float | int | bool | None) -> None:
        self.state[state_id] = value


# =============================================================================
# Value Resolver (inline fallback)
# =============================================================================


class ValueResolver:
    """Resolves IR value references to float values.

    Uses Pydantic ValueRef types for type-safe resolution.
    """

    def resolve(self, ref: ValueRef, ctx: EvalContext) -> float:
        """Resolve a value reference to a float."""
        match ref:
            case IndicatorRef():
                return self._resolve_indicator(ref, ctx)
            case IndicatorBandRef():
                return self._resolve_indicator_band(ref, ctx)
            case PriceRef():
                return self._resolve_price(ref, ctx)
            case VolumeRef():
                return ctx.price_bar.Volume
            case TimeRef():
                return self._resolve_time(ref, ctx)
            case StateRef():
                value = ctx.get_state(ref.state_id)
                return float(value) if value is not None else 0.0
            case LiteralRef():
                return ref.value
            case IRExpression():
                return self._resolve_expression(ref, ctx)
            case _:
                raise ValueError(f"Unknown value ref type: {type(ref)}")

    def _resolve_indicator(self, ref: IndicatorRef, ctx: EvalContext) -> float:
        """Resolve an indicator reference to its current value.

        Uses indicator_id to look up pre-declared indicator from context.
        """
        ind = ctx.get_indicator(ref.indicator_id)
        return float(ind.Current.Value)

    def _resolve_indicator_band(self, ref: IndicatorBandRef, ctx: EvalContext) -> float:
        """Resolve a band indicator reference (BB upper, KC lower, etc.).

        Uses indicator_id to look up pre-declared indicator, then gets the specified band.
        """
        ind = ctx.get_indicator(ref.indicator_id)

        match ref.band:
            case "upper":
                return float(ind.UpperBand.Current.Value)
            case "middle":
                return float(ind.MiddleBand.Current.Value)
            case "lower":
                return float(ind.LowerBand.Current.Value)
            case _:
                raise ValueError(f"Unknown band: {ref.band}")

    def _resolve_price(self, ref: PriceRef, ctx: EvalContext) -> float:
        """Resolve a price reference."""
        match ref.field:
            case PriceField.OPEN:
                return ctx.price_bar.Open
            case PriceField.HIGH:
                return ctx.price_bar.High
            case PriceField.LOW:
                return ctx.price_bar.Low
            case PriceField.CLOSE:
                return ctx.price_bar.Close

    def _resolve_time(self, ref: TimeRef, ctx: EvalContext) -> float:
        """Resolve a time component reference."""
        match ref.component:
            case "hour":
                return float(ctx.hour)
            case "minute":
                return float(ctx.minute)
            case "day_of_week":
                return float(ctx.day_of_week)
        return 0.0

    def _resolve_expression(self, ref: IRExpression, ctx: EvalContext) -> float:
        """Resolve an arithmetic expression."""
        left_val = self.resolve(ref.left, ctx)
        right_val = self.resolve(ref.right, ctx)
        match ref.op:
            case "+":
                return left_val + right_val
            case "-":
                return left_val - right_val
            case "*":
                return left_val * right_val
            case "/":
                return left_val / right_val if right_val != 0 else 0.0


# =============================================================================
# Condition Evaluator
# =============================================================================


class ConditionEvaluator:
    """Evaluates IR conditions to boolean results.

    Uses Pydantic IRCondition types for type-safe evaluation.
    """

    def __init__(self):
        self.resolver = ValueResolver()
        # Track previous values for cross detection (keyed by condition hash)
        self._previous_values: dict[int, tuple[float, float]] = {}

    def evaluate(self, condition, ctx: EvalContext) -> bool:
        """Evaluate a condition to a boolean.

        Handles both primitive IR types (IRCompare, IRAllOf, etc.) and
        strategy_ir types (CompareCondition, AllOfCondition, etc.).
        """
        match condition:
            # Primitive IR types
            case IRCompare():
                return self._evaluate_compare(condition, ctx)
            case IRAllOf():
                return all(self.evaluate(c, ctx) for c in condition.conditions)
            case IRAnyOf():
                return any(self.evaluate(c, ctx) for c in condition.conditions)
            case IRNot():
                return not self.evaluate(condition.condition, ctx)
            case IRCross():
                return self._evaluate_cross(condition, ctx)
            # Strategy IR types (from strategy_ir.py)
            case CompareCondition():
                return self._evaluate_strategy_compare(condition, ctx)
            case AllOfCondition():
                return all(self.evaluate(c, ctx) for c in condition.conditions)
            case AnyOfCondition():
                return any(self.evaluate(c, ctx) for c in condition.conditions)
            case NotCondition():
                return not self.evaluate(condition.condition, ctx)
            case CrossCondition():
                return self._evaluate_strategy_cross(condition, ctx)
            case TimeFilterCondition():
                return self._evaluate_time_filter(condition, ctx)
            case _:
                raise ValueError(f"Unknown condition type: {type(condition)}")

    def _evaluate_compare(self, condition: IRCompare, ctx: EvalContext) -> bool:
        """Evaluate a typed compare condition."""
        left_val = self.resolver.resolve(condition.left, ctx)
        right_val = self.resolver.resolve(condition.right, ctx)
        return condition.op.apply(left_val, right_val)

    def _evaluate_cross(self, condition: IRCross, ctx: EvalContext) -> bool:
        """Evaluate a cross condition.

        Cross detection requires comparing current and previous values:
        - CROSS_ABOVE: prev_left <= prev_right AND curr_left > curr_right
        - CROSS_BELOW: prev_left >= prev_right AND curr_left < curr_right
        """
        left_val = self.resolver.resolve(condition.left, ctx)
        right_val = self.resolver.resolve(condition.right, ctx)

        # Use hash of frozen Pydantic model as key (stable and efficient)
        key = hash(condition)

        # Get previous values (default to current if first evaluation)
        prev_left, prev_right = self._previous_values.get(key, (left_val, right_val))

        # Store current values for next evaluation
        self._previous_values[key] = (left_val, right_val)

        match condition.direction:
            case CrossDirection.CROSS_ABOVE:
                return prev_left <= prev_right and left_val > right_val
            case CrossDirection.CROSS_BELOW:
                return prev_left >= prev_right and left_val < right_val

    def _evaluate_strategy_compare(self, condition: CompareCondition, ctx: EvalContext) -> bool:
        """Evaluate a CompareCondition from strategy_ir.

        Unlike IRCompare which uses CompareOp enum, CompareCondition uses
        string operators like ">", "<", ">=", "<=", "==", "!=".
        """
        left_val = self.resolver.resolve(condition.left, ctx)
        right_val = self.resolver.resolve(condition.right, ctx)

        match condition.op:
            case ">":
                return left_val > right_val
            case "<":
                return left_val < right_val
            case ">=":
                return left_val >= right_val
            case "<=":
                return left_val <= right_val
            case "==":
                return left_val == right_val
            case "!=":
                return left_val != right_val
            case _:
                raise ValueError(f"Unknown compare operator: {condition.op}")

    def _evaluate_strategy_cross(self, condition: CrossCondition, ctx: EvalContext) -> bool:
        """Evaluate a CrossCondition from strategy_ir.

        Similar to IRCross but uses string direction field.
        """
        left_val = self.resolver.resolve(condition.left, ctx)
        right_val = self.resolver.resolve(condition.right, ctx)

        key = hash((condition.left, condition.right, condition.direction))
        prev_left, prev_right = self._previous_values.get(key, (left_val, right_val))
        self._previous_values[key] = (left_val, right_val)

        match condition.direction:
            case "above":
                return prev_left <= prev_right and left_val > right_val
            case "below":
                return prev_left >= prev_right and left_val < right_val
            case _:
                raise ValueError(f"Unknown cross direction: {condition.direction}")

    def _evaluate_time_filter(self, condition: TimeFilterCondition, ctx: EvalContext) -> bool:
        """Evaluate a TimeFilterCondition.

        Checks if current time is within the allowed time window and day restrictions.

        TimeFilterCondition has:
        - time_window: str (format "HH:MM-HH:MM", e.g., "09:30-16:00")
        - days_of_week: list[int] (0=Monday, 6=Sunday)
        - days_of_month: list[int] (1-31)
        - timezone: str (not currently used in backtesting)
        """
        # Check day of week restriction if specified
        if condition.days_of_week:
            if ctx.day_of_week not in condition.days_of_week:
                return False

        # Check day of month restriction if specified
        if condition.days_of_month:
            # ctx doesn't have day_of_month, derive from timestamp if available
            # For now, skip this check in backtesting
            pass

        # Check time window if specified
        if condition.time_window:
            # Parse "HH:MM-HH:MM" format
            try:
                start_str, end_str = condition.time_window.split("-")
                start_hour, start_minute = map(int, start_str.split(":"))
                end_hour, end_minute = map(int, end_str.split(":"))

                # Convert to minutes since midnight for comparison
                current_minutes = ctx.hour * 60 + ctx.minute
                start_minutes = start_hour * 60 + start_minute
                end_minutes = end_hour * 60 + end_minute

                # Check if current time is within window
                if not (start_minutes <= current_minutes <= end_minutes):
                    return False
            except (ValueError, AttributeError):
                # If time_window is malformed, allow (fail open)
                pass

        return True


# =============================================================================
# Main Strategy Runtime Algorithm
# =============================================================================


class StrategyRuntime(QCAlgorithm):
    """LEAN algorithm that executes a StrategyIR."""

    def Initialize(self):
        """Initialize the algorithm from StrategyIR.

        The StrategyIR can be provided via:
        - STRATEGY_IR_JSON environment variable (JSON string)
        - STRATEGY_IR_PATH environment variable (path to JSON file)
        - Default path: /Data/strategy_ir.json

        For live trading, the IR is typically loaded from GCS and passed
        via environment variable.
        """
        # Load IR from JSON (file path from environment)
        ir_path = os.getenv("STRATEGY_IR_PATH", "/Data/strategy_ir.json")
        ir_json = os.getenv("STRATEGY_IR_JSON", "")

        if ir_json:
            ir_dict = json.loads(ir_json)
            self.Log("Loaded strategy from STRATEGY_IR_JSON env")
        elif os.path.exists(ir_path):
            with open(ir_path) as f:
                ir_dict = json.load(f)
            self.Log(f"Loaded strategy from {ir_path}")
        else:
            raise ValueError(f"No StrategyIR found at {ir_path} or in STRATEGY_IR_JSON env")

        # Parse using Pydantic StrategyIR model (required)
        self.ir = StrategyIR.model_validate(ir_dict)
        self.Log("Parsed IR using Pydantic StrategyIR model")

        # Collect any inline IndicatorRefs from conditions
        inline_indicators = self.ir.collect_inline_indicators()
        if inline_indicators:
            self.Log(f"Collected {len(inline_indicators)} inline indicators")

        # Combine explicit and inline indicators
        all_indicators = list(self.ir.indicators) + inline_indicators

        self.Log(f"Strategy: {self.ir.strategy_name} (id: {self.ir.strategy_id})")

        # Set algorithm parameters from IR or environment
        start_date_str = os.getenv("START_DATE", self.ir.start_date or "2024-01-01")
        end_date_str = os.getenv("END_DATE", self.ir.end_date or "2024-01-01")

        start_parts = [int(p) for p in start_date_str.split("-")]
        end_parts = [int(p) for p in end_date_str.split("-")]

        self.SetStartDate(*start_parts)
        self.SetEndDate(*end_parts)
        self.SetCash(100000)
        # Use UTC timezone to match our data
        self.SetTimeZone(TimeZones.Utc)

        # Check mode
        self._use_custom_data = os.getenv("USE_CUSTOM_DATA_MODE", "false").lower() == "true"

        # Configure fee and slippage models (for realistic backtesting)
        if not self._use_custom_data:
            # Fee: 0.1% maker fee (Coinbase Pro)
            fee_rate = float(os.getenv("FEE_RATE", "0.001"))
            self.SetSecurityInitializer(lambda security: self._init_security(security, fee_rate))
            self.Log(f"Fee model: {fee_rate * 100:.2f}% per trade")

        # Add symbol
        self._add_symbol(self.ir.symbol)

        # Create resolution
        resolution = self._get_resolution(self.ir.resolution)

        # Calculate max indicator period for warmup
        max_period = 0
        for ind_spec in all_indicators:
            period = ind_spec.period if ind_spec.period else 20
            max_period = max(max_period, period)

        # Set warmup period (only for AddCrypto mode - custom data handles it differently)
        if not self._use_custom_data and max_period > 0:
            warmup_bars = max_period + 10  # Add buffer
            self.SetWarmUp(warmup_bars, resolution)
            self.Log(f"Warmup: {warmup_bars} bars (max indicator period: {max_period})")

        # Create indicators from Pydantic models
        self.lean_indicators = {}
        self._indicator_meta = {}
        for ind_spec in all_indicators:
            self._create_indicator_from_spec(ind_spec, resolution)

        # Initialize state from Pydantic models
        self.state = {}
        for state_var in self.ir.state:
            self.state[state_var.id] = state_var.default

        # Create evaluator
        self.evaluator = ConditionEvaluator()

        # Track position state
        self.is_invested = False

        self.Log(f"Initialized with {len(self.lean_indicators)} indicators, {len(self.state)} state vars")

    def _add_symbol(self, symbol_str: str):
        """Add a trading symbol with proper data control.

        Two modes:
        1. USE_CUSTOM_DATA_MODE=true: Use VibeTradeCryptoData (for testing with synthetic data)
        2. Default: Use AddCrypto with explicit conversion rate (for production-like behavior)

        In production, we want LEAN's crypto infrastructure (CashBook, portfolio accounting)
        but with all data coming from our controlled sources.
        """
        # Normalize symbol: BTC-USD -> BTCUSD
        normalized = symbol_str.replace("-", "").upper()
        base_currency = normalized.replace("USD", "")

        if self._use_custom_data:
            # Custom data mode for synthetic testing
            self.Log(f"Using VibeTradeCryptoData for symbol: {normalized}")
            custom_data = self.add_data(VibeTradeCryptoData, normalized, Resolution.Minute)
            self.symbol = custom_data.Symbol
            self.Log(f"Added custom data symbol: {self.symbol}")
            self.Log(f"  Symbol.Value: {self.symbol.Value}")
            self.Log(f"  Symbol.SecurityType: {self.symbol.SecurityType}")
        else:
            # Production mode: Use AddCrypto but control conversion
            # Set explicit cash holdings with conversion rate BEFORE AddCrypto
            # This prevents LEAN from looking up external conversion rates
            self.Log(f"Using AddCrypto for symbol: {normalized}")
            self.Log(f"  Setting explicit conversion rate for {base_currency}")

            # Set the base currency with explicit conversion rate
            # For testing: 1 TEST = 1 USD (no conversion)
            # For production: would come from our data feeds
            conversion_rate = float(os.getenv("CONVERSION_RATE", "1.0"))
            self.SetCash(base_currency, 0, conversion_rate)

            self.Log(f"  {base_currency} conversion rate set to: {conversion_rate}")

            try:
                crypto = self.AddCrypto(normalized, Resolution.Minute, Market.Coinbase)
                self.symbol = crypto.Symbol
                self.Log(f"Added crypto symbol: {self.symbol}")
                self.Log(f"  Symbol.Value: {self.symbol.Value}")
                self.Log(f"  Market: {self.symbol.ID.Market}")

                # Verify the conversion is set correctly
                if base_currency in self.Portfolio.CashBook:
                    cash = self.Portfolio.CashBook[base_currency]
                    self.Log(f"  CashBook[{base_currency}]: ConversionRate={cash.ConversionRate}")
            except Exception as e:
                self.Log(f"AddCrypto failed: {e}")
                raise

    def _get_resolution(self, res_str: str) -> Resolution:
        """Convert resolution string to LEAN Resolution."""
        res_map = {
            "Minute": Resolution.Minute,
            "Hour": Resolution.Hour,
            "Daily": Resolution.Daily,
        }
        return res_map.get(res_str, Resolution.Hour)

    def _init_security(self, security, fee_rate: float):
        """Initialize security with fee and slippage models.

        Args:
            security: The security to initialize
            fee_rate: Fee rate as decimal (0.001 = 0.1%)
        """
        # Fee model: percentage of trade value
        security.SetFeeModel(ConstantFeeModel(fee_rate))

        # Slippage model: small constant slippage for crypto
        # VolumeShareSlippageModel would be more realistic but requires volume data
        security.SetSlippageModel(ConstantSlippageModel(0))

        # Fill model: default immediate fill (good for backtesting liquid crypto)
        # For more realistic simulation, could use PartialFillModel

    def _create_indicator_from_spec(self, spec: IndicatorSpec, resolution):
        """Create a LEAN indicator from an IndicatorSpec Pydantic model.

        For custom data (VibeTradeCryptoData), we create indicators manually
        and update them in OnData since LEAN's helper methods don't auto-register
        for PythonData types.

        For AddCrypto mode, LEAN auto-registers indicators and updates them.
        """
        ind_type = spec.type
        ind_id = spec.id
        period = spec.period or 20

        if not self._use_custom_data:
            # AddCrypto mode - use LEAN's helper methods
            self._create_indicator_addcrypto_from_spec(spec, resolution)
            return

        # Look up indicator definition in registry
        if ind_type not in INDICATOR_REGISTRY:
            self.Log(f"Unknown indicator type: {ind_type}")
            return

        registry_entry = INDICATOR_REGISTRY[ind_type]
        class_name = registry_entry["class_name"]
        ind_class = INDICATOR_CLASSES.get(class_name)

        if ind_class is None:
            self.Log(f"Indicator class not found: {class_name}")
            return

        # Build constructor arguments from spec and registry defaults
        args = []
        for param in registry_entry.get("params", []):
            default = registry_entry.get("param_defaults", {}).get(param)
            if param == "period":
                args.append(period)
            elif param == "multiplier":
                args.append(spec.multiplier or default or 2.0)
            elif param == "fast_period":
                args.append(spec.fast_period or default or 12)
            elif param == "slow_period":
                args.append(spec.slow_period or default or 26)
            elif param == "signal_period":
                args.append(spec.signal_period or default or 9)
            elif default is not None:
                args.append(default)
            else:
                args.append(getattr(spec, param, None))

        # Create the indicator
        if class_name == "RollingWindow":
            self.lean_indicators[ind_id] = RollingWindow[float](args[0] if args else 10)
        else:
            self.lean_indicators[ind_id] = ind_class(*args)

        # Store metadata for update logic
        field = spec.field or registry_entry.get("default_field", "close")
        self._indicator_meta[ind_id] = {
            "type": ind_type,
            "update_mode": registry_entry.get("update_mode", UPDATE_SCALAR),
            "field": field,
        }

    def _create_indicator_addcrypto_from_spec(self, spec: IndicatorSpec, resolution):
        """Create indicator using LEAN's helper methods (for AddCrypto mode)."""
        ind_type = spec.type
        ind_id = spec.id
        period = spec.period or 20

        if ind_type == "EMA":
            self.lean_indicators[ind_id] = self.EMA(self.symbol, period, resolution=resolution)
        elif ind_type == "SMA":
            self.lean_indicators[ind_id] = self.SMA(self.symbol, period, resolution=resolution)
        elif ind_type == "BB":
            multiplier = spec.multiplier or 2.0
            self.lean_indicators[ind_id] = self.BB(self.symbol, period, multiplier, resolution=resolution)
        elif ind_type == "KC":
            multiplier = spec.multiplier or 2.0
            self.lean_indicators[ind_id] = self.KCH(self.symbol, period, multiplier, resolution=resolution)
        elif ind_type == "ATR":
            self.lean_indicators[ind_id] = self.ATR(self.symbol, period, resolution=resolution)
        elif ind_type == "RSI":
            self.lean_indicators[ind_id] = self.RSI(self.symbol, period, resolution=resolution)
        elif ind_type == "MACD":
            fast = spec.fast_period or 12
            slow = spec.slow_period or 26
            signal = spec.signal_period or 9
            self.lean_indicators[ind_id] = self.MACD(self.symbol, fast, slow, signal, resolution=resolution)
        elif ind_type == "MAX":
            self.lean_indicators[ind_id] = self.MAX(self.symbol, period, resolution=resolution)
        elif ind_type == "MIN":
            self.lean_indicators[ind_id] = self.MIN(self.symbol, period, resolution=resolution)
        elif ind_type == "ROC":
            self.lean_indicators[ind_id] = self.ROC(self.symbol, period, resolution=resolution)
        elif ind_type == "ADX":
            self.lean_indicators[ind_id] = self.ADX(self.symbol, period, resolution=resolution)
        elif ind_type == "DC":
            self.lean_indicators[ind_id] = self.DCH(self.symbol, period, resolution=resolution)
        elif ind_type == "VWAP":
            self.lean_indicators[ind_id] = self.VWAP(self.symbol)
        elif ind_type == "VOL_SMA":
            sma = SimpleMovingAverage(period)
            self.RegisterIndicator(self.symbol, sma, resolution)
            self.lean_indicators[ind_id] = sma
        elif ind_type == "RW":
            window_size = spec.period or 10
            self.lean_indicators[ind_id] = RollingWindow[float](window_size)
        else:
            self.Log(f"Unknown indicator type: {ind_type}")

    def OnData(self, data):
        """Process new data and execute strategy logic."""
        # Track bar count for logging
        if not hasattr(self, "_bar_count"):
            self._bar_count = 0
        self._bar_count += 1

        if self.symbol not in data or data[self.symbol] is None:
            return

        raw_bar = data[self.symbol]

        # Extract OHLCV from data - handle both custom data and TradeBar
        if self._use_custom_data:
            # Custom data (VibeTradeCryptoData) - access via dict-like interface
            bar_open = float(raw_bar["Open"])
            bar_high = float(raw_bar["High"])
            bar_low = float(raw_bar["Low"])
            bar_close = float(raw_bar["Close"])
            bar_volume = float(raw_bar["Volume"])
        else:
            # TradeBar - access via properties
            bar_open = float(raw_bar.Open)
            bar_high = float(raw_bar.High)
            bar_low = float(raw_bar.Low)
            bar_close = float(raw_bar.Close)
            bar_volume = float(raw_bar.Volume)

        # For custom data, manually update indicators
        if self._use_custom_data:
            self._update_indicators_manual(bar_open, bar_high, bar_low, bar_close, bar_volume)

        # Check if indicators are ready (warmup period)
        if not self._indicators_ready():
            return

        # Log first bar after warmup
        if not hasattr(self, "_indicators_warmup_complete"):
            self._indicators_warmup_complete = True
            self.Log(f"Indicators ready after {self._bar_count} bars, processing from bar at {raw_bar.Time}")

        # Update rolling windows with current values
        self._update_rolling_windows_values(bar_open, bar_high, bar_low, bar_close)

        # Build evaluation context
        ctx = EvalContext(
            indicators=self.lean_indicators,
            state=self.state,
            price_bar=PriceBar(
                Open=bar_open,
                High=bar_high,
                Low=bar_low,
                Close=bar_close,
                Volume=bar_volume
            ),
            hour=raw_bar.Time.hour,
            minute=raw_bar.Time.minute,
            day_of_week=raw_bar.Time.weekday()
        )

        # Execute on_bar state hooks
        for state_op in self.ir.on_bar:
            self._execute_state_op(state_op, ctx)

        # If invested, execute on_bar_invested hooks
        if self.is_invested:
            for state_op in self.ir.on_bar_invested:
                self._execute_state_op(state_op, ctx)

        # Check gates
        gates_pass = self._evaluate_gates(ctx)

        if not gates_pass:
            return

        # Check exits first (if invested)
        if self.is_invested:
            self._evaluate_exits(ctx)

        # Check entry (if not invested)
        if not self.is_invested:
            self._evaluate_entry(ctx)

    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        for _ind_id, ind in self.lean_indicators.items():
            if hasattr(ind, "IsReady") and not ind.IsReady:
                return False
        return True

    def _update_indicators_manual(self, open_val: float, high_val: float, low_val: float, close_val: float, volume_val: float):
        """Manually update indicators for custom data mode using registry metadata.

        For PythonData custom types, LEAN doesn't auto-feed indicators.
        We manually update each indicator with the new bar data.
        """
        # Field value mapping for scalar updates
        field_values = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
        }

        # Reusable TradeBar for tradebar updates
        trade_bar = None

        for ind_id, ind in self.lean_indicators.items():
            # Get metadata for this indicator
            meta = self._indicator_meta.get(ind_id)
            if meta is None:
                continue  # Skip indicators without metadata (shouldn't happen)

            update_mode = meta.get("update_mode")

            if update_mode == UPDATE_SCALAR:
                # Scalar update - use the appropriate field value
                field = meta.get("field", "close")
                value = field_values.get(field, close_val)
                ind.Update(self.Time, value)

            elif update_mode == UPDATE_TRADEBAR:
                # TradeBar update - create lazily
                if trade_bar is None:
                    trade_bar = TradeBar(self.Time, self.symbol, open_val, high_val, low_val, close_val, volume_val)
                ind.Update(trade_bar)

            # rolling_window mode handled separately in _update_rolling_windows_values

    def _update_rolling_windows_values(self, open_val: float, high_val: float, low_val: float, close_val: float):
        """Update rolling window indicators using metadata."""
        field_values = {
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
        }

        for ind_id, ind in self.lean_indicators.items():
            meta = self._indicator_meta.get(ind_id)
            if meta and meta.get("update_mode") == "rolling_window":
                field = meta.get("field", "close")
                ind.Add(field_values.get(field, close_val))

    def _evaluate_gates(self, ctx: EvalContext) -> bool:
        """Evaluate all gates. Returns True if allowed to proceed."""
        for gate in self.ir.gates:
            result = self.evaluator.evaluate(gate.condition, ctx)

            if gate.mode == "allow" and not result:
                return False
            elif gate.mode == "block" and result:
                return False

        return True

    def _evaluate_entry(self, ctx: EvalContext):
        """Evaluate entry condition and execute if true."""
        entry = self.ir.entry
        if not entry:
            return

        if self.evaluator.evaluate(entry.condition, ctx):
            self._execute_action(entry.action)
            self.is_invested = True

            # Execute on_fill hooks
            for state_op in entry.on_fill:
                self._execute_state_op(state_op, ctx)

            self.Log(f"ENTRY executed at bar={self._bar_count} price={ctx.price_bar.Close}")

    def _evaluate_exits(self, ctx: EvalContext):
        """Evaluate exit conditions and execute highest priority."""
        # Sort by priority (higher priority first)
        exits_sorted = sorted(self.ir.exits, key=lambda x: x.priority, reverse=True)

        for exit_rule in exits_sorted:
            if self.evaluator.evaluate(exit_rule.condition, ctx):
                self._execute_action(exit_rule.action)
                self.is_invested = False
                self.Log(f"EXIT ({exit_rule.id}) executed at bar={self._bar_count} price={ctx.price_bar.Close}")
                return  # Only execute first matching exit

    def _execute_action(self, action: SetHoldingsAction | LiquidateAction):
        """Execute a trading action."""
        match action:
            case SetHoldingsAction():
                self.SetHoldings(self.symbol, action.allocation)
            case LiquidateAction():
                self.Liquidate(self.symbol)
            case _:
                raise ValueError(f"Unknown action type: {type(action)}")

    def _execute_state_op(self, op: OnFillAction | OnBarAction, ctx: EvalContext):
        """Execute a state operation.

        Handles both typed actions (SetStateAction, IncrementStateAction) and
        generic OnBarAction which uses string type field for dispatch.
        """
        match op:
            case SetStateAction():
                value = self.evaluator.resolver.resolve(op.value, ctx)
                ctx.set_state(op.state_id, value)
                self.state[op.state_id] = value
            case IncrementStateAction():
                current = ctx.get_state(op.state_id) or 0
                # Resolve amount if it's a ValueRef, otherwise use directly
                if isinstance(op.amount, (int, float)):
                    amount = op.amount
                else:
                    amount = self.evaluator.resolver.resolve(op.amount, ctx)
                self.state[op.state_id] = current + amount
            case MaxStateAction():
                current = ctx.get_state(op.state_id) or 0
                new_value = self.evaluator.resolver.resolve(op.value, ctx)
                self.state[op.state_id] = max(current, new_value)
                ctx.set_state(op.state_id, self.state[op.state_id])
            case OnBarAction():
                # Generic action - dispatch based on type field
                self._execute_on_bar_action(op, ctx)
            case _:
                raise ValueError(f"Unknown state operation type: {type(op)}")

    def _execute_on_bar_action(self, op: OnBarAction, ctx: EvalContext):
        """Execute a generic OnBarAction by dispatching on its type field."""
        match op.type:
            case "increment":
                # Increment state variable by 1
                state_id = getattr(op, "state_id", None)
                if state_id:
                    current = ctx.get_state(state_id) or 0
                    self.state[state_id] = current + 1
            case "set_state":
                # Set state variable
                state_id = getattr(op, "state_id", None)
                value = getattr(op, "value", None)
                if state_id and value is not None:
                    # Value might be a dict representing a ValueRef
                    if isinstance(value, dict):
                        from vibe_trade_shared.models.ir.value_refs import ValueRef
                        from pydantic import TypeAdapter
                        adapter = TypeAdapter(ValueRef)
                        resolved = self.evaluator.resolver.resolve(adapter.validate_python(value), ctx)
                    else:
                        resolved = float(value)
                    ctx.set_state(state_id, resolved)
                    self.state[state_id] = resolved
            case _:
                # Unknown action type - log but don't fail
                pass

    def OnEndOfAlgorithm(self):
        """Log summary at end of backtest."""
        self.Log(f"Strategy {self.ir.strategy_name} completed")
        self.Log(f"Final state: {self.state}")

"""Test data builder for deterministic strategy testing.

This module creates synthetic price data designed to trigger specific
strategy conditions at known times, enabling exact trade assertions.

Usage:
    from test.lib.test_data_builder import TestDataBuilder, ExpectedTrade

    builder = TestDataBuilder(symbol="TESTUSD", start_date="2024-01-01")

    # Build price data that triggers specific conditions
    builder.add_uptrend(bars=30, start_price=50000, trend_strength=0.001)
    builder.add_pullback_to_bb_lower(bars=5, bb_period=20, bb_mult=2.0)
    builder.add_recovery(bars=20, trend_strength=0.002)

    # Get expected trades based on strategy
    expected = builder.get_expected_trades(
        strategy_ir=strategy_ir,
        entry_type="trend_pullback"
    )

    # Export data
    builder.export_csv("test/data/custom/testusd/20240101.csv")
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal


@dataclass
class OHLCV:
    """Single candle."""
    timestamp_ms: int  # ms since midnight
    open: float
    high: float
    low: float
    close: float
    volume: float = 1000.0

    def to_csv_line(self) -> str:
        return f"{self.timestamp_ms},{self.open},{self.high},{self.low},{self.close},{self.volume}"


@dataclass
class ExpectedTrade:
    """Expected trade for test assertion."""
    bar_index: int
    action: Literal["ENTRY", "EXIT"]
    direction: Literal["long", "short"]
    price: float
    reason: str  # Human-readable reason for the trade


@dataclass
class TestDataBuilder:
    """Builds synthetic price data for deterministic testing."""

    symbol: str
    start_date: str  # YYYY-MM-DD
    base_volume: float = 1000.0

    # Internal state
    candles: list[OHLCV] = field(default_factory=list)
    current_price: float = 0.0
    bar_index: int = 0

    # For indicator calculation
    _prices: list[float] = field(default_factory=list)  # Close prices for indicator calc

    def add_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float | None = None,
    ) -> int:
        """Add a single candle. Returns bar index."""
        ms = self.bar_index * 60000  # 1-minute bars
        candle = OHLCV(
            timestamp_ms=ms,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume or self.base_volume,
        )
        self.candles.append(candle)
        self._prices.append(close_price)
        self.current_price = close_price
        self.bar_index += 1
        return self.bar_index - 1

    def add_flat(self, bars: int, price: float, noise: float = 0.0) -> list[int]:
        """Add flat/sideways price action."""
        indices = []
        for _ in range(bars):
            high = price + noise
            low = price - noise
            idx = self.add_candle(price, high, low, price)
            indices.append(idx)
        return indices

    def add_uptrend(
        self,
        bars: int,
        start_price: float | None = None,
        end_price: float | None = None,
        trend_strength: float = 0.001,  # 0.1% per bar
    ) -> list[int]:
        """Add uptrend. Either specify end_price or trend_strength."""
        start = start_price or self.current_price or 50000.0

        if end_price:
            # Calculate trend strength from start/end
            total_change = (end_price - start) / start
            trend_strength = total_change / bars

        indices = []
        price = start
        for i in range(bars):
            change = price * trend_strength
            new_price = price + change

            # Realistic OHLC
            self.add_candle(
                open_price=price,
                high_price=new_price + abs(change) * 0.5,
                low_price=price - abs(change) * 0.2,
                close_price=new_price,
            )
            indices.append(self.bar_index - 1)
            price = new_price

        return indices

    def add_downtrend(
        self,
        bars: int,
        start_price: float | None = None,
        end_price: float | None = None,
        trend_strength: float = 0.001,
    ) -> list[int]:
        """Add downtrend."""
        start = start_price or self.current_price

        if end_price:
            total_change = (start - end_price) / start
            trend_strength = total_change / bars

        indices = []
        price = start
        for i in range(bars):
            change = price * trend_strength
            new_price = price - change

            self.add_candle(
                open_price=price,
                high_price=price + abs(change) * 0.2,
                low_price=new_price - abs(change) * 0.5,
                close_price=new_price,
            )
            indices.append(self.bar_index - 1)
            price = new_price

        return indices

    def add_pullback_to_bb_lower(
        self,
        bars: int,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        overshoot: float = 0.0,  # How far below BB lower (0 = touch, 0.01 = 1% below)
    ) -> tuple[list[int], int]:
        """Add pullback that touches/crosses BB lower band.

        Returns:
            (bar_indices, trigger_bar) - indices of all bars, and the bar where BB lower is touched
        """
        if len(self._prices) < bb_period:
            raise ValueError(f"Need at least {bb_period} bars before pullback")

        # Calculate current BB lower
        recent_prices = self._prices[-bb_period:]
        sma = sum(recent_prices) / bb_period
        variance = sum((p - sma) ** 2 for p in recent_prices) / bb_period
        std_dev = math.sqrt(variance)
        bb_lower = sma - bb_mult * std_dev

        # Target price = BB lower - overshoot
        target_price = bb_lower * (1 - overshoot)

        # Sharp drop to target
        start_price = self.current_price
        drop_per_bar = (start_price - target_price) / bars

        indices = []
        price = start_price
        trigger_bar = None

        for i in range(bars):
            new_price = price - drop_per_bar

            # Recalculate BB lower with updated prices
            if len(self._prices) >= bb_period:
                recent = self._prices[-(bb_period - 1):] + [new_price]
                current_sma = sum(recent) / bb_period
                current_var = sum((p - current_sma) ** 2 for p in recent) / bb_period
                current_std = math.sqrt(current_var)
                current_bb_lower = current_sma - bb_mult * current_std

                # Check if this bar crosses BB lower
                if trigger_bar is None and new_price <= current_bb_lower:
                    trigger_bar = self.bar_index

            self.add_candle(
                open_price=price,
                high_price=price + drop_per_bar * 0.2,
                low_price=new_price - drop_per_bar * 0.3,
                close_price=new_price,
                volume=self.base_volume * 2,  # Higher volume on pullback
            )
            indices.append(self.bar_index - 1)
            price = new_price

        return indices, trigger_bar or indices[-1]

    def add_recovery(
        self,
        bars: int,
        trend_strength: float = 0.002,
        target_price: float | None = None,
    ) -> list[int]:
        """Add recovery/continuation of uptrend."""
        if target_price:
            return self.add_uptrend(bars, end_price=target_price)
        return self.add_uptrend(bars, trend_strength=trend_strength)

    def calculate_ema(self, period: int) -> list[float]:
        """Calculate EMA for all prices."""
        if len(self._prices) < period:
            return []

        emas = []
        multiplier = 2 / (period + 1)

        # Initial SMA
        sma = sum(self._prices[:period]) / period
        emas.append(sma)

        # EMA calculation
        for i in range(period, len(self._prices)):
            ema = (self._prices[i] - emas[-1]) * multiplier + emas[-1]
            emas.append(ema)

        # Pad with None for initial bars
        return [None] * (period - 1) + emas

    def calculate_bb(self, period: int = 20, mult: float = 2.0) -> list[dict]:
        """Calculate Bollinger Bands for all prices."""
        if len(self._prices) < period:
            return []

        bands = []
        for i in range(len(self._prices)):
            if i < period - 1:
                bands.append(None)
                continue

            window = self._prices[i - period + 1 : i + 1]
            sma = sum(window) / period
            variance = sum((p - sma) ** 2 for p in window) / period
            std_dev = math.sqrt(variance)

            bands.append({
                "middle": sma,
                "upper": sma + mult * std_dev,
                "lower": sma - mult * std_dev,
                "std_dev": std_dev,
            })

        return bands

    def get_expected_trades_trend_pullback(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        profit_target_pct: float = 0.02,
        stop_loss_pct: float = 0.01,
    ) -> list[ExpectedTrade]:
        """Calculate expected trades for TrendPullback strategy.

        Entry: EMA_fast > EMA_slow AND close < BB_lower
        Exit: profit target or stop loss
        """
        ema_fast_values = self.calculate_ema(ema_fast)
        ema_slow_values = self.calculate_ema(ema_slow)
        bb_values = self.calculate_bb(bb_period, bb_mult)

        trades = []
        in_position = False
        entry_price = 0.0
        entry_bar = 0

        warmup_bars = max(ema_fast, ema_slow, bb_period)

        for i in range(warmup_bars, len(self._prices)):
            price = self._prices[i]
            ema_f = ema_fast_values[i] if i < len(ema_fast_values) else None
            ema_s = ema_slow_values[i] if i < len(ema_slow_values) else None
            bb = bb_values[i] if i < len(bb_values) else None

            if ema_f is None or ema_s is None or bb is None:
                continue

            if not in_position:
                # Check entry condition
                uptrend = ema_f > ema_s
                pullback = price < bb["lower"]

                if uptrend and pullback:
                    trades.append(ExpectedTrade(
                        bar_index=i,
                        action="ENTRY",
                        direction="long",
                        price=price,
                        reason=f"EMA{ema_fast}({ema_f:.2f}) > EMA{ema_slow}({ema_s:.2f}), close({price:.2f}) < BB_lower({bb['lower']:.2f})"
                    ))
                    in_position = True
                    entry_price = price
                    entry_bar = i
            else:
                # Check exit conditions
                profit_pct = (price - entry_price) / entry_price
                loss_pct = (entry_price - price) / entry_price

                if profit_pct >= profit_target_pct:
                    trades.append(ExpectedTrade(
                        bar_index=i,
                        action="EXIT",
                        direction="long",
                        price=price,
                        reason=f"Profit target hit: {profit_pct*100:.2f}% >= {profit_target_pct*100:.1f}%"
                    ))
                    in_position = False
                elif loss_pct >= stop_loss_pct:
                    trades.append(ExpectedTrade(
                        bar_index=i,
                        action="EXIT",
                        direction="long",
                        price=price,
                        reason=f"Stop loss hit: -{loss_pct*100:.2f}% >= {stop_loss_pct*100:.1f}%"
                    ))
                    in_position = False

        return trades

    def export_csv(self, filepath: str) -> None:
        """Export candles to LEAN CSV format."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [candle.to_csv_line() for candle in self.candles]
        path.write_text("\n".join(lines) + "\n")

    def export_zip(self, filepath: str) -> None:
        """Export candles to LEAN ZIP format."""
        import zipfile
        from io import StringIO

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # CSV content
        csv_content = "\n".join(candle.to_csv_line() for candle in self.candles)

        # Inner filename
        date_str = self.start_date.replace("-", "")
        symbol_lower = self.symbol.lower()
        csv_filename = f"{date_str}_{symbol_lower}_minute_trade.csv"

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(csv_filename, csv_content)

    def debug_print(self) -> None:
        """Print debug info about generated data."""
        print(f"Generated {len(self.candles)} candles for {self.symbol}")
        print(f"Price range: {min(c.close for c in self.candles):.2f} - {max(c.close for c in self.candles):.2f}")

        # Show indicator values at key points
        ema20 = self.calculate_ema(20)
        ema50 = self.calculate_ema(50)
        bb = self.calculate_bb(20, 2.0)

        print("\nKey bars:")
        for i in [0, 20, 40, 50, 55, 60, len(self.candles) - 1]:
            if i >= len(self.candles):
                continue
            c = self.candles[i]
            e20 = ema20[i] if i < len(ema20) and ema20[i] else "N/A"
            e50 = ema50[i] if i < len(ema50) and ema50[i] else "N/A"
            bb_l = bb[i]["lower"] if i < len(bb) and bb[i] else "N/A"

            print(f"  Bar {i}: close={c.close:.2f}, EMA20={e20}, EMA50={e50}, BB_lower={bb_l}")


def create_trend_pullback_test(
    symbol: str = "TESTUSD",
    start_date: str = "2024-01-01",
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    profit_target_pct: float = 0.02,
    stop_loss_pct: float = 0.01,
) -> tuple[TestDataBuilder, list[ExpectedTrade]]:
    """Create a standard trend pullback test scenario.

    Phases:
    1. Warmup: Establish uptrend (EMA20 > EMA50)
    2. Pullback: Sharp drop to BB lower (triggers entry)
    3. Recovery: Rise to profit target (triggers exit)

    Returns:
        (builder, expected_trades)
    """
    builder = TestDataBuilder(symbol=symbol, start_date=start_date)

    # Phase 1: Warmup - establish uptrend
    # Need at least ema_slow bars for indicator warmup
    warmup_bars = ema_slow + 10
    builder.add_uptrend(bars=warmup_bars, start_price=50000, trend_strength=0.0008)

    # Phase 2: Pullback to BB lower
    pullback_indices, trigger_bar = builder.add_pullback_to_bb_lower(
        bars=5,
        bb_period=bb_period,
        bb_mult=bb_mult,
        overshoot=0.005,  # 0.5% below BB lower
    )

    # Phase 3: Recovery to profit target
    builder.add_recovery(bars=20, trend_strength=0.003)

    # Calculate expected trades
    expected = builder.get_expected_trades_trend_pullback(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        bb_period=bb_period,
        bb_mult=bb_mult,
        profit_target_pct=profit_target_pct,
        stop_loss_pct=stop_loss_pct,
    )

    return builder, expected


if __name__ == "__main__":
    # Demo usage
    builder, expected_trades = create_trend_pullback_test()

    print("=" * 60)
    print("Test Data Builder Demo")
    print("=" * 60)

    builder.debug_print()

    print("\nExpected Trades:")
    for trade in expected_trades:
        print(f"  Bar {trade.bar_index}: {trade.action} {trade.direction} @ {trade.price:.2f}")
        print(f"    Reason: {trade.reason}")

    # Export
    builder.export_csv("test/data/custom/testusd/20240101.csv")
    print("\nExported to test/data/custom/testusd/20240101.csv")

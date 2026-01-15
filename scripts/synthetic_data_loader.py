#!/usr/bin/env python3
"""Synthetic Data Loader for LEAN Backtesting.

Generates deterministic test data with known patterns for testing strategies.
Uses the same DataLoader interface as BigQueryDataLoader to ensure identical
output format.

This loader is used by tests to avoid external dependencies while maintaining
confidence that the data format matches production.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from data_loader_base import Candle, DataLoader


@dataclass
class ExpectedTrade:
    """Expected trade for test assertion."""

    bar_index: int
    action: Literal["ENTRY", "EXIT"]
    direction: Literal["long", "short"]
    price: float
    reason: str


class SyntheticDataLoader(DataLoader):
    """Generate synthetic candle data for testing.

    Creates deterministic price patterns that trigger known strategy conditions.
    """

    def __init__(self):
        """Initialize synthetic data loader."""
        self._candles: list[Candle] = []
        self._current_price: float = 0.0
        self._prices: list[float] = []  # For indicator calculation

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Candle]:
        """Return the generated candles.

        Note: For SyntheticDataLoader, you should call generator methods like
        `generate_uptrend_with_pullback()` before calling `load()`.

        Args:
            symbol: Trading symbol (ignored - set during generation)
            start_date: Start date (used to timestamp candles)
            end_date: End date (ignored - length determined by generation)

        Returns:
            List of Candle objects
        """
        # Re-timestamp candles based on start_date
        result = []
        for i, candle in enumerate(self._candles):
            result.append(Candle(
                timestamp=start_date + timedelta(minutes=i),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            ))
        return result

    def reset(self) -> None:
        """Clear generated data for reuse."""
        self._candles = []
        self._current_price = 0.0
        self._prices = []

    def add_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 1000.0,
    ) -> int:
        """Add a single candle. Returns bar index."""
        # Use placeholder timestamp - will be replaced in load()
        candle = Candle(
            timestamp=datetime(2024, 1, 1),  # Placeholder
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        self._candles.append(candle)
        self._prices.append(close_price)
        self._current_price = close_price
        return len(self._candles) - 1

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
        trend_strength: float = 0.001,
    ) -> list[int]:
        """Add uptrend bars."""
        start = start_price or self._current_price or 50000.0

        if end_price:
            total_change = (end_price - start) / start
            trend_strength = total_change / bars

        indices = []
        price = start
        for _ in range(bars):
            change = price * trend_strength
            new_price = price + change

            self.add_candle(
                open_price=price,
                high_price=new_price + abs(change) * 0.5,
                low_price=price - abs(change) * 0.2,
                close_price=new_price,
            )
            indices.append(len(self._candles) - 1)
            price = new_price

        return indices

    def add_downtrend(
        self,
        bars: int,
        start_price: float | None = None,
        end_price: float | None = None,
        trend_strength: float = 0.001,
    ) -> list[int]:
        """Add downtrend bars."""
        start = start_price or self._current_price

        if end_price:
            total_change = (start - end_price) / start
            trend_strength = total_change / bars

        indices = []
        price = start
        for _ in range(bars):
            change = price * trend_strength
            new_price = price - change

            self.add_candle(
                open_price=price,
                high_price=price + abs(change) * 0.2,
                low_price=new_price - abs(change) * 0.5,
                close_price=new_price,
            )
            indices.append(len(self._candles) - 1)
            price = new_price

        return indices

    def add_pullback_to_bb_lower(
        self,
        bars: int,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        overshoot: float = 0.0,
    ) -> tuple[list[int], int]:
        """Add pullback that touches/crosses BB lower band.

        Returns:
            (bar_indices, trigger_bar)
        """
        if len(self._prices) < bb_period:
            raise ValueError(f"Need at least {bb_period} bars before pullback")

        # Calculate current BB lower
        recent_prices = self._prices[-bb_period:]
        sma = sum(recent_prices) / bb_period
        variance = sum((p - sma) ** 2 for p in recent_prices) / bb_period
        std_dev = math.sqrt(variance)
        bb_lower = sma - bb_mult * std_dev

        target_price = bb_lower * (1 - overshoot)
        start_price = self._current_price
        drop_per_bar = (start_price - target_price) / bars

        indices = []
        price = start_price
        trigger_bar = None

        for _ in range(bars):
            new_price = price - drop_per_bar

            if len(self._prices) >= bb_period:
                recent = self._prices[-(bb_period - 1) :] + [new_price]
                current_sma = sum(recent) / bb_period
                current_var = sum((p - current_sma) ** 2 for p in recent) / bb_period
                current_std = math.sqrt(current_var)
                current_bb_lower = current_sma - bb_mult * current_std

                if trigger_bar is None and new_price <= current_bb_lower:
                    trigger_bar = len(self._candles)

            self.add_candle(
                open_price=price,
                high_price=price + drop_per_bar * 0.2,
                low_price=new_price - drop_per_bar * 0.3,
                close_price=new_price,
                volume=2000.0,  # Higher volume on pullback
            )
            indices.append(len(self._candles) - 1)
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

    # =========================================================================
    # Pre-built scenarios
    # =========================================================================

    def generate_uptrend_with_pullback(
        self,
        n_bars: int = 200,
        pullback_bar: int = 100,
        start_price: float = 50000.0,
    ) -> list[ExpectedTrade]:
        """Generate uptrend with pullback scenario.

        Creates:
        1. Uptrend establishing EMA20 > EMA50
        2. Sharp pullback to BB lower
        3. Recovery

        Returns:
            List of expected trades
        """
        self.reset()

        # Phase 1: Establish uptrend
        self.add_uptrend(bars=pullback_bar, start_price=start_price, trend_strength=0.0008)

        # Phase 2: Pullback to BB lower
        self.add_pullback_to_bb_lower(bars=5, bb_period=20, bb_mult=2.0, overshoot=0.005)

        # Phase 3: Recovery
        remaining = n_bars - len(self._candles)
        if remaining > 0:
            self.add_recovery(bars=remaining, trend_strength=0.003)

        return self._calculate_expected_trades_trend_pullback()

    def generate_breakout(
        self,
        n_bars: int = 200,
        consolidation_bars: int = 100,
        start_price: float = 50000.0,
    ) -> list[ExpectedTrade]:
        """Generate consolidation followed by breakout.

        Returns:
            List of expected trades
        """
        self.reset()

        # Phase 1: Consolidation
        self.add_flat(bars=consolidation_bars, price=start_price, noise=start_price * 0.005)

        # Phase 2: Breakout
        self.add_uptrend(bars=10, trend_strength=0.005)

        # Phase 3: Continuation
        remaining = n_bars - len(self._candles)
        if remaining > 0:
            self.add_uptrend(bars=remaining, trend_strength=0.002)

        # Calculate expected entry
        max_50 = max(self._prices[:consolidation_bars])
        expected = []
        for i, price in enumerate(self._prices):
            if i > consolidation_bars and price > max_50:
                expected.append(
                    ExpectedTrade(
                        bar_index=i,
                        action="ENTRY",
                        direction="long",
                        price=price,
                        reason=f"Breakout: close({price:.2f}) > max_50({max_50:.2f})",
                    )
                )
                break

        return expected

    def _calculate_expected_trades_trend_pullback(
        self,
        ema_fast: int = 20,
        ema_slow: int = 50,
        bb_period: int = 20,
        bb_mult: float = 2.0,
    ) -> list[ExpectedTrade]:
        """Calculate expected trades for TrendPullback strategy."""
        ema_fast_values = self._calculate_ema(ema_fast)
        ema_slow_values = self._calculate_ema(ema_slow)
        bb_values = self._calculate_bb(bb_period, bb_mult)

        trades = []
        in_position = False
        warmup_bars = max(ema_fast, ema_slow, bb_period)

        for i in range(warmup_bars, len(self._prices)):
            price = self._prices[i]
            ema_f = ema_fast_values[i] if i < len(ema_fast_values) else None
            ema_s = ema_slow_values[i] if i < len(ema_slow_values) else None
            bb = bb_values[i] if i < len(bb_values) else None

            if ema_f is None or ema_s is None or bb is None:
                continue

            if not in_position:
                uptrend = ema_f > ema_s
                pullback = price < bb["lower"]

                if uptrend and pullback:
                    trades.append(
                        ExpectedTrade(
                            bar_index=i,
                            action="ENTRY",
                            direction="long",
                            price=price,
                            reason=f"EMA{ema_fast}({ema_f:.2f}) > EMA{ema_slow}({ema_s:.2f}), "
                            f"close({price:.2f}) < BB_lower({bb['lower']:.2f})",
                        )
                    )
                    in_position = True

        return trades

    def _calculate_ema(self, period: int) -> list[float | None]:
        """Calculate EMA for all prices."""
        if len(self._prices) < period:
            return []

        emas: list[float | None] = [None] * (period - 1)
        multiplier = 2 / (period + 1)

        sma = sum(self._prices[:period]) / period
        emas.append(sma)

        for i in range(period, len(self._prices)):
            ema = (self._prices[i] - emas[-1]) * multiplier + emas[-1]
            emas.append(ema)

        return emas

    def _calculate_bb(
        self, period: int = 20, mult: float = 2.0
    ) -> list[dict[str, float] | None]:
        """Calculate Bollinger Bands for all prices."""
        if len(self._prices) < period:
            return []

        bands: list[dict[str, float] | None] = []
        for i in range(len(self._prices)):
            if i < period - 1:
                bands.append(None)
                continue

            window = self._prices[i - period + 1 : i + 1]
            sma = sum(window) / period
            variance = sum((p - sma) ** 2 for p in window) / period
            std_dev = math.sqrt(variance)

            bands.append(
                {
                    "middle": sma,
                    "upper": sma + mult * std_dev,
                    "lower": sma - mult * std_dev,
                    "std_dev": std_dev,
                }
            )

        return bands


# Convenience functions for common scenarios
def create_uptrend_pullback_data(
    output_dir: Path,
    symbol: str = "BTCUSD",
    start_date: datetime = datetime(2024, 1, 1),
    n_bars: int = 200,
) -> tuple[list[Candle], list[ExpectedTrade]]:
    """Create uptrend+pullback test data and write to LEAN format.

    Args:
        output_dir: Directory to write LEAN data files
        symbol: Trading symbol
        start_date: Start date for candles
        n_bars: Number of bars to generate

    Returns:
        Tuple of (candles, expected_trades)
    """
    loader = SyntheticDataLoader()
    expected = loader.generate_uptrend_with_pullback(n_bars=n_bars)
    candles = loader.load(symbol, start_date, start_date)

    # Write using shared method
    loader.write_lean_format(candles, symbol, output_dir)

    return candles, expected


def create_breakout_data(
    output_dir: Path,
    symbol: str = "BTCUSD",
    start_date: datetime = datetime(2024, 1, 1),
    n_bars: int = 200,
) -> tuple[list[Candle], list[ExpectedTrade]]:
    """Create breakout test data and write to LEAN format.

    Args:
        output_dir: Directory to write LEAN data files
        symbol: Trading symbol
        start_date: Start date for candles
        n_bars: Number of bars to generate

    Returns:
        Tuple of (candles, expected_trades)
    """
    loader = SyntheticDataLoader()
    expected = loader.generate_breakout(n_bars=n_bars)
    candles = loader.load(symbol, start_date, start_date)

    # Write using shared method
    loader.write_lean_format(candles, symbol, output_dir)

    return candles, expected

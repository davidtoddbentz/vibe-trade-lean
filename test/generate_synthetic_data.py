#!/usr/bin/env python3
"""Generate synthetic test data for LEAN backtesting.

This script generates deterministic OHLCV data with known signal patterns
and writes it in LEAN-compatible format for end-to-end testing.

Uses SyntheticDataLoader which shares write_lean_format() with BigQueryDataLoader,
ensuring identical output format between test and production.

Usage:
    python generate_synthetic_data.py --scenario trend_pullback --output /Data
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add scripts to path for data loader imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from synthetic_data_loader import (
    SyntheticDataLoader,
    ExpectedTrade,
    create_uptrend_pullback_data,
    create_breakout_data,
)
from data_loader_base import Candle


def write_lean_data(candles: list, symbol: str, start_date: datetime, output_dir: Path) -> None:
    """Write candles to LEAN-compatible format using shared DataLoader method.

    This function is kept for backward compatibility with existing tests.
    It delegates to SyntheticDataLoader.write_lean_format().

    Args:
        candles: List of candles (either Candle objects or dicts)
        symbol: Symbol name (e.g., "BTCUSD")
        start_date: Start date for timestamping
        output_dir: Path to output directory
    """
    # Convert to Candle objects if needed
    candle_objects = []
    for i, c in enumerate(candles):
        if isinstance(c, Candle):
            # Re-timestamp
            candle_objects.append(Candle(
                timestamp=start_date + __import__('datetime').timedelta(minutes=i),
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            ))
        elif hasattr(c, 'open'):
            # OHLCV dataclass from old test_data_builder
            candle_objects.append(Candle(
                timestamp=start_date + __import__('datetime').timedelta(minutes=i),
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=getattr(c, 'volume', 1000.0),
            ))
        else:
            # Dict
            candle_objects.append(Candle(
                timestamp=start_date + __import__('datetime').timedelta(minutes=i),
                open=c['open'],
                high=c['high'],
                low=c['low'],
                close=c['close'],
                volume=c.get('volume', 1000.0),
            ))

    # Use shared write method
    loader = SyntheticDataLoader()
    loader.write_lean_format(candle_objects, symbol, output_dir)


def generate_uptrend_with_pullback(
    n_bars: int = 200,
    pullback_bar: int = 100,
    start_price: float = 50000.0,
) -> tuple[list[Candle], list[ExpectedTrade]]:
    """Generate uptrend data with a pullback at specified bar.

    Creates synthetic price data where:
    1. First `pullback_bar` bars establish an uptrend (EMA20 > EMA50)
    2. Sharp pullback to lower BB
    3. Recovery to profit target

    Args:
        n_bars: Total number of bars
        pullback_bar: Bar index where pullback starts
        start_price: Starting price

    Returns:
        Tuple of (candles, expected_trades)
    """
    loader = SyntheticDataLoader()

    # Phase 1: Establish uptrend
    loader.add_uptrend(bars=pullback_bar, start_price=start_price, trend_strength=0.0008)

    # Phase 2: Pullback to BB lower
    loader.add_pullback_to_bb_lower(bars=5, bb_period=20, bb_mult=2.0, overshoot=0.005)

    # Phase 3: Recovery
    remaining = n_bars - len(loader._candles)
    if remaining > 0:
        loader.add_recovery(bars=remaining, trend_strength=0.003)

    # Get expected trades
    expected = loader._calculate_expected_trades_trend_pullback()

    # Get candles with proper timestamps
    candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

    return candles, expected


def generate_breakout(
    n_bars: int = 200,
    consolidation_bars: int = 100,
    start_price: float = 50000.0,
) -> tuple[list[Candle], list[ExpectedTrade]]:
    """Generate consolidation followed by breakout.

    Creates synthetic price data where:
    1. First `consolidation_bars` bars are flat/ranging
    2. Strong breakout above recent high
    3. Continuation

    Args:
        n_bars: Total number of bars
        consolidation_bars: Bars of consolidation before breakout
        start_price: Starting price

    Returns:
        Tuple of (candles, expected_trades)
    """
    loader = SyntheticDataLoader()
    expected = loader.generate_breakout(
        n_bars=n_bars,
        consolidation_bars=consolidation_bars,
        start_price=start_price,
    )
    candles = loader.load("BTCUSD", datetime(2024, 1, 1), datetime(2024, 1, 1))

    return candles, expected


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data for LEAN")
    parser.add_argument(
        "--scenario",
        type=str,
        default="trend_pullback",
        choices=["trend_pullback", "breakout"],
        help="Test scenario to generate",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSD",
        help="Symbol name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test/data",
        help="Output directory",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )

    args = parser.parse_args()

    print(f"Generating {args.scenario} scenario...")

    output_dir = Path(args.output)
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    if args.scenario == "trend_pullback":
        candles, expected = create_uptrend_pullback_data(
            output_dir=output_dir,
            symbol=args.symbol,
            start_date=start_date,
        )
    elif args.scenario == "breakout":
        candles, expected = create_breakout_data(
            output_dir=output_dir,
            symbol=args.symbol,
            start_date=start_date,
        )
    else:
        print(f"Unknown scenario: {args.scenario}")
        sys.exit(1)

    print("\nGenerated data summary:")
    print(f"  Total bars: {len(candles)}")
    print(f"  Expected trades: {len(expected)}")

    for trade in expected:
        print(f"    Bar {trade.bar_index}: {trade.action} {trade.direction} @ {trade.price:.2f}")
        print(f"      Reason: {trade.reason}")

    print("\nData written to LEAN format!")
    print(f"  Path: {output_dir}/crypto/coinbase/minute/{args.symbol.lower()}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export BigQuery candle data to LEAN format.

Converts candle data from BigQuery to LEAN's expected format:
- Path: /Data/crypto/{market}/minute/{symbol}/{YYYYMMDD}_trade.zip
- Format: ms_since_midnight,open,high,low,close,volume (no header)

Usage:
    # Export single day
    python scripts/export_to_lean.py --symbol BTC-USD --date 2024-01-01 --output ./data

    # Export date range
    python scripts/export_to_lean.py --symbol BTC-USD --start 2024-01-01 --end 2024-01-07 --output ./data

    # Export multiple symbols
    python scripts/export_to_lean.py --symbols BTC-USD ETH-USD --days 30 --output ./data
"""

import argparse
import logging
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from google.cloud import bigquery

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# BigQuery configuration
DEFAULT_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704")
DATASET = "market_data"
TABLE = "candles_parsed"


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol for LEAN path (BTC-USD -> btcusd)."""
    return symbol.lower().replace("-", "")


def get_lean_path(output_dir: str, market: str, symbol: str, date: datetime) -> Path:
    """Get the LEAN data path for a symbol and date."""
    symbol_normalized = normalize_symbol(symbol)
    date_str = date.strftime("%Y%m%d")
    return Path(output_dir) / "crypto" / market / "minute" / symbol_normalized / f"{date_str}_trade.zip"


def query_candles(
    client: bigquery.Client,
    project_id: str,
    symbol: str,
    date: datetime,
    granularity: str = "1m",
) -> Iterator[dict]:
    """Query candles from BigQuery for a specific symbol and date."""

    date_str = date.strftime("%Y-%m-%d")

    query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM `{project_id}.{DATASET}.{TABLE}`
        WHERE symbol = @symbol
        AND granularity = @granularity
        AND DATE(timestamp) = @date
        ORDER BY timestamp
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
            bigquery.ScalarQueryParameter("granularity", "STRING", granularity),
            bigquery.ScalarQueryParameter("date", "DATE", date_str),
        ]
    )

    query_job = client.query(query, job_config=job_config)

    for row in query_job:
        yield {
            "timestamp": row.timestamp,
            "open": row.open,
            "high": row.high,
            "low": row.low,
            "close": row.close,
            "volume": row.volume,
        }


def timestamp_to_ms_since_midnight(ts: datetime) -> int:
    """Convert timestamp to milliseconds since midnight UTC."""
    midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
    delta = ts - midnight
    return int(delta.total_seconds() * 1000)


def export_day(
    client: bigquery.Client,
    project_id: str,
    symbol: str,
    date: datetime,
    output_dir: str,
    market: str = "coinbase",
) -> int:
    """Export one day of candle data to LEAN format.

    Returns:
        Number of candles exported
    """
    # Get output path
    zip_path = get_lean_path(output_dir, market, symbol, date)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Query candles
    candles = list(query_candles(client, project_id, symbol, date))

    if not candles:
        logger.warning(f"  No candles found for {symbol} on {date.date()}")
        return 0

    # Write to CSV in memory, then ZIP
    symbol_normalized = normalize_symbol(symbol)
    date_str = date.strftime("%Y%m%d")
    csv_filename = f"{date_str}_{symbol_normalized}_minute_trade.csv"

    csv_lines = []
    for candle in candles:
        ms = timestamp_to_ms_since_midnight(candle["timestamp"])
        # LEAN format: 4 decimal places for prices
        line = f"{ms},{candle['open']:.4f},{candle['high']:.4f},{candle['low']:.4f},{candle['close']:.4f},{candle['volume']}"
        csv_lines.append(line)

    csv_content = "\n".join(csv_lines)

    # Write ZIP file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_filename, csv_content)

    logger.info(f"  Exported {len(candles)} candles to {zip_path}")
    return len(candles)


def export_symbol(
    client: bigquery.Client,
    project_id: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: str,
    market: str = "coinbase",
) -> int:
    """Export a date range of candle data for a symbol.

    Returns:
        Total number of candles exported
    """
    logger.info(f"Exporting {symbol} from {start_date.date()} to {end_date.date()}")

    total_candles = 0
    current_date = start_date

    while current_date <= end_date:
        try:
            count = export_day(client, project_id, symbol, current_date, output_dir, market)
            total_candles += count
        except Exception as e:
            logger.error(f"  Error exporting {current_date.date()}: {e}")

        current_date += timedelta(days=1)

    return total_candles


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime (UTC)."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def main():
    parser = argparse.ArgumentParser(
        description="Export BigQuery candles to LEAN format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single day, single symbol
    python scripts/export_to_lean.py --symbol BTC-USD --date 2024-01-01 --output ./data

    # Date range
    python scripts/export_to_lean.py --symbol BTC-USD --start 2024-01-01 --end 2024-01-07 --output ./data

    # Multiple symbols, last N days
    python scripts/export_to_lean.py --symbols BTC-USD ETH-USD --days 30 --output ./data
        """
    )

    # Symbol selection
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument("--symbol", type=str, help="Single symbol (e.g., BTC-USD)")
    symbol_group.add_argument("--symbols", type=str, nargs="+", help="Multiple symbols")

    # Date selection
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument("--date", type=str, help="Single date (YYYY-MM-DD)")
    date_group.add_argument("--days", type=int, help="Export last N days")
    date_group.add_argument("--start", type=str, help="Start date (requires --end)")

    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")

    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (LEAN data folder)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="coinbase",
        help="Market name (default: coinbase)"
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"GCP project ID (default: {DEFAULT_PROJECT})"
    )

    args = parser.parse_args()

    # Validate date arguments
    if args.start and not args.end:
        parser.error("--start requires --end")

    # Determine date range
    if args.date:
        start_date = parse_date(args.date)
        end_date = start_date
    elif args.days:
        end_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=args.days)
    else:
        start_date = parse_date(args.start)
        end_date = parse_date(args.end)

    # Determine symbols
    symbols = args.symbols if args.symbols else [args.symbol]

    logger.info("=" * 60)
    logger.info("BigQuery to LEAN Export")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Market: {args.market}")
    logger.info(f"Project: {args.project_id}")
    logger.info("=" * 60)

    # Initialize BigQuery client
    client = bigquery.Client(project=args.project_id)

    # Export each symbol
    total_all = 0
    for symbol in symbols:
        try:
            count = export_symbol(
                client=client,
                project_id=args.project_id,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                output_dir=args.output,
                market=args.market,
            )
            total_all += count
            logger.info(f"Completed {symbol}: {count} candles")
        except Exception as e:
            logger.error(f"Failed to export {symbol}: {e}")

    logger.info("=" * 60)
    logger.info(f"Export complete! Total candles: {total_all}")
    logger.info("=" * 60)

    # Print LEAN data path for reference
    logger.info("")
    logger.info("LEAN data structure created at:")
    for symbol in symbols:
        symbol_norm = normalize_symbol(symbol)
        path = Path(args.output) / "crypto" / args.market / "minute" / symbol_norm
        logger.info(f"  {path}/")


if __name__ == "__main__":
    main()

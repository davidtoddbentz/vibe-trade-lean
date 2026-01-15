#!/usr/bin/env python3
"""BigQuery Data Loader for LEAN Backtesting.

Downloads historical candle data from BigQuery and writes it in LEAN-compatible format.
Runs before LEAN starts to prepare data for backtesting.

Environment Variables:
    GOOGLE_CLOUD_PROJECT: GCP project ID (required)
    SYMBOL: Trading symbol, e.g., "BTC-USD" (required)
    START_DATE: Backtest start date, format YYYYMMDD (required)
    END_DATE: Backtest end date, format YYYYMMDD (required)
    DATA_FOLDER: Output directory for LEAN data (default: /Data)
    GRANULARITY: Data granularity, e.g., "1m" (default: 1m)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import bigquery

from data_loader_base import Candle, DataLoader


class BigQueryDataLoader(DataLoader):
    """Load candle data from BigQuery."""

    def __init__(self, project_id: str, granularity: str = "1m"):
        """Initialize BigQuery data loader.

        Args:
            project_id: GCP project ID
            granularity: Data granularity (e.g., "1m", "5m", "1h")
        """
        self.project_id = project_id
        self.granularity = granularity
        self.client = bigquery.Client(project=project_id)

    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Candle]:
        """Load candles from BigQuery.

        Queries the market_data.candles_parsed view for the specified
        symbol and date range.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of Candle objects sorted by timestamp
        """
        query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM `{self.project_id}.market_data.candles_parsed`
        WHERE symbol = @symbol
          AND granularity = @granularity
          AND timestamp >= @start_date
          AND timestamp < @end_date
        ORDER BY timestamp
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("granularity", "STRING", self.granularity),
                bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date),
                bigquery.ScalarQueryParameter(
                    "end_date", "TIMESTAMP", end_date + timedelta(days=1)
                ),
            ]
        )

        print(f"Querying BigQuery for {symbol} from {start_date.date()} to {end_date.date()}...")
        query_job = self.client.query(query, job_config=job_config)
        results = list(query_job.result())
        print(f"Retrieved {len(results)} candles")

        return [
            Candle(
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in results
        ]


def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYYMMDD format."""
    return datetime.strptime(date_str, "%Y%m%d")


def main():
    """Main entry point for CLI usage."""
    # Read environment variables
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    symbol = os.environ.get("SYMBOL")
    start_date_str = os.environ.get("START_DATE")
    end_date_str = os.environ.get("END_DATE")
    data_folder = os.environ.get("DATA_FOLDER", "/Data")
    granularity = os.environ.get("GRANULARITY", "1m")

    # Validate required env vars
    missing = []
    if not project_id:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not symbol:
        missing.append("SYMBOL")
    if not start_date_str:
        missing.append("START_DATE")
    if not end_date_str:
        missing.append("END_DATE")

    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Parse dates
    try:
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
    except ValueError as e:
        print(f"ERROR: Invalid date format. Expected YYYYMMDD. Error: {e}")
        sys.exit(1)

    print("=== BigQuery Data Loader ===")
    print(f"Project: {project_id}")
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date.date()} to {end_date.date()}")
    print(f"Granularity: {granularity}")
    print(f"Output: {data_folder}")
    print()

    # Create loader and fetch data
    loader = BigQueryDataLoader(project_id=project_id, granularity=granularity)
    candles = loader.load(symbol=symbol, start_date=start_date, end_date=end_date)

    if not candles:
        print(f"WARNING: No data found for {symbol} in the specified date range")
        print("Continuing with empty data set...")
        return

    # Write LEAN-compatible files using shared method
    print("\nWriting LEAN data files...")
    files_written = loader.write_lean_format(
        candles=candles,
        symbol=symbol,
        output_dir=Path(data_folder),
    )

    print("\n=== Data Loader Complete ===")
    print(f"Total files written: {files_written}")
    print(f"Total candles: {len(candles)}")


if __name__ == "__main__":
    main()

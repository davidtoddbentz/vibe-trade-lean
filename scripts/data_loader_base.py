"""Base classes for LEAN data loaders.

This module defines the common interface for loading market data,
ensuring both BigQuery and synthetic data loaders produce identical output.
"""

import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Candle:
    """Unified candle representation.

    All data loaders must produce candles in this format.
    """
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_lean_csv_row(self, day_start: datetime) -> str:
        """Convert to LEAN CSV format.

        LEAN minute data format:
        - Column 1: Milliseconds since midnight UTC
        - Column 2-5: Open, High, Low, Close (raw values for crypto)
        - Column 6: Volume

        Args:
            day_start: Midnight UTC of the candle's day

        Returns:
            CSV row string
        """
        ms_since_midnight = int((self.timestamp - day_start).total_seconds() * 1000)
        return f"{ms_since_midnight},{self.open},{self.high},{self.low},{self.close},{self.volume}"


class DataLoader(ABC):
    """Abstract base class for LEAN data loaders.

    Implementations must provide `load()` to fetch candles from their source.
    The `write_lean_format()` method is shared to guarantee identical output.
    """

    @abstractmethod
    def load(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[Candle]:
        """Load candles from the data source.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            List of Candle objects sorted by timestamp
        """
        ...

    def write_lean_format(
        self,
        candles: list[Candle],
        symbol: str,
        output_dir: Path,
        market: str = "coinbase",
        resolution: str = "minute",
    ) -> int:
        """Write candles to LEAN-compatible zip files.

        Output path: {output_dir}/crypto/{market}/{resolution}/{symbol}/{date}_trade.zip
        Each zip contains a CSV file with the candle data.

        This method is shared across all loaders to guarantee identical output format.

        Args:
            candles: List of Candle objects to write
            symbol: Trading symbol (e.g., "BTC-USD")
            output_dir: Root data directory (e.g., /Data)
            market: Market name (default: "coinbase")
            resolution: Data resolution (default: "minute")

        Returns:
            Number of files written
        """
        if not candles:
            return 0

        # Normalize symbol for file paths (BTC-USD -> btcusd)
        symbol_normalized = symbol.lower().replace("-", "").replace(".", "").replace("/", "")

        # Create output directory
        base_path = output_dir / "crypto" / market / resolution / symbol_normalized
        base_path.mkdir(parents=True, exist_ok=True)

        # Group candles by date
        grouped = self._group_by_date(candles)
        files_written = 0

        for date_str, day_candles in grouped.items():
            # Sort by timestamp
            day_candles.sort(key=lambda c: c.timestamp)

            # Calculate day start for ms offset
            day_start = day_candles[0].timestamp.replace(
                hour=0, minute=0, second=0, microsecond=0
            )

            # Generate CSV content
            csv_lines = [c.to_lean_csv_row(day_start) for c in day_candles]
            csv_content = "\n".join(csv_lines)

            # Write to zip file
            zip_filename = f"{date_str}_trade.zip"
            zip_path = base_path / zip_filename
            csv_filename = f"{date_str}_trade.csv"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(csv_filename, csv_content)

            files_written += 1

        return files_written

    def _group_by_date(self, candles: list[Candle]) -> dict[str, list[Candle]]:
        """Group candles by date string (YYYYMMDD)."""
        grouped: dict[str, list[Candle]] = {}
        for candle in candles:
            date_str = candle.timestamp.strftime("%Y%m%d")
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(candle)
        return grouped

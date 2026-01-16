#!/usr/bin/env python3
"""HTTP service for running backtests on Cloud Run.

This wraps the backtest logic in a FastAPI endpoint for use with
Cloud Run Services (vs Jobs) to benefit from warm containers.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vibe Trade Backtest Service")

# Configuration from environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704")
RESULTS_BUCKET = os.environ.get("RESULTS_BUCKET", "vibe-trade-backtest-results")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "batch-save")


def generate_backtest_id() -> str:
    """Generate a structured backtest ID: {timestamp}_{short_uuid}."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_id}"


# =============================================================================
# Request Models - aligned with vibe-trade-execution's LEANBacktestRequest
# =============================================================================


class OHLCVBar(BaseModel):
    """OHLCV candle data from execution service.

    Field names match vibe-trade-shared's OHLCVBar:
    - t: timestamp in milliseconds since epoch
    - o, h, l, c, v: OHLCV values
    """
    t: int  # timestamp (ms since epoch)
    o: float  # open
    h: float  # high
    l: float  # low
    c: float  # close
    v: float  # volume


class BacktestDataInput(BaseModel):
    """Data input for backtest - inline or GCS reference."""
    symbol: str
    resolution: str
    bars: list[OHLCVBar] | None = None  # Inline data
    gcs_uri: str | None = None  # GCS reference (future)


class BacktestConfig(BaseModel):
    """Configuration for backtest execution."""
    start_date: str  # YYYY-MM-DD format
    end_date: str
    initial_cash: float = 100000.0


class LEANBacktestRequest(BaseModel):
    """Request format from vibe-trade-execution.

    This matches execution's LEANBacktestRequest model.
    """
    strategy_ir: dict[str, Any]
    data: BacktestDataInput
    config: BacktestConfig
    # Additional symbol data for multi-symbol strategies
    additional_data: list[BacktestDataInput] = []
    # Optional: skip GCS upload (for testing)
    skip_gcs_upload: bool = False


# Legacy format for backward compatibility
class BacktestRequest(BaseModel):
    """Legacy request format (for direct calls)."""
    strategy_id: str
    strategy_ir: dict[str, Any]
    symbol: str = "BTC-USD"
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    initial_cash: float = 100000.0
    market_data: list[dict] | None = None  # List of OHLCVBar dicts
    skip_gcs_upload: bool = False


# =============================================================================
# Response Models
# =============================================================================


class Trade(BaseModel):
    """Single trade from backtest - matches execution's Trade model."""
    entry_bar: int
    entry_price: float
    entry_time: datetime
    exit_bar: int | None = None
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_reason: str | None = None  # Exit rule ID or "end_of_backtest"
    direction: str  # "long" or "short"
    quantity: float
    pnl: float | None = None
    pnl_pct: float | None = None


class LEANBacktestSummary(BaseModel):
    """Summary metrics - matches execution's BacktestSummary model."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None


class LEANBacktestResponse(BaseModel):
    """Response format expected by execution service."""
    status: str  # "success" or "error"
    trades: list[Trade] = []
    summary: LEANBacktestSummary | None = None
    equity_curve: list[float] | None = None
    error: str | None = None


# Legacy response format
class BacktestSummary(BaseModel):
    """Legacy summary statistics."""
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float


class BacktestResponse(BaseModel):
    """Legacy response format."""
    backtest_id: str
    strategy_id: str
    status: str
    created_at: str
    results_path: str | None = None
    summary: BacktestSummary | None = None
    error: str | None = None
    duration_seconds: float | None = None
    strategy_output: dict[str, Any] | None = None


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/backtest", response_model=LEANBacktestResponse)
async def run_backtest(request: LEANBacktestRequest):
    """Run a backtest using the new LEAN request format.

    This endpoint matches vibe-trade-execution's expected interface.
    """
    backtest_id = generate_backtest_id()
    start_time = datetime.now()

    logger.info(f"Starting backtest {backtest_id}")

    try:
        result = await _run_lean_backtest(request)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Backtest {backtest_id} completed in {duration:.1f}s")
        return result
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
        return LEANBacktestResponse(
            status="error",
            error=str(e),
        )


async def _run_lean_backtest(request: LEANBacktestRequest) -> LEANBacktestResponse:
    """Execute backtest using new LEAN request format.

    This function:
    1. Writes strategy IR to temp file
    2. Converts inline bars to LEAN CSV format
    3. Runs LEAN via subprocess
    4. Parses output and returns LEANBacktestResponse
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Setup directories
        data_dir = temp_path / "Data"
        algo_dir = temp_path / "Algorithms"
        results_dir = temp_path / "Results"

        data_dir.mkdir()
        algo_dir.mkdir()
        results_dir.mkdir()

        # Write strategy IR to file
        ir_path = data_dir / "strategy_ir.json"
        with open(ir_path, "w") as f:
            json.dump(request.strategy_ir, f, indent=2)

        # Write inline market data for primary symbol
        if request.data.bars:
            candle_count = _write_ohlcv_bars_to_csv(
                request.data.bars, request.data.symbol, data_dir
            )
            logger.info(f"Using {candle_count} inline bars for primary symbol {request.data.symbol}")
        else:
            return LEANBacktestResponse(
                status="error",
                error="Inline bars required (GCS not yet supported)",
            )

        if candle_count == 0:
            return LEANBacktestResponse(
                status="error",
                error=f"No market data provided for {request.data.symbol}",
            )

        # Write inline market data for additional symbols (multi-symbol strategies)
        for additional in request.additional_data:
            if additional.bars:
                additional_count = _write_ohlcv_bars_to_csv(
                    additional.bars, additional.symbol, data_dir
                )
                logger.info(f"Using {additional_count} inline bars for {additional.symbol}")

        # Copy LEAN data files
        _copy_lean_data_files(data_dir)

        # Copy StrategyRuntime
        runtime_src = Path("/Lean/Algorithm.Python/StrategyRuntime.py")
        if not runtime_src.exists():
            runtime_src = Path(__file__).parent / "Algorithms" / "StrategyRuntime.py"
        shutil.copy(runtime_src, algo_dir / "StrategyRuntime.py")

        # Parse dates from config (YYYY-MM-DD format)
        start_date = request.config.start_date.replace("-", "")
        end_date = request.config.end_date.replace("-", "")

        # Run LEAN
        lean_result = _run_lean(
            algo_dir=algo_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            ir_path=str(ir_path),
            start_date=start_date,
            end_date=end_date,
            initial_cash=request.config.initial_cash,
        )

        # Capture strategy output
        strategy_output = None
        strategy_output_path = data_dir / "strategy_output.json"
        if strategy_output_path.exists():
            with open(strategy_output_path, "r") as f:
                strategy_output = json.load(f)

        if lean_result.get("status") == "error":
            return LEANBacktestResponse(
                status="error",
                error=lean_result.get("stderr", "Unknown LEAN error"),
            )

        # Build response from strategy output
        trades = []
        summary = None

        if strategy_output:
            # Convert trades from output
            raw_trades = strategy_output.get("trades", [])
            for i, t in enumerate(raw_trades):
                trades.append(Trade(
                    entry_bar=t.get("entry_bar", i),
                    entry_price=t.get("entry_price", 0),
                    entry_time=datetime.fromisoformat(t["entry_time"]) if t.get("entry_time") else datetime.now(timezone.utc),
                    exit_bar=t.get("exit_bar"),
                    exit_price=t.get("exit_price"),
                    exit_time=datetime.fromisoformat(t["exit_time"]) if t.get("exit_time") else None,
                    exit_reason=t.get("exit_reason"),
                    direction=t.get("direction", "long"),
                    quantity=t.get("quantity", 0),
                    pnl=t.get("pnl"),
                    pnl_pct=t.get("pnl_percent") or t.get("pnl_pct"),  # StrategyRuntime uses pnl_percent
                ))

            # Build summary
            stats = strategy_output.get("statistics", {})
            initial_cash = strategy_output.get("initial_cash", request.config.initial_cash)
            final_equity = strategy_output.get("final_equity", initial_cash)
            total_pnl = final_equity - initial_cash
            total_pnl_pct = (total_pnl / initial_cash) * 100 if initial_cash else 0

            summary = LEANBacktestSummary(
                total_trades=stats.get("total_trades", len(trades)),
                winning_trades=stats.get("winning_trades", 0),
                losing_trades=stats.get("losing_trades", 0),
                total_pnl=total_pnl,
                total_pnl_pct=round(total_pnl_pct, 2),
                max_drawdown_pct=round(strategy_output.get("max_drawdown_pct", 0), 2),
            )

            # Extract equity curve
            equity_curve = [e.get("equity", initial_cash) for e in strategy_output.get("equity_curve", [])]

        return LEANBacktestResponse(
            status="success",
            trades=trades,
            summary=summary,
            equity_curve=equity_curve if strategy_output else None,
        )


async def _run_backtest_internal(
    request: BacktestRequest,
    backtest_id: str,
    created_at: datetime,
) -> BacktestResponse:
    """Internal backtest execution (legacy format)."""
    storage_client = None if request.skip_gcs_upload else storage.Client()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Setup directories
        data_dir = temp_path / "Data"
        algo_dir = temp_path / "Algorithms"
        results_dir = temp_path / "Results"

        data_dir.mkdir()
        algo_dir.mkdir()
        results_dir.mkdir()

        # Write strategy IR to file
        ir_path = data_dir / "strategy_ir.json"
        with open(ir_path, "w") as f:
            json.dump(request.strategy_ir, f, indent=2)

        # Fetch or use inline market data
        if request.market_data:
            # Use inline market data (for testing)
            candle_count = _write_inline_market_data(
                request.market_data, request.symbol, data_dir
            )
            logger.info(f"Using {candle_count} inline market data bars")
        else:
            # Fetch from GCS
            start_date = datetime.strptime(request.start_date, "%Y%m%d")
            end_date = datetime.strptime(request.end_date, "%Y%m%d")

            candle_count = _fetch_market_data(
                DATA_BUCKET, request.symbol, start_date, end_date, data_dir
            )

        if candle_count == 0:
            return BacktestResponse(
                backtest_id=backtest_id,
                strategy_id=request.strategy_id,
                status="error",
                created_at=created_at.isoformat(),
                error=f"No market data found for {request.symbol}",
            )

        # Copy LEAN data files
        _copy_lean_data_files(data_dir)

        # Copy StrategyRuntime
        runtime_src = Path("/Lean/Algorithm.Python/StrategyRuntime.py")
        if not runtime_src.exists():
            runtime_src = Path(__file__).parent / "Algorithms" / "StrategyRuntime.py"
        shutil.copy(runtime_src, algo_dir / "StrategyRuntime.py")

        # Run LEAN
        lean_result = _run_lean(
            algo_dir=algo_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            ir_path=str(ir_path),
            start_date=request.start_date,
            end_date=request.end_date,
            initial_cash=request.initial_cash,
        )

        # Capture strategy output
        strategy_output = None
        strategy_output_path = data_dir / "strategy_output.json"
        if strategy_output_path.exists():
            with open(strategy_output_path, "r") as f:
                strategy_output = json.load(f)

        # Build summary from strategy output
        summary = None
        if strategy_output:
            stats = strategy_output.get("statistics", {})
            initial_cash = strategy_output.get("initial_cash", request.initial_cash)
            final_equity = strategy_output.get("final_equity", initial_cash)
            total_return = ((final_equity - initial_cash) / initial_cash) * 100 if initial_cash else 0

            summary = BacktestSummary(
                final_equity=final_equity,
                total_return_pct=round(total_return, 2),
                max_drawdown_pct=round(strategy_output.get("max_drawdown_pct", 0), 2),
                total_trades=stats.get("total_trades", 0),
                win_rate=round(stats.get("win_rate", 0), 1),
                profit_factor=round(stats.get("profit_factor", 0), 2),
            )

        # GCS path organized by strategy_id
        gcs_folder = f"backtests/{request.strategy_id}/{backtest_id}"
        results_gcs_path: str | None = None

        # Full result for GCS storage
        full_result = {
            "backtest_id": backtest_id,
            "strategy_id": request.strategy_id,
            "created_at": created_at.isoformat(),
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_cash": request.initial_cash,
            "candle_count": candle_count,
            "status": lean_result.get("status", "unknown"),
            "strategy_output": strategy_output,
        }

        # Upload to GCS unless skip_gcs_upload is set
        if storage_client and not request.skip_gcs_upload:
            bucket = storage_client.bucket(RESULTS_BUCKET)

            # Upload results.json
            blob = bucket.blob(f"{gcs_folder}/results.json")
            blob.upload_from_string(json.dumps(full_result, indent=2))

            # Also upload strategy_ir.json for reference
            ir_blob = bucket.blob(f"{gcs_folder}/strategy_ir.json")
            ir_blob.upload_from_string(json.dumps(request.strategy_ir, indent=2))

            results_gcs_path = f"gs://{RESULTS_BUCKET}/{gcs_folder}/"

        return BacktestResponse(
            backtest_id=backtest_id,
            strategy_id=request.strategy_id,
            status=lean_result.get("status", "unknown"),
            created_at=created_at.isoformat(),
            results_path=results_gcs_path,
            summary=summary,
            error=lean_result.get("stderr") if lean_result.get("status") == "error" else None,
            # Include full strategy output for test inspection when not uploading to GCS
            strategy_output=strategy_output if request.skip_gcs_upload else None,
        )


def _write_inline_market_data(
    bars: list[OHLCVBar],
    symbol: str,
    output_dir: Path,
) -> int:
    """Write inline market data to LEAN format.

    Args:
        bars: List of OHLCV bars from request
        symbol: Trading symbol
        output_dir: Data output directory

    Returns:
        Number of bars written
    """
    from src.data import LeanDataExporter
    from src.data.models import Candle

    # Convert OHLCVBar to Candle objects
    candles = [
        Candle(
            symbol=symbol,
            timestamp=datetime.fromisoformat(bar.timestamp.replace("Z", "+00:00")),
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            resolution="1m",
        )
        for bar in bars
    ]

    if not candles:
        return 0

    exporter = LeanDataExporter(output_dir)
    exporter.export_candles(candles, symbol)

    return len(candles)


def _write_ohlcv_bars_to_csv(
    bars: list[OHLCVBar],
    symbol: str,
    output_dir: Path,
) -> int:
    """Write OHLCVBar data (t,o,h,l,c,v format) to LEAN CSV.

    Args:
        bars: List of OHLCVBar with t (ms timestamp), o, h, l, c, v fields
        symbol: Trading symbol
        output_dir: Data output directory

    Returns:
        Number of bars written
    """
    import csv
    from datetime import timezone

    if not bars:
        return 0

    # Normalize symbol for filename (e.g., BTC-USD -> btc_usd_data.csv)
    symbol_normalized = symbol.lower().replace("-", "_")
    filename = f"{symbol_normalized}_data.csv"
    file_path = output_dir / filename

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "open", "high", "low", "close", "volume"])

        for bar in bars:
            # Convert ms timestamp to datetime
            ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            dt_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([
                dt_str,
                f"{bar.o:.2f}",
                f"{bar.h:.2f}",
                f"{bar.l:.2f}",
                f"{bar.c:.2f}",
                f"{bar.v:.2f}",
            ])

    logger.info(f"Wrote {len(bars)} bars to {file_path}")
    return len(bars)


def _fetch_market_data(
    bucket: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> int:
    """Fetch market data from GCS."""
    # Import here to avoid slow startup
    from src.data import DataFetcher, LeanDataExporter

    fetcher = DataFetcher(bucket_name=bucket)
    candles = fetcher.fetch_candles(symbol, start_date, end_date)

    if not candles:
        logger.warning(f"No candles found for {symbol}")
        return 0

    exporter = LeanDataExporter(output_dir)
    exporter.export_candles(candles, symbol)

    logger.info(f"Fetched and exported {len(candles)} candles for {symbol}")
    return len(candles)


def _copy_lean_data_files(dest_dir: Path) -> None:
    """Copy required LEAN data files."""
    lean_data = Path("/Lean/Data")

    for subdir in ["symbol-properties", "market-hours"]:
        src = lean_data / subdir
        if src.exists():
            shutil.copytree(src, dest_dir / subdir)
            logger.info(f"Copied {subdir} to {dest_dir}")


def _run_lean(
    algo_dir: Path,
    data_dir: Path,
    results_dir: Path,
    ir_path: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> dict:
    """Run LEAN backtest."""
    config = {
        "environment": "backtesting",
        "algorithm-type-name": "StrategyRuntime",
        "algorithm-language": "Python",
        "algorithm-location": str(algo_dir / "StrategyRuntime.py"),
        "data-folder": str(data_dir),
        "results-destination-folder": str(results_dir),
        "parameters": {
            "strategy_ir_path": ir_path,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": str(initial_cash),
            "data_folder": str(data_dir),
        },
        "log-handler": "QuantConnect.Logging.CompositeLogHandler",
        "messaging-handler": "QuantConnect.Messaging.Messaging",
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "api-handler": "QuantConnect.Api.Api",
        "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
        "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
        "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
        "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
        "data-aggregator": "QuantConnect.Lean.Engine.DataFeeds.AggregationManager",
    }

    config_path = algo_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Running LEAN backtest")

    cmd = [
        "dotnet",
        "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll",
        "--config", str(config_path),
    ]

    # Set up environment with PYTHONPATH pointing to LEAN's Python modules
    env = os.environ.copy()
    lean_python_path = "/Lean/Launcher/bin/Debug"
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{lean_python_path}:{existing_path}" if existing_path else lean_python_path

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=env,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "exit_code": result.returncode,
            "stdout": result.stdout[-5000:] if result.stdout else "",  # Truncate
            "stderr": result.stderr[-5000:] if result.stderr else "",
        }

    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

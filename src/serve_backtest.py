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
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vibe Trade Backtest Service")

# Algorithm source directory - Docker container path or local fallback
ALGO_SRC_DIR = (
    Path("/Lean/Algorithm.Python")
    if Path("/Lean/Algorithm.Python").exists()
    else Path(__file__).parent / "Algorithms"
)


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
    low: float = Field(alias="l")  # low - renamed from 'l' for lint compliance
    c: float  # close
    v: float  # volume


class BacktestDataInput(BaseModel):
    """Data input for backtest - inline or GCS reference."""
    symbol: str
    resolution: str
    bars: list[OHLCVBar] | None = None  # Inline data
    gcs_uri: str | None = None  # GCS reference (future)

    @model_validator(mode="after")
    def validate_data_source(self):
        """Require either bars or gcs_uri."""
        if self.bars is None and self.gcs_uri is None:
            raise ValueError("Must provide either 'bars' or 'gcs_uri'")
        return self


class BacktestConfig(BaseModel):
    """Configuration for backtest execution."""
    start_date: str  # YYYY-MM-DD format (LEAN data processing start, may include warmup)
    end_date: str
    initial_cash: float = 100000.0
    trading_start_date: str | None = None  # User's actual start date (prevents warmup trades)


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



class EquityPoint(BaseModel):
    """Single point on equity curve with full portfolio breakdown."""

    time: str
    equity: float
    cash: float
    holdings: float
    drawdown: float


class LEANBacktestSummary(BaseModel):
    """Summary metrics from LEAN backtest.

    Includes basic custom statistics and LEAN's PortfolioStatistics.
    All LEAN fields are optional for backward compatibility.
    """
    # Basic metrics (required)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0

    # LEAN PortfolioStatistics (all optional)
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    probabilistic_sharpe_ratio: float | None = None
    information_ratio: float | None = None
    treynor_ratio: float | None = None
    compounding_annual_return: float | None = None
    total_net_profit: float | None = None
    start_equity: float | None = None
    end_equity: float | None = None
    drawdown: float | None = None
    annual_standard_deviation: float | None = None
    annual_variance: float | None = None
    tracking_error: float | None = None
    value_at_risk_99: float | None = None
    value_at_risk_95: float | None = None
    alpha: float | None = None
    beta: float | None = None
    win_rate: float | None = None
    loss_rate: float | None = None
    average_win_rate: float | None = None
    average_loss_rate: float | None = None
    profit_loss_ratio: float | None = None
    expectancy: float | None = None
    portfolio_turnover: float | None = None


class LEANBacktestResponse(BaseModel):
    """Response format expected by execution service."""
    status: str  # "success" or "error"
    trades: list[Trade] = []
    summary: LEANBacktestSummary | None = None
    # Full equity curve data with cash/holdings breakdown
    equity_curve: list[EquityPoint] | list[dict] | None = None
    ohlcv_bars: list[dict[str, Any]] | None = None
    indicators: dict[str, list[dict[str, Any]]] | None = None
    error: str | None = None


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

        # Debug: log fee/slippage values
        fee_pct = request.strategy_ir.get("fee_pct", 0.0)
        slippage_pct = request.strategy_ir.get("slippage_pct", 0.0)
        if fee_pct > 0 or slippage_pct > 0:
            logger.info(f"Trading costs in IR: fee_pct={fee_pct}%, slippage_pct={slippage_pct}%")

        # Write inline market data for primary symbol (LEAN ZIP format for AddCrypto)
        if request.data.bars:
            candle_count = _write_ohlcv_bars_lean_zip(
                request.data.bars, request.data.symbol,
                request.data.resolution, data_dir,
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
                additional_count = _write_ohlcv_bars_lean_zip(
                    additional.bars, additional.symbol,
                    additional.resolution, data_dir,
                )
                logger.info(f"Using {additional_count} inline bars for {additional.symbol}")

        # Copy LEAN data files
        _copy_lean_data_files(data_dir)

        # Copy only StrategyRuntime.py to temp directory
        # All modules (indicators, conditions, trades, etc.) are already in the Docker image
        # at /Lean/Algorithm.Python/ and will be accessible via PYTHONPATH
        src_file = ALGO_SRC_DIR / "StrategyRuntime.py"
        if not src_file.exists():
            raise FileNotFoundError(f"Required algorithm file not found: {src_file}")
        shutil.copy(src_file, algo_dir / "StrategyRuntime.py")
        logger.info(f"Copied StrategyRuntime.py to {algo_dir}")
        
        # Verify modules exist in source directory (they should be in Docker image)
        required_modules = ["indicators", "conditions", "trades", "position", "gates", "costs", "symbols", "ir", "execution", "initialization", "state"]
        missing_modules = [m for m in required_modules if not (ALGO_SRC_DIR / m).exists()]
        if missing_modules:
            logger.warning(f"Missing modules in {ALGO_SRC_DIR}: {missing_modules}")
            logger.warning("Modules should be in Docker image at /Lean/Algorithm.Python/")

        # Parse dates from config (YYYY-MM-DD format)
        start_date = request.config.start_date.replace("-", "")
        end_date = request.config.end_date.replace("-", "")
        trading_start_date = None
        if request.config.trading_start_date:
            trading_start_date = request.config.trading_start_date.replace("-", "")

        # Run LEAN
        lean_result = _run_lean(
            algo_dir=algo_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            ir_path=str(ir_path),
            start_date=start_date,
            end_date=end_date,
            initial_cash=request.config.initial_cash,
            trading_start_date=trading_start_date,
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
        ohlcv_bars = None
        indicators = None

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
                # Basic metrics from custom calculations
                total_trades=stats.get("total_trades", len(trades)),
                winning_trades=stats.get("winning_trades", 0),
                losing_trades=stats.get("losing_trades", 0),
                total_pnl=total_pnl,
                total_pnl_pct=round(total_pnl_pct, 2),
                max_drawdown_pct=round(strategy_output.get("max_drawdown_pct", 0), 2),

                # LEAN native statistics (extracted in StrategyRuntime)
                sharpe_ratio=stats.get("sharpe_ratio"),
                sortino_ratio=stats.get("sortino_ratio"),
                probabilistic_sharpe_ratio=stats.get("probabilistic_sharpe_ratio"),
                information_ratio=stats.get("information_ratio"),
                treynor_ratio=stats.get("treynor_ratio"),
                compounding_annual_return=stats.get("compounding_annual_return"),
                total_net_profit=stats.get("total_net_profit"),
                start_equity=stats.get("start_equity"),
                end_equity=stats.get("end_equity"),
                drawdown=stats.get("drawdown"),
                annual_standard_deviation=stats.get("annual_standard_deviation"),
                annual_variance=stats.get("annual_variance"),
                tracking_error=stats.get("tracking_error"),
                value_at_risk_99=stats.get("value_at_risk_99"),
                value_at_risk_95=stats.get("value_at_risk_95"),
                alpha=stats.get("alpha"),
                beta=stats.get("beta"),
                win_rate=stats.get("win_rate"),
                loss_rate=stats.get("loss_rate"),
                average_win_rate=stats.get("average_win_rate"),
                average_loss_rate=stats.get("average_loss_rate"),
                profit_loss_ratio=stats.get("profit_loss_ratio"),
                expectancy=stats.get("expectancy"),
                portfolio_turnover=stats.get("portfolio_turnover"),
            )

            # Pass through full equity curve data (time, equity, cash, holdings, drawdown)
            equity_curve = strategy_output.get("equity_curve", [])
            ohlcv_bars = strategy_output.get("ohlcv_bars")
            indicators = strategy_output.get("indicators")

        return LEANBacktestResponse(
            status="success",
            trades=trades,
            summary=summary,
            equity_curve=equity_curve if strategy_output else None,
            ohlcv_bars=ohlcv_bars if strategy_output else None,
            indicators=indicators if strategy_output else None,
        )


def _write_ohlcv_bars_lean_zip(
    bars: list[OHLCVBar],
    symbol: str,
    resolution: str,
    output_dir: Path,
    market: str = "coinbase",
) -> int:
    """Write OHLCVBar data in LEAN-native ZIP format for AddCrypto().

    Minute data: /Data/crypto/{market}/minute/{symbol}/{YYYYMMDD}_trade.zip
      CSV rows (no header): ms_since_midnight,open,high,low,close,volume

    Hourly data: /Data/crypto/{market}/hour/{symbol}_trade.zip
      CSV rows (no header): YYYYMMDD HH:mm,open,high,low,close,volume

    Daily data: /Data/crypto/{market}/daily/{symbol}_trade.zip
      CSV rows (no header): YYYYMMDD 00:00,open,high,low,close,volume

    Args:
        bars: List of OHLCVBar with t (ms timestamp), o, h, l, c, v fields
        symbol: Trading symbol (e.g., "BTC-USD", "TESTUSD")
        resolution: Data resolution ("1m", "5m", "15m", "1h", "4h", "1d")
        output_dir: Root data directory
        market: Market name for LEAN path (default: "coinbase")

    Returns:
        Number of bars written
    """
    if not bars:
        return 0

    # Normalize symbol for LEAN paths: BTC-USD -> btcusd, TESTUSD -> testusd
    symbol_normalized = symbol.lower().replace("-", "").replace("_", "")

    # Determine resolution type
    is_minute = resolution in ("1m", "minute", "5m", "15m")
    is_daily = resolution in ("1d", "daily")

    if is_minute:
        # Group bars by date for per-day ZIP files
        grouped: dict[str, list[OHLCVBar]] = defaultdict(list)
        for bar in bars:
            ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            date_str = ts.strftime("%Y%m%d")
            grouped[date_str].append(bar)

        # Write per-day ZIP files
        base_path = output_dir / "crypto" / market / "minute" / symbol_normalized
        base_path.mkdir(parents=True, exist_ok=True)

        for date_str, day_bars in grouped.items():
            csv_lines = []
            for bar in day_bars:
                ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
                day_start = ts.replace(hour=0, minute=0, second=0, microsecond=0)
                ms_since_midnight = int((ts - day_start).total_seconds() * 1000)
                csv_lines.append(
                    f"{ms_since_midnight},{bar.o},{bar.h},{bar.low},{bar.c},{bar.v}"
                )

            zip_path = base_path / f"{date_str}_trade.zip"
            csv_filename = f"{date_str}_trade.csv"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(csv_filename, "\n".join(csv_lines))

        logger.info(
            f"Wrote {len(bars)} minute bars across {len(grouped)} ZIPs "
            f"to {base_path}"
        )
    elif is_daily:
        # Daily: /crypto/{market}/daily/{symbol}_trade.zip
        base_path = output_dir / "crypto" / market / "daily"
        base_path.mkdir(parents=True, exist_ok=True)

        csv_lines = []
        for bar in bars:
            ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            # Daily format: YYYYMMDD 00:00 (always midnight)
            time_str = ts.strftime("%Y%m%d") + " 00:00"
            csv_lines.append(
                f"{time_str},{bar.o},{bar.h},{bar.low},{bar.c},{bar.v}"
            )

        zip_path = base_path / f"{symbol_normalized}_trade.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{symbol_normalized}.csv", "\n".join(csv_lines))

        logger.info(
            f"Wrote {len(bars)} daily bars to {zip_path}"
        )
    else:
        # Hourly (1h, 4h): /crypto/{market}/hour/{symbol}_trade.zip
        base_path = output_dir / "crypto" / market / "hour"
        base_path.mkdir(parents=True, exist_ok=True)

        csv_lines = []
        for bar in bars:
            ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            time_str = ts.strftime("%Y%m%d %H:%M")
            csv_lines.append(
                f"{time_str},{bar.o},{bar.h},{bar.low},{bar.c},{bar.v}"
            )

        zip_path = base_path / f"{symbol_normalized}_trade.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{symbol_normalized}.csv", "\n".join(csv_lines))

        logger.info(
            f"Wrote {len(bars)} hourly bars to {zip_path}"
        )

    return len(bars)


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
    trading_start_date: str | None = None,
) -> dict:
    """Run LEAN backtest."""
    parameters = {
        "strategy_ir_path": ir_path,
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": str(initial_cash),
        "data_folder": str(data_dir),
    }
    # Add trading_start_date if provided (prevents trades during warmup)
    if trading_start_date:
        parameters["trading_start_date"] = trading_start_date

    config = {
        "environment": "backtesting",
        "algorithm-type-name": "StrategyRuntime",
        "algorithm-language": "Python",
        "algorithm-location": str(algo_dir / "StrategyRuntime.py"),
        "data-folder": str(data_dir),
        "results-destination-folder": str(results_dir),
        "parameters": parameters,
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

    logger.info("Running LEAN backtest")

    cmd = [
        "dotnet",
        "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll",
        "--config", str(config_path),
    ]

    # Set up environment with PYTHONPATH pointing to:
    # 1. LEAN's Python modules (/Lean/Launcher/bin/Debug)
    # 2. Algorithm modules (/Lean/Algorithm.Python) - so StrategyRuntime can import them
    env = os.environ.copy()
    lean_python_path = "/Lean/Launcher/bin/Debug"
    algo_python_path = "/Lean/Algorithm.Python"
    existing_path = env.get("PYTHONPATH", "")
    # Add both paths, with algo path first so it takes precedence
    env["PYTHONPATH"] = f"{algo_python_path}:{lean_python_path}:{existing_path}" if existing_path else f"{algo_python_path}:{lean_python_path}"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
        env=env,
    )

    # Log LEAN output for debugging
    if result.stdout:
        for line in result.stdout.split("\n"):
            if "TRACE::" in line or "ERROR::" in line:
                logger.info(f"LEAN: {line}")

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
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=120)

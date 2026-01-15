#!/bin/bash
# LEAN Entrypoint Script
#
# This script runs before LEAN to prepare data:
# 1. If START_DATE is set (backtest mode): Download historical data from BigQuery
# 2. Then run LEAN with the provided arguments
#
# Environment Variables:
#   START_DATE - If set, triggers historical data download
#   END_DATE - End date for historical data
#   SYMBOL - Trading symbol (e.g., BTC-USD)
#   GOOGLE_CLOUD_PROJECT - GCP project ID
#   BACKTEST_MODE - If "true", use backtest config instead of live config

set -e

echo "=== LEAN Entrypoint ==="
echo "Working directory: $(pwd)"
echo "START_DATE: ${START_DATE:-not set}"
echo "END_DATE: ${END_DATE:-not set}"
echo "SYMBOL: ${SYMBOL:-not set}"
echo "BACKTEST_MODE: ${BACKTEST_MODE:-not set}"
echo ""

# Run data loader if in backtest mode (START_DATE is set)
# Skip if SKIP_DATA_DOWNLOAD is set (for local testing with pre-loaded data)
if [ -n "$START_DATE" ] && [ -z "$SKIP_DATA_DOWNLOAD" ]; then
    echo "Backtest mode detected - downloading historical data..."

    # Ensure Data directory exists
    mkdir -p /Data/crypto/coinbase/minute

    # Run the Python data loader
    python3 /scripts/data_loader.py

    if [ $? -ne 0 ]; then
        echo "ERROR: Data loader failed"
        exit 1
    fi

    echo ""
    echo "Data download complete. Starting LEAN..."
    echo ""
elif [ -n "$SKIP_DATA_DOWNLOAD" ]; then
    echo "SKIP_DATA_DOWNLOAD set - using pre-loaded data"
fi

# Check which LEAN launcher to use
if [ -f "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.exe" ]; then
    # Mono-based LEAN (older images)
    echo "Running LEAN with Mono..."
    exec mono /Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.exe "$@"
elif [ -f "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll" ]; then
    # .NET Core LEAN (newer images)
    echo "Running LEAN with .NET..."
    exec dotnet /Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll "$@"
else
    echo "ERROR: Could not find LEAN launcher"
    ls -la /Lean/Launcher/bin/Debug/
    exit 1
fi

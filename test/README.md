# Test Suite

This directory contains tests for the Pub/Sub data queue handler integration with LEAN.

## Test Algorithms

### `algorithm.py` - Basic Data Flow Test
Simple algorithm that only receives and logs data from Pub/Sub. Used to verify:
- Handler connects to Pub/Sub
- Data flows from Pub/Sub to LEAN
- Messages are parsed correctly

### `strategy_algorithm.py` - Strategy Execution Test
Full trading strategy that:
- Receives data from Pub/Sub
- Places orders based on data
- Executes trades
- Tracks metrics

Used to verify:
- End-to-end data flow
- Order execution
- Portfolio management
- Result generation

## Test Commands

### Basic Test (Data Flow Only)
```bash
make test-local
```
- Tests that data flows from Pub/Sub to LEAN
- Verifies handler connection and subscription
- Checks that data is received
- Minimal result verification (data received, no errors)

### Strategy Test (Full Trading)
```bash
make test-strategy
```
- Runs strategy algorithm that actually trades
- Verifies orders are placed and filled
- Validates result files contain expected metrics
- More comprehensive verification

## Result Verification

The `verify_results.py` script checks that:
- ✅ Result file exists and is parseable
- ✅ Algorithm ran successfully (no runtime errors)
- ✅ Data was received (minimum log count)
- ✅ Orders were placed (for strategy test)
- ✅ Equity curve is tracked
- ✅ Runtime statistics are present
- ✅ Holdings and cash data are available

### Usage
```bash
python3 test/verify_results.py <results_dir> [min_data_points] [min_orders]
```

Example:
```bash
# Verify at least 10 data points, 1 order
python3 test/verify_results.py test/results 10 1
```

## Test Results

Results are written to `test/results/` directory:
- `AlgorithmName.json` - Main result file with state, metrics, holdings
- `AlgorithmName-summary.json` - Summary statistics
- `AlgorithmName-*_Strategy Equity.json` - Equity curve data
- Other chart/auxiliary files

## Local Testing Setup

Tests use the Pub/Sub emulator (no GCP credentials needed):
1. Starts emulator via `docker-compose.test.yml`
2. Seeds test data via `seed-emulator.sh`
3. Publishes messages continuously via `publish-messages.sh`
4. Runs LEAN with test algorithm
5. Verifies results

## Production Testing

For production testing (requires GCP credentials):
```bash
make test
```

Requires:
- `GOOGLE_CLOUD_PROJECT` environment variable
- `GOOGLE_APPLICATION_CREDENTIALS` path to service account JSON
- Pub/Sub topics and subscriptions must exist

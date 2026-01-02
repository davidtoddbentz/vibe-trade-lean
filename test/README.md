# Testing

## Local Test (No Credentials Required)

Test with local Pub/Sub emulator:

```bash
make test-local
```

This will:
- Start a local Pub/Sub emulator
- Seed test data (20 messages)
- Run the handler against the emulator
- Verify connection, subscription, and data reception

## Production Test (Requires Credentials)

Test with production Pub/Sub:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
export PUBSUB_TEST_SYMBOL=BTC-USD  # Optional
export PUBSUB_TEST_SUBSCRIPTION=test_local  # Optional

make test
```

## What It Tests

- ✅ Handler connects to Pub/Sub (emulator or production)
- ✅ Handler subscribes to correct topic/subscription
- ✅ Data is received from Pub/Sub
- ✅ Data reaches algorithm

## Test Files

- `algorithm.py` - Test algorithm that subscribes to data
- `config.json` - LEAN configuration with Pub/Sub handler
- `data/` - LEAN data files (market hours, symbol properties)

## Results

- Logs: `/tmp/lean-test-*.log` or `/tmp/lean-test-local-*.log`
- Results: `test/results/`

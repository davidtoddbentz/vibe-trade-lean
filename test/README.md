# Testing vibe-trade-lean

This directory contains test files to verify the custom LEAN image works with Pub/Sub.

## Quick Test

Quick verification that the handler is installed correctly:

```bash
./test/quick-test.sh
```

This checks:
- Docker image exists
- `PubSubDataQueueHandler.dll` is present
- LEAN launcher works

## Full Integration Test

Run a complete test with Pub/Sub integration:

```bash
# Set required environment variables
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json

# Run the test
./test/run-test.sh
```

Or use the Makefile:

```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json
make test-integration
```

## What the Test Does

1. **Builds test image**: Creates a Docker image with test algorithm
2. **Runs algorithm**: Starts LEAN with Pub/Sub data queue handler
3. **Subscribes to Pub/Sub**: Attempts to connect to `vibe-trade-candles-btc-usd-1m` topic
4. **Receives data**: Logs incoming data from Pub/Sub
5. **Verifies handler**: Confirms the handler is working

## Expected Output

You should see:
- ✅ Algorithm initialized
- 📊 Symbol: BTC-USD.TradeBar
- 🔍 Waiting for data from Pub/Sub...
- 📈 Received data messages (if Pub/Sub topic has data)
- 🟢 LIVE MODE confirmation (if handler is working)

## Test Files

- `algorithm.py`: Test algorithm that subscribes to BTC-USD data
- `config.json`: LEAN configuration with Pub/Sub handler
- `data/`: Test data files (market hours, symbol properties)
- `quick-test.sh`: Quick verification script
- `run-test.sh`: Full integration test script

## Troubleshooting

### Handler Not Loading
- Check that `PubSubDataQueueHandler.dll` exists in the image
- Verify config.json has correct handler name
- Check LEAN logs for loading errors

### Pub/Sub Connection Issues
- Verify `GOOGLE_CLOUD_PROJECT` is set
- Check GCP credentials are valid
- Ensure Pub/Sub API is enabled
- Verify topic `vibe-trade-candles-btc-usd-1m` exists

### No Data Received
- Check if topic has messages
- Verify subscription was created
- Check Pub/Sub console for message flow
- Review LEAN logs for errors

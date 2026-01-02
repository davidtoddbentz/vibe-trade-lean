# Testing vibe-trade-lean

This directory contains test files to verify the custom LEAN image works with Pub/Sub.

## Quick Test

```bash
# Set GCP project
export GOOGLE_CLOUD_PROJECT=your-project-id

# Set credentials (if not using default)
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json

# Run test
make test-integration

# Or manually
./test/run-test.sh
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
- 📊 Symbol: BTCUSD
- 🔍 Waiting for data from Pub/Sub...
- 📈 Received data messages (if Pub/Sub topic has data)
- 🟢 LIVE MODE confirmation (if handler is working)

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

## Files

- `algorithm.py`: Test algorithm that uses Pub/Sub data
- `config.json`: LEAN configuration with Pub/Sub handler
- `Dockerfile`: Test image definition
- `run-test.sh`: Test execution script


# vibe-trade-lean

Custom QuantConnect LEAN Docker image with Google Cloud Pub/Sub integration for real-time data feeds.

## Overview

This repository extends the base `quantconnect/lean:latest` image with a custom `IDataQueueHandler` implementation that subscribes to Google Cloud Pub/Sub topics and feeds real-time market data to LEAN's algorithmic trading engine.

## Features

- **Pub/Sub Data Queue Handler (C#)**: Implements `IDataQueueHandler` to subscribe to GCP Pub/Sub topics
- **REST API Implementation**: Uses direct REST API calls (no gRPC) for Linux compatibility
- **Manual JWT Authentication**: Creates and signs JWTs manually for GCP authentication
- **EnqueueableEnumerator Pattern**: Uses LEAN's standard enumerator pattern for data delivery
- **Thread-Safe**: Properly synchronized for concurrent access
- **Production Ready**: Clean, tested, and follows LEAN best practices

## Architecture

The handler implements LEAN's `IDataQueueHandler` interface:

1. **Subscribe**: Creates an `EnqueueableEnumerator` for each symbol and starts a background task to pull messages from Pub/Sub
2. **Message Processing**: Parses Pub/Sub messages (JSON) into LEAN `TradeBar` objects
3. **Data Delivery**: Enqueues data into the enumerator, which LEAN's `LiveSubscriptionEnumerator` consumes via `MoveNext()`
4. **Event Notification**: Invokes event handlers to notify LEAN when new data is available

## Project Structure

```
vibe-trade-lean/
├── src/
│   └── DataFeeds/
│       └── PubSubDataQueueHandler.cs    # Main C# implementation
├── csproj/
│   └── PubSubDataQueueHandler.csproj    # C# project file
├── test/
│   ├── algorithm.py                     # Test algorithm
│   ├── config.json                      # LEAN configuration
│   └── data/                            # Test data files
├── Dockerfile                           # Docker image build
├── Makefile                             # Build commands
└── README.md
```

## Building

### Prerequisites

- Docker
- Access to `quantconnect/lean:latest` base image
- GCP project with Pub/Sub enabled (for runtime)

### Build the Image

```bash
# Using Makefile
make build

# Or directly with Docker
docker build -t vibe-trade-lean:latest .
```

## Usage

### Environment Variables

**Required:**
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to GCP service account JSON file

**Optional - Subscription Configuration:**
- `PUBSUB_TEST_SUBSCRIPTION`: Global subscription name override (backward compatibility)
- `PUBSUB_SUBSCRIPTION_{SYMBOL}`: Per-symbol subscription override (e.g., `PUBSUB_SUBSCRIPTION_BTC_USD=test_local_btc`)
- `PUBSUB_TOPIC_{SYMBOL}`: Per-symbol topic override (e.g., `PUBSUB_TOPIC_ETH_USD=vibe-trade-candles-eth-usd-1m`)
- `PUBSUB_TOPIC_PATTERN`: Global topic pattern with `{symbol}` and `{resolution}` placeholders

**Optional - Algorithm Configuration:**
- `PUBSUB_TEST_SYMBOL`: Symbol to use in test algorithm (default: `BTC-USD`)

### Subscription Name Resolution

The handler resolves subscription names in this order:
1. `PUBSUB_SUBSCRIPTION_{SYMBOL}` (e.g., `PUBSUB_SUBSCRIPTION_ETH_USD=test_local_eth`)
2. `PUBSUB_TEST_SUBSCRIPTION` (global override)
3. Auto-generated: `vibe-trade-lean-{symbol}-{resolution}`

### Topic Name Resolution

The handler resolves topic names in this order:
1. `PUBSUB_TOPIC_{SYMBOL}` (per-symbol override)
2. `PUBSUB_TOPIC_PATTERN` with placeholders
3. Default: `vibe-trade-candles-{symbol}-{resolution}`

### Examples

**Using ETH-USD with custom subscription:**
```bash
export PUBSUB_SUBSCRIPTION_ETH_USD=test_local_eth
export PUBSUB_TEST_SYMBOL=ETH-USD
```

**Using custom topic pattern:**
```bash
export PUBSUB_TOPIC_PATTERN="my-topic-{symbol}-{resolution}"
# Results in: my-topic-btc-usd-1m
```

### Running LEAN with Pub/Sub

```bash
docker run --rm \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcp-credentials.json \
  -e PUBSUB_TEST_SUBSCRIPTION=test_local \
  -v /path/to/credentials.json:/tmp/gcp-credentials.json:ro \
  -v /path/to/algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro \
  -v /path/to/config.json:/Lean/Launcher/bin/Debug/config.json:ro \
  -v /path/to/data:/Data:ro \
  -v /path/to/results:/Results \
  vibe-trade-lean:latest \
  --config /Lean/Launcher/bin/Debug/config.json
```

### Configuration

In your LEAN `config.json`, set the data queue handler:

```json
{
  "environments": {
    "live-pubsub": {
      "live-mode": true,
      "data-queue-handler": "QuantConnect.Lean.Engine.DataFeeds.PubSubDataQueueHandler",
      "data-feed-handler": "QuantConnect.Lean.Engine.DataFeeds.LiveTradingDataFeed"
    }
  }
}
```

## Subscription Management

**Important**: Subscriptions must be created **before** running LEAN. The handler will fail fast if a subscription does not exist.

### For Production

Use Terraform, Cloud Console, or `gcloud` CLI to create subscriptions:

```bash
# Example: Create subscription for BTC-USD 1-minute candles
gcloud pubsub subscriptions create vibe-trade-lean-btc-usd-1m \
  --topic=vibe-trade-candles-btc-usd-1m \
  --ack-deadline=60
```

Or via Terraform:
```hcl
resource "google_pubsub_subscription" "btc_usd_1m" {
  name  = "vibe-trade-lean-btc-usd-1m"
  topic = "vibe-trade-candles-btc-usd-1m"
  
  ack_deadline_seconds = 60
}
```

### For Testing

The test setup automatically creates subscriptions via `docker-compose.test.yml` and `test/seed-emulator.sh`. See the Testing section below.

## Pub/Sub Topic Format

The handler automatically generates topic and subscription names based on the symbol and resolution:

**Default Format:**
- **Topic**: `vibe-trade-candles-{symbol}-{resolution}` (e.g., `vibe-trade-candles-btc-usd-1m`)
- **Subscription**: `vibe-trade-lean-{symbol}-{resolution}` (e.g., `vibe-trade-lean-btc-usd-1m`)

**Symbol Normalization:**
- `BTCUSD` → `BTC-USD` → `btc-usd` (for topic names)
- `ETH.USD` → `ETH-USD` → `eth-usd`
- Dots and dashes are normalized automatically

**Resolution Format:**
- Seconds: `30s`, `60s`
- Minutes: `1m`, `5m`, `15m`, `60m`
- Hours: `1h`, `4h`, `24h`
- Days: `1d`

**Customization:**
You can override topic/subscription names per symbol using environment variables (see Configuration section above).

## Message Format

Pub/Sub messages must be JSON with the following structure:

```json
{
  "symbol": "BTC-USD",
  "timestamp": "2024-01-01T00:00:00Z",
  "open": 42000.0,
  "high": 42100.0,
  "low": 41900.0,
  "close": 42050.0,
  "volume": 123.45,
  "granularity": "1m"
}
```

The message data is base64-encoded in the Pub/Sub message's `data` field.

## Algorithm Example

```python
from AlgorithmImports import *

class MyAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetCash(100000)
        
        # Add symbol using AddData (bypasses symbol database)
        properties = SymbolProperties("Bitcoin", "USD", 1, 0.01, 0.01, "BTC-USD")
        exchangeHours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
        
        self.symbol = self.AddData(
            TradeBar,
            "BTC-USD",
            properties,
            exchangeHours,
            Resolution.Minute
        ).Symbol
    
    def OnData(self, data):
        if self.symbol in data:
            bar = data[self.symbol]
            self.Log(f"Received: {bar.Time} Close=${bar.Close}")
```

## Development

### Testing

```bash
# Test build
make test-build

# Run integration test (requires GCP credentials)
make test-integration
```

### Local Development

1. Build the image: `make build`
2. Test with your algorithm: Mount your algorithm and config files
3. Check logs: Results are written to `/Results`

## Implementation Details

### Authentication

The handler uses manual JWT creation and signing:
- Creates JWT with RS256 algorithm
- Signs with service account private key
- Exchanges JWT for OAuth2 access token
- Refreshes token automatically before expiry

### Data Flow

1. Algorithm calls `AddData()` → LEAN calls `Subscribe()`
2. Handler creates `EnqueueableEnumerator` and starts background pull loop
3. Pub/Sub messages are pulled, parsed, and enqueued
4. Event handler notifies LEAN → `LiveSubscriptionEnumerator` calls `MoveNext()`
5. Data flows to algorithm's `OnData()` method

### Thread Safety

- All shared state protected with `lock (_lock)`
- Event handlers invoked safely from background threads
- Enumerators properly disposed on unsubscribe

## Troubleshooting

### Handler Not Loading

- Verify `PubSubDataQueueHandler.dll` exists in `/Lean/Launcher/bin/Debug/`
- Check namespace: `QuantConnect.Lean.Engine.DataFeeds.PubSubDataQueueHandler`
- Review LEAN logs for MEF loading errors

### No Data Received

- Verify `GOOGLE_CLOUD_PROJECT` is set
- Check GCP credentials are valid and have `pubsub.subscriber` role
- Ensure Pub/Sub topics exist and have messages
- Check subscription name matches (or use `PUBSUB_TEST_SUBSCRIPTION`)

### Authentication Errors

- Verify service account JSON is valid
- Check service account has `pubsub.subscriber` role
- Ensure Pub/Sub API is enabled in GCP project

## License

Apache 2.0 (same as QuantConnect LEAN)

## Contributing

This is a custom extension for internal use. For issues or questions, please contact the maintainers.

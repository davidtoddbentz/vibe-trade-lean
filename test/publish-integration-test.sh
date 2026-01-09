#!/bin/bash
# Publish a deterministic sequence of bars for integration testing
# This script publishes exactly N bars with predictable prices to test algorithm behavior

set -e

EMULATOR_HOST=${PUBSUB_EMULATOR_HOST:-localhost:8085}
PROJECT_ID=${PUBSUB_PROJECT_ID:-test-project}
TOPIC_NAME=${PUBSUB_TOPIC_NAME:-vibe-trade-candles-btc-usd-1m}
NUM_BARS=${PUBSUB_NUM_BARS:-15}  # Publish 15 bars by default (enough for buy on 1st, sell after 3rd)

echo "📤 Publishing $NUM_BARS bars to topic: $TOPIC_NAME"
echo "   Strategy: Buy on bar 1, sell after bar 3 (bar 4)"
echo "   Expected: 2 orders (1 buy, 1 sell)"

# IMPORTANT: This script is called AFTER LEAN has been running for 60 seconds
# and the subscription has been seeked to clear any backlog.
# We should NOT wait here - we should start publishing immediately with real-time timestamps.
# The Makefile coordinates the timing: it waits 60s, seeks subscription, waits 10s, then calls this script.
# No additional wait needed - start publishing immediately.

# Base price - start at $42000
BASE_PRICE=42000

# Publish bars with incrementing prices
for i in $(seq 1 $NUM_BARS); do
  # Price increases by $10 per bar
  price=$((BASE_PRICE + (i * 10)))
  open=$price
  high=$((price + 5))
  low=$((price - 5))
  close=$price
  volume=$((100 + i))
  
  # Timestamp: current UTC time
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  
  message_data="{\"symbol\":\"BTC-USD\",\"timestamp\":\"${timestamp}\",\"open\":${open},\"high\":${high},\"low\":${low},\"close\":${close},\"volume\":${volume}}"
  
  # Base64 encode the message
  encoded=$(echo -n "$message_data" | base64)
  
  # Publish to Pub/Sub
  curl -s -X POST "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/topics/${TOPIC_NAME}:publish" \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"data\": \"${encoded}\"}]}" > /dev/null
  
  echo "  📊 Bar $i: $timestamp | Close: \$${close} | Volume: ${volume}"
  
  # Send at 2 bars per second (0.5 second intervals)
  sleep 0.5
done

echo "✅ Published $NUM_BARS bars"
echo "   Expected algorithm behavior:"
echo "   - Bar 1: BUY order placed"
echo "   - Bar 4: SELL order placed (after 3 bars)"
echo "   - Total: 2 orders expected"


#!/bin/bash
# Continuously publish test messages to Pub/Sub emulator

EMULATOR_HOST=${PUBSUB_EMULATOR_HOST:-localhost:8085}
PROJECT_ID=${PUBSUB_PROJECT_ID:-test-project}
TOPIC_NAME="vibe-trade-candles-btc-usd-1m"

# Wait a bit for handler to start
echo "Waiting for handler to start..."
sleep 8

echo "📤 Starting to publish messages..."

i=0
while true; do
  # Use current timestamp for realistic data
  timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  
  # Generate price data with some variation
  base_price=42000
  price_variation=$((i % 100))
  open=$((base_price + price_variation))
  high=$((open + 50))
  low=$((open - 50))
  close=$((open + (i % 20)))
  volume=$((100 + (i % 50)))
  
  message_data="{\"symbol\":\"BTC-USD\",\"timestamp\":\"${timestamp}\",\"open\":${open},\"high\":${high},\"low\":${low},\"close\":${close},\"volume\":${volume}}"
  
  # Base64 encode the message
  encoded=$(echo -n "$message_data" | base64)
  
  curl -s -X POST "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/topics/${TOPIC_NAME}:publish" \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"data\": \"${encoded}\"}]}" > /dev/null
  
  if [ $((i % 10)) -eq 0 ]; then
    echo "  Published message #$((i + 1))"
  fi
  
  i=$((i + 1))
  sleep 1
done


#!/bin/bash
# Seed Pub/Sub emulator with test data using REST API

set -e

EMULATOR_HOST=${PUBSUB_EMULATOR_HOST:-localhost:8085}
PROJECT_ID=${PUBSUB_PROJECT_ID:-test-project}
TOPIC_NAME="vibe-trade-candles-btc-usd-1m"
SUBSCRIPTION_NAME="test_local"

echo "Waiting for emulator to be ready..."
sleep 5

# Create topic
echo "Creating topic: $TOPIC_NAME"
curl -s -X PUT "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/topics/${TOPIC_NAME}" \
  -H "Content-Type: application/json" \
  -d '{}' > /dev/null || echo "Topic may already exist"

# Create subscription
echo "Creating subscription: $SUBSCRIPTION_NAME"
curl -s -X PUT "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/subscriptions/${SUBSCRIPTION_NAME}" \
  -H "Content-Type: application/json" \
  -d "{\"topic\": \"projects/${PROJECT_ID}/topics/${TOPIC_NAME}\"}" > /dev/null || echo "Subscription may already exist"

# Publish test messages
echo "Publishing test messages..."
i=0
while [ $i -lt 20 ]; do
  open=$((42000 + i))
  high=$((42100 + i))
  low=$((41900 + i))
  close=$((42050 + i))
  volume=$((100 + i))
  
  message_data="{\"symbol\":\"BTC-USD\",\"timestamp\":\"2024-01-01T00:00:00Z\",\"open\":${open},\"high\":${high},\"low\":${low},\"close\":${close},\"volume\":${volume}}"
  
  # Base64 encode the message
  encoded=$(echo -n "$message_data" | base64)
  
  curl -s -X POST "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/topics/${TOPIC_NAME}:publish" \
    -H "Content-Type: application/json" \
    -d "{\"messages\": [{\"data\": \"${encoded}\"}]}" > /dev/null
  
  i=$((i + 1))
  sleep 0.1
done

echo "✅ Published 20 test messages"


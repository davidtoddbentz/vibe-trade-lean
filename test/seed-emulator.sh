#!/bin/bash
# Seed Pub/Sub emulator with test data using REST API

set -e

EMULATOR_HOST=${PUBSUB_EMULATOR_HOST:-localhost:8085}
PROJECT_ID=${PUBSUB_PROJECT_ID:-test-project}
TOPIC_NAME="vibe-trade-candles-btc-usd-1m"
SUBSCRIPTION_NAME="test_local"

echo "Waiting for emulator to be ready..."
# Wait for emulator to be ready (check if it responds)
i=0
while [ $i -lt 30 ]; do
  if curl -s "http://${EMULATOR_HOST}" > /dev/null 2>&1; then
    echo "  ✅ Emulator is ready"
    break
  fi
  i=$((i + 1))
  if [ $i -eq 30 ]; then
    echo "  ❌ Emulator did not become ready after 30 attempts"
    exit 1
  fi
  sleep 1
done

# Create topic
echo "Creating topic: $TOPIC_NAME"
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/topics/${TOPIC_NAME}" \
  -H "Content-Type: application/json" \
  -d '{}')
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "409" ]; then
  echo "  ✅ Topic created/exists (HTTP $HTTP_CODE)"
else
  echo "  ❌ Failed to create topic (HTTP $HTTP_CODE)"
  echo "     Response: $(echo "$RESPONSE" | head -n -1)"
  exit 1
fi

# Create subscription
echo "Creating subscription: $SUBSCRIPTION_NAME"
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT "http://${EMULATOR_HOST}/v1/projects/${PROJECT_ID}/subscriptions/${SUBSCRIPTION_NAME}" \
  -H "Content-Type: application/json" \
  -d "{\"topic\": \"projects/${PROJECT_ID}/topics/${TOPIC_NAME}\"}")
HTTP_CODE=$(echo "$RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ] || [ "$HTTP_CODE" = "409" ]; then
  echo "  ✅ Subscription created/exists (HTTP $HTTP_CODE)"
else
  echo "  ❌ Failed to create subscription (HTTP $HTTP_CODE)"
  echo "     Response: $(echo "$RESPONSE" | head -n -1)"
  exit 1
fi

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


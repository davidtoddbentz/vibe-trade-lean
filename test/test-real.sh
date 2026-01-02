#!/bin/bash
# Test the handler with real GCP credentials

set -e

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-vibe-trade-475704}"
export PUBSUB_TEST_SUBSCRIPTION="${PUBSUB_TEST_SUBSCRIPTION:-test_local}"

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "❌ GOOGLE_APPLICATION_CREDENTIALS not set or file not found"
    echo "   Set it to the path of your service account JSON file"
    exit 1
fi

echo "=== Testing PubSubDataQueueHandler ==="
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Subscription: $PUBSUB_TEST_SUBSCRIPTION"
echo "Credentials: $GOOGLE_APPLICATION_CREDENTIALS"
echo ""

# Build test image
echo "Building test image..."
docker build -t vibe-trade-lean-test:latest -f test/Dockerfile . > /dev/null 2>&1

echo "Running LEAN with Pub/Sub handler..."
docker run --rm \
    -e GOOGLE_CLOUD_PROJECT="$GOOGLE_CLOUD_PROJECT" \
    -e PUBSUB_TEST_SUBSCRIPTION="$PUBSUB_TEST_SUBSCRIPTION" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcp-credentials.json" \
    -v "$GOOGLE_APPLICATION_CREDENTIALS:/tmp/gcp-credentials.json:ro" \
    -v "$(pwd)/test/data:/Data:ro" \
    -v "$(pwd)/test/results:/Results" \
    vibe-trade-lean-test:latest \
    --config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee /tmp/lean-test.log

echo ""
echo "=== Test Complete ==="
echo "Logs saved to: /tmp/lean-test.log"

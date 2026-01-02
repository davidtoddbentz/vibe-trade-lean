#!/bin/bash
# Test handler with real GCP credentials

set -e

export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-vibe-trade-475704}"
GCP_CREDENTIALS="${GCP_CREDENTIALS:-}"

if [ -z "$GCP_CREDENTIALS" ] || [ ! -f "$GCP_CREDENTIALS" ]; then
    echo "⚠️  GCP credentials not provided. Set GCP_CREDENTIALS env var to path of service account JSON"
    echo "   Example: export GCP_CREDENTIALS=/path/to/credentials.json"
    exit 1
fi

echo "=== Testing PubSubDataQueueHandler with GCP Credentials ==="
echo "Project: $GOOGLE_CLOUD_PROJECT"
echo "Credentials: $GCP_CREDENTIALS"
echo ""

docker run --rm \
    -e GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}" \
    -e GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcp-credentials.json" \
    -v "$(pwd)/data:/Data:ro" \
    -v "$(pwd)/results:/Results" \
    -v "$GCP_CREDENTIALS:/tmp/gcp-credentials.json:ro" \
    vibe-trade-lean-test:latest \
    --config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee /tmp/lean-gcp-test.log

echo ""
echo "=== Test Complete ==="
echo "Logs saved to: /tmp/lean-gcp-test.log"

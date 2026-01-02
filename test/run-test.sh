#!/bin/bash
# Test script to run the custom LEAN image with Pub/Sub handler

set -e

echo "🧪 Testing vibe-trade-lean image with Pub/Sub handler"
echo ""

# Check if GOOGLE_CLOUD_PROJECT is set
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "⚠️  Warning: GOOGLE_CLOUD_PROJECT not set"
    echo "   Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"
    echo ""
fi

# Check if credentials file exists
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Warning: GOOGLE_APPLICATION_CREDENTIALS not set"
    echo "   Set it with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json"
    echo ""
fi

# Build test image
echo "🔨 Building test image..."
docker build -t vibe-trade-lean-test:latest -f test/Dockerfile .

# Run test container
echo ""
echo "🚀 Running test container..."
echo "   This will attempt to connect to Pub/Sub and receive data"
echo "   Press Ctrl+C to stop early"
echo ""

# Generate log filename with timestamp
LOG_FILE="/tmp/lean-test-$(date +%Y%m%d-%H%M%S).log"
echo "   Logging to: $LOG_FILE"
echo "   Results will be saved to: test/results/"
echo ""

docker run --rm -it \
    -e GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}" \
    -e GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS}" \
    -v "$(pwd)/test/data:/Data:ro" \
    -v "$(pwd)/test/results:/Results" \
    ${GOOGLE_APPLICATION_CREDENTIALS:+-v "$GOOGLE_APPLICATION_CREDENTIALS:$GOOGLE_APPLICATION_CREDENTIALS:ro"} \
    vibe-trade-lean-test:latest \
    --config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Test complete! Check test/results/ for output files"


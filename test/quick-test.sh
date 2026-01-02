#!/bin/bash
# Quick test to verify the handler DLL exists and LEAN can start

set -e

echo "🔍 Quick Test: Verifying handler is installed"
echo ""

# Check if base image exists
if ! docker images | grep -q "vibe-trade-lean.*latest"; then
    echo "❌ vibe-trade-lean:latest image not found"
    echo "   Run: make build"
    exit 1
fi

echo "✅ Image found: vibe-trade-lean:latest"
echo ""

# Check if DLL exists
echo "🔍 Checking for PubSubDataQueueHandler.dll..."
if docker run --rm --entrypoint="" vibe-trade-lean:latest \
    ls -la /Lean/Launcher/bin/Debug/PubSubDataQueueHandler.dll > /dev/null 2>&1; then
    echo "✅ PubSubDataQueueHandler.dll found"
    docker run --rm --entrypoint="" vibe-trade-lean:latest \
        ls -lh /Lean/Launcher/bin/Debug/PubSubDataQueueHandler.dll
else
    echo "❌ PubSubDataQueueHandler.dll not found"
    exit 1
fi

echo ""
echo "🔍 Testing LEAN launcher..."
if docker run --rm vibe-trade-lean:latest --help > /dev/null 2>&1; then
    echo "✅ LEAN launcher works"
else
    echo "❌ LEAN launcher failed"
    exit 1
fi

echo ""
echo "✅ Quick test passed! Handler is installed correctly."
echo ""
echo "Next step: Run full integration test with:"
echo "  export GOOGLE_CLOUD_PROJECT=your-project-id"
echo "  make test-integration"


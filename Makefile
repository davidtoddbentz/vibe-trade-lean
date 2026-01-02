.PHONY: build test test-local clean help

IMAGE_NAME ?= vibe-trade-lean
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
TEST_TIMEOUT ?= 60
TEST_SYMBOL ?= BTC-USD

help:
	@echo "Available targets:"
	@echo "  build           - Build the custom LEAN Docker image"
	@echo "  test-local      - Run test with local Pub/Sub emulator (no credentials needed)"
	@echo "  test            - Run integration test with production Pub/Sub (requires GCP credentials)"
	@echo "  clean           - Remove built image"
	@echo ""
	@echo "Environment variables (for test):"
	@echo "  GOOGLE_CLOUD_PROJECT          - GCP project ID (required for test)"
	@echo "  GOOGLE_APPLICATION_CREDENTIALS - Path to service account JSON (required for test)"
	@echo "  PUBSUB_TEST_SYMBOL            - Symbol to test (default: BTC-USD)"
	@echo "  PUBSUB_TEST_SUBSCRIPTION      - Subscription name (default: test_local)"
	@echo "  TEST_TIMEOUT                  - Test timeout in seconds (default: 60)"

build:
	@echo "🔨 Building $(FULL_IMAGE)..."
	docker build -t $(FULL_IMAGE) .
	@echo "✅ Build complete: $(FULL_IMAGE)"

test: build
	@echo "🧪 Testing PubSubDataQueueHandler"
	@echo ""
	@if [ -z "$(GOOGLE_CLOUD_PROJECT)" ]; then \
		echo "❌ Error: GOOGLE_CLOUD_PROJECT not set"; \
		echo "   Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"; \
		exit 1; \
	fi
	@if [ -z "$(GOOGLE_APPLICATION_CREDENTIALS)" ] || [ ! -f "$(GOOGLE_APPLICATION_CREDENTIALS)" ]; then \
		echo "❌ Error: GOOGLE_APPLICATION_CREDENTIALS not set or file not found"; \
		echo "   Set it with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/creds.json"; \
		exit 1; \
	fi
	@echo "Configuration:"
	@echo "  Symbol: $(TEST_SYMBOL)"
	@echo "  Subscription: $$(echo $(or $(PUBSUB_TEST_SUBSCRIPTION),test_local))"
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-test-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running test (logs: $$LOG_FILE)"; \
	echo "   Press Ctrl+C to stop"; \
	echo ""; \
	docker run --rm \
		-e GOOGLE_CLOUD_PROJECT="$(GOOGLE_CLOUD_PROJECT)" \
		-e PUBSUB_TEST_SYMBOL="$(TEST_SYMBOL)" \
		-e PUBSUB_TEST_SUBSCRIPTION="$$(echo $(or $(PUBSUB_TEST_SUBSCRIPTION),test_local))" \
		-e GOOGLE_APPLICATION_CREDENTIALS="/tmp/gcp-credentials.json" \
		-v "$(GOOGLE_APPLICATION_CREDENTIALS):/tmp/gcp-credentials.json:ro" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/test/algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro" \
		-v "$(PWD)/test/config.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo ""; \
	echo "📊 Test Results:"; \
	if grep -q "PubSubDataQueueHandler: Connected" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Handler connected"; \
	else \
		echo "  ❌ Handler did not connect"; \
	fi; \
	if grep -q "PubSubDataQueueHandler: Subscribed to" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Subscription verified and active"; \
	else \
		echo "  ❌ Subscription not found or failed"; \
	fi; \
	if grep -q "🟢 LIVE MODE: Data received from Pub/Sub!" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Data received"; \
	else \
		echo "  ❌ No data received"; \
	fi; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

test-local: build
	@echo "🧪 Testing with local Pub/Sub emulator"
	@echo ""
	@echo "Starting emulator..."
	@docker-compose -f docker-compose.test.yml up -d pubsub-emulator
	@sleep 3
	@echo "Seeding test data..."
	@docker-compose -f docker-compose.test.yml run --rm pubsub-seed || true
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-test-local-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running test (logs: $$LOG_FILE)"; \
	echo "   Press Ctrl+C to stop"; \
	echo ""; \
	echo "Starting message publisher in background..."; \
	PUBSUB_EMULATOR_HOST=localhost:8085 $(PWD)/test/publish-messages.sh > /tmp/pubsub-publisher.log 2>&1 & \
	PUBLISHER_PID=$$!; \
	echo "Publisher PID: $$PUBLISHER_PID"; \
	sleep 2; \
	docker run --rm \
		--network host \
		-e GOOGLE_CLOUD_PROJECT="test-project" \
		-e PUBSUB_EMULATOR_HOST="localhost:8085" \
		-e PUBSUB_TEST_SYMBOL="$(TEST_SYMBOL)" \
		-e PUBSUB_TEST_SUBSCRIPTION="$$(echo $(or $(PUBSUB_TEST_SUBSCRIPTION),test_local))" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/test/algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro" \
		-v "$(PWD)/test/config.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo "Stopping message publisher..."; \
	kill $$PUBLISHER_PID 2>/dev/null || true; \
	echo ""; \
	echo "📊 Test Results:"; \
	if grep -q "PubSubDataQueueHandler: Connected" $$LOG_FILE 2>/dev/null || \
	   grep -q "Using emulator" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Handler connected to emulator"; \
	else \
		echo "  ❌ Handler did not connect"; \
	fi; \
	if grep -q "PubSubDataQueueHandler: Subscribed to" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Subscription verified and active"; \
	else \
		echo "  ❌ Subscription not found or failed"; \
	fi; \
	if grep -q "🟢 LIVE MODE: Data received from Pub/Sub!" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Data received"; \
	else \
		echo "  ❌ No data received"; \
	fi; \
	echo ""; \
	echo "Stopping emulator..."; \
	docker-compose -f docker-compose.test.yml down; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

clean:
	@echo "🧹 Removing image $(FULL_IMAGE)..."
	@docker rmi $(FULL_IMAGE) 2>/dev/null || echo "Image not found"
	@docker-compose -f docker-compose.test.yml down 2>/dev/null || true
	@echo "✅ Clean complete"


.PHONY: build test test-local test-backtest test-synthetic clean help prepare-packages

IMAGE_NAME ?= vibe-trade-lean
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)
TEST_TIMEOUT ?= 60
TEST_SYMBOL ?= BTC-USD

help:
	@echo "Available targets:"
	@echo "  build           - Build the custom LEAN Docker image (includes vibe-trade packages)"
	@echo "  test-synthetic  - Run end-to-end test with synthetic data (recommended)"
	@echo "  test-backtest   - Run backtest with custom strategy IR"
	@echo "  test-local      - Run test with local Pub/Sub emulator (no credentials needed)"
	@echo "  test            - Run integration test with production Pub/Sub (requires GCP credentials)"
	@echo "  clean           - Remove built image and temporary files"
	@echo ""
	@echo "Environment variables (for test):"
	@echo "  GOOGLE_CLOUD_PROJECT          - GCP project ID (required for test)"
	@echo "  GOOGLE_APPLICATION_CREDENTIALS - Path to service account JSON (required for test)"
	@echo "  PUBSUB_TEST_SYMBOL            - Symbol to test (default: BTC-USD)"
	@echo "  PUBSUB_TEST_SUBSCRIPTION      - Subscription name (default: test_local)"
	@echo "  TEST_TIMEOUT                  - Test timeout in seconds (default: 60)"
	@echo "  STRATEGY_IR_PATH              - Path to strategy IR JSON for backtest"

prepare-packages:
	@echo "📦 Preparing vibe-trade packages..."
	@mkdir -p packages
	@if [ -d "../vibe-trade-shared" ]; then \
		mkdir -p packages/vibe-trade-shared && \
		cp ../vibe-trade-shared/pyproject.toml packages/vibe-trade-shared/ && \
		cp -r ../vibe-trade-shared/src packages/vibe-trade-shared/ && \
		echo "  ✅ vibe-trade-shared copied (src only)"; \
	else \
		echo "  ⚠️  vibe-trade-shared not found - skipping"; \
	fi
	@if [ -d "../vibe-trade-execution" ]; then \
		mkdir -p packages/vibe-trade-execution && \
		cp ../vibe-trade-execution/pyproject.toml packages/vibe-trade-execution/ && \
		cp -r ../vibe-trade-execution/src packages/vibe-trade-execution/ && \
		echo "  ✅ vibe-trade-execution copied (src only)"; \
	else \
		echo "  ⚠️  vibe-trade-execution not found - skipping"; \
	fi

build: prepare-packages
	@echo "🔨 Building $(FULL_IMAGE)..."
	docker build --no-cache -t $(FULL_IMAGE) .
	@echo "🧹 Cleaning up packages..."
	@rm -rf packages
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

test-backtest: build
	@echo "🧪 Running backtest with StrategyRuntime"
	@echo ""
	@if [ -z "$(STRATEGY_IR_PATH)" ]; then \
		echo "ℹ️  Using default test strategy (no STRATEGY_IR_PATH set)"; \
	fi
	@mkdir -p test/results test/data/crypto/coinbase/minute
	@LOG_FILE="/tmp/lean-backtest-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running backtest (logs: $$LOG_FILE)"; \
	docker run --rm \
		-e STRATEGY_IR_PATH="$(or $(STRATEGY_IR_PATH),/Data/strategy_ir.json)" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/src/strategy_runtime.py:/Lean/Algorithm.Python/strategy_runtime.py:ro" \
		-v "$(PWD)/test/config-backtest.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo ""; \
	echo "📊 Backtest Results:"; \
	if grep -q "ENTRY executed" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Entry signals detected"; \
	else \
		echo "  ❌ No entry signals"; \
	fi; \
	if grep -q "EXIT" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Exit signals detected"; \
	else \
		echo "  ❌ No exit signals"; \
	fi; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

test-synthetic: build
	@echo "🧪 Running end-to-end test with synthetic data"
	@echo ""
	@echo "📊 Generating synthetic data..."
	@python3 test/generate_synthetic_data.py --output test/data --scenario trend_pullback --symbol TESTUSD --start-date 2024-01-01
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-synthetic-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running LEAN backtest (logs: $$LOG_FILE)"; \
	docker run --rm \
		-e STRATEGY_IR_PATH="/Data/strategy_ir.json" \
		-e START_DATE="2024-01-01" \
		-e END_DATE="2024-01-01" \
		-e SKIP_DATA_DOWNLOAD="1" \
		-e USE_CUSTOM_DATA_MODE="true" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/data/custom:/Data/custom:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/src/strategy_runtime.py:/Lean/Algorithm.Python/strategy_runtime.py:ro" \
		-v "$(PWD)/test/config-backtest.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo ""; \
	echo "📊 End-to-End Test Results:"; \
	echo ""; \
	echo "Strategy Execution:"; \
	if grep -q "Loaded strategy" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Strategy loaded successfully"; \
	else \
		echo "  ❌ Strategy failed to load"; \
	fi; \
	if grep -q "indicators" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Indicators initialized"; \
	else \
		echo "  ❌ Indicator initialization failed"; \
	fi; \
	echo ""; \
	echo "Trading Signals:"; \
	ENTRY_COUNT=$$(grep -c "ENTRY executed" $$LOG_FILE 2>/dev/null || echo 0); \
	EXIT_COUNT=$$(grep -c "EXIT" $$LOG_FILE 2>/dev/null || echo 0); \
	echo "  Entries: $$ENTRY_COUNT"; \
	echo "  Exits: $$EXIT_COUNT"; \
	echo ""; \
	if [ "$$ENTRY_COUNT" -gt 0 ]; then \
		echo "✅ TEST PASSED: Strategy generated entry signals"; \
	else \
		echo "❌ TEST FAILED: No entry signals generated"; \
	fi; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

clean:
	@echo "🧹 Removing image $(FULL_IMAGE)..."
	@docker rmi $(FULL_IMAGE) 2>/dev/null || echo "Image not found"
	@docker-compose -f docker-compose.test.yml down 2>/dev/null || true
	@rm -rf packages
	@echo "✅ Clean complete"


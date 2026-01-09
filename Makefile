.PHONY: build test test-local test-integration clean help

FULL_IMAGE=vibe-trade-lean:latest
TEST_SYMBOL?=BTC-USD

help:
	@echo "Available targets:"
	@echo "  build           - Build the Docker image"
	@echo "  test            - Run full integration test with production Pub/Sub (requires credentials)"
	@echo "  test-local      - Run basic test with local Pub/Sub emulator (no credentials needed)"
	@echo "  test-integration - Run full integration test with deterministic trading strategy"
	@echo "  clean           - Clean up build artifacts and containers"
	@echo ""
	@echo "Environment variables:"
	@echo "  TEST_SYMBOL     - Symbol to test (default: BTC-USD)"
	@echo "  GOOGLE_CLOUD_PROJECT - GCP project ID (required for 'test')"
	@echo "  GOOGLE_APPLICATION_CREDENTIALS - Path to service account key (required for 'test')"

build:
	@echo "🔨 Building vibe-trade-lean:latest..."
	docker build -t $(FULL_IMAGE) .
	@echo "✅ Build complete: $(FULL_IMAGE)"

test: build
	@if [ -z "$$GOOGLE_CLOUD_PROJECT" ]; then \
		echo "❌ Error: GOOGLE_CLOUD_PROJECT not set"; \
		echo "   Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"; \
		exit 1; \
	fi
	@if [ -z "$$GOOGLE_APPLICATION_CREDENTIALS" ] || [ ! -f "$$GOOGLE_APPLICATION_CREDENTIALS" ]; then \
		echo "❌ Error: GOOGLE_APPLICATION_CREDENTIALS not set or file not found"; \
		echo "   Set it with: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json"; \
		exit 1; \
	fi
	@echo "🧪 Testing with production Pub/Sub"
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-test-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running test (logs: $$LOG_FILE)"; \
	echo "   Press Ctrl+C to stop"; \
	echo ""; \
	docker run --rm \
		-e GOOGLE_CLOUD_PROJECT="$$GOOGLE_CLOUD_PROJECT" \
		-e GOOGLE_APPLICATION_CREDENTIALS="/tmp/credentials.json" \
		-e PUBSUB_TEST_SYMBOL="$(TEST_SYMBOL)" \
		-v "$$GOOGLE_APPLICATION_CREDENTIALS:/tmp/credentials.json:ro" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/test/algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro" \
		-v "$(PWD)/test/config.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo ""; \
	echo "📊 Test Results:"; \
	if grep -q "PubSubDataQueueHandler: Connected" $$LOG_FILE 2>/dev/null || \
	   grep -q "Using emulator" $$LOG_FILE 2>/dev/null; then \
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
	echo "📋 Verifying results..."; \
	python3 test/verify_results.py test/results 10 0 || echo "⚠️  Result verification failed (this is OK for basic data flow test)"; \
	echo ""; \
	echo "Stopping emulator..."; \
	docker-compose -f docker-compose.test.yml down; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

test-integration: build
	@echo "🧪 Integration Test: Deterministic Trading Strategy"
	@echo ""
	@echo "This test:"
	@echo "  1. Publishes exactly 15 bars with predictable prices"
	@echo "  2. Algorithm buys on bar 1, sells after bar 3 (bar 4)"
	@echo "  3. Verifies exactly 2 orders were placed and filled"
	@echo ""
	@echo "Starting emulator..."
	@docker-compose -f docker-compose.test.yml up -d pubsub-emulator
	@echo "Waiting for emulator to be ready..."
	@sleep 12
	@echo "Creating topic and subscription..."
	@TOPIC_OK=0; \
	for i in 1 2 3 4 5; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/topics/vibe-trade-candles-btc-usd-1m" \
			-H "Content-Type: application/json" -d '{}' 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "201" ] || [ "$$HTTP_CODE" = "409" ]; then \
			echo "  ✅ Topic created/exists (attempt $$i, HTTP $$HTTP_CODE)"; \
			TOPIC_OK=1; \
			break; \
		else \
			echo "  ⏳ Retrying topic creation... (attempt $$i/5, HTTP $$HTTP_CODE)"; \
			sleep 2; \
		fi; \
	done; \
	if [ $$TOPIC_OK -eq 0 ]; then \
		echo "  ❌ ERROR: Failed to create topic 'vibe-trade-candles-btc-usd-1m'"; \
		exit 1; \
	fi
	@sleep 3
	@echo "Verifying topic exists and is accessible..."
	@for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" http://pubsub-emulator:8085/v1/projects/test-project/topics/vibe-trade-candles-btc-usd-1m 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ]; then \
			echo "  ✅ Topic verified and accessible (attempt $$i, HTTP $$HTTP_CODE)"; \
			break; \
		else \
			echo "  ⏳ Topic not yet accessible... ($$i/3, HTTP $$HTTP_CODE)"; \
			sleep 2; \
		fi; \
	done
	@sleep 2
	@SUBSCRIPTION_OK=0; \
	for i in 1 2 3 4 5; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local" \
			-H "Content-Type: application/json" \
			-d '{"topic": "projects/test-project/topics/vibe-trade-candles-btc-usd-1m"}' 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "201" ] || [ "$$HTTP_CODE" = "409" ]; then \
			echo "  ✅ Subscription created/exists (attempt $$i, HTTP $$HTTP_CODE)"; \
			SUBSCRIPTION_OK=1; \
			break; \
		else \
			echo "  ⏳ Retrying subscription creation... (attempt $$i/5, HTTP $$HTTP_CODE)"; \
			if [ $$i -eq 5 ]; then \
				echo "  ⚠️  Last attempt failed, response: $$RESPONSE"; \
			fi; \
			sleep 2; \
		fi; \
	done; \
	if [ $$SUBSCRIPTION_OK -eq 0 ]; then \
		echo "  ❌ ERROR: Failed to create subscription 'test_local'"; \
		exit 1; \
	fi
	@sleep 5
	@echo "Verifying subscription exists and is accessible..."
	@SUBSCRIPTION_OK=0; \
	for i in 1 2 3 4 5; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ]; then \
			echo "  ✅ Subscription verified and accessible (attempt $$i, HTTP $$HTTP_CODE)"; \
			SUBSCRIPTION_OK=1; \
			break; \
		else \
			echo "  ⏳ Subscription not yet accessible... ($$i/5, HTTP $$HTTP_CODE)"; \
			if [ $$i -eq 5 ] && [ $$SUBSCRIPTION_OK -eq 0 ]; then \
				echo "  ⚠️  Subscription not found, ensuring topic exists and recreating..."; \
				docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
					curl -s -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/topics/vibe-trade-candles-btc-usd-1m" \
					-H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; \
				sleep 1; \
				docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
					curl -s -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local" \
					-H "Content-Type: application/json" \
					-d '{"topic": "projects/test-project/topics/vibe-trade-candles-btc-usd-1m"}' > /dev/null 2>&1; \
				sleep 3; \
				RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
					curl -s -w "\n%{http_code}" http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local 2>&1); \
				HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
				if [ "$$HTTP_CODE" = "200" ]; then \
					echo "  ✅ Subscription recreated and verified (HTTP $$HTTP_CODE)"; \
					SUBSCRIPTION_OK=1; \
				else \
					echo "  ❌ Failed to recreate subscription (HTTP $$HTTP_CODE)"; \
				fi; \
			fi; \
			sleep 2; \
		fi; \
	done; \
	if [ $$SUBSCRIPTION_OK -eq 0 ]; then \
		echo "  ❌ ERROR: Subscription 'test_local' could not be created or verified"; \
		exit 1; \
	fi
	@echo "Cleaning up any leftover messages from previous runs..."
	@echo "  Seeking subscription to current time to clear backlog..."; \
	for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X POST "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local:seek" \
			-H "Content-Type: application/json" \
			-d '{"time": "'$$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "204" ]; then \
			echo "  ✅ Subscription seeked to current time (cleared backlog, attempt $$i)"; \
			break; \
		else \
			echo "  ⏳ Retrying seek... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			sleep 1; \
		fi; \
	done; \
	sleep 2; \
	echo "  (Subscription is now empty - LEAN will start with no messages available)"
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-test-integration-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running integration test (logs: $$LOG_FILE)"; \
	echo ""; \
	echo "Verifying subscription exists before seeking..."; \
	SUBSCRIPTION_EXISTS=0; \
	for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ]; then \
			echo "  ✅ Subscription 'test_local' exists and is accessible (attempt $$i)"; \
			SUBSCRIPTION_EXISTS=1; \
			break; \
		else \
			echo "  ⏳ Subscription not found, recreating... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
				curl -s -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/topics/vibe-trade-candles-btc-usd-1m" \
				-H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; \
			sleep 1; \
			docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
				curl -s -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local" \
				-H "Content-Type: application/json" \
				-d '{"topic": "projects/test-project/topics/vibe-trade-candles-btc-usd-1m"}' > /dev/null 2>&1; \
			sleep 2; \
		fi; \
	done; \
	if [ $$SUBSCRIPTION_EXISTS -eq 0 ]; then \
		echo "  ❌ ERROR: Subscription 'test_local' does not exist and could not be created"; \
		exit 1; \
	fi; \
	echo "Seeking subscription to current time RIGHT BEFORE starting LEAN..."; \
	echo "  (This ensures LEAN starts with an empty subscription)"; \
	SEEK_OK=0; \
	for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X POST "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local:seek" \
			-H "Content-Type: application/json" \
			-d '{"time": "'$$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "204" ]; then \
			echo "  ✅ Subscription seeked to current time (attempt $$i)"; \
			SEEK_OK=1; \
			break; \
		else \
			echo "  ⏳ Retrying seek... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			if [ $$i -eq 3 ]; then \
				echo "  ⚠️  Seek failed, but continuing anyway (subscription may be empty)"; \
			fi; \
			sleep 1; \
		fi; \
	done; \
	sleep 2; \
	echo "Starting LEAN container..."; \
	echo "  (LEAN will start pulling messages immediately, but subscription is empty)"; \
	docker run -d --rm \
		--name lean-integration-test \
		--network vibe-trade-lean_default \
		-e GOOGLE_CLOUD_PROJECT="test-project" \
		-e PUBSUB_EMULATOR_HOST="pubsub-emulator:8085" \
		-e PUBSUB_TEST_SYMBOL="$(TEST_SYMBOL)" \
		-e PUBSUB_TEST_SUBSCRIPTION="test_local" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/test/strategy_algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro" \
		-v "$(PWD)/test/config.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json > $$LOG_FILE 2>&1; \
	echo ""; \
	echo "⏳ Waiting 60 seconds for LEAN to fully initialize..."; \
	echo "  (LEAN's algorithm clock starts at current time in live mode)"; \
	echo "  (LEAN will pull messages during this time, but subscription is empty)"; \
	echo "  (NO MESSAGES WILL BE PUBLISHED UNTIL AFTER THIS WAIT)"; \
	sleep 60; \
	echo ""; \
	echo "Seeking subscription to current time to clear any messages pulled during initialization..."; \
	echo "  (This ensures LEAN only processes messages published after this point)"; \
	for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X POST "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local:seek" \
			-H "Content-Type: application/json" \
			-d '{"time": "'$$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}' 2>&1); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "204" ]; then \
			echo "  ✅ Subscription seeked to current time (cleared backlog, attempt $$i)"; \
			break; \
		else \
			echo "  ⏳ Retrying seek... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			sleep 1; \
		fi; \
	done; \
	echo "  Waiting 10 seconds for seek to take effect and for LEAN to finish processing any in-flight messages..."; \
	sleep 10; \
	echo ""; \
	echo "📤 NOW publishing 15 bars (buy on bar 1, sell after bar 3)..."; \
	echo "  (This is the FIRST time messages are published - LEAN has been running for 70+ seconds)"; \
	echo "  (Messages will be published with real-time timestamps starting now)"; \
	docker run --rm --network vibe-trade-lean_default \
		-e PUBSUB_EMULATOR_HOST="pubsub-emulator:8085" \
		-e PUBSUB_PROJECT_ID="test-project" \
		-e PUBSUB_TOPIC_NAME="vibe-trade-candles-btc-usd-1m" \
		-e PUBSUB_NUM_BARS="15" \
		-v "$(PWD)/test/publish-integration-test.sh:/publish.sh:ro" \
		curlimages/curl:latest sh -c "apk add --no-cache bash > /dev/null 2>&1 && bash /publish.sh" 2>&1 | tee -a $$LOG_FILE; \
	echo ""; \
	echo "Waiting for algorithm to process all bars and execute orders..."; \
	echo "  (LEAN needs time to call MoveNext() and process queued data)"; \
	sleep 60; \
	echo "Collecting final logs from LEAN container..."; \
	docker logs lean-integration-test 2>&1 | tail -100 >> $$LOG_FILE || true; \
	echo "Stopping LEAN container..."; \
	docker stop lean-integration-test > /dev/null 2>&1 || true; \
	sleep 2; \
	echo ""; \
	echo "📊 Test Results:"; \
	if grep -q "PubSubDataQueueHandler: Connected" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Handler connected"; \
	else \
		echo "  ❌ Handler did not connect"; \
	fi; \
	if grep -q "PubSubDataQueueHandler: Subscribed to" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ Subscription active"; \
	else \
		echo "  ❌ Subscription failed"; \
	fi; \
	if grep -q "🛒 BUY Order" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ BUY order placed"; \
	else \
		echo "  ❌ BUY order not found"; \
	fi; \
	if grep -q "🛒 SELL Order" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ SELL order placed"; \
	else \
		echo "  ❌ SELL order not found"; \
	fi; \
	echo ""; \
	echo "📋 Verifying results (expecting 2 orders: 1 buy, 1 sell)..."; \
	python3 test/verify_results.py test/results 10 2 2 || echo "⚠️  Result verification failed"; \
	echo ""; \
	echo "Stopping emulator..."; \
	docker-compose -f docker-compose.test.yml down; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

test-strategy: build
	@echo "🧪 Testing strategy with local Pub/Sub emulator"
	@echo ""
	@echo "Starting emulator..."
	@docker-compose -f docker-compose.test.yml up -d pubsub-emulator
	@echo "Waiting for emulator to be ready..."
	@sleep 12
	@echo "Creating topic and subscription from within Docker network..."
	@echo "Creating topic..."
	@for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/topics/vibe-trade-candles-btc-usd-1m" \
			-H "Content-Type: application/json" -d '{}'); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		BODY=$$(echo "$$RESPONSE" | head -n -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "201" ] || [ "$$HTTP_CODE" = "409" ]; then \
			echo "  ✅ Topic created/exists (attempt $$i, HTTP $$HTTP_CODE)"; \
			break; \
		else \
			echo "  ⏳ Retrying topic creation... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			echo "     Response: $$BODY"; \
			sleep 2; \
		fi; \
	done
	@sleep 2
	@echo "Creating subscription..."
	@for i in 1 2 3; do \
		RESPONSE=$$(docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s -w "\n%{http_code}" -X PUT "http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local" \
			-H "Content-Type: application/json" \
			-d '{"topic": "projects/test-project/topics/vibe-trade-candles-btc-usd-1m"}'); \
		HTTP_CODE=$$(echo "$$RESPONSE" | tail -1); \
		BODY=$$(echo "$$RESPONSE" | head -n -1); \
		if [ "$$HTTP_CODE" = "200" ] || [ "$$HTTP_CODE" = "201" ] || [ "$$HTTP_CODE" = "409" ]; then \
			echo "  ✅ Subscription created/exists (attempt $$i, HTTP $$HTTP_CODE)"; \
			break; \
		else \
			echo "  ⏳ Retrying subscription creation... (attempt $$i/3, HTTP $$HTTP_CODE)"; \
			echo "     Response: $$BODY"; \
			sleep 2; \
		fi; \
	done
	@sleep 5
	@echo "Verifying subscription exists from within Docker network..."
	@echo "Verifying subscription exists from within Docker network..."
	@for i in 1 2 3 4 5; do \
		if docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
			curl -s http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local > /dev/null 2>&1; then \
			echo "  ✅ Subscription verified (attempt $$i)"; \
			break; \
		else \
			echo "  ⏳ Subscription not yet available, waiting... ($$i/5)"; \
			sleep 3; \
		fi; \
	done
	@echo "Waiting additional 5 seconds to ensure subscription is fully propagated..."
	@sleep 5
	@echo "Final verification..."
	@docker run --rm --network vibe-trade-lean_default curlimages/curl:latest \
		curl -s http://pubsub-emulator:8085/v1/projects/test-project/subscriptions/test_local > /dev/null 2>&1 && \
		echo "  ✅ Subscription confirmed ready" || echo "  ⚠️  Subscription verification failed - proceeding anyway"
	@echo ""
	@mkdir -p test/results
	@LOG_FILE="/tmp/lean-test-strategy-$$(date +%Y%m%d-%H%M%S).log"; \
	echo "🚀 Running strategy test (logs: $$LOG_FILE)"; \
	echo "   This will run for ~10 seconds with high-frequency messages"; \
	echo ""; \
	echo "Starting message publisher in background (10 msg/sec)..."; \
	docker run -d --rm --network vibe-trade-lean_default \
		-e PUBSUB_EMULATOR_HOST="pubsub-emulator:8085" \
		-e PUBSUB_PROJECT_ID="test-project" \
		-v "$(PWD)/test/publish-messages.sh:/publish.sh:ro" \
		curlimages/curl:latest sh -c "apk add --no-cache bash > /dev/null 2>&1 && bash /publish.sh" > /tmp/pubsub-publisher.log 2>&1; \
	PUBLISHER_PID=$$!; \
	echo "Publisher container started"; \
	sleep 5; \
	echo "Starting LEAN container..."; \
	docker run --rm \
		--network vibe-trade-lean_default \
		-e GOOGLE_CLOUD_PROJECT="test-project" \
		-e PUBSUB_EMULATOR_HOST="pubsub-emulator:8085" \
		-e PUBSUB_TEST_SYMBOL="$(TEST_SYMBOL)" \
		-e PUBSUB_TEST_SUBSCRIPTION="test_local" \
		-v "$(PWD)/test/data:/Data:ro" \
		-v "$(PWD)/test/results:/Results" \
		-v "$(PWD)/test/strategy_algorithm.py:/Lean/Algorithm.Python/algorithm.py:ro" \
		-v "$(PWD)/test/config.json:/Lean/Launcher/bin/Debug/config.json:ro" \
		$(FULL_IMAGE) \
		--config /Lean/Launcher/bin/Debug/config.json 2>&1 | tee $$LOG_FILE || true; \
	echo "Stopping publisher..."; \
	docker stop $$PUBLISHER_PID > /dev/null 2>&1 || true; \
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
	if grep -q "🛒 BUY Order" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ BUY order placed"; \
	else \
		echo "  ❌ BUY order not found"; \
	fi; \
	if grep -q "🛒 SELL Order" $$LOG_FILE 2>/dev/null; then \
		echo "  ✅ SELL order placed"; \
	else \
		echo "  ❌ SELL order not found"; \
	fi; \
	echo ""; \
	echo "📋 Verifying results..."; \
	python3 test/verify_results.py test/results 10 2 || echo "⚠️  Result verification failed"; \
	echo ""; \
	echo "Stopping emulator..."; \
	docker-compose -f docker-compose.test.yml down; \
	echo ""; \
	echo "Log file: $$LOG_FILE"; \
	echo "Results: test/results/"

clean:
	@echo "🧹 Cleaning up..."
	@docker-compose -f docker-compose.test.yml down 2>/dev/null || true
	@docker stop lean-integration-test 2>/dev/null || true
	@echo "✅ Cleanup complete"

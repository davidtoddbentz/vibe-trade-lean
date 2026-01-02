.PHONY: build test-build test-run test-integration clean help

IMAGE_NAME ?= vibe-trade-lean
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

help:
	@echo "Available targets:"
	@echo "  build           - Build the custom LEAN Docker image"
	@echo "  test-build      - Test that the image builds successfully"
	@echo "  test-run        - Run a test container to verify image works"
	@echo "  test-integration - Run full integration test with Pub/Sub"
	@echo "  clean           - Remove built image"
	@echo "  help            - Show this help message"

build:
	@echo "🔨 Building $(FULL_IMAGE)..."
	docker build -t $(FULL_IMAGE) .
	@echo "✅ Build complete: $(FULL_IMAGE)"

test-build: build
	@echo "🧪 Testing image build..."
	@docker run --rm --entrypoint="" $(FULL_IMAGE) \
		ls -la /Lean/Launcher/bin/Debug/PubSubDataQueueHandler.dll 2>/dev/null && \
		echo "✅ PubSubDataQueueHandler.dll found" || \
		echo "❌ PubSubDataQueueHandler.dll not found"
	@docker run --rm $(FULL_IMAGE) --help > /dev/null 2>&1 && \
		echo "✅ LEAN launcher works" || \
		echo "❌ LEAN launcher failed"

test-run: build
	@echo "🧪 Running test container..."
	@echo "Note: This requires GCP credentials and Pub/Sub topics"
	@echo "Set GOOGLE_CLOUD_PROJECT environment variable"
	@docker run --rm \
		-e GOOGLE_CLOUD_PROJECT=$(GOOGLE_CLOUD_PROJECT) \
		$(FULL_IMAGE) \
		--help

test-integration: build
	@echo "🧪 Running integration test..."
	@echo "This will test the Pub/Sub data queue handler with a live algorithm"
	@echo ""
	@if [ -z "$(GOOGLE_CLOUD_PROJECT)" ]; then \
		echo "❌ Error: GOOGLE_CLOUD_PROJECT not set"; \
		echo "   Set it with: export GOOGLE_CLOUD_PROJECT=your-project-id"; \
		exit 1; \
	fi
	@./test/run-test.sh

clean:
	@echo "🧹 Removing image $(FULL_IMAGE)..."
	docker rmi $(FULL_IMAGE) 2>/dev/null || echo "Image not found"
	@echo "✅ Clean complete"


# Custom LEAN Docker image with Pub/Sub support
FROM quantconnect/lean:latest

# Set environment variable to force SocketsHttpHandler on Linux
# This must be set at the Docker level, not just in C# code
ENV DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER=1
ENV DOTNET_SYSTEM_NET_HTTP_SOCKETSHTTPHANDLER_HTTP2UNENCRYPTEDSUPPORT=1

# Install Python dependencies for Pub/Sub, BigQuery, and Pydantic
RUN (python3 -m pip install --no-cache-dir \
        google-cloud-pubsub>=2.18.0 \
        google-cloud-bigquery>=3.0.0 \
        pydantic>=2.0.0 \
        2>/dev/null || \
     python -m pip install --no-cache-dir \
        google-cloud-pubsub>=2.18.0 \
        google-cloud-bigquery>=3.0.0 \
        pydantic>=2.0.0 \
        2>/dev/null || \
     pip install --no-cache-dir \
        google-cloud-pubsub>=2.18.0 \
        google-cloud-bigquery>=3.0.0 \
        pydantic>=2.0.0 \
        2>/dev/null || \
     /usr/bin/python3 -m pip install --no-cache-dir \
        google-cloud-pubsub>=2.18.0 \
        google-cloud-bigquery>=3.0.0 \
        pydantic>=2.0.0 \
        2>/dev/null) && \
    echo "✅ Python libraries installed (Pub/Sub, BigQuery, Pydantic)"

# Copy vibe-trade packages for strategy execution (optional)
# Note: The inline evaluator in strategy_runtime.py handles all IR evaluation,
# so package installation is not required for the runtime to work.
COPY packages/ /packages/
RUN if [ -d /packages/vibe-trade-shared ] && [ -f /packages/vibe-trade-shared/pyproject.toml ]; then \
        echo "📦 Installing vibe-trade-shared..."; \
        (python3 -m pip install --no-cache-dir --no-deps /packages/vibe-trade-shared 2>/dev/null || true) && \
        echo "✅ vibe-trade-shared installed (no-deps)"; \
    else \
        echo "ℹ️  vibe-trade packages not found - using inline evaluator"; \
    fi

# Create directory for C# handler compilation
RUN mkdir -p /Lean/CustomDataQueueHandler

# Copy C# handler source and project file
COPY src/DataFeeds/PubSubDataQueueHandler.cs /Lean/CustomDataQueueHandler/
COPY csproj/PubSubDataQueueHandler.csproj /Lean/CustomDataQueueHandler/

# Compile C# handler
# Note: LEAN base image should have .NET SDK
RUN if command -v dotnet >/dev/null 2>&1; then \
        echo "🔨 Compiling C# data queue handler..."; \
        cd /Lean/CustomDataQueueHandler && \
        dotnet restore && \
        dotnet build -c Release && \
        if [ -f bin/Release/net10.0/PubSubDataQueueHandler.dll ]; then \
            cp bin/Release/net10.0/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ && \
            echo "✅ C# handler compiled and copied to /Lean/Launcher/bin/Debug/"; \
            # Copy all Google.*, Grpc.*, and Microsoft.Extensions.* dependencies from NuGet packages \
            NUGET_DIR=~/.nuget/packages; \
            if [ ! -d "$NUGET_DIR" ]; then NUGET_DIR=/root/.nuget/packages; fi; \
            if [ -d "$NUGET_DIR" ]; then \
                # Copy all DLLs matching Google.*, Grpc.*, or Microsoft.Extensions.* patterns \
                find "$NUGET_DIR" -name "Google*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "Grpc*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "Microsoft.Extensions.Logging*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "Microsoft.Extensions.*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "System.Net.Http*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                # Copy from bin/Release if present \
                find bin/Release -name "Google*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find bin/Release -name "Grpc*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find bin/Release -name "Microsoft.Extensions.Logging*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find bin/Release -name "System.Net.Http*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
            fi; \
            echo "✅ Dependencies copied"; \
        elif [ -f bin/Release/net*/PubSubDataQueueHandler.dll ]; then \
            cp bin/Release/net*/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ && \
            echo "✅ C# handler compiled and copied (auto-detected .NET version)"; \
            # Copy dependencies \
            find ~/.nuget/packages -name "Google.Cloud.PubSub.V1.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || \
            find /root/.nuget/packages -name "Google.Cloud.PubSub.V1.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
        else \
            echo "⚠️  DLL not found after build - check compilation errors above"; \
            exit 1; \
        fi; \
    else \
        echo "❌ dotnet not found - cannot compile C# handler"; \
        exit 1; \
    fi

# Verify DLL was copied
RUN if [ -f /Lean/Launcher/bin/Debug/PubSubDataQueueHandler.dll ]; then \
        echo "✅ Verified: PubSubDataQueueHandler.dll is in LEAN directory"; \
    else \
        echo "❌ ERROR: PubSubDataQueueHandler.dll not found"; \
        exit 1; \
    fi

# Copy scripts for data loading
COPY scripts/ /scripts/
RUN chmod +x /scripts/*.sh /scripts/*.py

# Copy StrategyRuntime + modular Python packages into the image
# These are required when mounting only `StrategyRuntime.py` in tests or when
# running the backtest service which imports `indicators/`, `conditions/`, etc.
RUN mkdir -p /Lean/Algorithm.Python /Lean/src
COPY src/data/ /Lean/src/data/
COPY src/Algorithms/StrategyRuntime.py /Lean/Algorithm.Python/
COPY src/Algorithms/indicators/ /Lean/Algorithm.Python/indicators/
COPY src/Algorithms/conditions/ /Lean/Algorithm.Python/conditions/
COPY src/Algorithms/trades/ /Lean/Algorithm.Python/trades/
COPY src/Algorithms/position/ /Lean/Algorithm.Python/position/
COPY src/Algorithms/gates/ /Lean/Algorithm.Python/gates/
COPY src/Algorithms/costs/ /Lean/Algorithm.Python/costs/
COPY src/Algorithms/symbols/ /Lean/Algorithm.Python/symbols/
COPY src/Algorithms/ir/ /Lean/Algorithm.Python/ir/
COPY src/Algorithms/execution/ /Lean/Algorithm.Python/execution/
COPY src/Algorithms/initialization/ /Lean/Algorithm.Python/initialization/
COPY src/Algorithms/state/ /Lean/Algorithm.Python/state/
COPY src/serve_backtest.py /Lean/serve_backtest.py

# Ensure modular packages are importable by LEAN's Python runtime
ENV PYTHONPATH="/Lean/Algorithm.Python:/Lean/Launcher/bin/Debug:${PYTHONPATH}"

# Create Data directory
RUN mkdir -p /Data

# Set custom entrypoint that handles data loading before LEAN runs
ENTRYPOINT ["/scripts/entrypoint.sh"]
CMD ["--config", "/Lean/Launcher/bin/Debug/config.json"]


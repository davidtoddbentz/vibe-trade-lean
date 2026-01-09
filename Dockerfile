# Custom LEAN Docker image with Pub/Sub support and backtest capabilities
FROM quantconnect/lean:latest

# Set environment variable to force SocketsHttpHandler on Linux
ENV DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER=1
ENV DOTNET_SYSTEM_NET_HTTP_SOCKETSHTTPHANDLER_HTTP2UNENCRYPTEDSUPPORT=1

# Install Python dependencies
# - google-cloud-pubsub: For real-time data streaming (live trading)
# - google-cloud-storage: For GCS access (backtesting)
# - pyarrow/fastavro: For parsing candle data from GCS
# - pydantic: For data models
RUN (python3 -m pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null || \
     python -m pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null || \
     pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null) && \
    echo "✅ Python dependencies installed"

# Create directories
RUN mkdir -p /Lean/CustomDataQueueHandler /Lean/src

# Copy Python modules for data handling and strategy execution
COPY src/data/ /Lean/src/data/
COPY src/Algorithms/StrategyRuntime.py /Lean/Algorithm.Python/
COPY src/run_backtest.py /Lean/

# Create __init__.py for src module
RUN touch /Lean/src/__init__.py

# Copy C# handler source and project file
COPY src/DataFeeds/PubSubDataQueueHandler.cs /Lean/CustomDataQueueHandler/
COPY csproj/PubSubDataQueueHandler.csproj /Lean/CustomDataQueueHandler/

# Compile C# handler for live trading Pub/Sub support
RUN if command -v dotnet >/dev/null 2>&1; then \
        echo "🔨 Compiling C# data queue handler..."; \
        cd /Lean/CustomDataQueueHandler && \
        dotnet restore && \
        dotnet build -c Release && \
        if [ -f bin/Release/net10.0/PubSubDataQueueHandler.dll ]; then \
            cp bin/Release/net10.0/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ && \
            echo "✅ C# handler compiled and copied"; \
            NUGET_DIR=~/.nuget/packages; \
            if [ ! -d "$NUGET_DIR" ]; then NUGET_DIR=/root/.nuget/packages; fi; \
            if [ -d "$NUGET_DIR" ]; then \
                find "$NUGET_DIR" -name "Google*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "Grpc*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find "$NUGET_DIR" -name "Microsoft.Extensions.*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find bin/Release -name "Google*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
                find bin/Release -name "Grpc*.dll" -exec cp {} /Lean/Launcher/bin/Debug/ \; 2>/dev/null || true; \
            fi; \
            echo "✅ Dependencies copied"; \
        elif [ -f bin/Release/net*/PubSubDataQueueHandler.dll ]; then \
            cp bin/Release/net*/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ && \
            echo "✅ C# handler compiled (auto-detected .NET version)"; \
        else \
            echo "⚠️  DLL not found after build"; \
            exit 1; \
        fi; \
    else \
        echo "❌ dotnet not found"; \
        exit 1; \
    fi

# Verify C# DLL was copied
RUN if [ -f /Lean/Launcher/bin/Debug/PubSubDataQueueHandler.dll ]; then \
        echo "✅ Verified: PubSubDataQueueHandler.dll installed"; \
    else \
        echo "❌ ERROR: PubSubDataQueueHandler.dll not found"; \
        exit 1; \
    fi

# Copy LEAN data files (symbol properties, market hours)
COPY data/ /Lean/Data/

# Set PYTHONPATH for imports
ENV PYTHONPATH="/Lean:${PYTHONPATH}"

# Default: run as LEAN engine (for Cloud Run Jobs / backtests)
# Override CMD for different use cases (e.g., serve_backtest.py for HTTP service)
WORKDIR /Lean

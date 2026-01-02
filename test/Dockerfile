# Test Dockerfile that uses the custom LEAN image
FROM vibe-trade-lean:latest

# Copy test algorithm
COPY test/algorithm.py /Lean/Algorithm.Python/algorithm.py

# Copy config
COPY test/config.json /Lean/Launcher/bin/Debug/config.json

# Set working directory
WORKDIR /Lean/Launcher/bin/Debug


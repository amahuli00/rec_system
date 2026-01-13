# Multi-stage Dockerfile for Recommendation Service
#
# This Dockerfile uses a multi-stage build pattern to:
# 1. Keep the final image small (only production dependencies)
# 2. Separate build-time dependencies from runtime dependencies
# 3. Improve security (fewer packages = smaller attack surface)
# 4. Speed up builds with layer caching

# ============================================================================
# Stage 1: Builder
# ============================================================================
# This stage installs all dependencies including build tools
FROM python:3.9-slim as builder

# Install system dependencies needed for building Python packages
# - gcc, g++: C/C++ compilers for building native extensions
# - libomp-dev: OpenMP library for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy only requirements first (layer caching optimization)
# If requirements don't change, Docker reuses this layer
COPY requirements.txt .

# Install Python dependencies into a virtual environment
# Why virtual environment in Docker? Makes it easy to copy to runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
# --no-cache-dir: Don't cache pip downloads (saves space)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# ============================================================================
# This stage contains only what's needed to run the service
FROM python:3.9-slim

# Install only runtime dependencies (no build tools)
# libomp5: OpenMP runtime library for XGBoost (smaller than dev package)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libomp5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# Running as non-root prevents privilege escalation attacks
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
# .dockerignore controls what gets copied (excludes venv, __pycache__, etc.)
COPY candidate_gen/ ./candidate_gen/
COPY ranking/ ./ranking/
COPY serving/ ./serving/

# Copy model artifacts
# In production, you might fetch these from cloud storage instead
COPY candidate_gen/artifacts/ ./candidate_gen/artifacts/
COPY ranking/models/ ./ranking/models/
COPY ranking/features/*.parquet ./ranking/features/

# Set Python path so imports work
ENV PYTHONPATH=/app

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
# This is documentation - doesn't actually publish the port
EXPOSE 8000

# Health check
# Docker/Kubernetes will call this to determine if container is healthy
# --fail: Return non-zero exit code if HTTP status is not 2xx
# --silent: Don't show progress
# --show-error: Show errors if any
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:8000/health || exit 1

# Set OpenMP library path for XGBoost
# This tells the dynamic linker where to find libomp
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Default command
# --host 0.0.0.0: Listen on all interfaces (required for Docker)
# --port 8000: Port to listen on
# --workers 1: Single worker process (increase for more throughput)
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

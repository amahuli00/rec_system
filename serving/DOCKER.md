# Docker Deployment Guide

This guide explains how to build and run the recommendation service using Docker.

## Prerequisites

Install Docker Desktop:
- **Mac**: Download from https://www.docker.com/products/docker-desktop/
- **Linux**: `sudo apt-get install docker.io` (Ubuntu) or equivalent
- **Windows**: Download Docker Desktop from official website

Verify installation:
```bash
docker --version
docker-compose --version  # Optional but recommended
```

## Building the Docker Image

### Build Command

From the project root:

```bash
docker build -t recommendation-service:latest .
```

**What this does:**
- `-t recommendation-service:latest`: Tags the image with name and version
- `.`: Uses current directory as build context (looks for Dockerfile here)

### Build Process Explained

The Dockerfile uses a **multi-stage build**:

**Stage 1: Builder**
- Base: `python:3.9-slim`
- Installs build tools (gcc, g++, libomp-dev)
- Creates virtual environment
- Installs all Python dependencies
- ~500MB total size

**Stage 2: Runtime** (final image)
- Base: `python:3.9-slim`
- Copies only the virtual environment from builder (no build tools)
- Installs only runtime dependencies (libomp5)
- Copies application code and model artifacts
- Runs as non-root user for security
- ~300MB total size

**Benefits:**
- Final image is 40% smaller (no build tools)
- Faster deployments (smaller image to push/pull)
- More secure (fewer packages = smaller attack surface)
- Layer caching speeds up rebuilds

### Build Options

**Fast rebuild (use cached layers):**
```bash
docker build -t recommendation-service:latest .
```

**Force clean build (no cache):**
```bash
docker build --no-cache -t recommendation-service:latest .
```

**Build with specific tag:**
```bash
docker build -t recommendation-service:v1.0.0 .
```

**Build and view progress:**
```bash
docker build --progress=plain -t recommendation-service:latest .
```

## Running the Container

### Basic Run

```bash
docker run -p 8000:8000 recommendation-service:latest
```

**What this does:**
- `-p 8000:8000`: Maps host port 8000 to container port 8000
- Service accessible at `http://localhost:8000`

### Run in Background (Detached Mode)

```bash
docker run -d --name rec-service -p 8000:8000 recommendation-service:latest
```

**Options:**
- `-d`: Detached mode (runs in background)
- `--name rec-service`: Name the container for easy reference
- Container ID returned (use to stop/remove later)

### View Logs

```bash
# Follow logs in real-time
docker logs -f rec-service

# View last 100 lines
docker logs --tail 100 rec-service

# View logs since 10 minutes ago
docker logs --since 10m rec-service
```

### Stop and Remove Container

```bash
# Stop the container
docker stop rec-service

# Remove the container
docker rm rec-service

# Stop and remove in one command
docker rm -f rec-service
```

## Advanced Configuration

### Environment Variables

Pass configuration via environment variables:

```bash
docker run -d \
  -p 8000:8000 \
  -e CANDIDATE_POOL_SIZE=200 \
  -e RANKING_MODEL_NAME=xgboost_v2 \
  recommendation-service:latest
```

**Common variables:**
- `CANDIDATE_POOL_SIZE`: Number of candidates before ranking (default: 100)
- `RANKING_MODEL_NAME`: Which ranking model to use (default: xgboost_tuned)
- `LOG_LEVEL`: Logging verbosity (default: INFO)

### Resource Limits

Limit CPU and memory usage:

```bash
docker run -d \
  -p 8000:8000 \
  --cpus=2 \
  --memory=4g \
  recommendation-service:latest
```

**Why limit resources?**
- Prevents one container from consuming all host resources
- Important in production with multiple services
- Helps identify performance bottlenecks

### Health Checks

View container health status:

```bash
# Check health status
docker ps

# View health check logs
docker inspect --format='{{json .State.Health}}' rec-service | jq
```

The Dockerfile includes a built-in health check:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s CMD curl --fail http://localhost:8000/health
```

### Volume Mounts (Development)

Mount code for live development:

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/serving:/app/serving \
  recommendation-service:latest \
  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

**Use cases:**
- Development: Edit code without rebuilding image
- Custom models: Mount different model directories
- Logs: Mount logs directory to host

## Production Deployment

### Multi-Worker Configuration

For higher throughput, run multiple Uvicorn workers:

```bash
docker run -d \
  -p 8000:8000 \
  recommendation-service:latest \
  uvicorn serving.api:app --host 0.0.0.0 --port 8000 --workers 4
```

**Worker recommendations:**
- CPU-bound workload: workers = (2 x CPU cores) + 1
- I/O-bound workload: workers = (4 x CPU cores)
- Start conservative (2-4 workers) and measure

### Gunicorn + Uvicorn

For production, use Gunicorn as process manager:

```bash
# In Dockerfile, change CMD to:
CMD ["gunicorn", "serving.api:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

**Benefits:**
- Graceful restarts (zero downtime deploys)
- Worker management (auto-restart on crash)
- Better signal handling

### Container Orchestration

For production at scale, use orchestration:

**Docker Compose** (single machine, multiple services):
```yaml
# docker-compose.yml
version: '3.8'
services:
  recommendation-service:
    image: recommendation-service:latest
    ports:
      - "8000:8000"
    environment:
      - CANDIDATE_POOL_SIZE=200
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
```

Run with: `docker-compose up -d`

**Kubernetes** (multi-machine, production):
- See Phase 4 of the infrastructure plan for K8s deployment
- Includes: Deployments, Services, Ingress, HPA, ConfigMaps

## Testing the Deployed Service

### Quick Test

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}'
```

### Load Testing

Use the test client:

```bash
# From host machine
python serving/test_client.py
```

Or use a load testing tool:

```bash
# Install hey (HTTP load testing)
# Mac: brew install hey
# Linux: go install github.com/rakyll/hey@latest

# Test with 100 requests, 10 concurrent
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}' \
  http://localhost:8000/recommend
```

## Troubleshooting

### Container won't start

**Check logs:**
```bash
docker logs rec-service
```

**Common issues:**
- Port 8000 already in use: Use different port `-p 8001:8000`
- Model files missing: Verify `COPY` commands in Dockerfile
- XGBoost fails: Ensure libomp is installed in runtime stage

### Container running but no response

**Check if port is exposed:**
```bash
docker ps  # Look for port mapping
```

**Check from inside container:**
```bash
docker exec -it rec-service curl http://localhost:8000/health
```

**Check firewall:**
```bash
# Mac/Linux
sudo lsof -i :8000
```

### High memory usage

**Check container stats:**
```bash
docker stats rec-service
```

**Reduce memory:**
- Decrease candidate_pool_size
- Use fewer workers
- Profile with memory_profiler

### Slow startup

Models are loaded at startup (takes 2-3 seconds). This is normal.

**To verify:**
```bash
# Watch logs
docker logs -f rec-service

# Should see: "RecommendationService initialized in 0.31s"
```

## Image Size Optimization

Current image size: ~300MB

**Further optimizations:**

1. **Use alpine base** (smallest Python image):
   ```dockerfile
   FROM python:3.9-alpine
   ```
   Warning: alpine needs additional build tools and may have compatibility issues

2. **Use distroless** (Google's minimal images):
   ```dockerfile
   FROM gcr.io/distroless/python3
   ```
   More secure but harder to debug (no shell)

3. **Remove unnecessary artifacts:**
   - Delete test files: `rm -rf */test/`
   - Remove notebooks: `rm -rf notebooks/`
   - Compress model files: Use HDF5 instead of JSON

## Security Best Practices

✅ **What we're doing:**
- Running as non-root user
- Multi-stage build (no build tools in final image)
- Minimal base image (slim, not full)
- Health checks for failure detection

⚠️ **Production additions:**
- Scan images for vulnerabilities: `docker scan recommendation-service:latest`
- Use private registry (AWS ECR, Google GCR, Docker Hub private)
- Sign images with Docker Content Trust
- Implement secrets management (not hardcoded in image)
- Regular base image updates (security patches)

## Next Steps

After Docker works locally:

1. **Push to registry:**
   ```bash
   docker tag recommendation-service:latest your-registry/recommendation-service:v1.0.0
   docker push your-registry/recommendation-service:v1.0.0
   ```

2. **Deploy to cloud:**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - Kubernetes cluster

3. **Add monitoring:**
   - Prometheus metrics endpoint
   - Grafana dashboards
   - CloudWatch/Stackdriver logs

4. **CI/CD pipeline:**
   - Auto-build on git push
   - Auto-deploy to staging
   - Manual approval for production

See the main plan document for Phase 3 (Monitoring) and Phase 4 (Deployment) details.

# Recommendation Service API

Production-ready REST API for serving movie recommendations.

## Overview

This service provides a unified HTTP API for the recommendation system, orchestrating:
1. **Candidate Generation**: Two-tower model + FAISS ANN retrieval
2. **Ranking**: XGBoost model with feature engineering (when trained)

## Architecture

```
HTTP Request
    ↓
FastAPI Server
    ↓
RecommendationService (orchestrator)
    ↓
┌─────────────────────┬───────────────────┐
│ CandidateGeneration │  RankingService   │
│  (Two-Tower + FAISS)│    (XGBoost)      │
└─────────────────────┴───────────────────┘
    ↓
HTTP Response
```

## Quick Start

### Option 1: Local Python (Development)

Using your local Python environment:

```bash
# From project root
source venv/bin/activate

# Set library path for XGBoost
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

# Start server
PYTHONPATH=$(pwd) uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

### Option 2: Docker (Production)

Using Docker container (recommended for deployment):

```bash
# Build image
docker build -t recommendation-service:latest .

# Run container
docker run -d -p 8000:8000 --name rec-service recommendation-service:latest

# View logs
docker logs -f rec-service
```

See [DOCKER.md](DOCKER.md) for complete Docker documentation.

## API Endpoints

### GET /
Simple root endpoint to verify service is running.

```bash
curl http://localhost:8000/
```

### POST /recommend
Get personalized recommendations for a user.

**Request:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}'
```

**Response:**
```json
{
  "user_id": 635,
  "recommendations": [
    {
      "movie_id": 3183,
      "similarity_score": 0.470,
      "rank": 1
    },
    ...
  ],
  "count": 10,
  "latency_ms": 1.3
}
```

**Parameters:**
- `user_id` (int, required): User ID to generate recommendations for
- `k` (int, optional): Number of recommendations to return (default: 10, max: 100)

### GET /health
Health check endpoint for load balancers.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "checks": {
    "candidate_service": "healthy",
    "ranking_service": "healthy"
  }
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service is unhealthy (not ready for traffic)

### GET /ready
Readiness check endpoint.

```bash
curl http://localhost:8000/ready
```

Used by orchestration systems (Kubernetes) to determine if service is ready to accept traffic.

### GET /info
Service information and configuration.

```bash
curl http://localhost:8000/info
```

Returns details about loaded models, number of users/items, and configuration.

### GET /docs
Interactive API documentation (Swagger UI).

Open in browser: http://localhost:8000/docs

### GET /redoc
Alternative API documentation (ReDoc).

Open in browser: http://localhost:8000/redoc

## Performance

**Latency (demo mode - candidate generation only):**
- k=10: ~1-5ms
- k=50: ~5-15ms
- k=100: ~10-30ms

**Latency (full mode - candidate + ranking):**
- k=10: ~10-20ms
- k=50: ~20-40ms
- k=100: ~30-60ms

**Initialization time:** 2-3 seconds (loads models once at startup)

## Error Handling

The API returns appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request format
- `422 Unprocessable Entity`: Valid format but failed validation (e.g., k=0)
- `404 Not Found`: Resource doesn't exist
- `500 Internal Server Error`: Server-side error
- `503 Service Unavailable`: Service not ready

**Example validation error:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 640, "k": 0}'
```

Response (422):
```json
{
  "detail": [
    {
      "type": "greater_than",
      "loc": ["body", "k"],
      "msg": "Input should be greater than 0",
      "input": 0
    }
  ]
}
```

## Testing

### Manual Testing

Use the provided test client:

```bash
# Install requests if needed
pip install requests

# Run test client
python serving/test_client.py
```

### cURL Examples

```bash
# Valid user
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}'

# Cold-start user (returns empty list)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 999999, "k": 10}'

# Large k
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 640, "k": 50}'

# Invalid k (validation error)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 640, "k": 0}'
```

## Configuration

### Environment Variables

Currently, configuration is hardcoded in the lifespan function. For production, consider:

```python
# In api.py lifespan function
candidate_pool_size = int(os.getenv("CANDIDATE_POOL_SIZE", "100"))
model_name = os.getenv("RANKING_MODEL_NAME", "xgboost_tuned")
```

### Model Configuration

- **Candidate Pool Size**: Number of candidates to retrieve before ranking (default: 100)
  - Trade-off: Larger = better quality, higher latency
  - Recommendation: 5-10x your target k

- **Ranking Model**: Which ranking model to use (default: "xgboost_tuned")
  - Useful for A/B testing and canary deployments

## Files

- `recommendation_service.py`: Core orchestrator service
- `api.py`: FastAPI wrapper (full mode with ranking)
- `demo_api.py`: Demo FastAPI wrapper (candidate generation only)
- `run_server.py`: CLI script to start server
- `test_client.py`: Test client for manual testing
- `README.md`: This file

## Next Steps

1. **Containerization**: Add Dockerfile and Docker Compose
2. **Testing**: Add pytest unit and integration tests
3. **Monitoring**: Add Prometheus metrics collection
4. **Deployment**: Add Kubernetes manifests or deployment scripts
5. **Features**:
   - Request/response logging for audit trail
   - Rate limiting
   - Authentication (API keys, JWT)
   - Caching layer (Redis)
   - Model versioning and A/B testing

## Troubleshooting

**Service won't start:**
- Check that models are trained and artifacts exist
- Verify PYTHONPATH includes project root
- Check logs for detailed error messages

**No recommendations returned:**
- Verify user_id exists in system (check `candidate_gen/artifacts/data/user_to_idx.json`)
- Cold-start users return empty list (expected behavior)

**Slow latency:**
- Check candidate_pool_size (larger = slower)
- Profile with `/info` endpoint to see model details
- Consider reducing k for faster responses

## Production Considerations

**Before deploying to production:**

1. ✅ Train ranking model (currently using demo mode)
2. ⚠️ Add authentication/authorization
3. ⚠️ Add rate limiting
4. ⚠️ Configure CORS properly (restrict origins)
5. ⚠️ Add request logging and monitoring
6. ⚠️ Set up health check monitoring
7. ⚠️ Use production ASGI server (Gunicorn + Uvicorn workers)
8. ⚠️ Add load balancer in front
9. ⚠️ Set up auto-scaling based on traffic
10. ⚠️ Implement circuit breakers for dependencies

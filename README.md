# Recommendation System

An end-to-end recommendation system implementing industry-standard patterns: two-stage retrieval (candidate generation + ranking), feature stores, and drift detection.

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│                  RecommendationService                       │
│  ┌───────────────────────┐    ┌──────────────────────────┐ │
│  │ Candidate Generation  │    │     Ranking Service      │ │
│  │  (Two-Tower + FAISS)  │───▶│  (XGBoost + Features)    │ │
│  │                       │    │           │              │ │
│  │  • User embeddings    │    │    FeatureStore          │ │
│  │  • Item embeddings    │    │  (Redis + Fallback)      │ │
│  │  • ANN search         │    │                          │ │
│  └───────────────────────┘    └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
Top-K Ranked Recommendations
```

## Components

### Candidate Generation (`candidate_gen/`)
- **Two-Tower Model**: Neural network with separate user and item towers
- **FAISS Index**: Approximate nearest neighbor search for fast retrieval
- **Output**: ~100 candidates from 3,600+ items in <5ms

### Ranking (`ranking/`)
- **XGBoost Model**: Gradient-boosted trees for precise scoring
- **Feature Engineering**: User stats, movie stats, demographics, genres
- **Feature Store**: Redis-backed with circuit breaker fallback

### Serving (`serving/`)
- **FastAPI**: REST API with health checks and observability
- **Drift Detection**: Real-time feature monitoring with KL divergence, PSI metrics
- **Docker Compose**: Multi-container setup with Redis

## Quick Start

### Prerequisites
- Python 3.9+
- Docker (for Redis)

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd rec_system
pip install -r requirements.txt

# Start Redis
docker-compose up -d redis

# Migrate features to Redis (optional - can use parquet mode)
PYTHONPATH=$(pwd) python ranking/serving/scripts/migrate_to_redis.py

# Compute drift baselines (optional)
PYTHONPATH=$(pwd) python ranking/serving/scripts/compute_baseline.py
```

### Run the Service

```bash
# Parquet mode (no Redis needed)
PYTHONPATH=$(pwd) uvicorn serving.api:app --host 0.0.0.0 --port 8000

# Redis mode (requires Redis running)
FEATURE_STORE_MODE=redis PYTHONPATH=$(pwd) uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 635, "k": 10}'

# Health check
curl http://localhost:8000/health

# Drift status
curl http://localhost:8000/drift/status
```

## Project Structure

```
rec_system/
├── candidate_gen/           # Candidate generation module
│   ├── artifacts/           # Trained models, embeddings, FAISS index
│   ├── retrieval/           # Two-tower model, index builder
│   ├── serving/             # CandidateGenerationService
│   └── shared_utils/        # Paths, model loading utilities
│
├── ranking/                 # Ranking module
│   ├── features/            # Pre-computed features, metadata
│   ├── models/              # Trained XGBoost models
│   ├── serving/             # RankerService, FeatureBuilder
│   │   ├── feature_store/   # Redis + Fallback + Circuit Breaker
│   │   ├── drift/           # Drift detection (baseline, online stats, metrics)
│   │   └── scripts/         # Migration, baseline computation
│   └── shared_utils/        # Feature loading utilities
│
├── serving/                 # Unified serving layer
│   ├── api.py               # FastAPI application
│   └── recommendation_service.py
│
├── data/                    # Raw data (MovieLens)
├── notebooks/               # Exploration notebooks
├── docker-compose.yml       # Redis + service orchestration
├── Dockerfile               # Service container
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/recommend` | POST | Get recommendations for a user |
| `/health` | GET | Service health check |
| `/ready` | GET | Readiness probe |
| `/info` | GET | Service configuration |
| `/drift/status` | GET | Drift detection status |
| `/drift/check` | POST | Trigger drift analysis |
| `/drift/report` | GET | Latest drift report |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FEATURE_STORE_MODE` | `parquet` | `redis` or `parquet` |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `LOG_LEVEL` | `INFO` | Logging level |

## ML Infrastructure Concepts

This project demonstrates several production ML patterns:

1. **Two-Stage Retrieval**: Candidate generation (fast, approximate) + Ranking (slow, precise)
2. **Feature Store**: Centralized feature serving with Redis
3. **Circuit Breaker**: Automatic fallback when Redis fails
4. **Drift Detection**: Monitoring feature distributions with Welford's algorithm
5. **Dependency Injection**: Pluggable backends for testing and flexibility

## Data

Uses the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/):
- 6,040 users
- 3,706 movies
- 1,000,209 ratings

## Performance

- **Initialization**: 2-3 seconds (loads models, embeddings, index)
- **Per-request latency**: 10-20ms for k=10
  - Candidate retrieval: 1-5ms
  - Ranking: 5-15ms
  - Feature lookup (Redis): <1ms

## Development

See [CLAUDE.md](CLAUDE.md) for development guidelines and how to work with this codebase.

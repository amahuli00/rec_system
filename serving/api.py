"""
FastAPI REST API for the recommendation service.

This module provides HTTP endpoints for serving recommendations. It wraps the
RecommendationService class and exposes it via a REST API with proper request
validation, error handling, and observability.

Endpoints:
    POST /recommend - Get recommendations for a user
    GET /health - Health check endpoint
    GET /ready - Readiness probe endpoint
    GET /info - Service information and configuration

Architecture:
    HTTP Request → FastAPI → Pydantic Validation → RecommendationService → HTTP Response

Design Decisions:
- Pydantic models for request/response validation (type safety + automatic docs)
- Proper HTTP status codes (200, 400, 404, 500, 503)
- Structured error responses with details
- Health vs Readiness: Different semantics for load balancers
- CORS enabled for browser-based clients (optional, can be disabled)

Performance:
- Service initialized once at startup (stateful pattern)
- Request latency = service latency + ~1ms HTTP overhead
- No middleware overhead beyond FastAPI defaults
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from serving.recommendation_service import RecommendationService, RecommendedItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances (initialized at startup)
recommendation_service: Optional[RecommendationService] = None
drift_detector: Optional[Any] = None  # Will be DriftDetector when initialized


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class RecommendationRequest(BaseModel):
    """
    Request schema for getting recommendations.

    Attributes:
        user_id: The user to generate recommendations for
        k: Number of recommendations to return (default: 10, max: 100)
    """
    user_id: int = Field(..., description="User ID to generate recommendations for", gt=0)
    k: int = Field(10, description="Number of recommendations to return", gt=0, le=100)

    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "k": 10
            }
        }


class RecommendationItemResponse(BaseModel):
    """
    Single recommendation item in the response.

    Attributes:
        movie_id: The recommended movie ID
        score: Ranking score (higher is better)
        rank: Position in the ranking (1-indexed)
        candidate_similarity: Similarity score from candidate generation
    """
    movie_id: int
    score: float
    rank: int
    candidate_similarity: float

    class Config:
        schema_extra = {
            "example": {
                "movie_id": 527,
                "score": 4.234,
                "rank": 1,
                "candidate_similarity": 0.856
            }
        }


class RecommendationResponse(BaseModel):
    """
    Response schema for recommendation requests.

    Attributes:
        user_id: The user ID from the request
        recommendations: List of recommended items
        count: Number of recommendations returned
        latency_ms: Request processing time in milliseconds
    """
    user_id: int
    recommendations: List[RecommendationItemResponse]
    count: int
    latency_ms: float

    class Config:
        schema_extra = {
            "example": {
                "user_id": 123,
                "recommendations": [
                    {"movie_id": 527, "score": 4.234, "rank": 1, "candidate_similarity": 0.856},
                    {"movie_id": 1891, "score": 4.156, "rank": 2, "candidate_similarity": 0.823}
                ],
                "count": 2,
                "latency_ms": 15.3
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    checks: dict

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "checks": {
                    "candidate_service": "healthy",
                    "ranking_service": "healthy"
                }
            }
        }


class ServiceInfoResponse(BaseModel):
    """Service information response schema."""
    service_name: str
    version: str
    candidate_generation: dict
    ranking: dict
    config: dict

    class Config:
        schema_extra = {
            "example": {
                "service_name": "recommendation-service",
                "version": "1.0.0",
                "candidate_generation": {
                    "num_users": 943,
                    "num_items": 1682
                },
                "ranking": {
                    "model_name": "xgboost_tuned"
                },
                "config": {
                    "candidate_pool_size": 100
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    status_code: int

    class Config:
        schema_extra = {
            "example": {
                "error": "User not found",
                "detail": "User ID 999999 does not exist in the system",
                "status_code": 404
            }
        }


# ============================================================================
# Lifespan Event Handler (Startup/Shutdown)
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.

    This replaces the deprecated @app.on_event("startup") pattern.
    """
    # Startup
    global recommendation_service, drift_detector
    logger.info("Starting recommendation service...")

    try:
        recommendation_service = RecommendationService(
            candidate_pool_size=100,
            model_name="xgboost_tuned"
        )
        logger.info("Recommendation service started successfully")
    except Exception as e:
        logger.error(f"Failed to start recommendation service: {e}")
        raise

    # Initialize drift detector if baseline exists
    try:
        drift_detector = _initialize_drift_detector()
        if drift_detector:
            logger.info("Drift detector initialized")
        else:
            logger.info("Drift detector not initialized (no baseline found)")
    except Exception as e:
        logger.warning(f"Failed to initialize drift detector: {e}")
        drift_detector = None

    yield

    # Shutdown
    logger.info("Shutting down recommendation service...")
    recommendation_service = None
    drift_detector = None


def _initialize_drift_detector():
    """
    Initialize drift detector if baseline statistics exist.

    Returns:
        DriftDetector instance or None if baseline not found
    """
    from ranking.serving.drift import BaselineStatistics, DriftDetector
    from ranking.shared_utils import FEATURES_DIR

    baseline_path = FEATURES_DIR / "baseline_stats.json"

    if not baseline_path.exists():
        logger.info(f"No baseline statistics found at {baseline_path}")
        return None

    try:
        baseline = BaselineStatistics.load(baseline_path)
        detector = DriftDetector(baseline)
        return detector
    except Exception as e:
        logger.warning(f"Failed to load baseline: {e}")
        return None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Recommendation Service API",
    description="Production-ready recommendation service with candidate generation and ranking",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc alternative documentation
)

# Add CORS middleware (optional - enable if you need browser access)
# In production, you'd restrict origins to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Middleware for Request Logging
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all requests with timing information.

    This is useful for debugging and monitoring. In production, you'd also
    log this to a structured logging system (e.g., JSON logs to stdout for
    collection by Fluentd/Logstash).
    """
    start_time = time.time()

    # Log request
    logger.info(f"→ {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response with timing
    duration = (time.time() - start_time) * 1000
    logger.info(
        f"← {request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.1f}ms"
    )

    return response


# ============================================================================
# API Endpoints
# ============================================================================

@app.post(
    "/recommend",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get recommendations for a user",
    description="Returns top-K ranked movie recommendations for a given user",
    responses={
        200: {"description": "Successful response with recommendations"},
        400: {"description": "Invalid request parameters", "model": ErrorResponse},
        404: {"description": "User not found", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        503: {"description": "Service unavailable", "model": ErrorResponse}
    }
)
async def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    """
    Get top-K recommendations for a user.

    This endpoint:
    1. Validates the request (Pydantic handles this automatically)
    2. Calls the recommendation service
    3. Returns ranked recommendations with scores

    Args:
        request: RecommendationRequest with user_id and k

    Returns:
        RecommendationResponse with recommendations and metadata

    Raises:
        HTTPException: 404 if user not found, 503 if service unavailable
    """
    if recommendation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not initialized"
        )

    start_time = time.time()

    try:
        # Get recommendations
        recommendations = recommendation_service.get_recommendations(
            user_id=request.user_id,
            k=request.k
        )

        # Handle cold-start user (empty recommendations)
        if not recommendations:
            logger.warning(f"No recommendations found for user {request.user_id}")
            # Return 200 with empty list (not 404 - user might exist but be cold-start)
            return RecommendationResponse(
                user_id=request.user_id,
                recommendations=[],
                count=0,
                latency_ms=(time.time() - start_time) * 1000
            )

        # Convert to response format
        recommendation_items = [
            RecommendationItemResponse(
                movie_id=rec.movie_id,
                score=rec.score,
                rank=rec.rank,
                candidate_similarity=rec.candidate_similarity
            )
            for rec in recommendations
        ]

        latency_ms = (time.time() - start_time) * 1000

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendation_items,
            count=len(recommendation_items),
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns the health status of the service and its dependencies",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy", "model": HealthResponse}
    }
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    This is used by load balancers and orchestration systems (Kubernetes, ECS)
    to determine if the service is healthy. If unhealthy, traffic is not routed
    to this instance.

    Returns:
        HealthResponse with status and detailed checks

    Status Codes:
        200: Service is healthy
        503: Service is unhealthy (one or more checks failed)
    """
    if recommendation_service is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "checks": {
                    "recommendation_service": "not initialized"
                }
            }
        )

    health = recommendation_service.health_check()

    # Return 503 if unhealthy (important for load balancers)
    if health["status"] != "healthy":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health
        )

    return HealthResponse(**health)


@app.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Returns whether the service is ready to accept traffic",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"}
    }
)
async def readiness_check():
    """
    Readiness check endpoint.

    This is different from /health:
    - /health: Is the service running properly? (used for restarts)
    - /ready: Is the service ready to accept traffic? (used for startup/rollout)

    During startup, /health might be OK but /ready returns 503 until models
    are loaded. This prevents routing traffic before the service is ready.

    Returns:
        200 if ready, 503 if not ready
    """
    if recommendation_service is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not ready", "reason": "service not initialized"}
        )

    # Additional readiness checks can be added here
    # For example: check if model files exist, database is reachable, etc.

    return {"status": "ready"}


@app.get(
    "/info",
    response_model=ServiceInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Service information",
    description="Returns information about the service configuration and loaded models"
)
async def service_info() -> ServiceInfoResponse:
    """
    Get service information and configuration.

    Useful for debugging, monitoring, and understanding what's deployed.

    Returns:
        ServiceInfoResponse with service metadata
    """
    if recommendation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not initialized"
        )

    info = recommendation_service.get_service_info()

    return ServiceInfoResponse(
        service_name="recommendation-service",
        version="1.0.0",
        candidate_generation=info["candidate_generation"],
        ranking=info["ranking"],
        config=info["config"]
    )


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Root endpoint",
    description="Simple endpoint to verify the service is running"
)
async def root():
    """
    Root endpoint - simple health indicator.

    Returns:
        Simple message indicating service is running
    """
    return {
        "message": "Recommendation Service API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# Drift Detection Endpoints
# ============================================================================

@app.get(
    "/drift/status",
    status_code=status.HTTP_200_OK,
    summary="Get drift detection status",
    description="Returns the current status of drift detection"
)
async def drift_status():
    """
    Get current drift detection status.

    Returns:
        Drift detector status including monitored features and last report
    """
    if drift_detector is None:
        return {
            "enabled": False,
            "message": "Drift detection not initialized (no baseline found)"
        }

    return {
        "enabled": True,
        **drift_detector.get_status()
    }


@app.post(
    "/drift/check",
    status_code=status.HTTP_200_OK,
    summary="Run drift check",
    description="Manually trigger a drift check and return the report"
)
async def drift_check(reset_after: bool = False):
    """
    Manually trigger a drift check.

    Args:
        reset_after: Reset statistics after check (for time-windowed checks)

    Returns:
        Drift report with per-feature results
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detection not initialized"
        )

    report = drift_detector.check_drift(reset_after=reset_after)
    return report.to_dict()


@app.get(
    "/drift/report",
    status_code=status.HTTP_200_OK,
    summary="Get last drift report",
    description="Returns the most recent drift detection report"
)
async def drift_report():
    """
    Get the most recent drift report.

    Returns:
        Last drift report or error if no report available
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detection not initialized"
        )

    report = drift_detector.get_last_report()
    if report is None:
        return {"message": "No drift report available. Run /drift/check first."}

    return report.to_dict()


@app.get(
    "/drift/history",
    status_code=status.HTTP_200_OK,
    summary="Get drift report history",
    description="Returns recent drift detection reports"
)
async def drift_history(limit: int = 10):
    """
    Get recent drift reports.

    Args:
        limit: Maximum number of reports to return

    Returns:
        List of recent drift reports
    """
    if drift_detector is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift detection not initialized"
        )

    history = drift_detector.get_report_history(limit=limit)
    return {
        "count": len(history),
        "reports": [r.to_dict() for r in history]
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.

    This catches any exception that isn't explicitly handled and returns
    a structured error response. In production, you'd also log these to
    an error tracking service (Sentry, Rollbar, etc.).
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# ============================================================================
# Run Server (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run with: python -m serving.api
    # or: uvicorn serving.api:app --reload
    uvicorn.run(
        "serving.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level="info"
    )

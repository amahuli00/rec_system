"""
Unified recommendation service that orchestrates candidate generation and ranking.

This module provides the RecommendationService class which serves as the main entry point
for getting recommendations. It coordinates between the candidate generation service
(two-tower model + FAISS) and the ranking service (XGBoost) to produce final ranked
recommendations.

Architecture:
    User Request
        ↓
    RecommendationService.get_recommendations(user_id, k)
        ↓
    CandidateGenerationService.retrieve(user_id, candidate_pool_size)
        ↓ (returns ~100-500 candidates)
    RankerService.rank(user_id, candidate_ids)
        ↓                ↓
        │        FeatureStore (Redis + Fallback)
        ↓ (re-ranks candidates)
    Return top K ranked items

Design Decisions:
- Stateful initialization: Both services are loaded once at startup to minimize latency
- Configurable candidate pool: Retrieve more candidates than needed, then rank to top-K
- Graceful degradation: Handle cold-start users and missing data elegantly
- Error isolation: Service initialization failures are surfaced clearly
- Feature store integration: Optional Redis feature store with circuit breaker

Performance:
- Initialization: 2-3 seconds (loads models, embeddings, FAISS index)
- Per-request latency: 10-20ms for k=10, 15-30ms for k=50
  - Candidate retrieval: 1-5ms
  - Ranking: 5-15ms (depends on candidate pool size)
  - Feature lookup (Redis): <1ms with connection pooling
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from candidate_gen.serving.candidate_service import CandidateGenerationService
from ranking.serving.ranker_service import RankerService

if TYPE_CHECKING:
    from ranking.serving.feature_store import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RecommendedItem:
    """
    A single recommended item with all relevant metadata.

    Attributes:
        movie_id: The unique identifier for the movie
        score: The final ranking score (higher is better)
        rank: The position in the final ranking (1-indexed)
        candidate_similarity: The similarity score from candidate generation stage
    """
    movie_id: int
    score: float
    rank: int
    candidate_similarity: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'movie_id': self.movie_id,
            'score': float(self.score),
            'rank': self.rank,
            'candidate_similarity': float(self.candidate_similarity)
        }


class RecommendationService:
    """
    Unified recommendation service orchestrating candidate generation and ranking.

    This is the main entry point for serving recommendations. It wraps both the
    candidate generation service (two-tower model) and ranking service (XGBoost)
    into a single, easy-to-use interface.

    Usage:
        # Initialize once (at service startup)
        service = RecommendationService()

        # Get recommendations for a user
        recommendations = service.get_recommendations(user_id=123, k=10)

        # Each recommendation contains:
        for item in recommendations:
            print(f"Rank {item.rank}: Movie {item.movie_id} (score: {item.score:.4f})")

    Configuration:
        candidate_pool_size: How many candidates to retrieve before ranking
            - Default: 100
            - Trade-off: Larger pool = better ranking quality but higher latency
            - Recommendation: 5-10x your target k value

        model_name: Which ranking model to use (for A/B testing, canary deployments)
            - Default: "xgboost_tuned"
            - Can be changed to support multiple model versions
    """

    def __init__(
        self,
        candidate_pool_size: int = 100,
        model_name: str = "xgboost_tuned",
        feature_store: Optional["FeatureStore"] = None,
    ):
        """
        Initialize the recommendation service.

        This performs one-time setup:
        1. Initialize feature store (Redis + fallback) if configured
        2. Load candidate generation service (two-tower model + FAISS index)
        3. Load ranking service (XGBoost model + feature builder)
        4. Validate all services initialized successfully

        Args:
            candidate_pool_size: Number of candidates to retrieve before ranking.
                                 Should be larger than the k you'll request.
            model_name: Name of the ranking model to use.
            feature_store: Optional FeatureStore for feature retrieval.
                          If None and FEATURE_STORE_MODE=redis, creates one automatically.

        Raises:
            RuntimeError: If any service fails to initialize
        """
        logger.info("Initializing RecommendationService...")
        start_time = time.time()

        self.candidate_pool_size = candidate_pool_size
        self.model_name = model_name

        # Initialize feature store (if configured)
        self.feature_store = feature_store
        if self.feature_store is None:
            self.feature_store = self._create_feature_store_from_env()

        # Initialize candidate generation service
        try:
            logger.info("Loading candidate generation service...")
            self.candidate_service = CandidateGenerationService()
            logger.info(
                f"Candidate service loaded: {self.candidate_service.num_users} users, "
                f"{self.candidate_service.num_items} items"
            )
        except Exception as e:
            logger.error(f"Failed to initialize candidate generation service: {e}")
            raise RuntimeError(f"Candidate service initialization failed: {e}")

        # Initialize ranking service (with feature store if available)
        try:
            logger.info(f"Loading ranking service (model: {model_name})...")
            self.ranker_service = RankerService(
                model_name=model_name,
                feature_store=self.feature_store
            )
            mode = "feature store" if self.feature_store else "parquet"
            logger.info(f"Ranking service loaded successfully (mode: {mode})")
        except Exception as e:
            logger.error(f"Failed to initialize ranking service: {e}")
            raise RuntimeError(f"Ranking service initialization failed: {e}")

        init_time = time.time() - start_time
        logger.info(f"RecommendationService initialized in {init_time:.2f}s")

    def _create_feature_store_from_env(self) -> Optional["FeatureStore"]:
        """
        Create a feature store based on environment variables.

        Environment Variables:
            FEATURE_STORE_MODE: "redis" or "parquet" (default: parquet)
            REDIS_HOST: Redis host (default: localhost)
            REDIS_PORT: Redis port (default: 6379)

        Returns:
            FeatureStore if mode is "redis", None otherwise
        """
        mode = os.getenv("FEATURE_STORE_MODE", "parquet").lower()

        if mode != "redis":
            logger.info(f"Feature store mode: {mode} (using parquet)")
            return None

        # Import here to avoid circular dependencies
        from ranking.serving.feature_store import (
            RedisFeatureStore,
            FallbackFeatureStore,
            LayeredFeatureStore,
        )

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))

        logger.info(f"Initializing Redis feature store at {redis_host}:{redis_port}")

        try:
            primary = RedisFeatureStore(host=redis_host, port=redis_port)
            fallback = FallbackFeatureStore()
            feature_store = LayeredFeatureStore(primary=primary, fallback=fallback)
            logger.info("Feature store initialized (Redis + fallback)")
            return feature_store
        except Exception as e:
            logger.warning(f"Failed to initialize Redis feature store: {e}")
            logger.warning("Falling back to parquet mode")
            return None

    def get_recommendations(
        self,
        user_id: int,
        k: int = 10,
        return_scores: bool = True
    ) -> List[RecommendedItem]:
        """
        Get top-K ranked recommendations for a user.

        This is the main API method. It:
        1. Retrieves candidates using the two-tower model + FAISS
        2. Re-ranks candidates using the XGBoost ranking model
        3. Returns the top-K highest-scoring items

        Args:
            user_id: The user to generate recommendations for
            k: Number of recommendations to return
            return_scores: Whether to include similarity scores in the response

        Returns:
            List of RecommendedItem objects, sorted by score (highest first).
            Empty list if user is unknown or no candidates available.

        Example:
            >>> service = RecommendationService()
            >>> recs = service.get_recommendations(user_id=123, k=5)
            >>> for item in recs:
            ...     print(f"{item.rank}. Movie {item.movie_id}: {item.score:.3f}")
            1. Movie 527: 4.234
            2. Movie 1891: 4.156
            3. Movie 318: 4.089
            ...
        """
        # Validate inputs
        if k <= 0:
            logger.warning(f"Invalid k={k}, must be > 0")
            return []

        if k > self.candidate_pool_size:
            logger.warning(
                f"Requested k={k} exceeds candidate_pool_size={self.candidate_pool_size}. "
                f"Consider increasing candidate_pool_size for better quality."
            )

        start_time = time.time()

        # Stage 1: Candidate Generation
        logger.debug(f"Retrieving {self.candidate_pool_size} candidates for user {user_id}")
        candidate_start = time.time()

        candidates = self.candidate_service.retrieve(
            user_id=user_id,
            k=self.candidate_pool_size
        )

        candidate_time = time.time() - candidate_start

        if not candidates:
            logger.info(f"No candidates found for user {user_id} (likely cold-start user)")
            return []

        logger.debug(
            f"Retrieved {len(candidates)} candidates in {candidate_time*1000:.1f}ms"
        )

        # Stage 2: Ranking
        ranking_start = time.time()
        candidate_ids = [c.movie_id for c in candidates]

        ranked_items = self.ranker_service.rank(
            user_id=user_id,
            candidate_ids=candidate_ids
        )

        ranking_time = time.time() - ranking_start
        logger.debug(f"Ranked {len(ranked_items)} items in {ranking_time*1000:.1f}ms")

        # Stage 3: Merge scores and return top-K
        # Create a lookup for candidate similarities
        candidate_sim_map = {c.movie_id: c.similarity_score for c in candidates}

        # Build final recommendations with both scores
        recommendations = []
        for ranked_item in ranked_items[:k]:
            rec = RecommendedItem(
                movie_id=ranked_item.movie_id,
                score=ranked_item.predicted_score,
                rank=ranked_item.rank,
                candidate_similarity=candidate_sim_map.get(ranked_item.movie_id, 0.0)
            )
            recommendations.append(rec)

        total_time = time.time() - start_time
        logger.info(
            f"Generated {len(recommendations)} recommendations for user {user_id} "
            f"in {total_time*1000:.1f}ms "
            f"(candidate: {candidate_time*1000:.1f}ms, ranking: {ranking_time*1000:.1f}ms)"
        )

        return recommendations

    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models and service configuration.

        Useful for debugging and monitoring.

        Returns:
            Dictionary with service metadata including:
            - Number of users and items
            - Model information
            - Configuration settings
        """
        return {
            'candidate_generation': {
                'num_users': self.candidate_service.num_users,
                'num_items': self.candidate_service.num_items,
                'model_info': self.candidate_service.model_info
            },
            'ranking': {
                'model_name': self.model_name,
            },
            'config': {
                'candidate_pool_size': self.candidate_pool_size,
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the service.

        Returns:
            Dictionary with health status:
            - status: "healthy" or "unhealthy"
            - checks: Status of individual components
        """
        checks = {}

        # Check candidate service
        try:
            assert self.candidate_service.num_users > 0
            assert self.candidate_service.num_items > 0
            checks['candidate_service'] = 'healthy'
        except Exception as e:
            checks['candidate_service'] = f'unhealthy: {e}'

        # Check ranking service
        try:
            assert self.ranker_service is not None
            checks['ranking_service'] = 'healthy'
        except Exception as e:
            checks['ranking_service'] = f'unhealthy: {e}'

        # Check feature store (if configured)
        if self.feature_store is not None:
            try:
                fs_health = self.feature_store.health_check()
                if fs_health.get('healthy', False):
                    checks['feature_store'] = 'healthy'
                else:
                    # Feature store degraded but service can still work with fallback
                    checks['feature_store'] = f"degraded: {fs_health.get('message', 'unknown')}"
            except Exception as e:
                checks['feature_store'] = f'degraded: {e}'
        else:
            checks['feature_store'] = 'not configured (using parquet)'

        # Overall status: healthy if candidate and ranking are healthy
        # Feature store can be degraded (fallback mode) and service is still healthy
        core_healthy = (
            checks.get('candidate_service') == 'healthy' and
            checks.get('ranking_service') == 'healthy'
        )

        return {
            'status': 'healthy' if core_healthy else 'unhealthy',
            'checks': checks
        }

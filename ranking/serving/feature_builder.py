"""
Feature builder for real-time serving.

This module builds features for a single user and multiple candidate movies
at inference time. It supports two modes:

1. **Feature Store Mode** (recommended for production):
   - Uses a FeatureStore (Redis + fallback) for feature retrieval
   - Enables fast lookup with graceful degradation
   - Supports real-time feature updates

2. **Parquet Mode** (legacy, for local development):
   - Loads features from parquet files into memory
   - Simple but doesn't scale or support updates

The feature store mode is preferred because it:
- Separates storage from computation
- Enables circuit breaker for reliability
- Provides observability (source tracking, latency)
"""

import logging
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from ranking.shared_utils import (
    FEATURES_DIR,
    DATA_DIR,
    load_feature_columns,
    load_feature_metadata,
)

# Conditional import to avoid circular dependency
if TYPE_CHECKING:
    from ranking.serving.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class ServingFeatureBuilder:
    """
    Builds features for real-time serving (single user, multiple candidates).

    This class supports two modes of operation:

    1. **Feature Store Mode** (production):
       Pass a FeatureStore to the constructor. Features are fetched from
       the store (Redis + fallback) with circuit breaker protection.

    2. **Parquet Mode** (development/legacy):
       Don't pass a FeatureStore. Features are loaded from parquet files
       into memory dictionaries. Simple but doesn't scale.

    All feature computation mirrors the batch FeatureBuilder logic to ensure
    consistency between training and serving.

    Example (Feature Store Mode):
        store = LayeredFeatureStore(RedisFeatureStore(), FallbackFeatureStore())
        builder = ServingFeatureBuilder(feature_store=store)
        features = builder.build_features(user_id=123, candidate_movie_ids=[1, 50])

    Example (Parquet Mode - legacy):
        builder = ServingFeatureBuilder()
        features = builder.build_features(user_id=123, candidate_movie_ids=[1, 50])
    """

    def __init__(self, feature_store: Optional["FeatureStore"] = None):
        """
        Initialize the serving feature builder.

        Args:
            feature_store: Optional FeatureStore for feature retrieval.
                          If None, falls back to loading parquet files.

        Loads:
        - Feature column order (always required)
        - Feature metadata with cold-start defaults
        - If no feature_store: parquet files into memory dicts
        """
        logger.info("Initializing ServingFeatureBuilder...")

        self.feature_store = feature_store
        self.use_feature_store = feature_store is not None

        # Load feature metadata (always needed for column order)
        self.metadata = load_feature_metadata()
        self.cold_start_defaults = self.metadata.get("cold_start_defaults", {})

        # Load feature column order (critical for inference)
        self.feature_columns = load_feature_columns()

        if self.use_feature_store:
            logger.info("Using FeatureStore mode")
            # Still need genre columns for building features
            self.genre_cols = self.metadata.get("feature_groups", {}).get("genre", [])
            # Initialize empty dicts (not used in feature store mode)
            self.user_stats_dict = {}
            self.movie_stats_dict = {}
            self.users_dict = {}
            self.movies_dict = {}
        else:
            logger.info("Using Parquet mode (legacy)")
            self._load_parquet_data()

        logger.info("ServingFeatureBuilder initialized")

    def _load_parquet_data(self) -> None:
        """Load feature data from parquet files (legacy mode)."""
        # Load pre-computed stats
        self.user_stats = pd.read_parquet(FEATURES_DIR / "user_stats.parquet")
        self.movie_stats = pd.read_parquet(FEATURES_DIR / "movie_stats.parquet")

        # Create lookup dictionaries for fast access
        self.user_stats_dict = self.user_stats.set_index("user_id").to_dict("index")
        self.movie_stats_dict = self.movie_stats.set_index("movie_id").to_dict("index")

        # Load user demographics
        self.users = pd.read_parquet(DATA_DIR / "users.parquet")
        # Encode gender: M=1, F=0
        self.users["gender"] = (self.users["gender"] == "M").astype(int)
        self.users = self.users.rename(columns={"age": "age_group"})
        self.users_dict = self.users.set_index("user_id").to_dict("index")

        # Load and process movies with genres
        self.movies = pd.read_parquet(DATA_DIR / "movies.parquet")
        self._prepare_genre_features()
        self.movies_dict = self.movies_with_genres.set_index("movie_id").to_dict("index")

        logger.info(
            f"Loaded parquet data: "
            f"{len(self.user_stats_dict)} users, "
            f"{len(self.movie_stats_dict)} movies"
        )

    def _prepare_genre_features(self) -> None:
        """Create multi-hot encoding of movie genres (mirrors batch FeatureBuilder)."""
        # Extract all unique genres
        all_genres = set()
        for genres_str in self.movies["genres"]:
            all_genres.update(genres_str.split("|"))
        all_genres = sorted(all_genres)

        # Create binary columns for each genre
        self.movies_with_genres = self.movies.copy()
        for genre in all_genres:
            genre_clean = genre.lower().replace("-", "_").replace("'", "")
            col_name = f"genre_{genre_clean}"
            self.movies_with_genres[col_name] = (
                self.movies_with_genres["genres"]
                .str.contains(genre, regex=False)
                .astype(int)
            )

        self.genre_cols = [
            col for col in self.movies_with_genres.columns if col.startswith("genre_")
        ]

    def build_features(
        self,
        user_id: int,
        candidate_movie_ids: List[int],
    ) -> pd.DataFrame:
        """
        Build feature matrix for one user and multiple candidate movies.

        Args:
            user_id: User to build features for
            candidate_movie_ids: List of movie IDs to score

        Returns:
            DataFrame with len(candidate_movie_ids) rows and 35 feature columns,
            ordered as required for model inference.
        """
        rows = []

        # Get user features (or cold-start defaults)
        user_features = self._get_user_features(user_id)

        for movie_id in candidate_movie_ids:
            # Get movie features (or cold-start defaults)
            movie_features = self._get_movie_features(movie_id)

            # Combine all features
            row = {
                **user_features,
                **movie_features,
            }

            # Compute interaction features
            row["user_movie_rating_diff"] = (
                row["user_avg_rating"] - row["movie_avg_rating"]
            )
            row["user_rating_count_norm"] = np.log1p(row["user_rating_count"])
            row["movie_rating_count_norm"] = np.log1p(row["movie_rating_count"])
            row["user_movie_activity_product"] = (
                row["user_rating_count_norm"] * row["movie_rating_count_norm"]
            )

            # Store movie_id for tracking (not used in features)
            row["movie_id"] = movie_id

            rows.append(row)

        # Create DataFrame
        features_df = pd.DataFrame(rows)

        # Return only the feature columns in correct order
        return features_df[self.feature_columns]

    def _get_user_features(self, user_id: int) -> Dict:
        """
        Get user features, using cold-start defaults if user not found.

        In feature store mode, fetches from the store (with circuit breaker).
        In parquet mode, uses in-memory dictionaries.
        """
        if self.use_feature_store:
            return self._get_user_features_from_store(user_id)
        else:
            return self._get_user_features_from_parquet(user_id)

    def _get_user_features_from_store(self, user_id: int) -> Dict:
        """Get user features from the feature store."""
        vec = self.feature_store.get_user_features(user_id)

        if vec is not None:
            # Feature store returned values (either from Redis or fallback)
            return vec.features.copy()
        else:
            # This shouldn't happen with layered store (fallback always returns)
            # But handle it gracefully just in case
            logger.warning(f"Feature store returned None for user {user_id}, using defaults")
            return {
                "user_rating_count": self.cold_start_defaults.get("user_rating_count", 0),
                "user_avg_rating": self.cold_start_defaults.get("user_avg_rating", 3.5),
                "user_rating_std": self.cold_start_defaults.get("user_rating_std", 1.0),
                "user_rating_min": self.cold_start_defaults.get("user_rating_min", 1.0),
                "user_rating_max": self.cold_start_defaults.get("user_rating_max", 5.0),
                "gender": 0,
                "age_group": 25,
                "occupation": 0,
            }

    def _get_user_features_from_parquet(self, user_id: int) -> Dict:
        """Get user features from in-memory parquet dictionaries (legacy)."""
        features = {}

        # User aggregation stats
        if user_id in self.user_stats_dict:
            stats = self.user_stats_dict[user_id]
            features["user_rating_count"] = stats["user_rating_count"]
            features["user_avg_rating"] = stats["user_avg_rating"]
            features["user_rating_std"] = stats["user_rating_std"]
            features["user_rating_min"] = stats["user_rating_min"]
            features["user_rating_max"] = stats["user_rating_max"]
        else:
            # Cold-start user
            features["user_rating_count"] = self.cold_start_defaults.get("user_rating_count", 0)
            features["user_avg_rating"] = self.cold_start_defaults.get("user_avg_rating", 3.5)
            features["user_rating_std"] = self.cold_start_defaults.get("user_rating_std", 1.0)
            features["user_rating_min"] = self.cold_start_defaults.get("user_rating_min", 1.0)
            features["user_rating_max"] = self.cold_start_defaults.get("user_rating_max", 5.0)

        # User demographics
        if user_id in self.users_dict:
            user_demo = self.users_dict[user_id]
            features["gender"] = user_demo["gender"]
            features["age_group"] = user_demo["age_group"]
            features["occupation"] = user_demo["occupation"]
        else:
            # Cold-start user demographics (use defaults)
            features["gender"] = 0
            features["age_group"] = 25
            features["occupation"] = 0

        return features

    def _get_movie_features(self, movie_id: int) -> Dict:
        """
        Get movie features, using cold-start defaults if movie not found.

        In feature store mode, fetches from the store (with circuit breaker).
        In parquet mode, uses in-memory dictionaries.
        """
        if self.use_feature_store:
            return self._get_movie_features_from_store(movie_id)
        else:
            return self._get_movie_features_from_parquet(movie_id)

    def _get_movie_features_from_store(self, movie_id: int) -> Dict:
        """Get movie features from the feature store."""
        vec = self.feature_store.get_movie_features(movie_id)

        if vec is not None:
            # Feature store returned values (either from Redis or fallback)
            return vec.features.copy()
        else:
            # This shouldn't happen with layered store (fallback always returns)
            # But handle it gracefully just in case
            logger.warning(f"Feature store returned None for movie {movie_id}, using defaults")
            features = {
                "movie_rating_count": self.cold_start_defaults.get("movie_rating_count", 0),
                "movie_avg_rating": self.cold_start_defaults.get("movie_avg_rating", 3.5),
                "movie_rating_std": self.cold_start_defaults.get("movie_rating_std", 1.0),
                "movie_rating_min": self.cold_start_defaults.get("movie_rating_min", 1.0),
                "movie_rating_max": self.cold_start_defaults.get("movie_rating_max", 5.0),
            }
            # Add zero genres
            for genre_col in self.genre_cols:
                features[genre_col] = 0
            return features

    def _get_movie_features_from_parquet(self, movie_id: int) -> Dict:
        """Get movie features from in-memory parquet dictionaries (legacy)."""
        features = {}

        # Movie aggregation stats
        if movie_id in self.movie_stats_dict:
            stats = self.movie_stats_dict[movie_id]
            features["movie_rating_count"] = stats["movie_rating_count"]
            features["movie_avg_rating"] = stats["movie_avg_rating"]
            features["movie_rating_std"] = stats["movie_rating_std"]
            features["movie_rating_min"] = stats["movie_rating_min"]
            features["movie_rating_max"] = stats["movie_rating_max"]
        else:
            # Cold-start movie
            features["movie_rating_count"] = self.cold_start_defaults.get("movie_rating_count", 0)
            features["movie_avg_rating"] = self.cold_start_defaults.get("movie_avg_rating", 3.5)
            features["movie_rating_std"] = self.cold_start_defaults.get("movie_rating_std", 1.0)
            features["movie_rating_min"] = self.cold_start_defaults.get("movie_rating_min", 1.0)
            features["movie_rating_max"] = self.cold_start_defaults.get("movie_rating_max", 5.0)

        # Movie genres
        if movie_id in self.movies_dict:
            movie_data = self.movies_dict[movie_id]
            for genre_col in self.genre_cols:
                features[genre_col] = movie_data.get(genre_col, 0)
        else:
            # Cold-start movie (no genres)
            for genre_col in self.genre_cols:
                features[genre_col] = 0

        return features

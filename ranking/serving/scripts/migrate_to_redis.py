#!/usr/bin/env python
"""
Migrate feature data from parquet files to Redis.

This script performs a one-time migration of pre-computed features from
parquet files to Redis. After migration, the recommendation service can
use Redis for feature lookups instead of loading parquet files into memory.

Usage:
    # Start Redis first
    docker-compose up -d redis

    # Run migration (from project root)
    PYTHONPATH=$(pwd) python ranking/serving/scripts/migrate_to_redis.py

    # Verify migration
    docker-compose exec redis redis-cli HGETALL features:user:635:stats

Design Decisions:

1. **Batch writes via pipeline**: Redis pipelines batch multiple commands
   into a single network round-trip. Without this, migration would take
   minutes instead of seconds.

2. **HMSET for HASHes**: We use Redis HASHes (not JSON strings) because:
   - Field-level access: Get just user_avg_rating without deserializing
   - Atomic field updates: Update one field without affecting others
   - Memory efficient: Redis optimizes small HASHes

3. **Key naming convention**:
   - features:user:{user_id}:stats → aggregation stats
   - features:user:{user_id}:demographics → demographic features
   - features:movie:{movie_id}:stats → aggregation stats
   - features:movie:{movie_id}:genres → genre flags

4. **Incremental migration**: This script is idempotent - you can run it
   multiple times safely. Existing keys are overwritten with latest data.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import redis

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.shared_utils import FEATURES_DIR, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisMigrator:
    """
    Migrates feature data from parquet files to Redis.

    This class handles the migration of user and movie features from parquet
    files to Redis. It uses pipelines for efficient batch writes and provides
    progress logging.

    Example:
        >>> migrator = RedisMigrator(host='localhost', port=6379)
        >>> migrator.migrate_all()
        Migrated 943 users, 1682 movies in 2.3s
    """

    # Redis key patterns (must match RedisFeatureStore)
    USER_STATS_KEY = "features:user:{user_id}:stats"
    USER_DEMOGRAPHICS_KEY = "features:user:{user_id}:demographics"
    MOVIE_STATS_KEY = "features:movie:{movie_id}:stats"
    MOVIE_GENRES_KEY = "features:movie:{movie_id}:genres"

    # Batch size for pipeline writes
    PIPELINE_BATCH_SIZE = 500

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize the migrator.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify Redis connection is working."""
        try:
            self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis.connection_pool.connection_kwargs['host']}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def migrate_all(self) -> Dict[str, int]:
        """
        Migrate all feature data to Redis.

        Returns:
            Dictionary with counts of migrated entities
        """
        start_time = time.time()

        counts = {
            'users': 0,
            'movies': 0,
        }

        # Migrate user features
        counts['users'] = self.migrate_user_features()

        # Migrate movie features
        counts['movies'] = self.migrate_movie_features()

        elapsed = time.time() - start_time
        logger.info(
            f"Migration complete: {counts['users']} users, {counts['movies']} movies "
            f"in {elapsed:.1f}s"
        )

        return counts

    def migrate_user_features(self) -> int:
        """
        Migrate user features (stats + demographics) to Redis.

        Returns:
            Number of users migrated
        """
        logger.info("Migrating user features...")

        # Load user stats
        user_stats = pd.read_parquet(FEATURES_DIR / "user_stats.parquet")
        logger.info(f"Loaded {len(user_stats)} user stats from parquet")

        # Load user demographics
        users = pd.read_parquet(DATA_DIR / "users.parquet")
        # Encode gender: M=1, F=0 (matching ServingFeatureBuilder)
        users["gender"] = (users["gender"] == "M").astype(int)
        users = users.rename(columns={"age": "age_group"})
        logger.info(f"Loaded {len(users)} user demographics from parquet")

        # Create user lookup
        users_dict = users.set_index("user_id").to_dict("index")

        # Migrate in batches using pipeline
        pipeline = self.redis.pipeline()
        batch_count = 0
        migrated = 0

        for _, row in user_stats.iterrows():
            user_id = int(row["user_id"])

            # User stats
            stats_key = self.USER_STATS_KEY.format(user_id=user_id)
            stats_data = {
                "user_rating_count": float(row["user_rating_count"]),
                "user_avg_rating": float(row["user_avg_rating"]),
                "user_rating_std": float(row["user_rating_std"]),
                "user_rating_min": float(row["user_rating_min"]),
                "user_rating_max": float(row["user_rating_max"]),
            }
            pipeline.hset(stats_key, mapping=stats_data)

            # User demographics (if available)
            if user_id in users_dict:
                demo_key = self.USER_DEMOGRAPHICS_KEY.format(user_id=user_id)
                demo_data = users_dict[user_id]
                pipeline.hset(demo_key, mapping={
                    "gender": int(demo_data["gender"]),
                    "age_group": int(demo_data["age_group"]),
                    "occupation": int(demo_data["occupation"]),
                })

            batch_count += 1
            if batch_count >= self.PIPELINE_BATCH_SIZE:
                pipeline.execute()
                migrated += batch_count
                logger.debug(f"Migrated {migrated} users...")
                pipeline = self.redis.pipeline()
                batch_count = 0

        # Execute remaining
        if batch_count > 0:
            pipeline.execute()
            migrated += batch_count

        logger.info(f"Migrated {migrated} users to Redis")
        return migrated

    def migrate_movie_features(self) -> int:
        """
        Migrate movie features (stats + genres) to Redis.

        Returns:
            Number of movies migrated
        """
        logger.info("Migrating movie features...")

        # Load movie stats
        movie_stats = pd.read_parquet(FEATURES_DIR / "movie_stats.parquet")
        logger.info(f"Loaded {len(movie_stats)} movie stats from parquet")

        # Load movies with genres
        movies = pd.read_parquet(DATA_DIR / "movies.parquet")
        movies_with_genres = self._prepare_genre_features(movies)
        logger.info(f"Loaded {len(movies)} movies with genres from parquet")

        # Create movie lookup
        movies_dict = movies_with_genres.set_index("movie_id").to_dict("index")

        # Get genre columns
        genre_cols = [col for col in movies_with_genres.columns if col.startswith("genre_")]

        # Migrate in batches using pipeline
        pipeline = self.redis.pipeline()
        batch_count = 0
        migrated = 0

        for _, row in movie_stats.iterrows():
            movie_id = int(row["movie_id"])

            # Movie stats
            stats_key = self.MOVIE_STATS_KEY.format(movie_id=movie_id)
            stats_data = {
                "movie_rating_count": float(row["movie_rating_count"]),
                "movie_avg_rating": float(row["movie_avg_rating"]),
                "movie_rating_std": float(row["movie_rating_std"]),
                "movie_rating_min": float(row["movie_rating_min"]),
                "movie_rating_max": float(row["movie_rating_max"]),
            }
            pipeline.hset(stats_key, mapping=stats_data)

            # Movie genres (if available)
            if movie_id in movies_dict:
                genres_key = self.MOVIE_GENRES_KEY.format(movie_id=movie_id)
                movie_data = movies_dict[movie_id]
                genre_data = {col: int(movie_data.get(col, 0)) for col in genre_cols}
                pipeline.hset(genres_key, mapping=genre_data)

            batch_count += 1
            if batch_count >= self.PIPELINE_BATCH_SIZE:
                pipeline.execute()
                migrated += batch_count
                logger.debug(f"Migrated {migrated} movies...")
                pipeline = self.redis.pipeline()
                batch_count = 0

        # Execute remaining
        if batch_count > 0:
            pipeline.execute()
            migrated += batch_count

        logger.info(f"Migrated {migrated} movies to Redis")
        return migrated

    def _prepare_genre_features(self, movies: pd.DataFrame) -> pd.DataFrame:
        """Create multi-hot encoding of movie genres (mirrors batch FeatureBuilder)."""
        # Extract all unique genres
        all_genres = set()
        for genres_str in movies["genres"]:
            all_genres.update(genres_str.split("|"))
        all_genres = sorted(all_genres)

        # Create binary columns for each genre
        movies_with_genres = movies.copy()
        for genre in all_genres:
            genre_clean = genre.lower().replace("-", "_").replace("'", "")
            col_name = f"genre_{genre_clean}"
            movies_with_genres[col_name] = (
                movies_with_genres["genres"]
                .str.contains(genre, regex=False)
                .astype(int)
            )

        return movies_with_genres

    def verify_migration(self, sample_size: int = 5) -> bool:
        """
        Verify migration was successful by spot-checking some records.

        Args:
            sample_size: Number of random records to verify

        Returns:
            True if verification passed
        """
        logger.info("Verifying migration...")

        # Load source data
        user_stats = pd.read_parquet(FEATURES_DIR / "user_stats.parquet")
        movie_stats = pd.read_parquet(FEATURES_DIR / "movie_stats.parquet")

        # Sample some users
        sample_users = user_stats.sample(min(sample_size, len(user_stats)))
        for _, row in sample_users.iterrows():
            user_id = int(row["user_id"])
            key = self.USER_STATS_KEY.format(user_id=user_id)
            redis_data = self.redis.hgetall(key)

            if not redis_data:
                logger.error(f"User {user_id} not found in Redis")
                return False

            # Verify a field
            if abs(float(redis_data.get("user_avg_rating", 0)) - row["user_avg_rating"]) > 0.001:
                logger.error(f"User {user_id} avg_rating mismatch")
                return False

        # Sample some movies
        sample_movies = movie_stats.sample(min(sample_size, len(movie_stats)))
        for _, row in sample_movies.iterrows():
            movie_id = int(row["movie_id"])
            key = self.MOVIE_STATS_KEY.format(movie_id=movie_id)
            redis_data = self.redis.hgetall(key)

            if not redis_data:
                logger.error(f"Movie {movie_id} not found in Redis")
                return False

        logger.info("Verification passed!")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the Redis data."""
        # Count keys by pattern
        user_stats_keys = len(list(self.redis.scan_iter("features:user:*:stats")))
        user_demo_keys = len(list(self.redis.scan_iter("features:user:*:demographics")))
        movie_stats_keys = len(list(self.redis.scan_iter("features:movie:*:stats")))
        movie_genre_keys = len(list(self.redis.scan_iter("features:movie:*:genres")))

        # Get memory usage
        info = self.redis.info("memory")

        return {
            "user_stats": user_stats_keys,
            "user_demographics": user_demo_keys,
            "movie_stats": movie_stats_keys,
            "movie_genres": movie_genre_keys,
            "memory_used_mb": info.get("used_memory", 0) / (1024 * 1024),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Migrate features from parquet to Redis"
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6379,
        help="Redis port (default: 6379)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify migration after completion"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Show Redis statistics after migration"
    )

    args = parser.parse_args()

    try:
        migrator = RedisMigrator(host=args.host, port=args.port)
        counts = migrator.migrate_all()

        if args.verify:
            success = migrator.verify_migration()
            if not success:
                logger.error("Verification failed!")
                return 1

        if args.stats:
            stats = migrator.get_stats()
            logger.info(f"Redis stats: {stats}")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

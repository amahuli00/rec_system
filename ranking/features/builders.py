"""
Feature builder for GBDT ranking model.

This module contains the FeatureBuilder class that handles the complete
feature materialization pipeline: loading data, computing aggregations,
preparing encodings, assembling feature matrices, and saving outputs.

Design Decision:
    All feature computation methods are consolidated in a single class rather
    than split into separate builder classes. This is intentional given low
    likelihood of adding new feature types and no need for unit testing
    individual builders.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FeatureConfig, FeatureMetadata


logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Builds ranking features from MovieLens data splits.

    This class handles the complete feature materialization pipeline:
    1. Load train/val/test splits and metadata
    2. Compute aggregation features from training data
    3. Prepare genre and demographic encodings
    4. Assemble final feature matrices with cold-start handling
    5. Save outputs with metadata

    All aggregation features are computed from training data only to prevent
    data leakage. Cold-start users/movies receive global default values.

    Example:
        config = FeatureConfig(
            data_dir=Path("data/splits"),
            output_dir=Path("ranking/features")
        )
        builder = FeatureBuilder(config)
        train_features, val_features, test_features = builder.run()
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize the feature builder.

        Args:
            config: Feature configuration specifying paths and options.
        """
        self.config = config

        # Data attributes (populated by _load_data)
        self.train_ratings: Optional[pd.DataFrame] = None
        self.val_ratings: Optional[pd.DataFrame] = None
        self.test_ratings: Optional[pd.DataFrame] = None
        self.movies: Optional[pd.DataFrame] = None
        self.users: Optional[pd.DataFrame] = None

        # Computed feature components (populated during build)
        self.user_stats: Optional[pd.DataFrame] = None
        self.movie_stats: Optional[pd.DataFrame] = None
        self.cold_start_defaults: Optional[Dict[str, float]] = None
        self.movies_with_genres: Optional[pd.DataFrame] = None
        self.genre_cols: Optional[List[str]] = None
        self.users_encoded: Optional[pd.DataFrame] = None

        # Output attributes
        self.train_features: Optional[pd.DataFrame] = None
        self.val_features: Optional[pd.DataFrame] = None
        self.test_features: Optional[pd.DataFrame] = None
        self.metadata: Optional[FeatureMetadata] = None

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete feature building pipeline.

        Returns:
            Tuple of (train_features, val_features, test_features) DataFrames.
        """
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING FOR RANKING MODEL")
        logger.info("=" * 60)

        # Stage 1: Load data
        self._load_data()

        # Stage 2: Compute feature components (from train only)
        self._compute_aggregation_features()
        self._compute_cold_start_defaults()
        self._prepare_genre_features()
        self._prepare_demographic_features()

        # Stage 3: Assemble feature matrices
        self._build_all_features()

        # Stage 4: Save outputs
        self._save_outputs()

        logger.info("=" * 60)
        logger.info("Feature engineering complete!")
        logger.info("=" * 60)
        logger.info(f"Features saved to: {self.config.output_dir}")

        return self.train_features, self.val_features, self.test_features

    # =========================================================================
    # Stage 1: Data Loading
    # =========================================================================

    def _load_data(self) -> None:
        """Load train/val/test splits and metadata from parquet files."""
        logger.info("Loading data...")

        data_dir = self.config.data_dir

        self.train_ratings = pd.read_parquet(data_dir / "train_ratings.parquet")
        self.val_ratings = pd.read_parquet(data_dir / "val_ratings.parquet")
        self.test_ratings = pd.read_parquet(data_dir / "test_ratings.parquet")
        self.movies = pd.read_parquet(data_dir / "movies.parquet")
        self.users = pd.read_parquet(data_dir / "users.parquet")

        logger.info(f"Train: {len(self.train_ratings):,} ratings")
        logger.info(f"Val:   {len(self.val_ratings):,} ratings")
        logger.info(f"Test:  {len(self.test_ratings):,} ratings")
        logger.info(f"Movies: {len(self.movies):,}")
        logger.info(f"Users:  {len(self.users):,}")

    # =========================================================================
    # Stage 2: Feature Computation (from training data only)
    # =========================================================================

    def _compute_aggregation_features(self) -> None:
        """
        Compute user and movie aggregation features from training data.

        These statistics are computed ONLY from training data to prevent
        data leakage. The same statistics are then applied to val/test.
        """
        logger.info("Computing aggregation features (from train only)...")

        # User statistics
        self.user_stats = self.train_ratings.groupby("user_id")["rating"].agg(
            [
                ("user_rating_count", "count"),
                ("user_avg_rating", "mean"),
                ("user_rating_std", "std"),
                ("user_rating_min", "min"),
                ("user_rating_max", "max"),
            ]
        ).reset_index()

        # Fill NaN std (happens when user has only 1 rating)
        self.user_stats["user_rating_std"] = self.user_stats["user_rating_std"].fillna(0)

        # Movie statistics
        self.movie_stats = self.train_ratings.groupby("movie_id")["rating"].agg(
            [
                ("movie_rating_count", "count"),
                ("movie_avg_rating", "mean"),
                ("movie_rating_std", "std"),
                ("movie_rating_min", "min"),
                ("movie_rating_max", "max"),
            ]
        ).reset_index()

        # Fill NaN std (happens when movie has only 1 rating)
        self.movie_stats["movie_rating_std"] = self.movie_stats["movie_rating_std"].fillna(0)

        logger.info(f"User features computed for {len(self.user_stats):,} users")
        logger.info(f"Movie features computed for {len(self.movie_stats):,} movies")

    def _compute_cold_start_defaults(self) -> None:
        """
        Compute global defaults for cold-start users/movies.

        When a user or movie in val/test wasn't seen in training,
        we use global statistics as fallback values.
        """
        logger.info("Computing cold-start defaults...")

        global_mean = self.train_ratings["rating"].mean()
        global_std = self.train_ratings["rating"].std()
        global_min = self.train_ratings["rating"].min()
        global_max = self.train_ratings["rating"].max()

        self.cold_start_defaults = {
            "user_rating_count": 0,
            "user_avg_rating": global_mean,
            "user_rating_std": global_std,
            "user_rating_min": global_min,
            "user_rating_max": global_max,
            "movie_rating_count": 0,
            "movie_avg_rating": global_mean,
            "movie_rating_std": global_std,
            "movie_rating_min": global_min,
            "movie_rating_max": global_max,
        }

        logger.info("Cold-start defaults:")
        for key, value in self.cold_start_defaults.items():
            logger.info(f"  {key}: {value:.3f}")

    def _prepare_genre_features(self) -> None:
        """
        Create multi-hot encoding of movie genres.

        Each genre becomes a binary column (1 if movie has that genre, 0 otherwise).
        """
        logger.info("Preparing genre features...")

        # Extract all unique genres
        all_genres = set()
        for genres_str in self.movies["genres"]:
            all_genres.update(genres_str.split("|"))
        all_genres = sorted(all_genres)

        logger.info(f"Found {len(all_genres)} unique genres: {all_genres}")

        # Create binary columns for each genre
        self.movies_with_genres = self.movies.copy()
        for genre in all_genres:
            # Clean genre name for column name
            genre_clean = genre.lower().replace("-", "_").replace("'", "")
            col_name = f"genre_{genre_clean}"
            self.movies_with_genres[col_name] = (
                self.movies_with_genres["genres"]
                .str.contains(genre, regex=False)
                .astype(int)
            )

        # Get list of genre columns
        self.genre_cols = [
            col for col in self.movies_with_genres.columns if col.startswith("genre_")
        ]
        logger.info(f"Created {len(self.genre_cols)} genre features")

    def _prepare_demographic_features(self) -> None:
        """
        Encode user demographic features.

        - gender: Binary (M=1, F=0)
        - age: Renamed to age_group for clarity
        - occupation: Kept as integer codes
        """
        logger.info("Preparing demographic features...")

        self.users_encoded = self.users.copy()

        # Encode gender: M=1, F=0
        self.users_encoded["gender"] = (self.users_encoded["gender"] == "M").astype(int)

        # Rename age to age_group for clarity
        self.users_encoded = self.users_encoded.rename(columns={"age": "age_group"})

        logger.info("Demographic features prepared:")
        logger.info("  gender: binary (M=1, F=0)")
        logger.info(f"  age_group: {sorted(self.users_encoded['age_group'].unique())}")
        logger.info(f"  occupation: {sorted(self.users_encoded['occupation'].unique())}")

    # =========================================================================
    # Stage 3: Feature Assembly
    # =========================================================================

    def _build_features_for_split(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build complete feature matrix for a single ratings split.

        The output contains PAIRS from the input ratings_df, but all aggregation
        STATISTICS come from training data (via self.user_stats, self.movie_stats).

        For example, when ratings_df is test data:
        - Pairs (user_id, movie_id, rating) come from test_ratings
        - Features (user_avg_rating, movie_rating_count, etc.) are frozen from train

        This mirrors production: new pairs use pre-computed statistics.

        Args:
            ratings_df: Ratings DataFrame (train, val, or test).

        Returns:
            Feature DataFrame with all features merged.
        """
        features = ratings_df.copy()

        # 1. Merge user aggregation features (left join to handle cold-start)
        features = features.merge(self.user_stats, on="user_id", how="left")

        # 2. Merge movie aggregation features (left join)
        features = features.merge(self.movie_stats, on="movie_id", how="left")

        # 3. Fill cold-start defaults
        for col, default_val in self.cold_start_defaults.items():
            if col in features.columns:
                features[col] = features[col].fillna(default_val)

        # 4. Merge demographics
        features = features.merge(
            self.users_encoded[["user_id", "gender", "age_group", "occupation"]],
            on="user_id",
            how="left",
        )

        # 5. Merge genres
        genre_features = self.movies_with_genres[["movie_id"] + self.genre_cols]
        features = features.merge(genre_features, on="movie_id", how="left")

        # 6. Compute interaction features
        if self.config.compute_interaction_features:
            features["user_movie_rating_diff"] = (
                features["user_avg_rating"] - features["movie_avg_rating"]
            )
            features["user_rating_count_norm"] = np.log1p(features["user_rating_count"])
            features["movie_rating_count_norm"] = np.log1p(features["movie_rating_count"])
            features["user_movie_activity_product"] = (
                features["user_rating_count_norm"] * features["movie_rating_count_norm"]
            )

        # 7. Fill any remaining NaN with 0
        if self.config.fill_missing_with_zero:
            missing = features.isnull().sum()
            if missing.sum() > 0:
                logger.warning("Missing values found:")
                for col, count in missing[missing > 0].items():
                    logger.warning(f"  {col}: {count}")
                features = features.fillna(0)

        return features

    def _build_all_features(self) -> None:
        """Build feature matrices for all splits."""
        logger.info("=" * 60)
        logger.info("Building features for all splits")
        logger.info("=" * 60)

        self.train_features = self._build_features_for_split(self.train_ratings)
        logger.info(f"Train: {self.train_features.shape}")

        self.val_features = self._build_features_for_split(self.val_ratings)
        logger.info(f"Val:   {self.val_features.shape}")

        self.test_features = self._build_features_for_split(self.test_ratings)
        logger.info(f"Test:  {self.test_features.shape}")

    # =========================================================================
    # Stage 4: Output Persistence
    # =========================================================================

    def _save_outputs(self) -> None:
        """Save feature parquets, metadata, and README."""
        logger.info("=" * 60)
        logger.info("Saving features...")
        logger.info("=" * 60)

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save parquets
        self.train_features.to_parquet(output_dir / "train_features.parquet", index=False)
        self.val_features.to_parquet(output_dir / "val_features.parquet", index=False)
        self.test_features.to_parquet(output_dir / "test_features.parquet", index=False)

        logger.info("Saved parquet files:")
        for file in sorted(output_dir.glob("*.parquet")):
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"  {file.name:30} ({size_mb:6.2f} MB)")

        # Save metadata
        self._save_metadata(output_dir)

        # Save README
        self._save_readme(output_dir)

    def _save_metadata(self, output_dir: Path) -> None:
        """Save feature metadata JSON."""
        feature_cols = [
            col
            for col in self.train_features.columns
            if col not in ["user_id", "movie_id", "rating", "timestamp"]
        ]

        metadata_dict = {
            "version": "1.0",
            "created_at": pd.Timestamp.now().isoformat(),
            "train_timestamp_max": int(self.train_ratings["timestamp"].max()),
            "num_features": len(feature_cols),
            "feature_groups": {
                "user_agg": [
                    "user_rating_count",
                    "user_avg_rating",
                    "user_rating_std",
                    "user_rating_min",
                    "user_rating_max",
                ],
                "movie_agg": [
                    "movie_rating_count",
                    "movie_avg_rating",
                    "movie_rating_std",
                    "movie_rating_min",
                    "movie_rating_max",
                ],
                "interaction": [
                    "user_movie_rating_diff",
                    "user_rating_count_norm",
                    "movie_rating_count_norm",
                    "user_movie_activity_product",
                ],
                "demographic": ["gender", "age_group", "occupation"],
                "genre": self.genre_cols,
            },
            "cold_start_defaults": {k: float(v) for k, v in self.cold_start_defaults.items()},
            "split_sizes": {
                "train": len(self.train_features),
                "val": len(self.val_features),
                "test": len(self.test_features),
            },
        }

        with open(output_dir / "feature_metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info("Metadata saved")

    def save_serving_data(self, output_dir: Optional[Path] = None) -> None:
        """
        Save pre-computed data for serving (real-time inference).

        This saves user and movie statistics as parquet files for fast lookup
        during serving. These are the same stats computed in
        _compute_aggregation_features() from training data.

        Args:
            output_dir: Directory to save serving data. Defaults to config.output_dir.

        Note:
            Must be called after run() or after _load_data() and
            _compute_aggregation_features() have been called.
        """
        if self.user_stats is None or self.movie_stats is None:
            raise RuntimeError(
                "User/movie stats not computed. Call run() first or "
                "call _load_data() and _compute_aggregation_features()."
            )

        output_dir = output_dir or self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save user stats
        user_stats_path = output_dir / "user_stats.parquet"
        self.user_stats.to_parquet(user_stats_path, index=False)
        logger.info(f"Saved user stats: {user_stats_path} ({len(self.user_stats):,} users)")

        # Save movie stats
        movie_stats_path = output_dir / "movie_stats.parquet"
        self.movie_stats.to_parquet(movie_stats_path, index=False)
        logger.info(f"Saved movie stats: {movie_stats_path} ({len(self.movie_stats):,} movies)")

        logger.info("Serving data saved successfully")

    def _save_readme(self, output_dir: Path) -> None:
        """Save README documentation."""
        feature_cols = [
            col
            for col in self.train_features.columns
            if col not in ["user_id", "movie_id", "rating", "timestamp"]
        ]

        readme_content = f"""# Ranking Model Features

## Feature Set v1.0

Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files

- `train_features.parquet`: Training features ({len(self.train_features):,} rows)
- `val_features.parquet`: Validation features ({len(self.val_features):,} rows)
- `test_features.parquet`: Test features ({len(self.test_features):,} rows)
- `feature_metadata.json`: Feature schema and metadata

## Feature Categories ({len(feature_cols)} features total)

### User Aggregation Features (5)
Computed from training data only:
- user_rating_count: Number of ratings given
- user_avg_rating: Mean rating given
- user_rating_std: Std dev of ratings
- user_rating_min: Minimum rating
- user_rating_max: Maximum rating

### Movie Aggregation Features (5)
Computed from training data only:
- movie_rating_count: Number of ratings received (popularity)
- movie_avg_rating: Mean rating received (quality)
- movie_rating_std: Std dev (polarization)
- movie_rating_min: Minimum rating
- movie_rating_max: Maximum rating

### User-Movie Interaction Features (4)
- user_movie_rating_diff: user_avg - movie_avg
- user_rating_count_norm: log1p(user_rating_count)
- movie_rating_count_norm: log1p(movie_rating_count)
- user_movie_activity_product: user_norm * movie_norm

### Demographic Features (3)
- gender: Binary (M=1, F=0)
- age_group: 7 age bins (1, 18, 25, 35, 45, 50, 56)
- occupation: 21 occupation codes (0-20)

### Genre Features ({len(self.genre_cols)})
Multi-hot encoding of movie genres:
{', '.join(self.genre_cols)}

## Data Leakage Prevention

- User/movie aggregations computed **only from training data**
- Same statistics applied to val/test (frozen at train time)
- Cold-start users/movies receive global defaults
- No future information used in feature computation

## Usage

```python
import pandas as pd

# Load features
train = pd.read_parquet('train_features.parquet')
val = pd.read_parquet('val_features.parquet')
test = pd.read_parquet('test_features.parquet')

# Separate features and target
feature_cols = [col for col in train.columns
                if col not in ['user_id', 'movie_id', 'rating', 'timestamp']]
X_train = train[feature_cols]
y_train = train['rating']
```

## Generation

This feature set was generated by `build_features.py`.
To regenerate: `python -m ranking.features.build_features`
"""

        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

        logger.info("README saved")

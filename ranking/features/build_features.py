"""
Build features for GBDT ranking model.

This script materializes features from the train/val/test splits.
Features are computed only from training data to prevent data leakage.

Strategy:
- User/movie aggregations from train only
- Cold-start defaults for unseen users/movies
- Demographic and genre features
- Interaction features (user-movie crosses)

Output:
- train_features.parquet
- val_features.parquet
- test_features.parquet
- feature_metadata.json
- README.md
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple


def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits and metadata."""
    print("Loading data...")

    train_ratings = pd.read_parquet(data_dir / 'train_ratings.parquet')
    val_ratings = pd.read_parquet(data_dir / 'val_ratings.parquet')
    test_ratings = pd.read_parquet(data_dir / 'test_ratings.parquet')
    movies = pd.read_parquet(data_dir / 'movies.parquet')
    users = pd.read_parquet(data_dir / 'users.parquet')

    print(f"Train: {len(train_ratings):,} ratings")
    print(f"Val:   {len(val_ratings):,} ratings")
    print(f"Test:  {len(test_ratings):,} ratings")
    print(f"Movies: {len(movies):,}")
    print(f"Users:  {len(users):,}")

    return train_ratings, val_ratings, test_ratings, movies, users


def compute_user_features(train_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute user aggregation features from training data only.

    Returns:
        DataFrame with user_id and features: count, avg, std, min, max
    """
    print("\nComputing user features (from train only)...")

    user_stats = train_ratings.groupby('user_id')['rating'].agg([
        ('user_rating_count', 'count'),
        ('user_avg_rating', 'mean'),
        ('user_rating_std', 'std'),
        ('user_rating_min', 'min'),
        ('user_rating_max', 'max')
    ]).reset_index()

    # Fill NaN std (happens when user has only 1 rating)
    user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

    print(f"User features computed for {len(user_stats):,} users")

    return user_stats


def compute_movie_features(train_ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute movie aggregation features from training data only.

    Returns:
        DataFrame with movie_id and features: count, avg, std, min, max
    """
    print("\nComputing movie features (from train only)...")

    movie_stats = train_ratings.groupby('movie_id')['rating'].agg([
        ('movie_rating_count', 'count'),
        ('movie_avg_rating', 'mean'),
        ('movie_rating_std', 'std'),
        ('movie_rating_min', 'min'),
        ('movie_rating_max', 'max')
    ]).reset_index()

    # Fill NaN std (happens when movie has only 1 rating)
    movie_stats['movie_rating_std'] = movie_stats['movie_rating_std'].fillna(0)

    print(f"Movie features computed for {len(movie_stats):,} movies")

    return movie_stats


def compute_cold_start_defaults(train_ratings: pd.DataFrame) -> Dict[str, float]:
    """Compute default values for cold-start users/movies."""
    print("\nComputing cold-start defaults...")

    global_mean_rating = train_ratings['rating'].mean()
    global_std_rating = train_ratings['rating'].std()
    global_min_rating = train_ratings['rating'].min()
    global_max_rating = train_ratings['rating'].max()

    defaults = {
        'user_rating_count': 0,
        'user_avg_rating': global_mean_rating,
        'user_rating_std': global_std_rating,
        'user_rating_min': global_min_rating,
        'user_rating_max': global_max_rating,
        'movie_rating_count': 0,
        'movie_avg_rating': global_mean_rating,
        'movie_rating_std': global_std_rating,
        'movie_rating_min': global_min_rating,
        'movie_rating_max': global_max_rating
    }

    print("Cold-start defaults:")
    for key, value in defaults.items():
        print(f"  {key}: {value:.3f}")

    return defaults


def prepare_genre_features(movies: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """
    Create multi-hot encoding of movie genres.

    Returns:
        movies_with_genres: DataFrame with genre columns added
        genre_cols: List of genre column names
    """
    print("\nPreparing genre features...")

    # Extract all unique genres
    all_genres = set()
    for genres_str in movies['genres']:
        all_genres.update(genres_str.split('|'))

    all_genres = sorted(all_genres)
    print(f"Found {len(all_genres)} unique genres: {all_genres}")

    # Create binary features for each genre
    movies_with_genres = movies.copy()
    for genre in all_genres:
        # Clean genre name for column name
        genre_clean = genre.lower().replace('-', '_').replace("'", '')
        col_name = f'genre_{genre_clean}'
        movies_with_genres[col_name] = movies_with_genres['genres'].str.contains(
            genre, regex=False
        ).astype(int)

    # Get list of genre columns
    genre_cols = [col for col in movies_with_genres.columns if col.startswith('genre_')]
    print(f"Created {len(genre_cols)} genre features")

    return movies_with_genres, genre_cols


def prepare_demographic_features(users: pd.DataFrame) -> pd.DataFrame:
    """Encode user demographic features."""
    print("\nPreparing demographic features...")

    users_encoded = users.copy()

    # Encode gender: M=1, F=0
    users_encoded['gender'] = (users_encoded['gender'] == 'M').astype(int)

    # Rename age to age_group for clarity
    users_encoded = users_encoded.rename(columns={'age': 'age_group'})

    print("Demographic features:")
    print(f"  gender: binary (M=1, F=0)")
    print(f"  age_group: {sorted(users_encoded['age_group'].unique())}")
    print(f"  occupation: {sorted(users_encoded['occupation'].unique())}")

    return users_encoded


def build_features(
    ratings_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    movie_stats: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    genre_cols: list,
    defaults: Dict[str, float]
) -> pd.DataFrame:
    """
    Build feature matrix for a ratings dataframe.

    Args:
        ratings_df: Ratings to build features for
        user_stats: Pre-computed user statistics (from train only)
        movie_stats: Pre-computed movie statistics (from train only)
        movies_df: Movie metadata with genre features
        users_df: User demographics
        genre_cols: List of genre column names
        defaults: Cold-start default values

    Returns:
        DataFrame with all features
    """
    print(f"Building features for {len(ratings_df):,} ratings...")

    # Start with ratings
    features = ratings_df.copy()

    # 1. Merge user features (left join to handle cold-start)
    features = features.merge(user_stats, on='user_id', how='left')

    # 2. Merge movie features (left join)
    features = features.merge(movie_stats, on='movie_id', how='left')

    # 3. Fill NaN with cold-start defaults
    for col, default_val in defaults.items():
        if col in features.columns:
            features[col] = features[col].fillna(default_val)

    # 4. Merge demographics
    features = features.merge(
        users_df[['user_id', 'gender', 'age_group', 'occupation']],
        on='user_id',
        how='left'
    )

    # 5. Merge genres (keep only genre columns + movie_id for merge)
    genre_features = movies_df[['movie_id'] + genre_cols]
    features = features.merge(genre_features, on='movie_id', how='left')

    # 6. Compute interaction features
    features['user_movie_rating_diff'] = features['user_avg_rating'] - features['movie_avg_rating']
    features['user_rating_count_norm'] = np.log1p(features['user_rating_count'])
    features['movie_rating_count_norm'] = np.log1p(features['movie_rating_count'])
    features['user_movie_activity_product'] = (
        features['user_rating_count_norm'] * features['movie_rating_count_norm']
    )

    # 7. Fill any remaining NaN with 0 (e.g., for genre features of new movies)
    missing = features.isnull().sum()
    if missing.sum() > 0:
        print("\nWarning: Missing values found:")
        print(missing[missing > 0])
        features = features.fillna(0)

    print(f"Features built: {features.shape}")

    return features


def save_features(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    test_features: pd.DataFrame,
    genre_cols: list,
    defaults: Dict[str, float],
    train_max_timestamp: int,
    output_dir: Path
):
    """Save feature dataframes and metadata."""
    print("\n" + "="*60)
    print("Saving features...")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save parquet files
    train_features.to_parquet(output_dir / 'train_features.parquet', index=False)
    val_features.to_parquet(output_dir / 'val_features.parquet', index=False)
    test_features.to_parquet(output_dir / 'test_features.parquet', index=False)

    print("\nSaved parquet files:")
    for file in sorted(output_dir.glob('*.parquet')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name:30} ({size_mb:6.2f} MB)")

    # Feature schema
    feature_cols = [col for col in train_features.columns
                    if col not in ['user_id', 'movie_id', 'rating', 'timestamp']]

    # Save metadata
    metadata = {
        "version": "1.0",
        "created_at": pd.Timestamp.now().isoformat(),
        "train_timestamp_max": int(train_max_timestamp),
        "num_features": len(feature_cols),
        "feature_groups": {
            "user_agg": [
                "user_rating_count", "user_avg_rating", "user_rating_std",
                "user_rating_min", "user_rating_max"
            ],
            "movie_agg": [
                "movie_rating_count", "movie_avg_rating", "movie_rating_std",
                "movie_rating_min", "movie_rating_max"
            ],
            "interaction": [
                "user_movie_rating_diff", "user_rating_count_norm",
                "movie_rating_count_norm", "user_movie_activity_product"
            ],
            "demographic": ["gender", "age_group", "occupation"],
            "genre": genre_cols
        },
        "cold_start_defaults": {k: float(v) for k, v in defaults.items()},
        "split_sizes": {
            "train": len(train_features),
            "val": len(val_features),
            "test": len(test_features)
        }
    }

    with open(output_dir / 'feature_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✓ Metadata saved")

    # Save README
    readme_content = f"""# Ranking Model Features

## Feature Set v1.0

Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files

- `train_features.parquet`: Training features ({len(train_features):,} rows)
- `val_features.parquet`: Validation features ({len(val_features):,} rows)
- `test_features.parquet`: Test features ({len(test_features):,} rows)
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

### Genre Features ({len(genre_cols)})
Multi-hot encoding of movie genres:
{', '.join(genre_cols)}

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
To regenerate: `python build_features.py`
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print("✓ README saved")


def main():
    """Main function to build and save features."""
    # Set up paths
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'splits'
    output_dir = Path(__file__).parent

    print("="*60)
    print("FEATURE ENGINEERING FOR RANKING MODEL")
    print("="*60)

    # Load data
    train_ratings, val_ratings, test_ratings, movies, users = load_data(data_dir)

    # Compute aggregation features (from train only)
    user_stats = compute_user_features(train_ratings)
    movie_stats = compute_movie_features(train_ratings)

    # Compute cold-start defaults
    defaults = compute_cold_start_defaults(train_ratings)

    # Prepare genre features
    movies_with_genres, genre_cols = prepare_genre_features(movies)

    # Prepare demographic features
    users_encoded = prepare_demographic_features(users)

    # Build features for all splits
    print("\n" + "="*60)
    print("Building features for all splits")
    print("="*60)

    train_features = build_features(
        train_ratings, user_stats, movie_stats,
        movies_with_genres, users_encoded, genre_cols, defaults
    )

    val_features = build_features(
        val_ratings, user_stats, movie_stats,
        movies_with_genres, users_encoded, genre_cols, defaults
    )

    test_features = build_features(
        test_ratings, user_stats, movie_stats,
        movies_with_genres, users_encoded, genre_cols, defaults
    )

    print("\n" + "="*60)
    print("Feature Materialization Complete")
    print("="*60)
    print(f"Train: {train_features.shape}")
    print(f"Val:   {val_features.shape}")
    print(f"Test:  {test_features.shape}")

    # Save features
    save_features(
        train_features, val_features, test_features,
        genre_cols, defaults, train_ratings['timestamp'].max(),
        output_dir
    )

    print("\n" + "="*60)
    print("✓ Feature engineering complete!")
    print("="*60)
    print(f"\nFeatures saved to: {output_dir}")
    print("\nNext step: Train GBDT ranking model using these features")


if __name__ == '__main__':
    main()

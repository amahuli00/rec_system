"""
Create train/validation/test splits for the MovieLens 1M dataset.

Strategy:
- Time-based split: Most realistic for recommendation systems
- Train: earliest 80% of interactions (by timestamp)
- Validation: next 10%
- Test: latest 10%

This mimics real-world scenarios where we predict future interactions.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_movielens_data(data_dir: Path):
    """Load MovieLens 1M data files."""
    print("Loading MovieLens 1M data...")

    # Load movies
    movies = pd.read_csv(
        data_dir / 'movies.dat',
        sep='::',
        engine='python',
        encoding='latin-1',
        names=['movie_id', 'title', 'genres'],
        header=None
    )

    # Load ratings
    ratings = pd.read_csv(
        data_dir / 'ratings.dat',
        sep='::',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        header=None
    )

    # Load users
    users = pd.read_csv(
        data_dir / 'users.dat',
        sep='::',
        engine='python',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
        header=None
    )

    print(f"Loaded {len(ratings):,} ratings from {users['user_id'].nunique():,} users "
          f"on {movies['movie_id'].nunique():,} movies")

    return movies, ratings, users


def create_time_based_split(ratings: pd.DataFrame, train_ratio=0.8, val_ratio=0.1):
    """
    Create time-based train/val/test splits.

    Args:
        ratings: DataFrame with ratings
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation (test gets the rest)

    Returns:
        train_df, val_df, test_df
    """
    print("\nCreating time-based splits...")

    # Sort by timestamp
    ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)

    # Calculate split indices
    n = len(ratings_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # Split the data
    train_df = ratings_sorted.iloc[:train_end].copy()
    val_df = ratings_sorted.iloc[train_end:val_end].copy()
    test_df = ratings_sorted.iloc[val_end:].copy()

    print(f"Train: {len(train_df):,} ratings ({len(train_df)/n*100:.1f}%)")
    print(f"Val:   {len(val_df):,} ratings ({len(val_df)/n*100:.1f}%)")
    print(f"Test:  {len(test_df):,} ratings ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


def analyze_splits(train_df, val_df, test_df):
    """Analyze the characteristics of the splits."""
    print("\n" + "="*60)
    print("Split Analysis")
    print("="*60)

    # Time ranges
    print("\nTime ranges:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        min_ts = pd.to_datetime(df['timestamp'].min(), unit='s')
        max_ts = pd.to_datetime(df['timestamp'].max(), unit='s')
        print(f"{name:6} {min_ts} to {max_ts}")

    # User overlap
    train_users = set(train_df['user_id'].unique())
    val_users = set(val_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())

    print("\nUser statistics:")
    print(f"Train users:     {len(train_users):,}")
    print(f"Val users:       {len(val_users):,}")
    print(f"Test users:      {len(test_users):,}")
    print(f"Val new users:   {len(val_users - train_users):,} ({len(val_users - train_users)/len(val_users)*100:.1f}%)")
    print(f"Test new users:  {len(test_users - train_users):,} ({len(test_users - train_users)/len(test_users)*100:.1f}%)")

    # Movie overlap
    train_movies = set(train_df['movie_id'].unique())
    val_movies = set(val_df['movie_id'].unique())
    test_movies = set(test_df['movie_id'].unique())

    print("\nMovie statistics:")
    print(f"Train movies:    {len(train_movies):,}")
    print(f"Val movies:      {len(val_movies):,}")
    print(f"Test movies:     {len(test_movies):,}")
    print(f"Val new movies:  {len(val_movies - train_movies):,} ({len(val_movies - train_movies)/len(val_movies)*100:.1f}%)")
    print(f"Test new movies: {len(test_movies - train_movies):,} ({len(test_movies - train_movies)/len(test_movies)*100:.1f}%)")

    # Rating distribution
    print("\nRating distribution:")
    print("       ", "  ".join([f"{r}star" for r in range(1, 6)]))
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df['rating'].value_counts().sort_index()
        counts = [dist.get(i, 0) for i in range(1, 6)]
        pcts = [f"{c/len(df)*100:5.1f}%" for c in counts]
        print(f"{name:6}", "  ".join(pcts))


def save_splits(train_df, val_df, test_df, movies, users, output_dir: Path):
    """Save splits as parquet files."""
    print("\n" + "="*60)
    print("Saving splits to parquet files...")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save rating splits
    train_df.to_parquet(output_dir / 'train_ratings.parquet', index=False)
    val_df.to_parquet(output_dir / 'val_ratings.parquet', index=False)
    test_df.to_parquet(output_dir / 'test_ratings.parquet', index=False)

    # Save movies and users (same across all splits)
    movies.to_parquet(output_dir / 'movies.parquet', index=False)
    users.to_parquet(output_dir / 'users.parquet', index=False)

    print(f"\nSaved files to {output_dir}:")
    for file in sorted(output_dir.glob('*.parquet')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name:25} ({size_mb:6.2f} MB)")

    # Save a README
    readme_content = """# MovieLens 1M Splits

## Split Strategy

Time-based split to simulate real-world recommendation scenarios:
- **Train**: Earliest 80% of interactions by timestamp
- **Validation**: Next 10% of interactions
- **Test**: Latest 10% of interactions

## Files

- `train_ratings.parquet`: Training ratings (user_id, movie_id, rating, timestamp)
- `val_ratings.parquet`: Validation ratings
- `test_ratings.parquet`: Test ratings
- `movies.parquet`: Movie metadata (movie_id, title, genres)
- `users.parquet`: User metadata (user_id, gender, age, occupation, zip_code)

## Notes

- Some users/movies in validation/test may not appear in training (cold-start scenarios)
- Timestamps represent when the rating was made
- Ratings are on a 1-5 scale (integer values)
"""

    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print("\n✓ Splits created successfully!")


def main():
    """Main function to create and save splits."""
    # Set up paths
    data_dir = Path(__file__).parent / 'ml-1m'
    output_dir = Path(__file__).parent / 'splits'

    # Load data
    movies, ratings, users = load_movielens_data(data_dir)

    # Create splits
    train_df, val_df, test_df = create_time_based_split(
        ratings,
        train_ratio=0.8,
        val_ratio=0.1
    )

    # Analyze splits
    analyze_splits(train_df, val_df, test_df)

    # Save splits
    save_splits(train_df, val_df, test_df, movies, users, output_dir)


if __name__ == '__main__':
    main()

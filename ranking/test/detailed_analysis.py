"""
Detailed model analysis script.

Provides in-depth analysis beyond basic metrics:
- Per-user NDCG breakdown
- Error distribution analysis
- Feature importance visualization

Usage:
    python -m ranking.test.detailed_analysis --model xgboost_tuned
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.test.model_loader import (
    load_feature_columns,
    load_model,
    load_test_data,
)
from ranking.training.metrics import compute_ndcg_at_k


def compute_per_user_ndcg(
    df: pd.DataFrame, predictions: np.ndarray, k: int = 10
) -> pd.DataFrame:
    """
    Compute NDCG@K for each user individually.

    Args:
        df: DataFrame with user_id and rating columns
        predictions: Predicted scores
        k: Top-K for NDCG

    Returns:
        DataFrame with user_id, ndcg, and num_ratings columns
    """
    df_eval = df[["user_id", "rating"]].copy()
    df_eval["prediction"] = predictions

    results = []
    for user_id, user_df in df_eval.groupby("user_id"):
        if len(user_df) < 2:
            continue

        # Sort by prediction
        user_df_sorted = user_df.sort_values("prediction", ascending=False)
        top_k = user_df_sorted.head(k)

        # DCG
        relevances = top_k["rating"].values
        positions = np.arange(1, len(relevances) + 1)
        dcg = np.sum(relevances / np.log2(positions + 1))

        # IDCG
        ideal_relevances = np.sort(user_df["rating"].values)[::-1][:k]
        ideal_positions = np.arange(1, len(ideal_relevances) + 1)
        idcg = np.sum(ideal_relevances / np.log2(ideal_positions + 1))

        if idcg > 0:
            ndcg = dcg / idcg
            results.append(
                {"user_id": user_id, "ndcg": ndcg, "num_ratings": len(user_df)}
            )

    return pd.DataFrame(results)


def analyze_per_user_performance(
    df_test: pd.DataFrame, y_pred: np.ndarray, top_n: int = 10
) -> None:
    """
    Analyze and print per-user NDCG breakdown.
    """
    print("\n" + "=" * 60)
    print("PER-USER NDCG ANALYSIS")
    print("=" * 60)

    per_user_ndcg = compute_per_user_ndcg(df_test, y_pred, k=10)

    print(f"\nTotal users evaluated: {len(per_user_ndcg):,}")
    print(f"\nNDCG@10 Distribution:")
    print(f"  Mean:   {per_user_ndcg['ndcg'].mean():.4f}")
    print(f"  Median: {per_user_ndcg['ndcg'].median():.4f}")
    print(f"  Std:    {per_user_ndcg['ndcg'].std():.4f}")
    print(f"  Min:    {per_user_ndcg['ndcg'].min():.4f}")
    print(f"  Max:    {per_user_ndcg['ndcg'].max():.4f}")

    # Percentile breakdown
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(per_user_ndcg["ndcg"], p)
        print(f"  P{p:02d}: {val:.4f}")

    # Worst performing users
    print(f"\n{'-'*40}")
    print(f"Bottom {top_n} Users (Lowest NDCG@10):")
    print(f"{'-'*40}")
    worst = per_user_ndcg.nsmallest(top_n, "ndcg")
    for _, row in worst.iterrows():
        print(
            f"  User {int(row['user_id']):>5}: NDCG={row['ndcg']:.4f} "
            f"({int(row['num_ratings'])} ratings)"
        )

    # Best performing users
    print(f"\n{'-'*40}")
    print(f"Top {top_n} Users (Highest NDCG@10):")
    print(f"{'-'*40}")
    best = per_user_ndcg.nlargest(top_n, "ndcg")
    for _, row in best.iterrows():
        print(
            f"  User {int(row['user_id']):>5}: NDCG={row['ndcg']:.4f} "
            f"({int(row['num_ratings'])} ratings)"
        )

    # Users with low activity
    low_activity = per_user_ndcg[per_user_ndcg["num_ratings"] <= 5]
    if len(low_activity) > 0:
        print(f"\n{'-'*40}")
        print(f"Low Activity Users (<=5 ratings): {len(low_activity)}")
        print(f"  Their mean NDCG@10: {low_activity['ndcg'].mean():.4f}")


def analyze_error_distribution(
    y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """
    Analyze prediction error distribution.
    """
    print("\n" + "=" * 60)
    print("ERROR DISTRIBUTION ANALYSIS")
    print("=" * 60)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    print(f"\nError Statistics (Predicted - Actual):")
    print(f"  Mean Error:     {errors.mean():+.4f}")
    print(f"  Std Error:      {errors.std():.4f}")
    print(f"  Mean Abs Error: {abs_errors.mean():.4f}")

    print(f"\nError Distribution:")
    print(f"  Min: {errors.min():+.4f}")
    print(f"  Max: {errors.max():+.4f}")

    # Error buckets
    print(f"\nError Buckets:")
    buckets = [
        ("< -2.0 (severe under)", errors < -2.0),
        ("-2.0 to -1.0", (errors >= -2.0) & (errors < -1.0)),
        ("-1.0 to -0.5", (errors >= -1.0) & (errors < -0.5)),
        ("-0.5 to 0.0", (errors >= -0.5) & (errors < 0.0)),
        ("0.0 to 0.5", (errors >= 0.0) & (errors < 0.5)),
        ("0.5 to 1.0", (errors >= 0.5) & (errors < 1.0)),
        ("1.0 to 2.0", (errors >= 1.0) & (errors < 2.0)),
        ("> 2.0 (severe over)", errors >= 2.0),
    ]
    for label, mask in buckets:
        count = mask.sum()
        pct = 100 * count / len(errors)
        print(f"  {label:20s}: {count:>6,} ({pct:5.1f}%)")

    # Check for systematic bias by rating level
    print(f"\n{'-'*40}")
    print("Mean Error by Actual Rating:")
    print(f"{'-'*40}")
    for rating in sorted(np.unique(y_true)):
        mask = y_true == rating
        mean_err = errors[mask].mean()
        count = mask.sum()
        print(f"  Rating {int(rating)}: {mean_err:+.4f} (n={count:,})")


def analyze_feature_importance(model: xgb.Booster, feature_cols: List[str]) -> None:
    """
    Extract and display feature importance from the model.
    """
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)

    # Get feature importance (gain-based)
    # XGBoost Booster stores feature names, so importance keys are actual names
    importance = model.get_score(importance_type="gain")

    # Sort by importance
    sorted_importance = sorted(
        importance.items(), key=lambda x: x[1], reverse=True
    )

    total_gain = sum(importance.values())

    print(f"\nTop 15 Features by Gain:")
    print(f"{'-'*50}")
    for i, (feat, gain) in enumerate(sorted_importance[:15], 1):
        pct = 100 * gain / total_gain if total_gain > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {i:2d}. {feat:30s} {gain:>10.1f} ({pct:5.1f}%) {bar}")

    # Group by feature type
    print(f"\n{'-'*50}")
    print("Importance by Feature Group:")
    print(f"{'-'*50}")

    groups = {
        "User Aggregation": [
            "user_rating_count",
            "user_avg_rating",
            "user_rating_std",
            "user_rating_min",
            "user_rating_max",
        ],
        "Movie Aggregation": [
            "movie_rating_count",
            "movie_avg_rating",
            "movie_rating_std",
            "movie_rating_min",
            "movie_rating_max",
        ],
        "Demographics": ["gender", "age_group", "occupation"],
        "Interaction": [
            "user_movie_rating_diff",
            "user_rating_count_norm",
            "movie_rating_count_norm",
            "user_movie_activity_product",
        ],
        "Genre": [f for f in feature_cols if f.startswith("genre_")],
    }

    group_totals = []
    for group_name, features in groups.items():
        group_gain = sum(importance.get(f, 0) for f in features)
        group_totals.append((group_name, group_gain))

    group_totals.sort(key=lambda x: x[1], reverse=True)
    for group_name, gain in group_totals:
        pct = 100 * gain / total_gain if total_gain > 0 else 0
        print(f"  {group_name:20s}: {pct:5.1f}%")


def run_detailed_analysis(model_name: str) -> None:
    """
    Run all detailed analyses on a model.
    """
    print(f"\n{'#'*60}")
    print(f"# DETAILED ANALYSIS: {model_name}")
    print(f"{'#'*60}")

    # Load model and data
    print("\nLoading model and data...")
    model = load_model(model_name)
    X_test, df_test, y_test = load_test_data()
    feature_cols = load_feature_columns()

    print(f"  Test samples: {len(X_test):,}")
    print(f"  Unique users: {df_test['user_id'].nunique():,}")

    # Make predictions
    dtest = xgb.DMatrix(X_test[feature_cols])
    y_pred = model.predict(dtest)

    # Run analyses
    analyze_per_user_performance(df_test, y_pred)
    analyze_error_distribution(y_test.values, y_pred)
    analyze_feature_importance(model, feature_cols)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run detailed analysis on a trained ranking model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost_tuned",
        help="Model name to analyze (default: xgboost_tuned)",
    )

    args = parser.parse_args()
    run_detailed_analysis(args.model)


if __name__ == "__main__":
    main()

"""
Evaluation metrics for ranking model training.

This module contains pure functions for computing evaluation metrics:
- NDCG@K: Normalized Discounted Cumulative Gain (ranking quality)
- RMSE: Root Mean Squared Error (prediction accuracy)
- MAE: Mean Absolute Error (prediction accuracy)

Primary metric for ranking: NDCG@K (measures how well we rank items per user)
Secondary metrics: RMSE/MAE (measure how accurate our rating predictions are)
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_ndcg_at_k(df: pd.DataFrame, predictions: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@K for ranking evaluation.

    NDCG (Normalized Discounted Cumulative Gain) measures ranking quality.
    - Range: [0, 1], higher is better
    - 1.0 = perfect ranking (best items at top)
    - Accounts for position: items at top matter more
    - Handles graded relevance: ratings 1-5, not just binary

    Algorithm:
    1. For each user, sort items by predicted score (descending)
    2. Take top K items
    3. Compute DCG using actual ratings as relevance
    4. Compute ideal DCG (best possible ranking)
    5. NDCG = DCG / IDCG
    6. Average across all users

    Args:
        df: DataFrame with 'user_id' and 'rating' columns
        predictions: Array of predicted ratings (same length as df)
        k: Number of top items to consider

    Returns:
        Average NDCG@K across all users
    """
    # Add predictions to dataframe
    df_eval = df[["user_id", "rating"]].copy()
    df_eval["prediction"] = predictions

    ndcg_scores = []

    # Compute NDCG for each user
    for user_id, user_df in df_eval.groupby("user_id"):
        # Skip users with too few items (less than 2)
        if len(user_df) < 2:
            continue

        # Sort by prediction (descending) - this is our ranking
        user_df_sorted = user_df.sort_values("prediction", ascending=False)

        # Take top K
        top_k = user_df_sorted.head(k)

        # Compute DCG@K
        # DCG = sum(rel_i / log2(i+1)) for i in 1..K
        # rel_i is the actual rating at position i
        relevances = top_k["rating"].values
        positions = np.arange(1, len(relevances) + 1)
        dcg = np.sum(relevances / np.log2(positions + 1))

        # Compute Ideal DCG (IDCG) - best possible ranking
        # Sort by actual rating (descending)
        ideal_relevances = np.sort(user_df["rating"].values)[::-1][:k]
        ideal_positions = np.arange(1, len(ideal_relevances) + 1)
        idcg = np.sum(ideal_relevances / np.log2(ideal_positions + 1))

        # NDCG = DCG / IDCG (handle division by zero)
        if idcg > 0:
            ndcg = dcg / idcg
            ndcg_scores.append(ndcg)

    # Return average NDCG across all users
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    prefix: str = "",
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluate model on a dataset with all metrics.

    Args:
        model: Trained XGBoost model with predict() method
        X: Feature DataFrame
        y: Target Series
        df: Original DataFrame with user_id column (for NDCG computation)
        prefix: Prefix for metric names (e.g., "train_", "val_")

    Returns:
        Tuple of:
            - Dictionary with all metrics
            - Array of predictions
    """
    # Make predictions
    y_pred = model.predict(X)

    # Compute metrics
    rmse = compute_rmse(y.values, y_pred)
    mae = compute_mae(y.values, y_pred)
    ndcg_10 = compute_ndcg_at_k(df, y_pred, k=10)
    ndcg_20 = compute_ndcg_at_k(df, y_pred, k=20)

    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}ndcg_10": ndcg_10,
        f"{prefix}ndcg_20": ndcg_20,
    }

    return metrics, y_pred

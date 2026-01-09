"""
Basic model evaluation script.

Computes standard metrics (NDCG@K, RMSE, MAE) on test data.

Usage:
    python -m ranking.test.evaluate_model --model xgboost_tuned
"""

import argparse
import sys
from pathlib import Path

import xgboost as xgb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.test.model_loader import (
    get_available_models,
    load_feature_columns,
    load_model,
    load_model_info,
    load_test_data,
)
from ranking.training.metrics import compute_mae, compute_ndcg_at_k, compute_rmse


def evaluate_on_test(model_name: str) -> dict:
    """
    Evaluate a model on the test set.

    Args:
        model_name: Name of the model to evaluate

    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = load_model(model_name)

    # Load test data
    print("Loading test data...")
    X_test, df_test, y_test = load_test_data()
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {len(X_test.columns)}")

    # Make predictions using DMatrix (XGBoost Booster API)
    print("\nMaking predictions...")
    feature_cols = load_feature_columns()
    dtest = xgb.DMatrix(X_test[feature_cols])
    y_pred = model.predict(dtest)

    # Compute metrics
    print("Computing metrics...\n")
    metrics = {
        "ndcg_10": compute_ndcg_at_k(df_test, y_pred, k=10),
        "ndcg_20": compute_ndcg_at_k(df_test, y_pred, k=20),
        "rmse": compute_rmse(y_test.values, y_pred),
        "mae": compute_mae(y_test.values, y_pred),
    }

    # Print results
    print("=" * 40)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 40)
    print(f"\nRanking Quality:")
    print(f"  NDCG@10: {metrics['ndcg_10']:.4f}")
    print(f"  NDCG@20: {metrics['ndcg_20']:.4f}")
    print(f"\nPrediction Accuracy:")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")

    # Compare with validation metrics if available
    model_info = load_model_info(model_name)
    if model_info and "metrics" in model_info:
        print(f"\n{'-'*40}")
        print("Comparison with Validation Metrics:")
        print(f"{'-'*40}")
        val_metrics = model_info["metrics"]
        print(f"  Val NDCG@10: {val_metrics.get('val_ndcg_10', 'N/A'):.4f}")
        print(f"  Test NDCG@10: {metrics['ndcg_10']:.4f}")
        diff = metrics["ndcg_10"] - val_metrics.get("val_ndcg_10", 0)
        print(f"  Difference: {diff:+.4f}")

    print()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ranking model on test data"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost_tuned",
        help="Model name to evaluate (default: xgboost_tuned)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        models = get_available_models()
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        return

    evaluate_on_test(args.model)


if __name__ == "__main__":
    main()

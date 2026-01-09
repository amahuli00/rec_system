"""
Model comparison script.

Compares multiple models side-by-side on test data.

Usage:
    python -m ranking.test.compare_models
    python -m ranking.test.compare_models --models xgboost_baseline xgboost_tuned
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import xgboost as xgb

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ranking.test.model_loader import (
    get_available_models,
    load_feature_columns,
    load_model,
    load_test_data,
)
from ranking.training.metrics import compute_mae, compute_ndcg_at_k, compute_rmse


def evaluate_single_model(
    model: xgb.Booster,
    X_test,
    df_test,
    y_test,
    feature_cols: List[str],
) -> Dict[str, float]:
    """
    Evaluate a single model and return metrics.
    """
    dtest = xgb.DMatrix(X_test[feature_cols])
    y_pred = model.predict(dtest)

    return {
        "ndcg_10": compute_ndcg_at_k(df_test, y_pred, k=10),
        "ndcg_20": compute_ndcg_at_k(df_test, y_pred, k=20),
        "rmse": compute_rmse(y_test.values, y_pred),
        "mae": compute_mae(y_test.values, y_pred),
    }


def compare_models(model_names: List[str]) -> None:
    """
    Compare multiple models on test data.
    """
    print(f"\n{'='*70}")
    print("MODEL COMPARISON ON TEST SET")
    print(f"{'='*70}\n")

    # Load test data once
    print("Loading test data...")
    X_test, df_test, y_test = load_test_data()
    feature_cols = load_feature_columns()
    print(f"  Test samples: {len(X_test):,}\n")

    # Evaluate each model
    results = {}
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        try:
            model = load_model(model_name)
            results[model_name] = evaluate_single_model(
                model, X_test, df_test, y_test, feature_cols
            )
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

    if len(results) < 2:
        print("\nNeed at least 2 models to compare.")
        return

    # Print comparison table
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")

    # Header
    model_width = max(len(name) for name in results.keys()) + 2
    print(f"{'Metric':<15}", end="")
    for name in results.keys():
        print(f"{name:>{model_width}}", end="")
    print(f"{'Diff':>12}")
    print("-" * (15 + model_width * len(results) + 12))

    # Metrics (higher is better for NDCG)
    metrics_higher_better = ["ndcg_10", "ndcg_20"]
    metrics_lower_better = ["rmse", "mae"]
    all_metrics = metrics_higher_better + metrics_lower_better

    model_list = list(results.keys())

    for metric in all_metrics:
        print(f"{metric:<15}", end="")
        values = []
        for name in model_list:
            val = results[name][metric]
            values.append(val)
            print(f"{val:>{model_width}.4f}", end="")

        # Compute diff (last - first model)
        if len(values) >= 2:
            diff = values[-1] - values[0]
            # Determine if diff is good or bad
            if metric in metrics_higher_better:
                indicator = "+" if diff > 0 else ""
                status = "better" if diff > 0 else ("worse" if diff < 0 else "same")
            else:  # lower is better
                indicator = "" if diff >= 0 else ""
                status = "better" if diff < 0 else ("worse" if diff > 0 else "same")

            print(f"{indicator}{diff:>+11.4f} ({status})")
        else:
            print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    # Find best model for each metric
    for metric in all_metrics:
        values = [(name, results[name][metric]) for name in model_list]
        if metric in metrics_higher_better:
            best = max(values, key=lambda x: x[1])
            print(f"Best {metric}: {best[0]} ({best[1]:.4f})")
        else:
            best = min(values, key=lambda x: x[1])
            print(f"Best {metric}: {best[0]} ({best[1]:.4f})")

    # Overall recommendation
    print(f"\n{'-'*70}")
    print("RECOMMENDATION:")
    print(f"{'-'*70}")

    # Score each model: +1 for being best in a metric
    scores = {name: 0 for name in model_list}
    for metric in all_metrics:
        values = [(name, results[name][metric]) for name in model_list]
        if metric in metrics_higher_better:
            best_name = max(values, key=lambda x: x[1])[0]
        else:
            best_name = min(values, key=lambda x: x[1])[0]
        scores[best_name] += 1

    best_overall = max(scores.items(), key=lambda x: x[1])
    print(
        f"  {best_overall[0]} wins on {best_overall[1]}/{len(all_metrics)} metrics"
    )

    # Primary metric recommendation (NDCG is the ranking metric)
    ndcg_values = [(name, results[name]["ndcg_10"]) for name in model_list]
    best_ndcg = max(ndcg_values, key=lambda x: x[1])
    print(f"  For ranking quality (NDCG@10), use: {best_ndcg[0]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple ranking models on test data"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to compare (default: all available)",
    )

    args = parser.parse_args()

    if args.models:
        model_names = args.models
    else:
        # Default: compare all available models
        model_names = get_available_models()
        if not model_names:
            print("No models found in models directory.")
            return

    print(f"Comparing models: {model_names}")
    compare_models(model_names)


if __name__ == "__main__":
    main()

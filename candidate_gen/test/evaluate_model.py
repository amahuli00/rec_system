"""
Basic model evaluation script for candidate generation.

Computes Recall@K on test data using the trained Two-Tower model.

Usage:
    python -m candidate_gen.test.evaluate_model
    python -m candidate_gen.test.evaluate_model --k 10 50 100 200
"""

import argparse
import logging

from candidate_gen.test.model_loader import (
    get_available_models,
    load_model,
    load_model_info,
    load_id_mapper,
    load_test_user_positives,
)
from candidate_gen.training.metrics import compute_recall_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_on_test(k_values: list, device: str = "cpu") -> dict:
    """
    Evaluate the Two-Tower model on the test set.

    Args:
        k_values: List of K values for Recall@K
        device: Computation device

    Returns:
        Dictionary with all metrics
    """
    print(f"\n{'='*60}")
    print("CANDIDATE GENERATION MODEL EVALUATION")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model = load_model(device=device)
    model_info = load_model_info()
    print(f"  Model: {model_info['num_users']:,} users, {model_info['num_items']:,} items")
    print(f"  Embedding dim: {model_info['embedding_dim']}")

    # Load test data
    print("Loading test data...")
    test_user_positives = load_test_user_positives()
    id_mapper = load_id_mapper()
    print(f"  Test users: {len(test_user_positives):,}")

    # Compute Recall@K
    print(f"\nComputing Recall@K for K={k_values}...")
    metrics = compute_recall_at_k(
        model=model,
        user_positives=test_user_positives,
        num_items=id_mapper.num_items,
        k_values=k_values,
        device=device,
        show_progress=True,
    )

    # Print results
    print("\n" + "=" * 40)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 40)
    print("\nRetrieval Quality (Recall@K):")
    for k in k_values:
        metric_key = f"recall@{k}"
        print(f"  Recall@{k:3d}: {metrics[metric_key]:.4f}")

    # Compare with training validation metrics if available
    if "best_val_metric" in model_info:
        print(f"\n{'-'*40}")
        print("Comparison with Training:")
        print(f"{'-'*40}")
        print(f"  Best Val Recall@50: {model_info['best_val_metric']:.4f}")
        test_recall_50 = metrics.get("recall@50", 0.0)
        diff = test_recall_50 - model_info["best_val_metric"]
        print(f"  Test Recall@50:     {test_recall_50:.4f}")
        print(f"  Difference:         {diff:+.4f}")

    print()
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Two-Tower candidate generation model on test data"
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[10, 50, 100, 200],
        help="K values for Recall@K (default: 10 50 100 200)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
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

    evaluate_on_test(k_values=args.k, device=args.device)


if __name__ == "__main__":
    main()

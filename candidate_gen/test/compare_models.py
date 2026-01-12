"""
Model comparison script for candidate generation.

Compares multiple Two-Tower model checkpoints side-by-side on test data.

Usage:
    python -m candidate_gen.test.compare_models
    python -m candidate_gen.test.compare_models --models two_tower_v1 two_tower_v2
"""

import argparse
import logging
from typing import Dict, List, Set

import torch

from candidate_gen.test.model_loader import (
    get_available_models,
    load_id_mapper,
    load_model,
    load_test_user_positives,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_recall_at_k(
    model: torch.nn.Module,
    user_positives: Dict[int, Set[int]],
    num_items: int,
    k: int,
    device: str = "cpu",
) -> float:
    """
    Compute Recall@K for a model on test users.

    Args:
        model: Trained Two-Tower model
        user_positives: Dict mapping user_idx to set of positive item_idx
        num_items: Total number of items
        k: K value for Recall@K
        device: Computation device

    Returns:
        Mean Recall@K across all test users
    """
    model.eval()

    # Pre-compute all item embeddings
    with torch.no_grad():
        all_item_ids = torch.arange(num_items, device=device)
        item_embeddings = model.get_item_embedding(all_item_ids)

    total_recall = 0.0
    num_users = 0

    for user_idx, positive_items in user_positives.items():
        if not positive_items:
            continue

        with torch.no_grad():
            user_tensor = torch.tensor([user_idx], device=device)
            user_emb = model.get_user_embedding(user_tensor)
            scores = torch.matmul(user_emb, item_embeddings.T).squeeze()
            top_k_indices = torch.topk(scores, k).indices.cpu().numpy()

        top_k_set = set(top_k_indices)
        hits = len(top_k_set & positive_items)
        recall = hits / len(positive_items)
        total_recall += recall
        num_users += 1

    return total_recall / num_users if num_users > 0 else 0.0


def evaluate_single_model(
    model: torch.nn.Module,
    user_positives: Dict[int, Set[int]],
    num_items: int,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a single model and return metrics at multiple K values.
    """
    metrics = {}
    for k in [10, 20, 50, 100]:
        recall = compute_recall_at_k(model, user_positives, num_items, k, device)
        metrics[f"recall_{k}"] = recall
    return metrics


def compare_models(model_names: List[str], device: str = "cpu") -> None:
    """
    Compare multiple Two-Tower models on test data.
    """
    print(f"\n{'='*70}")
    print("CANDIDATE GENERATION MODEL COMPARISON")
    print(f"{'='*70}\n")

    # Load test data once
    print("Loading test data...")
    id_mapper = load_id_mapper()
    user_positives = load_test_user_positives()
    print(f"  Test users: {len(user_positives):,}")
    print(f"  Items: {id_mapper.num_items:,}\n")

    # Evaluate each model
    results = {}
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        try:
            model = load_model(model_name=model_name, device=device)
            results[model_name] = evaluate_single_model(
                model, user_positives, id_mapper.num_items, device
            )
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")
            continue

    if len(results) == 0:
        print("\nNo models successfully evaluated.")
        return

    if len(results) == 1:
        # Single model - just show results
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}\n")

        model_name = list(results.keys())[0]
        print(f"Model: {model_name}\n")
        for metric, value in results[model_name].items():
            print(f"  {metric}: {value:.4f}")
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

    # All recall metrics (higher is better)
    all_metrics = ["recall_10", "recall_20", "recall_50", "recall_100"]
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
            status = "better" if diff > 0 else ("worse" if diff < 0 else "same")
            print(f"{diff:>+11.4f} ({status})")
        else:
            print()

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    # Find best model for each metric
    for metric in all_metrics:
        values = [(name, results[name][metric]) for name in model_list]
        best = max(values, key=lambda x: x[1])
        print(f"Best {metric}: {best[0]} ({best[1]:.4f})")

    # Overall recommendation
    print(f"\n{'-'*70}")
    print("RECOMMENDATION:")
    print(f"{'-'*70}")

    # Score each model: +1 for being best in a metric
    scores = {name: 0 for name in model_list}
    for metric in all_metrics:
        values = [(name, results[name][metric]) for name in model_list]
        best_name = max(values, key=lambda x: x[1])[0]
        scores[best_name] += 1

    best_overall = max(scores.items(), key=lambda x: x[1])
    print(
        f"  {best_overall[0]} wins on {best_overall[1]}/{len(all_metrics)} metrics"
    )

    # Primary metric recommendation (Recall@50 is typical for candidate gen)
    recall_50_values = [(name, results[name]["recall_50"]) for name in model_list]
    best_recall = max(recall_50_values, key=lambda x: x[1])
    print(f"  For candidate generation (Recall@50), use: {best_recall[0]}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple Two-Tower models on test data"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to compare (default: all available)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
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
    compare_models(model_names, device=args.device)


if __name__ == "__main__":
    main()

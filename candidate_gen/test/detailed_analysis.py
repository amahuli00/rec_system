"""
Detailed model analysis script for candidate generation.

Provides in-depth analysis beyond basic metrics:
- Per-user Recall@K breakdown
- Embedding distribution analysis
- Retrieval quality by user activity level

Usage:
    python -m candidate_gen.test.detailed_analysis
"""

import argparse
import logging
from typing import Dict, Set

import numpy as np
import pandas as pd
import torch

from candidate_gen.test.model_loader import (
    load_model,
    load_id_mapper,
    load_test_user_positives,
    load_user_embeddings,
    load_item_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_per_user_recall(
    model: torch.nn.Module,
    user_positives: Dict[int, Set[int]],
    num_items: int,
    k: int = 50,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Compute Recall@K for each user individually.

    Args:
        model: Trained Two-Tower model
        user_positives: Dict mapping user_idx to positive item_idx set
        num_items: Total number of items
        k: K value for Recall@K
        device: Computation device

    Returns:
        DataFrame with user_idx, recall, num_positives columns
    """
    model.eval()

    # Pre-compute item embeddings
    with torch.no_grad():
        all_item_ids = torch.arange(num_items, device=device)
        item_embeddings = model.get_item_embedding(all_item_ids)

    results = []
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

        results.append({
            "user_idx": user_idx,
            "recall": recall,
            "num_positives": len(positive_items),
        })

    return pd.DataFrame(results)


def analyze_per_user_performance(
    model: torch.nn.Module,
    user_positives: Dict[int, Set[int]],
    num_items: int,
    k: int = 50,
    top_n: int = 10,
    device: str = "cpu",
) -> None:
    """
    Analyze and print per-user Recall@K breakdown.
    """
    print("\n" + "=" * 60)
    print(f"PER-USER RECALL@{k} ANALYSIS")
    print("=" * 60)

    per_user_recall = compute_per_user_recall(
        model, user_positives, num_items, k, device
    )

    print(f"\nTotal users evaluated: {len(per_user_recall):,}")
    print(f"\nRecall@{k} Distribution:")
    print(f"  Mean:   {per_user_recall['recall'].mean():.4f}")
    print(f"  Median: {per_user_recall['recall'].median():.4f}")
    print(f"  Std:    {per_user_recall['recall'].std():.4f}")
    print(f"  Min:    {per_user_recall['recall'].min():.4f}")
    print(f"  Max:    {per_user_recall['recall'].max():.4f}")

    # Percentile breakdown
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(per_user_recall["recall"], p)
        print(f"  P{p:02d}: {val:.4f}")

    # Worst performing users
    print(f"\n{'-'*40}")
    print(f"Bottom {top_n} Users (Lowest Recall@{k}):")
    print(f"{'-'*40}")
    worst = per_user_recall.nsmallest(top_n, "recall")
    for _, row in worst.iterrows():
        print(
            f"  User {int(row['user_idx']):>5}: Recall={row['recall']:.4f} "
            f"({int(row['num_positives'])} positives)"
        )

    # Best performing users
    print(f"\n{'-'*40}")
    print(f"Top {top_n} Users (Highest Recall@{k}):")
    print(f"{'-'*40}")
    best = per_user_recall.nlargest(top_n, "recall")
    for _, row in best.iterrows():
        print(
            f"  User {int(row['user_idx']):>5}: Recall={row['recall']:.4f} "
            f"({int(row['num_positives'])} positives)"
        )

    # Recall by user activity level
    print(f"\n{'-'*40}")
    print("Recall by User Activity Level:")
    print(f"{'-'*40}")

    # Bin users by number of positives
    bins = [0, 5, 10, 20, 50, float('inf')]
    labels = ["1-5", "6-10", "11-20", "21-50", "50+"]
    per_user_recall["activity_bin"] = pd.cut(
        per_user_recall["num_positives"],
        bins=bins,
        labels=labels
    )

    for label in labels:
        subset = per_user_recall[per_user_recall["activity_bin"] == label]
        if len(subset) > 0:
            print(f"  {label:>6} positives: Recall={subset['recall'].mean():.4f} (n={len(subset)})")


def analyze_embeddings() -> None:
    """
    Analyze embedding distributions and properties.
    """
    print("\n" + "=" * 60)
    print("EMBEDDING ANALYSIS")
    print("=" * 60)

    user_embeddings = load_user_embeddings()
    item_embeddings = load_item_embeddings()

    print(f"\nUser Embeddings: {user_embeddings.shape}")
    print(f"Item Embeddings: {item_embeddings.shape}")

    # Check L2 norms
    user_norms = np.linalg.norm(user_embeddings, axis=1)
    item_norms = np.linalg.norm(item_embeddings, axis=1)

    print(f"\nL2 Norms:")
    print(f"  User: mean={user_norms.mean():.4f}, std={user_norms.std():.4f}")
    print(f"  Item: mean={item_norms.mean():.4f}, std={item_norms.std():.4f}")

    # Embedding value statistics
    print(f"\nEmbedding Value Statistics:")
    print(f"  User: mean={user_embeddings.mean():.4f}, std={user_embeddings.std():.4f}")
    print(f"  Item: mean={item_embeddings.mean():.4f}, std={item_embeddings.std():.4f}")

    # Dimension-wise variance
    user_dim_var = np.var(user_embeddings, axis=0)
    item_dim_var = np.var(item_embeddings, axis=0)

    print(f"\nDimension-wise Variance:")
    print(f"  User: mean={user_dim_var.mean():.4f}, min={user_dim_var.min():.4f}, max={user_dim_var.max():.4f}")
    print(f"  Item: mean={item_dim_var.mean():.4f}, min={item_dim_var.min():.4f}, max={item_dim_var.max():.4f}")

    # Sample similarity distribution
    print(f"\n{'-'*40}")
    print("Sample User-Item Similarities:")
    print(f"{'-'*40}")

    # Random sample of user-item similarities
    sample_users = np.random.choice(len(user_embeddings), min(1000, len(user_embeddings)), replace=False)
    sample_items = np.random.choice(len(item_embeddings), min(1000, len(item_embeddings)), replace=False)

    sample_similarities = np.dot(
        user_embeddings[sample_users],
        item_embeddings[sample_items].T
    ).flatten()

    print(f"  Mean:   {sample_similarities.mean():.4f}")
    print(f"  Std:    {sample_similarities.std():.4f}")
    print(f"  Min:    {sample_similarities.min():.4f}")
    print(f"  Max:    {sample_similarities.max():.4f}")


def run_detailed_analysis(device: str = "cpu") -> None:
    """
    Run all detailed analyses.
    """
    print(f"\n{'#'*60}")
    print("# DETAILED CANDIDATE GENERATION ANALYSIS")
    print(f"{'#'*60}")

    # Load model and data
    print("\nLoading model and data...")
    model = load_model(device=device)
    id_mapper = load_id_mapper()
    test_user_positives = load_test_user_positives()

    print(f"  Test users: {len(test_user_positives):,}")
    print(f"  Items: {id_mapper.num_items:,}")

    # Run analyses
    analyze_per_user_performance(
        model, test_user_positives, id_mapper.num_items, k=50, device=device
    )
    analyze_embeddings()

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run detailed analysis on Two-Tower candidate generation model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device (default: cpu)",
    )

    args = parser.parse_args()
    run_detailed_analysis(device=args.device)


if __name__ == "__main__":
    main()

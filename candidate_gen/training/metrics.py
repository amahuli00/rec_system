"""
Evaluation metrics for Two-Tower candidate generation.

Primary metric: Recall@K
- Measures what fraction of relevant items are retrieved in top-K
- Standard metric for candidate generation / retrieval

Also provides baseline comparisons:
- Popularity baseline: Recommend most popular items
- Random baseline: Recommend random items
"""

from typing import Dict, List, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


def compute_recall_at_k(
    model: nn.Module,
    user_positives: Dict[int, Set[int]],
    num_items: int,
    k_values: List[int] = [10, 50, 100],
    device: str = "cpu",
    max_users: int = None,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Compute Recall@K for candidate generation evaluation.

    Recall@K = (# of relevant items in top-K) / (# of total relevant items)

    For each user:
    1. Compute user embedding
    2. Compute similarity with all items
    3. Get top-K items by similarity
    4. Check overlap with user's positive items

    Args:
        model: Trained TwoTowerModel
        user_positives: Dict mapping user_idx to set of positive item_idx
        num_items: Total number of items
        k_values: List of K values to compute (default: [10, 50, 100])
        device: Computation device
        max_users: Maximum users to evaluate (for speed, None = all)
        show_progress: Whether to show progress bar

    Returns:
        Dict with recall@k for each k value
        Example: {"recall@10": 0.15, "recall@50": 0.35, "recall@100": 0.52}
    """
    model.eval()

    # Pre-compute all item embeddings once
    with torch.no_grad():
        all_item_ids = torch.arange(num_items, device=device)
        item_embeddings = model.get_item_embedding(all_item_ids)  # [num_items, dim]

    recalls = {f"recall@{k}": [] for k in k_values}
    users_to_eval = list(user_positives.keys())

    if max_users is not None:
        users_to_eval = users_to_eval[:max_users]

    iterator = users_to_eval
    if show_progress:
        iterator = tqdm(iterator, desc="Computing Recall@K")

    max_k = max(k_values)

    with torch.no_grad():
        for user_idx in iterator:
            positive_items = user_positives[user_idx]
            if not positive_items:
                continue

            # Get user embedding
            user_tensor = torch.tensor([user_idx], device=device)
            user_emb = model.get_user_embedding(user_tensor)  # [1, dim]

            # Compute similarity with all items
            scores = torch.matmul(user_emb, item_embeddings.T).squeeze()  # [num_items]

            # Get top-K items
            top_k_indices = torch.topk(scores, max_k).indices.cpu().numpy()

            # Compute recall for each K
            for k in k_values:
                top_k_set = set(top_k_indices[:k])
                hits = len(top_k_set & positive_items)
                recall = hits / len(positive_items)
                recalls[f"recall@{k}"].append(recall)

    # Average across users
    return {k: float(np.mean(v)) for k, v in recalls.items()}


def compute_baselines(
    user_positives: Dict[int, Set[int]],
    train_positive_df: pd.DataFrame,
    item_to_idx: Dict[int, int],
    num_items: int,
    k_values: List[int] = [10, 50, 100],
) -> Dict[str, Dict[str, float]]:
    """
    Compute baseline metrics for comparison.

    Baselines:
    - Popularity: Recommend the most rated items to everyone
    - Random: Recommend random items

    Args:
        user_positives: Dict mapping user_idx to positive item_idx set
        train_positive_df: DataFrame with positive training interactions
        item_to_idx: Mapping from movie_id to item_idx
        num_items: Total number of items
        k_values: List of K values to compute

    Returns:
        Dict with baseline metrics:
        {
            "popularity": {"recall@10": 0.12, ...},
            "random": {"recall@10": 0.01, ...}
        }
    """
    # Popularity baseline: most frequently rated items
    item_popularity = train_positive_df.groupby("movie_id").size().sort_values(ascending=False)
    popular_item_ids = item_popularity.index.tolist()
    popular_item_indices = [item_to_idx[iid] for iid in popular_item_ids if iid in item_to_idx]

    # Compute popularity baseline recall
    pop_recalls = {f"recall@{k}": [] for k in k_values}
    for user_idx, positive_items in user_positives.items():
        for k in k_values:
            top_k_set = set(popular_item_indices[:k])
            hits = len(top_k_set & positive_items)
            recall = hits / len(positive_items) if positive_items else 0.0
            pop_recalls[f"recall@{k}"].append(recall)

    popularity_metrics = {k: float(np.mean(v)) for k, v in pop_recalls.items()}

    # Random baseline
    max_k = max(k_values)
    random_recalls = {f"recall@{k}": [] for k in k_values}
    for user_idx, positive_items in user_positives.items():
        random_items = set(np.random.choice(num_items, size=max_k, replace=False))
        for k in k_values:
            top_k_set = set(list(random_items)[:k])
            hits = len(top_k_set & positive_items)
            recall = hits / len(positive_items) if positive_items else 0.0
            random_recalls[f"recall@{k}"].append(recall)

    random_metrics = {k: float(np.mean(v)) for k, v in random_recalls.items()}

    return {
        "popularity": popularity_metrics,
        "random": random_metrics,
    }

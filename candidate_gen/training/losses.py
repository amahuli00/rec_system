"""
Loss functions for Two-Tower model training.

Provides:
- in_batch_softmax_loss: InfoNCE with in-batch negatives (recommended)
- bpr_loss: Bayesian Personalized Ranking (alternative)
"""

import torch
import torch.nn.functional as F


def in_batch_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    In-batch softmax loss (InfoNCE with in-batch negatives).

    For each user, treats:
    - Their positive item as the correct class
    - All other items in the batch as negative classes

    This is the industry standard for Two-Tower models (YouTube, Google, Spotify).

    How it works:
    - Compute all-pairs similarity: scores[i][j] = user_i · item_j
    - The diagonal (i=j) represents positive pairs
    - Off-diagonal represents in-batch negatives
    - Apply softmax and cross-entropy with diagonal as target

    With batch_size=1024:
    - Each sample gets 1023 negatives "for free"
    - Different batches = different negatives (prevents overfitting)

    Args:
        user_emb: User embeddings, shape [batch_size, dim]
        item_emb: Item embeddings, shape [batch_size, dim]
        temperature: Softmax temperature (default: 0.1)
            Lower = sharper distribution, harder to match
            Higher = softer distribution, easier to match

    Returns:
        Scalar loss value
    """
    # Compute all-pairs similarity: [batch_size, batch_size]
    # scores[i][j] = cosine similarity between user_i and item_j
    scores = torch.matmul(user_emb, item_emb.T) / temperature

    # Target: diagonal (user_i should match item_i)
    # labels[i] = i for all i
    labels = torch.arange(scores.shape[0], device=scores.device)

    # Cross-entropy loss
    return F.cross_entropy(scores, labels)


def bpr_loss(
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    neg_item_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) loss.

    For each (user, pos_item, neg_item) triplet:
    - Maximize: score(user, pos) - score(user, neg)
    - Loss: -log(sigmoid(pos_score - neg_score))

    Note: This requires explicit negative sampling, unlike in_batch_softmax_loss.
    Not currently used in the main training loop but provided as an alternative.

    Args:
        user_emb: User embeddings, shape [batch_size, dim]
        pos_item_emb: Positive item embeddings, shape [batch_size, dim]
        neg_item_emb: Negative item embeddings, shape [batch_size, dim]

    Returns:
        Scalar loss value
    """
    # Compute positive and negative scores
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)  # [batch_size]
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)  # [batch_size]

    # BPR loss: -log(sigmoid(pos - neg))
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()

    return loss

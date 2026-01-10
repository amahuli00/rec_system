"""
Two-Tower model for candidate generation.

The Two-Tower architecture learns separate embeddings for users and items
that can be compared via dot product (cosine similarity when L2 normalized).
"""

from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .tower import Tower


class TwoTowerModel(nn.Module):
    """
    Two-Tower model for candidate generation.

    Architecture:
        User Tower: user_id → user_embedding
        Item Tower: item_id → item_embedding

    Training:
        - Forward pass returns (user_emb, item_emb) for computing loss
        - Uses in-batch negatives: all items in batch serve as negatives

    Inference:
        - Pre-compute all item embeddings once
        - For each user, compute user embedding and find nearest items via ANN

    Example:
        model = TwoTowerModel(num_users=1000, num_items=5000)

        # Training
        user_emb, item_emb = model(user_ids, pos_item_ids)
        loss = in_batch_softmax_loss(user_emb, item_emb)

        # Inference
        user_emb = model.get_user_embedding(user_ids)
        item_emb = model.get_item_embedding(all_item_ids)
        scores = user_emb @ item_emb.T  # cosine similarity
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        config: ModelConfig = None,
        embedding_dim: int = 128,
    ):
        """
        Initialize the Two-Tower model.

        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            config: ModelConfig (if provided, overrides embedding_dim)
            embedding_dim: Embedding dimension (default: 128)
        """
        super().__init__()

        if config is None:
            config = ModelConfig(embedding_dim=embedding_dim)

        self.config = config
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = config.embedding_dim

        # Create user and item towers with same architecture
        self.user_tower = Tower(
            num_entities=num_users,
            embedding_dim=config.embedding_dim,
            use_mlp=config.use_mlp,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            init_std=config.init_std,
        )

        self.item_tower = Tower(
            num_entities=num_items,
            embedding_dim=config.embedding_dim,
            use_mlp=config.use_mlp,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            init_std=config.init_std,
        )

    def get_user_embedding(self, user_ids: torch.Tensor) -> torch.Tensor:
        """
        Get user embeddings.

        Args:
            user_ids: User indices, shape [batch_size]

        Returns:
            User embeddings, shape [batch_size, embedding_dim]
        """
        return self.user_tower(user_ids)

    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Get item embeddings.

        Args:
            item_ids: Item indices, shape [batch_size] or [num_items]

        Returns:
            Item embeddings, shape [batch_size, embedding_dim] or [num_items, embedding_dim]
        """
        return self.item_tower(item_ids)

    def forward(
        self, user_ids: torch.Tensor, pos_item_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training with in-batch negatives.

        Args:
            user_ids: User indices, shape [batch_size]
            pos_item_ids: Positive item indices, shape [batch_size]

        Returns:
            Tuple of (user_emb, item_emb) for computing in-batch loss
        """
        user_emb = self.get_user_embedding(user_ids)
        item_emb = self.get_item_embedding(pos_item_ids)

        return user_emb, item_emb

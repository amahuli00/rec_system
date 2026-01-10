"""
Tower module for Two-Tower model.

A tower maps entity IDs (users or items) to dense embeddings.
The current implementation is a simple embedding lookup with L2 normalization.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    """
    Simple tower: Embedding → L2 Normalize.

    This is a simplified architecture that works well for collaborative
    filtering. The embedding layer learns representations, and L2 normalization
    ensures embeddings lie on a unit hypersphere for stable cosine similarity.

    Why no MLP?
    - Simpler to optimize
    - Fewer hyperparameters to tune
    - Works well for MovieLens scale (~4K items)
    - Larger embedding_dim compensates for reduced capacity

    Example:
        tower = Tower(num_entities=1000, embedding_dim=128)
        ids = torch.tensor([0, 1, 2])
        embeddings = tower(ids)  # [3, 128], L2-normalized
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int,
        use_mlp: bool = False,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        init_std: float = 0.1,
    ):
        """
        Initialize the tower.

        Args:
            num_entities: Number of unique entities (users or items)
            embedding_dim: Output embedding dimension
            use_mlp: Whether to add MLP layers (default: False)
            hidden_dims: MLP hidden dimensions (only used if use_mlp=True)
            dropout: Dropout probability for MLP (only used if use_mlp=True)
            init_std: Standard deviation for embedding initialization
        """
        super().__init__()

        self.use_mlp = use_mlp
        self.embedding_dim = embedding_dim

        if use_mlp and hidden_dims:
            # MLP architecture: Embedding(hidden_dims[0]) → MLP → embedding_dim
            self.embedding = nn.Embedding(num_entities, hidden_dims[0])

            layers = []
            for i in range(len(hidden_dims) - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.BatchNorm1d(hidden_dims[i + 1]),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
            layers.append(nn.Linear(hidden_dims[-1], embedding_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            # Simple architecture: just embedding
            self.embedding = nn.Embedding(num_entities, embedding_dim)
            self.mlp = None

        # Initialize embeddings with smaller values for stability
        nn.init.normal_(self.embedding.weight, mean=0, std=init_std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            ids: Entity IDs, shape [batch_size] or [batch_size, num_items]

        Returns:
            L2-normalized embeddings, shape [..., embedding_dim]
        """
        x = self.embedding(ids)

        if self.mlp is not None:
            # Handle both 1D and 2D inputs for MLP
            original_shape = x.shape
            if len(original_shape) > 2:
                x = x.view(-1, x.shape[-1])

            x = self.mlp(x)

            if len(original_shape) > 2:
                x = x.view(*original_shape[:-1], -1)

        # L2 normalize
        x = F.normalize(x, p=2, dim=-1)

        return x

"""
Configuration for Two-Tower model architecture.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """
    Configuration for Two-Tower model architecture.

    The current implementation uses a simple embedding-only tower
    (no MLP layers) which is easier to optimize and performs well
    for collaborative filtering on MovieLens.

    Attributes:
        embedding_dim: Dimension of user/item embeddings (default: 128)
            Larger values increase model capacity but also memory usage.
            With no MLP, we use a larger embedding_dim to compensate.

        use_mlp: Whether to add MLP layers after embedding (default: False)
            If True, adds hidden layers specified by hidden_dims.
            Currently disabled as the simple version works well.

        hidden_dims: Hidden layer dimensions for MLP (default: [128, 64])
            Only used if use_mlp=True.

        dropout: Dropout probability for MLP layers (default: 0.1)
            Only used if use_mlp=True.

        init_std: Standard deviation for embedding initialization (default: 0.1)
            Smaller values help with training stability.
    """

    embedding_dim: int = 128
    use_mlp: bool = False
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    init_std: float = 0.1

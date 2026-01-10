"""
FAISS index building and embedding materialization.

Provides:
- EmbeddingMaterializer: Extract and save embeddings from trained model
- FAISSIndexBuilder: Build FAISS index from embeddings
"""

import json
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import FAISSConfig

logger = logging.getLogger(__name__)


class EmbeddingMaterializer:
    """
    Extract and save embeddings from a trained Two-Tower model.

    Materializes:
    - User embeddings: [num_users, embedding_dim]
    - Item embeddings: [num_items, embedding_dim]
    - ID mappings for lookup

    Example:
        materializer = EmbeddingMaterializer(model, num_users, num_items)
        materializer.materialize(output_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        num_users: int,
        num_items: int,
        device: str = "cpu",
    ):
        """
        Initialize the materializer.

        Args:
            model: Trained TwoTowerModel
            num_users: Number of users
            num_items: Number of items
            device: Computation device
        """
        self.model = model
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

    def materialize(self, output_dir: Path, batch_size: int = 1024) -> None:
        """
        Materialize all embeddings to disk.

        Args:
            output_dir: Directory to save embeddings
            batch_size: Batch size for embedding computation
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # Materialize user embeddings
        logger.info(f"Materializing user embeddings ({self.num_users:,} users)...")
        user_embeddings = self._materialize_embeddings(
            self.model.get_user_embedding, self.num_users, batch_size
        )
        np.save(output_dir / "user_embeddings.npy", user_embeddings)
        logger.info(f"User embeddings shape: {user_embeddings.shape}")

        # Materialize item embeddings
        logger.info(f"Materializing item embeddings ({self.num_items:,} items)...")
        item_embeddings = self._materialize_embeddings(
            self.model.get_item_embedding, self.num_items, batch_size
        )
        np.save(output_dir / "item_embeddings.npy", item_embeddings)
        logger.info(f"Item embeddings shape: {item_embeddings.shape}")

        # Save metadata
        metadata = {
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": user_embeddings.shape[1],
            "normalized": True,  # Embeddings are L2 normalized
            "created_at": pd.Timestamp.now().isoformat(),
        }
        with open(output_dir / "embedding_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Embeddings saved to: {output_dir}")

    def _materialize_embeddings(
        self,
        embedding_fn,
        num_entities: int,
        batch_size: int,
    ) -> np.ndarray:
        """
        Materialize embeddings in batches.

        Args:
            embedding_fn: Function that takes IDs and returns embeddings
            num_entities: Total number of entities
            batch_size: Batch size

        Returns:
            Numpy array of embeddings
        """
        embeddings = []

        with torch.no_grad():
            for start_idx in range(0, num_entities, batch_size):
                end_idx = min(start_idx + batch_size, num_entities)
                ids = torch.arange(start_idx, end_idx, device=self.device)
                batch_emb = embedding_fn(ids).cpu().numpy()
                embeddings.append(batch_emb)

        return np.vstack(embeddings).astype(np.float32)


class FAISSIndexBuilder:
    """
    Build FAISS index from item embeddings.

    Supports multiple index types:
    - Flat: Exact search (default for small catalogs)
    - IVF: Approximate search (for medium catalogs)
    - HNSW: Fast approximate search (for large catalogs)

    Example:
        builder = FAISSIndexBuilder(config=FAISSConfig(index_type="flat"))
        index = builder.build(item_embeddings)
        builder.save(index, "path/to/index.faiss")
    """

    def __init__(self, config: Optional[FAISSConfig] = None):
        """
        Initialize the index builder.

        Args:
            config: FAISS configuration (defaults to flat index)
        """
        self.config = config or FAISSConfig()

    def build(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Item embeddings [num_items, embedding_dim]

        Returns:
            FAISS index
        """
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        num_vectors = embeddings.shape[0]

        logger.info(f"Building FAISS index: type={self.config.index_type}, metric={self.config.metric}")
        logger.info(f"Vectors: {num_vectors:,}, Dimension: {dim}")

        if self.config.index_type == "flat":
            index = self._build_flat_index(dim)

        elif self.config.index_type == "ivf":
            index = self._build_ivf_index(dim, embeddings)

        elif self.config.index_type == "hnsw":
            index = self._build_hnsw_index(dim)

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

        # Add vectors to index
        index.add(embeddings)
        logger.info(f"Index built with {index.ntotal:,} vectors")

        return index

    def _build_flat_index(self, dim: int) -> faiss.Index:
        """Build flat (exact) index."""
        if self.config.metric == "ip":
            return faiss.IndexFlatIP(dim)
        else:
            return faiss.IndexFlatL2(dim)

    def _build_ivf_index(self, dim: int, embeddings: np.ndarray) -> faiss.Index:
        """Build IVF (inverted file) index."""
        # Create quantizer
        if self.config.metric == "ip":
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(
                quantizer, dim, self.config.nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, self.config.nlist)

        # Train index
        logger.info(f"Training IVF index with {self.config.nlist} clusters...")
        index.train(embeddings)
        index.nprobe = self.config.nprobe

        return index

    def _build_hnsw_index(self, dim: int) -> faiss.Index:
        """Build HNSW (hierarchical navigable small world) index."""
        # HNSW with inner product
        index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M parameter
        index.hnsw.efConstruction = self.config.ef_construction
        index.hnsw.efSearch = self.config.ef_search

        return index

    def save(self, index: faiss.Index, path: Path) -> None:
        """
        Save FAISS index to disk.

        Args:
            index: FAISS index
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(path))
        logger.info(f"Index saved to: {path}")

    @staticmethod
    def load(path: Path) -> faiss.Index:
        """
        Load FAISS index from disk.

        Args:
            path: Path to index file

        Returns:
            FAISS index
        """
        return faiss.read_index(str(path))

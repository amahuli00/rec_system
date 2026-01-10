"""
Candidate retrieval using FAISS index.

Provides fast retrieval of candidate items for a given user.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn

from ..data import IDMapper
from ..model import ModelConfig, TwoTowerModel
from .index_builder import FAISSIndexBuilder

logger = logging.getLogger(__name__)


class CandidateRetriever:
    """
    Retrieve candidate items for users using FAISS.

    Two modes of operation:
    1. Pre-computed embeddings: Load saved user/item embeddings
    2. Online inference: Compute user embedding on-the-fly

    Example:
        # Load from saved artifacts
        retriever = CandidateRetriever.load("candidate_gen/artifacts")
        candidates = retriever.retrieve(user_id=123, k=100)

        # Or with model for online inference
        retriever = CandidateRetriever.from_model(model, id_mapper, index)
        candidates = retriever.retrieve(user_id=123, k=100)
    """

    def __init__(
        self,
        index: faiss.Index,
        id_mapper: IDMapper,
        user_embeddings: Optional[np.ndarray] = None,
        model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        """
        Initialize the retriever.

        Args:
            index: FAISS index of item embeddings
            id_mapper: ID mapper for user/item ID conversion
            user_embeddings: Pre-computed user embeddings (optional)
            model: TwoTowerModel for online inference (optional)
            device: Device for model inference
        """
        self.index = index
        self.id_mapper = id_mapper
        self.user_embeddings = user_embeddings
        self.model = model
        self.device = device

        # Need either user_embeddings or model
        if user_embeddings is None and model is None:
            raise ValueError("Must provide either user_embeddings or model")

    @classmethod
    def load(cls, artifacts_dir: Path, device: str = "cpu") -> "CandidateRetriever":
        """
        Load retriever from saved artifacts.

        Expected directory structure:
            artifacts_dir/
            ├── models/
            │   └── two_tower_model.pt
            │   └── model_info.json
            ├── embeddings/
            │   └── user_embeddings.npy
            │   └── item_embeddings.npy
            ├── index/
            │   └── item_index.faiss
            └── data/
                └── user_to_idx.json
                └── item_to_idx.json

        Args:
            artifacts_dir: Base directory containing all artifacts
            device: Device for model inference

        Returns:
            CandidateRetriever instance
        """
        artifacts_dir = Path(artifacts_dir)

        # Load ID mapper
        data_dir = artifacts_dir / "data"
        id_mapper = IDMapper.load(data_dir)
        logger.info(f"Loaded ID mapper: {id_mapper.num_users} users, {id_mapper.num_items} items")

        # Load FAISS index
        index_path = artifacts_dir / "index" / "item_index.faiss"
        index = FAISSIndexBuilder.load(index_path)
        logger.info(f"Loaded FAISS index: {index.ntotal} vectors")

        # Load user embeddings
        embeddings_dir = artifacts_dir / "embeddings"
        user_embeddings = np.load(embeddings_dir / "user_embeddings.npy")
        logger.info(f"Loaded user embeddings: {user_embeddings.shape}")

        return cls(
            index=index,
            id_mapper=id_mapper,
            user_embeddings=user_embeddings,
            device=device,
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        id_mapper: IDMapper,
        index: faiss.Index,
        device: str = "cpu",
    ) -> "CandidateRetriever":
        """
        Create retriever with model for online inference.

        Args:
            model: Trained TwoTowerModel
            id_mapper: ID mapper
            index: FAISS index of item embeddings
            device: Device for inference

        Returns:
            CandidateRetriever instance
        """
        return cls(
            index=index,
            id_mapper=id_mapper,
            model=model,
            device=device,
        )

    def retrieve(
        self,
        user_id: int,
        k: int = 100,
    ) -> List[int]:
        """
        Retrieve top-K candidate items for a user.

        Args:
            user_id: Original user ID (not internal index)
            k: Number of candidates to retrieve

        Returns:
            List of movie_ids (original IDs, not internal indices)
        """
        # Convert to internal index
        user_idx = self.id_mapper.user_to_idx.get(user_id)
        if user_idx is None:
            logger.warning(f"Unknown user_id: {user_id}")
            return []

        # Get user embedding
        user_emb = self._get_user_embedding(user_idx)

        # Search FAISS index
        distances, indices = self.index.search(user_emb, k)

        # Convert to original movie IDs
        movie_ids = [
            self.id_mapper.idx_to_item.get(int(idx))
            for idx in indices[0]
            if idx >= 0  # FAISS returns -1 for not found
        ]

        return [mid for mid in movie_ids if mid is not None]

    def retrieve_batch(
        self,
        user_ids: List[int],
        k: int = 100,
    ) -> Dict[int, List[int]]:
        """
        Retrieve candidates for multiple users.

        Args:
            user_ids: List of original user IDs
            k: Number of candidates per user

        Returns:
            Dict mapping user_id to list of movie_ids
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.retrieve(user_id, k)
        return results

    def retrieve_with_scores(
        self,
        user_id: int,
        k: int = 100,
    ) -> List[Tuple[int, float]]:
        """
        Retrieve candidates with similarity scores.

        Args:
            user_id: Original user ID
            k: Number of candidates

        Returns:
            List of (movie_id, score) tuples
        """
        user_idx = self.id_mapper.user_to_idx.get(user_id)
        if user_idx is None:
            return []

        user_emb = self._get_user_embedding(user_idx)
        distances, indices = self.index.search(user_emb, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx >= 0:
                movie_id = self.id_mapper.idx_to_item.get(int(idx))
                if movie_id is not None:
                    results.append((movie_id, float(score)))

        return results

    def _get_user_embedding(self, user_idx: int) -> np.ndarray:
        """
        Get user embedding for internal index.

        Uses pre-computed embeddings if available, otherwise
        computes on-the-fly using the model.
        """
        if self.user_embeddings is not None:
            # Use pre-computed embedding
            return self.user_embeddings[user_idx : user_idx + 1].astype(np.float32)
        else:
            # Compute on-the-fly
            self.model.eval()
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx], device=self.device)
                user_emb = self.model.get_user_embedding(user_tensor)
                return user_emb.cpu().numpy().astype(np.float32)

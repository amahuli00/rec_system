"""
Candidate generation service for real-time retrieval.

This module provides the CandidateGenerationService class that orchestrates
candidate retrieval using pre-computed embeddings and FAISS index.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List

from candidate_gen.shared_utils import (
    load_faiss_index,
    load_id_mapper,
    load_user_embeddings,
    load_model_info,
)
from candidate_gen.retrieval import CandidateRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievedCandidate:
    """A retrieved candidate item with its similarity score and rank."""

    movie_id: int
    similarity_score: float
    rank: int


class CandidateGenerationService:
    """
    Candidate generation service for retrieval.

    This service provides end-to-end candidate retrieval functionality:
    1. Looks up user embedding from pre-computed embeddings
    2. Searches FAISS index for nearest item embeddings
    3. Returns top-K candidate movie IDs

    The service is stateful - it loads all artifacts once at initialization
    for fast repeated inference.

    Example:
        service = CandidateGenerationService()
        candidates = service.retrieve(user_id=123, k=100)
        for item in candidates:
            print(f"Rank {item.rank}: Movie {item.movie_id} ({item.similarity_score:.4f})")

        # Or just get movie IDs
        movie_ids = service.retrieve_ids(user_id=123, k=100)
    """

    def __init__(self):
        """
        Initialize the candidate generation service.

        Loads:
            - FAISS index for ANN search
            - ID mapper for ID conversion
            - Pre-computed user embeddings
        """
        logger.info("Initializing CandidateGenerationService...")

        # Load artifacts
        index = load_faiss_index()
        id_mapper = load_id_mapper()
        user_embeddings = load_user_embeddings()

        # Create retriever
        self.retriever = CandidateRetriever(
            index=index,
            id_mapper=id_mapper,
            user_embeddings=user_embeddings,
        )

        # Store metadata
        self.model_info = load_model_info()
        self.num_users = self.model_info["num_users"]
        self.num_items = self.model_info["num_items"]

        logger.info(
            f"CandidateGenerationService initialized: "
            f"{self.num_users:,} users, {self.num_items:,} items"
        )

    def retrieve(
        self,
        user_id: int,
        k: int = 100,
    ) -> List[RetrievedCandidate]:
        """
        Retrieve top-K candidate items for a user with scores.

        Args:
            user_id: Original user ID (not internal index)
            k: Number of candidates to retrieve

        Returns:
            List of RetrievedCandidate sorted by similarity score (descending).
            The best candidate has rank=1.

        Note:
            - Unknown users return empty list
            - Uses inner product similarity (cosine for L2-normalized embeddings)
        """
        results = self.retriever.retrieve_with_scores(user_id, k)

        return [
            RetrievedCandidate(
                movie_id=movie_id,
                similarity_score=score,
                rank=rank,
            )
            for rank, (movie_id, score) in enumerate(results, 1)
        ]

    def retrieve_ids(
        self,
        user_id: int,
        k: int = 100,
    ) -> List[int]:
        """
        Retrieve top-K candidate movie IDs for a user.

        This is a lightweight alternative to retrieve() when you only need
        movie IDs without scores.

        Args:
            user_id: Original user ID
            k: Number of candidates to retrieve

        Returns:
            List of movie_ids in order of relevance (best first).
        """
        return self.retriever.retrieve(user_id, k)

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
        return self.retriever.retrieve_batch(user_ids, k)

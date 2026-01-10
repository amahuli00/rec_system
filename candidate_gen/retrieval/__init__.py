"""
Retrieval module for Two-Tower candidate generation.

This module provides:
- FAISSIndexBuilder: Build FAISS index from item embeddings
- EmbeddingMaterializer: Extract embeddings from trained model
- CandidateRetriever: Retrieve candidates for users
"""

from .config import FAISSConfig
from .index_builder import EmbeddingMaterializer, FAISSIndexBuilder
from .retriever import CandidateRetriever

__all__ = [
    "FAISSConfig",
    "FAISSIndexBuilder",
    "EmbeddingMaterializer",
    "CandidateRetriever",
]

"""
Candidate Generation module for recommendation system.

This module implements a Two-Tower neural network for candidate generation:
- User tower: Maps user_id → user embedding
- Item tower: Maps item_id → item embedding
- FAISS: Fast approximate nearest neighbor search for retrieval

Usage:
    # 1. Prepare data
    python -m candidate_gen.data.prepare_data

    # 2. Train model
    python -m candidate_gen.training.train_model

    # 3. Build index
    python -m candidate_gen.retrieval.build_index

    # 4. Retrieve candidates
    from candidate_gen.retrieval import CandidateRetriever
    retriever = CandidateRetriever.load("candidate_gen/artifacts")
    candidates = retriever.retrieve(user_id=123, k=100)
"""

__version__ = "1.0.0"

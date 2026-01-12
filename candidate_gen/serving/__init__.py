"""
Serving module for the candidate generation system.

This module provides real-time candidate retrieval capabilities:
- CandidateGenerationService: End-to-end retrieval service
- RetrievedCandidate: Data class for retrieved items
"""

from .candidate_service import CandidateGenerationService, RetrievedCandidate

__all__ = [
    "CandidateGenerationService",
    "RetrievedCandidate",
]

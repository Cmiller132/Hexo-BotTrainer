"""Compatibility exports for Hexformer candidate-frontier construction."""

from .candidates import (
    TAG_FRONTIER,
    TAG_IMMEDIATE_WIN,
    TAG_LEGAL,
    TAG_MUST_BLOCK,
    TAG_RECENT,
    TAG_TACTICAL,
    Candidate,
    CandidateSet,
    build_candidate_frontier,
)

__all__ = [
    "Candidate",
    "CandidateSet",
    "TAG_FRONTIER",
    "TAG_IMMEDIATE_WIN",
    "TAG_LEGAL",
    "TAG_MUST_BLOCK",
    "TAG_RECENT",
    "TAG_TACTICAL",
    "build_candidate_frontier",
]

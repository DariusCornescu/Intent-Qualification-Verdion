"""
Stage 2: Soft Scoring Package

  scoring_index.py  — Pre-computed indexes (ScoringIndex, embeddings, BM25)
  signals.py        — Five independent scoring signals
  scorer.py         — RRF fusion orchestrator
"""

from code.soft_scoring.scorer import score_and_rank
from code.soft_scoring.scoring_index import ( ScoringIndex, build_scoring_index, warm_up_embedding_model )

__all__ = [ "score_and_rank", "ScoringIndex", "build_scoring_index", "warm_up_embedding_model" ]
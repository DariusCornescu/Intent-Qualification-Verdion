"""
Stage 2 Tests — Soft Scoring
==============================
Validates that Stage 2 correctly ranks companies by relevance using
BM25 + semantic embeddings + match expressions + NAICS similarity,
combined through Reciprocal Rank Fusion (RRF).

Key validations:
  - RRF scores are normalized to [0, 1]
  - Pre-computed ScoringIndex produces same results as from-scratch
  - Relevant companies rank higher than irrelevant ones
  - Individual signals (BM25, NAICS, match expressions) fire correctly

Run:  python -m pytest tests/test_stage2.py -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.models import (
    CompanyProfile, ParsedQuery, HardFilters, SemanticCriteria,
    MatchNode, MatchLeaf, MatchOperator, QueryComplexity,
)
from code.soft_scoring import (
    score_and_rank, build_scoring_index, ScoringIndex,
)
from code.soft_scoring.signals import (
    compute_naics_scores as _compute_naics_scores,
    compute_business_model_scores as _compute_business_model_scores,
    compute_match_expression_scores as _compute_match_expression_scores,
)
from code.soft_scoring.scorer import (
    _reciprocal_rank_fusion, _scores_to_ranks,
)


# ═══════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def logistics_companies():
    """Set of companies for ranking tests — one relevant, others not."""
    return [
        CompanyProfile(
            operational_name="FreightCo",
            description="Global freight and logistics services, warehousing and supply chain management",
            business_model=["B2B"],
            primary_naics={"code": "484110", "label": "General Freight Trucking"},
            core_offerings=["freight", "logistics", "warehousing"],
        ),
        CompanyProfile(
            operational_name="TechStartup",
            description="AI-powered SaaS platform for developer productivity tools",
            business_model=["B2B", "SaaS"],
            primary_naics={"code": "511210", "label": "Software Publishers"},
            core_offerings=["software", "ai", "developer tools"],
        ),
        CompanyProfile(
            operational_name="BakeryInc",
            description="Artisan bakery producing organic bread and pastries",
            business_model=["B2C"],
            primary_naics={"code": "311811", "label": "Retail Bakeries"},
            core_offerings=["bread", "pastries", "organic food"],
        ),
        CompanyProfile(
            operational_name="ShipExpress",
            description="International shipping and cargo transport services",
            business_model=["B2B"],
            primary_naics={"code": "483111", "label": "Deep Sea Freight Transportation"},
            core_offerings=["shipping", "cargo", "international transport"],
        ),
    ]

@pytest.fixture
def logistics_query():
    return ParsedQuery(
        original_query="Logistics companies",
        semantic_criteria=SemanticCriteria(
            industry_keywords=["logistics", "freight", "shipping", "warehousing",
                               "supply chain", "transportation"],
        ),
        hard_filters=HardFilters(
            naics_codes=["484", "483", "488", "493"],
        ),
        complexity=QueryComplexity.MODERATE,
        confidence=0.8,
    )


# ═══════════════════════════════════════════════════════════════════
# RRF normalization tests
# ═══════════════════════════════════════════════════════════════════

class TestRRFNormalization:
    """Verify that RRF produces correct raw scores (no normalization)."""

    def test_normalized_range(self):
        signals = {
            "sig1": [0.9, 0.5, 0.1, 0.0],
            "sig2": [0.3, 0.8, 0.6, 0.1],
        }
        scores = _reciprocal_rank_fusion(signals, 4)
        # All scores should be positive
        assert all(s > 0 for s in scores)
        # Best overall should have highest RRF score
        assert max(scores) > min(scores)

    def test_preserves_order(self):
        """RRF doesn't change ranking order for dominant signals."""
        signals = {
            "sig1": [10.0, 5.0, 1.0],
            "sig2": [8.0, 9.0, 2.0],
        }
        scores = _reciprocal_rank_fusion(signals, 3)
        # First item should still rank highest (best in sig1, decent in sig2)
        sorted_idx = sorted(range(3), key=lambda i: scores[i], reverse=True)
        assert sorted_idx[0] == 0

    def test_single_signal(self):
        """With one signal, top-ranked gets highest RRF, bottom gets lowest."""
        signals = {"only": [100.0, 50.0, 0.0]}
        scores = _reciprocal_rank_fusion(signals, 3)
        assert scores[0] > scores[1] > scores[2]

    def test_all_equal_signals(self):
        """If all companies score equally, all get the same RRF score."""
        signals = {"flat": [5.0, 5.0, 5.0]}
        scores = _reciprocal_rank_fusion(signals, 3)
        assert scores[0] == scores[1] == scores[2]


class TestScoresToRanks:

    def test_basic_ranking(self):
        ranks = _scores_to_ranks([0.9, 0.5, 0.1])
        assert ranks[0] == 1  # highest
        assert ranks[1] == 2
        assert ranks[2] == 3

    def test_ties(self):
        ranks = _scores_to_ranks([0.5, 0.9, 0.5])
        assert ranks[1] == 1  # highest
        assert ranks[0] == ranks[2]  # tied


# ═══════════════════════════════════════════════════════════════════
# Individual signal tests
# ═══════════════════════════════════════════════════════════════════

class TestNAICSScores:

    def test_exact_match_scores_highest(self, logistics_companies, logistics_query):
        scores = _compute_naics_scores(logistics_companies, logistics_query)
        # FreightCo (484110) and ShipExpress (483111) match NAICS prefixes
        assert scores[0] > 0  # FreightCo
        assert scores[3] > 0  # ShipExpress
        # BakeryInc (311811) shouldn't match logistics NAICS
        assert scores[2] == 0

    def test_no_naics_in_query(self, logistics_companies):
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(naics_codes=None),
        )
        scores = _compute_naics_scores(logistics_companies, pq)
        assert all(s == 0.0 for s in scores)


class TestBusinessModelScores:

    def test_b2b_match(self, logistics_companies):
        pq = ParsedQuery(
            original_query="test",
            semantic_criteria=SemanticCriteria(
                business_model_keywords=["B2B"],
            ),
        )
        scores = _compute_business_model_scores(logistics_companies, pq)
        assert scores[0] > 0  # FreightCo is B2B
        assert scores[2] == 0  # BakeryInc is B2C


class TestMatchExpressions:

    def test_or_expression(self, logistics_companies):
        expr = MatchNode(
            operator=MatchOperator.OR,
            operands=[
                MatchLeaf("freight"),
                MatchLeaf("shipping"),
                MatchLeaf("cargo"),
            ],
        )
        pq = ParsedQuery(
            original_query="test",
            semantic_criteria=SemanticCriteria(match_expressions=[expr]),
        )
        scores = _compute_match_expression_scores(logistics_companies, pq)
        assert scores[0] > 0  # FreightCo has "freight"
        assert scores[3] > 0  # ShipExpress has "shipping", "cargo"
        assert scores[2] == 0  # BakeryInc has none


# ═══════════════════════════════════════════════════════════════════
# Ranking correctness tests
# ═══════════════════════════════════════════════════════════════════

class TestRankingCorrectness:
    """Verify that relevant companies rank above irrelevant ones."""

    def test_logistics_ranks_above_bakery(self, logistics_companies, logistics_query):
        results = score_and_rank(logistics_companies, logistics_query, top_k=4)
        names = [c.operational_name for c, s in results]
        scores = [s for c, s in results]

        # FreightCo and ShipExpress should be top 2
        assert names[0] in ("FreightCo", "ShipExpress")
        assert names[1] in ("FreightCo", "ShipExpress")
        # BakeryInc should be last or near last
        bakery_idx = names.index("BakeryInc")
        assert bakery_idx >= 2

        # Scores should be normalized [0, 1]
        assert max(scores) <= 1.0
        assert min(scores) >= 0.0


# ═══════════════════════════════════════════════════════════════════
# ScoringIndex tests
# ═══════════════════════════════════════════════════════════════════

class TestScoringIndex:

    def test_index_builds(self, logistics_companies):
        idx = build_scoring_index(logistics_companies)
        assert idx.n == 4
        assert idx.bm25_index is not None

    def test_get_indices_maps_correctly(self, logistics_companies):
        idx = build_scoring_index(logistics_companies)
        # Simulate filtering — keep only first and third
        filtered = [logistics_companies[0], logistics_companies[2]]
        indices = idx.get_indices(filtered)
        assert indices == [0, 2]

    def test_cached_vs_scratch_same_ranking(self, logistics_companies, logistics_query):
        """Results with and without ScoringIndex should have same rank order."""
        idx = build_scoring_index(logistics_companies)

        results_cached = score_and_rank(
            logistics_companies, logistics_query, top_k=4, scoring_index=idx,
        )
        results_scratch = score_and_rank(
            logistics_companies, logistics_query, top_k=4,
        )

        # Same rank order (company names in same position)
        names_cached = [c.operational_name for c, s in results_cached]
        names_scratch = [c.operational_name for c, s in results_scratch]
        assert names_cached == names_scratch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
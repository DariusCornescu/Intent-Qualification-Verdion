"""
Stage 3 Tests — LLM Batch Qualification
=========================================
Validates the batching, prompt construction, result parsing, and
graceful degradation logic of Stage 3.

These tests mock the LLM client — they don't make real API calls.
They verify that the *orchestration* is correct: batching, parsing,
fallback on failure.

Run:  python -m pytest tests/test_stage3.py -v
"""

import sys
import os
import json
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.models import CompanyProfile, ParsedQuery, HardFilters, QualificationResult
from code.llm_qualifier import (
    qualify_with_llm, _create_batches, _build_batch_prompt,
    _format_company_profile, _parse_batch_results, BATCH_SIZE,
)


# ═══════════════════════════════════════════════════════════════════
# Test fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def candidates():
    """Top-K candidates from Stage 2 (company, score) tuples."""
    companies = [
        CompanyProfile(
            operational_name=f"Company_{i}",
            description=f"Description for company {i}",
            country_code="us",
            primary_naics={"code": "236220", "label": "Construction"},
            employee_count=100 * (i + 1),
        )
        for i in range(20)
    ]
    return [(c, 0.8 - i * 0.02) for i, c in enumerate(companies)]

@pytest.fixture
def parsed_query():
    return ParsedQuery(
        original_query="Construction companies in the US with revenue over $50M",
    )


# ═══════════════════════════════════════════════════════════════════
# Batching tests
# ═══════════════════════════════════════════════════════════════════

class TestBatching:

    def test_batch_sizes(self, candidates):
        batches = _create_batches(candidates, BATCH_SIZE)
        # 20 candidates / 8 per batch = 3 batches (8, 8, 4)
        assert len(batches) == 3
        assert len(batches[0]) == 8
        assert len(batches[1]) == 8
        assert len(batches[2]) == 4

    def test_single_batch(self):
        small = [(CompanyProfile(operational_name=f"C{i}"), 0.5) for i in range(3)]
        batches = _create_batches(small, BATCH_SIZE)
        assert len(batches) == 1
        assert len(batches[0]) == 3

    def test_exact_batch_size(self):
        exact = [(CompanyProfile(operational_name=f"C{i}"), 0.5) for i in range(BATCH_SIZE)]
        batches = _create_batches(exact, BATCH_SIZE)
        assert len(batches) == 1


# ═══════════════════════════════════════════════════════════════════
# Prompt construction tests
# ═══════════════════════════════════════════════════════════════════

class TestPromptConstruction:

    def test_prompt_contains_query(self, candidates, parsed_query):
        prompt = _build_batch_prompt(candidates[:3], parsed_query)
        assert parsed_query.original_query in prompt

    def test_prompt_contains_companies(self, candidates, parsed_query):
        batch = candidates[:3]
        prompt = _build_batch_prompt(batch, parsed_query)
        for i, (company, _) in enumerate(batch):
            assert company.operational_name in prompt
            assert f"[Company {i}]" in prompt

    def test_prompt_asks_for_json(self, candidates, parsed_query):
        prompt = _build_batch_prompt(candidates[:3], parsed_query)
        assert "JSON" in prompt

    def test_company_profile_formatting(self):
        company = CompanyProfile(
            operational_name="TestCo",
            description="A test company that does testing.",
            country_code="de",
            address="Berlin, Germany",
            primary_naics={"code": "511210", "label": "Software"},
            employee_count=42,
        )
        text = _format_company_profile(company, 0)
        assert "TestCo" in text
        assert "[Company 0]" in text
        assert "42" in text


# ═══════════════════════════════════════════════════════════════════
# Result parsing tests
# ═══════════════════════════════════════════════════════════════════

class TestResultParsing:

    def test_parse_valid_results(self, candidates):
        batch = candidates[:3]
        llm_results = {
            "results": [
                {"index": 0, "qualified": True, "confidence": 0.9, "reasoning": "Direct match"},
                {"index": 1, "qualified": False, "confidence": 0.2, "reasoning": "Wrong industry"},
                {"index": 2, "qualified": True, "confidence": 0.7, "reasoning": "Partial match"},
            ]
        }
        results = _parse_batch_results(batch, llm_results)
        assert len(results) == 3
        assert results[0].qualified is True
        assert results[1].qualified is False
        assert results[0].reasoning == "Direct match"
        assert results[0].stage_reached == "llm_qualified"

    def test_missing_llm_result_falls_back(self, candidates):
        """If LLM doesn't return a result for a company, use soft score."""
        batch = candidates[:3]
        llm_results = {
            "results": [
                {"index": 0, "qualified": True, "confidence": 0.9, "reasoning": "OK"},
                # Missing index 1 and 2
            ]
        }
        results = _parse_batch_results(batch, llm_results)
        assert len(results) == 3
        assert results[0].stage_reached == "llm_qualified"
        assert results[1].stage_reached == "soft_score_fallback"
        assert results[2].stage_reached == "soft_score_fallback"


# ═══════════════════════════════════════════════════════════════════
# Graceful degradation tests
# ═══════════════════════════════════════════════════════════════════

class TestGracefulDegradation:

    def test_llm_failure_falls_back(self, candidates, parsed_query):
        """When LLM call fails, results fall back to soft scores."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API timeout")

        results = qualify_with_llm(candidates[:3], parsed_query, mock_client)
        assert len(results) == 3
        assert all(r.stage_reached == "soft_score_fallback" for r in results)

    def test_successful_qualification(self, candidates, parsed_query):
        """Mock a successful LLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "results": [
                {"index": i, "qualified": True, "confidence": 0.8,
                 "reasoning": f"Match for company {i}"}
                for i in range(min(BATCH_SIZE, len(candidates[:3])))
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        results = qualify_with_llm(candidates[:3], parsed_query, mock_client)
        assert len(results) == 3
        assert all(r.qualified for r in results)
        assert all(r.stage_reached == "llm_qualified" for r in results)

    def test_results_sorted_by_score(self, candidates, parsed_query):
        """Results should come back sorted by score descending."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "results": [
                {"index": 0, "qualified": True, "confidence": 0.3, "reasoning": "low"},
                {"index": 1, "qualified": True, "confidence": 0.9, "reasoning": "high"},
                {"index": 2, "qualified": True, "confidence": 0.6, "reasoning": "mid"},
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        results = qualify_with_llm(candidates[:3], parsed_query, mock_client)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
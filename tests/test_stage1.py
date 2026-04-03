"""
Stage 1 Tests — Hard Filters
==============================
Validates that Stage 1 correctly filters companies based on structured
constraints from Stage 0: location, revenue, employees, NAICS codes,
year founded, public/private, and geo proximity.

Critical test: the US location bug — verifying that country_code="us"
correctly matches location="united states" even though "us" is filtered
from COUNTRY_ALIASES as an ambiguous English pronoun.

Run:  python -m pytest tests/test_stage1.py -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.models import CompanyProfile, ParsedQuery, HardFilters, QueryComplexity
from code.hard_filters import apply_hard_filters, _check_location, _check_naics


# ═══════════════════════════════════════════════════════════════════
# Test fixtures — reusable company profiles
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def us_construction():
    """US construction company with full data."""
    return CompanyProfile(
        operational_name="BuildCo",
        country_code="us",
        region_name="California",
        town="Los Angeles",
        latitude=34.05,
        longitude=-118.24,
        employee_count=500,
        revenue=75_000_000,
        year_founded=2005,
        is_public=False,
        primary_naics={"code": "236220", "label": "Commercial Building Construction"},
    )

@pytest.fixture
def de_software():
    """German software company."""
    return CompanyProfile(
        operational_name="SoftDE",
        country_code="de",
        employee_count=200,
        revenue=10_000_000,
        year_founded=2015,
        is_public=False,
        primary_naics={"code": "511210", "label": "Software Publishers"},
    )

@pytest.fixture
def sparse_company():
    """Company with minimal data — tests missing data tolerance."""
    return CompanyProfile(
        operational_name="MysteryInc",
        description="We do things.",
    )

@pytest.fixture
def sample_companies(us_construction, de_software, sparse_company):
    return [us_construction, de_software, sparse_company]


# ═══════════════════════════════════════════════════════════════════
# Location filter tests
# ═══════════════════════════════════════════════════════════════════

class TestLocationFilter:
    """Test _check_location with various country code / location name combos."""

    def test_us_matches_united_states(self, us_construction):
        """CRITICAL: The bug that caused 0 results for US construction query."""
        assert _check_location(us_construction, ["united states"]) is True

    def test_us_matches_usa(self, us_construction):
        assert _check_location(us_construction, ["usa"]) is True

    def test_de_matches_germany(self, de_software):
        assert _check_location(de_software, ["germany"]) is True

    def test_de_does_not_match_france(self, de_software):
        assert _check_location(de_software, ["france"]) is False

    def test_missing_location_passes(self, sparse_company):
        """Companies with no location data pass (missing data tolerance)."""
        assert _check_location(sparse_company, ["germany"]) is True

    def test_region_name_match(self, us_construction):
        """Fallback: substring match on region_name."""
        assert _check_location(us_construction, ["california"]) is True

    def test_multiple_locations_or(self, us_construction, de_software):
        """Company passes if it matches ANY of the target locations."""
        locs = ["germany", "united states"]
        assert _check_location(us_construction, locs) is True
        assert _check_location(de_software, locs) is True


# ═══════════════════════════════════════════════════════════════════
# NAICS filter tests
# ═══════════════════════════════════════════════════════════════════

class TestNAICSFilter:

    def test_exact_prefix_match(self, us_construction):
        """NAICS code 236220 starts with prefix '23'."""
        assert _check_naics(us_construction, ["23"]) is True

    def test_longer_prefix_match(self, us_construction):
        assert _check_naics(us_construction, ["2362"]) is True

    def test_no_match(self, us_construction):
        """Software NAICS prefix doesn't match construction company."""
        assert _check_naics(us_construction, ["5112"]) is False

    def test_missing_naics_passes(self, sparse_company):
        """Company with no NAICS codes passes (missing data tolerance)."""
        assert _check_naics(sparse_company, ["23"]) is True


# ═══════════════════════════════════════════════════════════════════
# Revenue filter tests
# ═══════════════════════════════════════════════════════════════════

class TestRevenueFilter:

    def test_revenue_min(self, sample_companies):
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(revenue_min=50_000_000),
        )
        results = apply_hard_filters(sample_companies, pq)
        names = [c.operational_name for c in results]
        assert "BuildCo" in names  # 75M > 50M
        assert "SoftDE" not in names  # 10M < 50M
        assert "MysteryInc" in names  # None → passes

    def test_revenue_max(self, sample_companies):
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(revenue_max=20_000_000),
        )
        results = apply_hard_filters(sample_companies, pq)
        names = [c.operational_name for c in results]
        assert "SoftDE" in names  # 10M < 20M
        assert "BuildCo" not in names  # 75M > 20M


# ═══════════════════════════════════════════════════════════════════
# Employee count filter tests
# ═══════════════════════════════════════════════════════════════════

class TestEmployeeFilter:

    def test_employee_min(self, sample_companies):
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(employee_count_min=300),
        )
        results = apply_hard_filters(sample_companies, pq)
        names = [c.operational_name for c in results]
        assert "BuildCo" in names  # 500 >= 300
        assert "SoftDE" not in names  # 200 < 300

    def test_missing_employee_passes(self, sparse_company):
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(employee_count_min=1000),
        )
        results = apply_hard_filters([sparse_company], pq)
        assert len(results) == 1  # Missing data → passes


# ═══════════════════════════════════════════════════════════════════
# NAICS relaxation tests
# ═══════════════════════════════════════════════════════════════════

class TestNAICSRelaxation:

    def test_relaxation_triggers(self, sample_companies):
        """When NAICS filter eliminates too many, it should relax."""
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(
                naics_codes=["9999"],  # Matches nothing
            ),
        )
        results = apply_hard_filters(sample_companies, pq)
        # With relaxation, should get back more results than 0
        # (MysteryInc passes because missing NAICS tolerance)
        assert pq.naics_was_relaxed is True

    def test_no_relaxation_when_enough_results(self):
        """NAICS relaxation should NOT trigger when enough companies pass."""
        companies = [
            CompanyProfile(
                operational_name=f"ConstructCo{i}",
                primary_naics={"code": "236220", "label": "Construction"},
            )
            for i in range(20)
        ]
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(naics_codes=["2362"]),
        )
        results = apply_hard_filters(companies, pq)
        assert len(results) == 20
        assert pq.naics_was_relaxed is False


# ═══════════════════════════════════════════════════════════════════
# Combined filter tests (the actual pipeline query)
# ═══════════════════════════════════════════════════════════════════

class TestCombinedFilters:

    def test_construction_us_revenue(self, us_construction, de_software, sparse_company):
        """Simulate: 'Construction companies in US with revenue > $50M'."""
        pq = ParsedQuery(
            original_query="Construction companies in the US with revenue over $50 million",
            hard_filters=HardFilters(
                locations=["united states"],
                revenue_min=50_000_000,
                naics_codes=["23"],
            ),
        )
        companies = [us_construction, de_software, sparse_company]
        results = apply_hard_filters(companies, pq)
        names = [c.operational_name for c in results]
        # BuildCo: US ✓, revenue 75M ✓, NAICS 236220 ✓ → passes
        assert "BuildCo" in names
        # SoftDE: DE ✗ → eliminated by location
        assert "SoftDE" not in names

    def test_all_filters_together(self):
        company = CompanyProfile(
            operational_name="PerfectMatch",
            country_code="de",
            employee_count=150,
            revenue=5_000_000,
            year_founded=2019,
            is_public=False,
            primary_naics={"code": "511210", "label": "Software"},
        )
        pq = ParsedQuery(
            original_query="test",
            hard_filters=HardFilters(
                locations=["germany"],
                employee_count_min=100,
                employee_count_max=500,
                revenue_min=1_000_000,
                year_founded_min=2018,
                is_public=False,
                naics_codes=["5112"],
            ),
        )
        results = apply_hard_filters([company], pq)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════
# Real dataset tests
# ═══════════════════════════════════════════════════════════════════

class TestWithRealData:
    """Tests against the actual companies.jsonl dataset."""

    @pytest.fixture
    def real_companies(self):
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "companies.jsonl")
        if not os.path.exists(data_path):
            data_path = os.path.join(os.path.dirname(__file__), "..", "companies.jsonl")
        if not os.path.exists(data_path):
            pytest.skip("companies.jsonl not found")
        with open(data_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]
        return [CompanyProfile.from_dict(c) for c in raw]

    def test_us_construction_revenue(self, real_companies):
        """The bug query: should return ~20 companies."""
        pq = ParsedQuery(
            original_query="Construction companies in the US with revenue over $50 million",
            hard_filters=HardFilters(
                locations=["united states"],
                revenue_min=50_000_000,
                naics_codes=["23"],
            ),
        )
        results = apply_hard_filters(real_companies, pq)
        assert len(results) >= 15, f"Expected ≥15, got {len(results)}"

    def test_logistics_romania(self, real_companies):
        pq = ParsedQuery(
            original_query="Logistics companies in Romania",
            hard_filters=HardFilters(
                locations=["romania"],
                naics_codes=["48", "49"],
            ),
        )
        results = apply_hard_filters(real_companies, pq)
        assert len(results) >= 1

    def test_no_filter_returns_all(self, real_companies):
        pq = ParsedQuery(
            original_query="all companies",
            hard_filters=HardFilters(),
        )
        results = apply_hard_filters(real_companies, pq)
        assert len(results) == len(real_companies)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
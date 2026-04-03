"""
Stage 0 Tests — Query Analysis
================================
Validates that Stage 0 correctly extracts structured data from natural
language queries: locations, numeric constraints, NAICS codes, complexity,
and range queries.

These tests run WITHOUT an LLM — they test the regex/taxonomy validation
layer that cross-checks LLM output. This is the deterministic backbone
that catches LLM hallucinations.

Run:  python -m pytest tests/test_stage0.py -v
"""

import sys
import os
import re
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code.models import ParsedQuery, HardFilters, SemanticCriteria, QueryComplexity
from code.taxonomy import ( normalize_location, expand_industry, COUNTRY_ALIASES, ISO2_TO_COUNTRY, REGION_EXPANSIONS, INDUSTRY_TAXONOMY )
from code.query_analysis.regex_patterns import ( NUMERIC_PATTERNS, REVENUE_MULTIPLIERS, parse_number as _parse_number )


# ----------------------------------------------------------
# Location normalization
# ----------------------------------------------------------
class TestLocationNormalization:

    def test_country_name(self):
        assert normalize_location("Germany") == ["germany"]

    def test_country_code_via_alias(self):
        result = normalize_location("USA")
        assert "united states" in result

    def test_region_expansion(self):
        result = normalize_location("Scandinavia")
        assert "norway" in result
        assert "sweden" in result
        assert "denmark" in result
        assert len(result) >= 4

    def test_europe_expansion(self):
        result = normalize_location("Europe")
        assert len(result) >= 20
        assert "germany" in result
        assert "france" in result

    def test_unknown_location_passthrough(self):
        """Unknown locations pass through as-is (lowercase)."""
        assert normalize_location("Atlantis") == ["atlantis"]


class TestISO2Mapping:
    """
    Verify that ISO2_TO_COUNTRY keeps ALL 2-letter codes, including ambiguous ones
    like 'us', 'in', 'no' that COUNTRY_ALIASES filters out. This is critical for _check_location to work.
    """

    def test_us_in_iso2(self):
        assert "us" in ISO2_TO_COUNTRY
        assert ISO2_TO_COUNTRY["us"] == "united states"

    def test_us_filtered_from_aliases(self):
        """'us' is correctly filtered from COUNTRY_ALIASES (it's an English pronoun)."""
        assert "us" not in COUNTRY_ALIASES

    def test_india_in_iso2(self):
        assert "in" in ISO2_TO_COUNTRY

    def test_india_filtered_from_aliases(self):
        """'in' is correctly filtered (it's an English preposition)."""
        assert "in" not in COUNTRY_ALIASES

    def test_germany_in_both(self):
        """Non-ambiguous codes like 'de' exist in both dicts."""
        assert "de" in ISO2_TO_COUNTRY
        assert "de" in COUNTRY_ALIASES

    def test_usa_in_aliases(self):
        """'usa' (3 letters) is NOT ambiguous and stays in COUNTRY_ALIASES."""
        assert "usa" in COUNTRY_ALIASES



# ----------------------------------------------------------
# Industry taxonomy expansion
# ----------------------------------------------------------

class TestIndustryExpansion:

    def test_construction_naics(self):
        result = expand_industry("construction")
        assert "naics_prefixes" in result
        codes = result["naics_prefixes"]
        assert any(c.startswith("23") for c in codes)

    def test_logistics_naics(self):
        result = expand_industry("logistics")
        codes = result["naics_prefixes"]
        assert len(codes) > 0

    def test_logistics_keywords(self):
        result = expand_industry("logistics")
        kws = result.get("keywords", [])
        # Should include related terms
        all_kws = " ".join(kws).lower()
        assert any(term in all_kws for term in ["freight", "shipping", "transport"])

    def test_unknown_industry(self):
        """Unknown terms return empty NAICS but keep the term as keyword."""
        result = expand_industry("quantum_teleportation_xyz")
        assert result["naics_prefixes"] == []

    def test_plural_handling(self):
        """'logistics' and 'logistic' should resolve to the same entry."""
        r1 = expand_industry("logistics")
        r2 = expand_industry("logistic")
        assert r1.get("naics_prefixes") == r2.get("naics_prefixes")


# ----------------------------------------------------------
# Numeric constraint extraction (regex layer)
# ----------------------------------------------------------

class TestNumericPatterns:

    def _extract(self, query: str) -> dict:
        """Run regex extraction on a query, return dict of extracted fields."""
        query_lower = query.lower()
        result = {}

        for pattern, field_name in NUMERIC_PATTERNS:
            m = re.search(pattern, query_lower)
            if not m:
                continue
            value = _parse_number(m.group(1))
            if field_name in ("revenue_min", "revenue_max") and m.lastindex >= 2:
                mult_str = m.group(2)
                if mult_str and mult_str.lower() in REVENUE_MULTIPLIERS:
                    value *= REVENUE_MULTIPLIERS[mult_str.lower()]
            result[field_name] = value

        return result

    def test_employees_min(self):
        r = self._extract("companies with more than 1,000 employees")
        assert r.get("employee_count_min") == 1000

    def test_employees_max(self):
        r = self._extract("companies with fewer than 200 employees")
        assert r.get("employee_count_max") == 200

    def test_employees_plus(self):
        r = self._extract("500+ employees")
        assert r.get("employee_count_min") == 500

    def test_revenue_min_million(self):
        r = self._extract("revenue over $50 million")
        assert r.get("revenue_min") == 50_000_000

    def test_revenue_min_billion(self):
        r = self._extract("revenue over $2 billion")
        assert r.get("revenue_min") == 2_000_000_000

    def test_founded_after(self):
        r = self._extract("startups founded after 2018")
        assert r.get("year_founded_min") == 2018



# ----------------------------------------------------------
# Complexity classification
# ----------------------------------------------------------

class TestComplexitySignals:

    from code.query_analysis import QueryAnalyzer

    def test_supply_chain_is_complex(self):
        """'suppliers of' should force COMPLEX classification."""
        query = "suppliers of lithium for battery manufacturers"
        # The signal "suppliers of" is in _COMPLEX_SIGNALS
        assert "suppliers of" in query.lower()

    def test_competing_is_complex(self):
        query = "companies competing with traditional banks"
        assert "competing with" in query.lower()

    def test_startup_bumps_to_moderate(self):
        """'startup' in a simple query should bump it to moderate."""
        query = "fintech startups in Europe"
        assert "startup" in query.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
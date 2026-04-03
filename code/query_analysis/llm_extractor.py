"""
LLM-based query extraction for Stage 0.

Handles communication with the LLM API: prompt template, API call with caching and timeout, and 
mapping the raw LLM JSON output onto ParsedQuery fields.

The LLM is the primary extractor - it sees the raw query and returns structured JSON. Downstream 
validators then cross-check and normalize this output against deterministic rules (regex, taxonomy).
"""

import json
import logging
from typing import Optional
from code.models import ParsedQuery, QueryComplexity

logger = logging.getLogger(__name__)


LLM_EXTRACTION_PROMPT = """
You are a query parser for a company search engine. Extract structured information from the user's query.

Return ONLY valid JSON with these fields (use null for fields not mentioned in the query):
{
  "locations": ["list of countries, regions, or cities mentioned — use EXACT names like 'Europe', 'Scandinavia', 'Southeast Asia', 'United States'"],
  "excluded_locations": ["locations explicitly EXCLUDED by the query, e.g. 'not in Germany' → ['Germany']. null if no exclusion"],
  "geo_reference": "city or place name if spatial proximity is implied (e.g. 'near Boston'), or null",
  "geo_modifier": "the spatial phrase used (e.g. 'near', 'close to', 'not far from') or null",
  "geo_radius_km": "estimated radius in km based on the modifier, or null",
  "employee_count_min": "number or null",
  "employee_count_max": "number or null",
  "revenue_min": "number in USD or null",
  "revenue_max": "number in USD or null",
  "year_founded_min": "number or null",
  "year_founded_max": "number or null",
  "is_public": "true/false/null",
  "industry_keywords": ["list of relevant industry terms the query WANTS — see rules below"],
  "excluded_industries": ["industries explicitly EXCLUDED, e.g. 'NOT in construction' → ['construction']. null if no exclusion"],
  "business_model_keywords": ["B2B", "B2C", "SaaS", "marketplace", "D2C", etc. or empty list],
  "complexity": "simple | moderate | complex — see classification rules below"
}

===== INDUSTRY KEYWORDS RULES =====

- Return ONLY 5-10 keywords directly related to the CORE industry in the query
- Quality over quantity: 5 focused keywords > 20 scattered ones
- Include the exact industry term + direct synonyms ONLY
- Do NOT pad with tangentially related terms from other industries
- WRONG for "Construction companies": returning "cloud services", "renewable energy", "platform as a service"
- RIGHT for "Construction companies": "construction", "building", "general contracting", "civil engineering", "infrastructure", "commercial construction"
- WRONG for "Pharmaceutical companies": returning "data analytics", "cloud computing"
- RIGHT for "Pharmaceutical companies": "pharmaceutical", "pharma", "drug manufacturing", "therapeutics", "biotech", "clinical trials"
- Every keyword you return should pass this test: "Would a company with this keyword in their description be in the industry the user asked about?"

===== COMPLEXITY CLASSIFICATION (critical — read carefully) =====

"simple" -> The query specifies ONLY concrete, filterable attributes with NO qualitative or subjective criteria. Every condition maps directly to a database field (country, employee count, revenue, year founded, public/private, industry sector).
  EXAMPLES of simple:
  - "Biotech companies in Germany" → location + industry = simple
  - "Publicly traded companies with revenue over $1 billion" → public + revenue = simple
  - "Private companies in Japan with fewer than 500 employees" → location + private + employee count = simple
  - "Chemical manufacturers in the Netherlands" → location + industry = simple

"moderate" — The query mentions a specific industry/sector AND at least one qualitative descriptor that needs semantic matching against company descriptions, but does NOT require reasoning about indirect relationships. The qualitative part goes beyond simple filters but can still be matched by checking what a company says about itself.
  EXAMPLES of moderate:
  - "Semiconductor companies in Asia focused on automotive chips" → industry + location + "focused on automotive" needs description matching = moderate
  - "B2C retail brands in Spain with an online presence" → business model + location + "online presence" needs semantic check = moderate
  - "Sustainable agriculture startups in Northern Europe" → industry + "sustainable" + "startups" qualifiers need description matching = moderate
  - "Consulting firms in the UK specializing in cybersecurity" → industry + location + "specializing in cybersecurity" = moderate
  - "Wholesale distributors in Canada serving the healthcare sector" → industry + location + "serving healthcare" = moderate

"complex" — The query requires REASONING about indirect relationships, supply chains, competitive dynamics, or implicit criteria that cannot be directly read from a company's own profile. The answer depends on understanding how companies relate to EACH OTHER or to external concepts.
  EXAMPLES of complex:
  - "Potential suppliers of lithium for battery manufacturers in China" → supply chain reasoning: who SUPPLIES raw materials TO battery makers = complex
  - "Companies that could replace Russian gas suppliers for Western European utilities" → competitive substitution + geopolitical reasoning = complex
  - "Firms whose services would benefit a hospital expanding into telemedicine" → indirect relationship: who could HELP a hospital = complex

KEY DECISION TEST: Ask yourself — can I answer this query by ONLY checking a company's own structured fields (country, size, industry code, public status)? If yes → simple. Does it also need checking the company's description text for specific capabilities or sub-sectors? → moderate. Does it need reasoning about relationships between companies, supply chains, competition, or capabilities not stated in their profile? → complex.

=== OTHER RULES ===
- For revenue, always convert to raw USD (e.g., "$50 million" → 50000000). Handle €/£ by converting at approximate rate.
- LOCATIONS (critical): Return the EXACT location phrase from the query as a single array element. NEVER expand regions into individual countries.
  CORRECT: "companies in Europe" → locations: ["Europe"]
  CORRECT: "companies in Southeast Asia" → locations: ["Southeast Asia"]
  CORRECT: "companies in Scandinavia" → locations: ["Scandinavia"]
  CORRECT: "companies in the United States" → locations: ["United States"]
  WRONG: "companies in Europe" → locations: ["France", "Germany", "Italy", ...]
  WRONG: "companies in Southeast Asia" → locations: ["Singapore", "Thailand", ...]

  The downstream system has a taxonomy that expands regions into countries. You MUST NOT do this yourself.
- Do NOT invent constraints that aren't in the query
- NEGATION (critical): Watch for exclusion language like "NOT in", "excluding", "except", "other than", "outside of", "non-", "apart from". When detected:
  - "NOT in construction" → industry_keywords: [], excluded_industries: ["construction"]
  - "companies in Europe excluding Germany" → locations: ["Europe"], excluded_locations: ["Germany"]
  - "non-public companies" → is_public: false (this is a direct filter, not an exclusion)

  Do NOT put excluded terms in industry_keywords — they go ONLY in excluded_industries/excluded_locations.
- SPATIAL PROXIMITY: When the query uses phrases like "near", "close to", "not far from", "around", "in the area of", "within X km of", "just outside", "in the greater area of", 
"surrounding", "based around", "headquartered near", etc., set geo_reference to the place name and geo_modifier to the phrase used. Estimate geo_radius_km: "near" → 50, "close to" → 50, 
"not far from" → 100, "in the greater area" → 75, "within driving distance" → 100, explicit distances like "within 50km" → 50. If NO spatial modifier is used (just "in Germany" or "in Boston"),
set all geo fields to null — it's a regular location filter, not proximity.

"""

CACHE_MAX = 200   # Max cached queries (LRU-style eviction)
LLM_TIMEOUT = 15  # Seconds before giving up on LLM call


class LLMExtractor:
    """
    Manages the API call, response caching (temperature=0 -> deterministic, safe to cache), and mapping raw JSON output to ParsedQuery fields.
    """

    def __init__(self, llm_client, llm_model: str = "gpt-4o-mini"):
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._cache: dict[str, dict] = {}

    def extract(self, query: str) -> Optional[dict]:
        """
        Call LLM to extract structured JSON from the raw query.
        Returns parsed dict or None on failure.

        Uses in-memory cache (temperature=0 -> deterministic output) and a timeout to prevent slow API calls from blocking.
        """

        cache_key = query.strip().lower()
        if cache_key in self._cache:
            logger.debug(f"LLM cache hit: {query[:50]}")
            return self._cache[cache_key]

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": LLM_EXTRACTION_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT,
            )
            raw = response.choices[0].message.content
            result = json.loads(raw)
            logger.debug(f"LLM extraction: {json.dumps(result, indent=2)}")

            # Cache the result (evict oldest if full)
            if len(self._cache) >= CACHE_MAX:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return None

    def apply_extraction(self, parsed: ParsedQuery, llm_output: dict):
        """
        Apply the raw LLM output to ParsedQuery fields (before validation).
        Sets numerics, locations, keywords, complexity, and geo fields.
        """
        trace = parsed.trace
        hard_filter = parsed.hard_filters

        # Locations (raw from LLM, will be normalized in validation step)
        raw_locations = llm_output.get("locations") or []
        if raw_locations:
            trace.add("stage_0", "llm_locations", str(raw_locations[:5]))

        for field in ("employee_count_min", "employee_count_max", "year_founded_min", "year_founded_max"):
            val = llm_output.get(field)
            if val is not None:
                try:
                    setattr(hard_filter, field, int(val))
                except (ValueError, TypeError):
                    pass

        for field in ("revenue_min", "revenue_max"):
            val = llm_output.get(field)
            if val is not None:
                try:
                    setattr(hard_filter, field, float(val))
                except (ValueError, TypeError):
                    pass

        is_public = llm_output.get("is_public")
        if is_public is not None:
            hard_filter.is_public = bool(is_public)
            trace.add("stage_0", "is_public", str(hard_filter.is_public))

        # Industry keywords (raw from LLM, will be expanded via taxonomy)
        raw_keywords = llm_output.get("industry_keywords") or []
        if raw_keywords:
            parsed.semantic_criteria.industry_keywords = [
                str(key_word) for key_word in raw_keywords if isinstance(key_word, str)
            ]

        excluded_ind = llm_output.get("excluded_industries") or []
        if excluded_ind:
            trace.add("stage_0", "negation", f"excluded industries: {excluded_ind}")

        excluded_loc = llm_output.get("excluded_locations") or []
        if excluded_loc:
            trace.add("stage_0", "negation", f"excluded locations: {excluded_loc}")

        complexity_str = llm_output.get("complexity", "moderate")
        complexity_map = {
            "simple": QueryComplexity.SIMPLE,
            "moderate": QueryComplexity.MODERATE,
            "complex": QueryComplexity.COMPLEX,
        }
        parsed.complexity = complexity_map.get(complexity_str, QueryComplexity.MODERATE)

        # Geo proximity — "near Boston" → geocode + radius
        geo_ref = llm_output.get("geo_reference")
        geo_mod = llm_output.get("geo_modifier")
        if geo_ref:
            trace.add("stage_0", "geo_intent", f"ref={geo_ref}, modifier={geo_mod}, "f"llm_radius={llm_output.get('geo_radius_km')}")
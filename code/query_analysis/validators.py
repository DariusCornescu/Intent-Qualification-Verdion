"""
Validation and normalization functions for Stage 0 query analysis.

These functions take the raw LLM extraction output and cross-check it against deterministic sources: 
regex patterns for numerics, taxonomy dictionaries for locations and industries, and heuristic rules for
complexity classification

Each function is a pure transformation: (query, parsed, llm_output) -> mutates parsed ( deterministic validation )
"""

import re
import logging
from code.models import ( ParsedQuery, MatchNode, MatchLeaf, MatchOperator, QueryComplexity )
from code.taxonomy import ( normalize_location, expand_industry, REGION_EXPANSIONS, INDUSTRY_TAXONOMY, BUSINESS_MODEL_KEYWORDS )
from code.query_analysis.regex_patterns import ( extract_numeric_constraints, detect_public_private, detect_business_model )

logger = logging.getLogger(__name__)


def validate_numerics_with_regex(query: str, parsed: ParsedQuery, llm_output: dict):
    """
    Run regex on the original query to cross-check LLM's numeric extractions
        > If regex finds something LLM missed -> add it
        > If regex and LLM disagree on a value -> regex wins (it's deterministic)
    """

    trace = parsed.trace
    regex_parsed = ParsedQuery(original_query=query)
    extract_numeric_constraints(query, regex_parsed)

    regex_hard_filters = regex_parsed.hard_filters
    hard_filters = parsed.hard_filters
    corrections = []

    # Cross-checking each numeric field
    for field in ("employee_count_min", "employee_count_max", "year_founded_min", "year_founded_max", "revenue_min", "revenue_max"):
        regex_val = getattr(regex_hard_filters, field)
        llm_val = getattr(hard_filters, field)

        if regex_val is not None and llm_val is None:
            setattr(hard_filters, field, regex_val)
            corrections.append(f"{field}: added {regex_val} (LLM missed)")
        elif regex_val is not None and llm_val is not None and regex_val != llm_val:
            setattr(hard_filters, field, regex_val)
            corrections.append(f"{field}: {llm_val}→{regex_val} (regex override)")

    # Cross-checking also public/private field with regex
    regex_pub = ParsedQuery(original_query=query)
    detect_public_private(query, regex_pub)
    if regex_pub.hard_filters.is_public is not None and hard_filters.is_public is None:
        hard_filters.is_public = regex_pub.hard_filters.is_public
        corrections.append(f"is_public: added {hard_filters.is_public} (regex found)")

    if corrections:
        trace.add("stage_0", "regex_validate", "; ".join(corrections))
    else:
        trace.add("stage_0", "regex_validate", "all numerics confirmed")

    numerics = []
    if hard_filters.employee_count_min: numerics.append(f"emp≥{hard_filters.employee_count_min}")
    if hard_filters.employee_count_max: numerics.append(f"emp≤{hard_filters.employee_count_max}")
    if hard_filters.revenue_min: numerics.append(f"rev≥{hard_filters.revenue_min:,.0f}")
    if hard_filters.revenue_max: numerics.append(f"rev≤{hard_filters.revenue_max:,.0f}")
    if hard_filters.year_founded_min: numerics.append(f"yr≥{hard_filters.year_founded_min}")
    if hard_filters.year_founded_max: numerics.append(f"yr≤{hard_filters.year_founded_max}")
    trace.add("stage_0", "numerics", ", ".join(numerics) if numerics else "none")


def validate_locations_with_taxonomy(parsed: ParsedQuery, llm_output: dict):
    """
    Normalize LLM-extracted locations through COUNTRY_ALIASES and REGION_EXPANSIONS.
        > "Scandinavia" -> ["norway", "sweden", "denmark", "finland", "iceland"]
        > "Germany" -> ["germany"]

    When geo_reference is present (something like "near Boston" ), geocode it to lat/lng and set proximity filter instead of country filter.
    """
    trace = parsed.trace
    llm_output = llm_output or {}
    geo_reference = llm_output.get("geo_reference")

    if geo_reference:

        from code.geocoding import geocode, infer_radius_km

        coords = geocode(geo_reference)
        if coords:
            latitude, longitude = coords
            radius = llm_output.get("geo_radius_km")
            if radius is None:
                geo_mod = llm_output.get("geo_modifier") or ""
                radius = infer_radius_km(geo_mod)

            parsed.hard_filters.geo_lat = latitude
            parsed.hard_filters.geo_lng = longitude
            parsed.hard_filters.geo_radius_km = float(radius)

            trace.add("stage_0", "geo_filter", f"{geo_reference} -> ({latitude:.4f}, {longitude:.4f}), radius={radius}km")

            # When geo filter is active, we are not setting country-level locations cause proximity replaces the broad country filter
            return
        else:
            trace.add("stage_0", "geo_filter", f"{geo_reference} -> could not geocode, falling back to country filter")


    raw_locations = llm_output.get("locations") or []

    if not raw_locations:
        trace.add("stage_0", "locations", "none detected")
        return

    all_locations = []
    for location in raw_locations:
        normalized = normalize_location(location)
        all_locations.extend(normalized)

    if all_locations:
        parsed.hard_filters.locations = list(set(all_locations))
        trace.add("stage_0", "locations", f"{raw_locations} -> {parsed.hard_filters.locations[:5]}")
    else:
        trace.add("stage_0", "locations", f"{raw_locations} -> could not normalize")

    raw_excluded = llm_output.get("excluded_locations") or []
    if raw_excluded:
        excluded = []
        for location in raw_excluded:
            excluded.extend(normalize_location(location))
        if excluded:

            parsed.hard_filters.excluded_locations = list(set(excluded))
            # Remove excluded from included locations
            if parsed.hard_filters.locations:
                parsed.hard_filters.locations = [
                    location for location in parsed.hard_filters.locations
                    if location not in parsed.hard_filters.excluded_locations
                ]
                if not parsed.hard_filters.locations:
                    parsed.hard_filters.locations = None

            trace.add("stage_0", "excluded_locations", f"{raw_excluded} -> {parsed.hard_filters.excluded_locations}")


def fix_region_from_query(query: str, parsed: ParsedQuery):
    """
    Safety net: if the original query mentions a known region name (Europe, Scandinavia, Southeast Asia, etc.) but the LLM expanded
    it into individual countries instead of passing it through, override with the full taxonomy expansion.

    Respects excluded_locations: if "not in Denmark" was detected, Denmark stays excluded even after region expansion.
    """
    query_lower = query.lower()
    trace = parsed.trace
    excluded = set(parsed.hard_filters.excluded_locations or [])

    for region_name in REGION_EXPANSIONS:
        # I'm using word boundary matching, not substring because without \b, "eu" matches inside "pharmaceutical", "asia" matches inside "Eurasian"
        if not re.search(r'\b' + re.escape(region_name) + r'\b', query_lower):
            continue

        full_expansion = normalize_location(region_name)
        if excluded:
            full_expansion = [location for location in full_expansion if location not in excluded]

        current = parsed.hard_filters.locations or []

        # If taxonomy gives more countries than what LLM returned, the LLM probably did a partial/wrong expansion
        if len(full_expansion) > len(current):
            parsed.hard_filters.locations = list(set(full_expansion))
            trace.add("stage_0", "region_fix", f"'{region_name}' in query -> override to {len(full_expansion)} countries (was {len(current)})" + (f", kept {len(excluded)} excluded" if excluded else ""))

        break 


def expand_industry_with_taxonomy(parsed: ParsedQuery, llm_output: dict):
    """
    Two separate jobs:
    1. NAICS codes (for hard filter): extracted from the ORIGINAL QUERY matched against the taxonomy. This prevents junk LLM keywords
       like "cloud services" from injecting wrong NAICS codes.
    2. Keywords (for soft scoring): expanded from ALL LLM keywords through the taxonomy. Junk keywords are harmless in soft scoring
       because they just don't match construction company descriptions.
    """
    trace = parsed.trace
    query_lower = parsed.original_query.lower()
    llm_keywords = llm_output.get("industry_keywords") or []

    query_naics = []
    query_matched_industries = []

    for industry_name in INDUSTRY_TAXONOMY:
        if industry_name in query_lower:
            expansion = expand_industry(industry_name)
            codes = expansion.get("naics_prefixes", [])
            for code in codes:
                if isinstance(code, list):
                    query_naics.extend(str(c) for c in code)
                else:
                    query_naics.append(str(code))
            query_matched_industries.append(industry_name)

    # Also try individual words from the query through expand_industry
    if not query_naics:
        query_words = query_lower.split()
        stop_words = { "the", "a", "an", "in", "on", "at", "for", "with", "and", "or", "of", "to", "than", "more", "less",
                      "fewer", "over", "under", "companies", "company", "that", "which", "are", "is", "find", "show", "list" }
        
        for word in query_words:
            if word in stop_words or len(word) < 3:
                continue

            expansion = expand_industry(word)
            if expansion.get("naics_prefixes"):
                codes = expansion["naics_prefixes"]
                for code in codes:
                    if isinstance(code, list):
                        query_naics.extend(str(c) for c in code)
                    else:
                        query_naics.append(str(code))
                query_matched_industries.append(word)

    if query_naics:
        parsed.hard_filters.naics_codes = list(set(query_naics))
        trace.add("stage_0", "naics_source", f"from query text: {query_matched_industries} -> {len(parsed.hard_filters.naics_codes)} codes")

    all_keywords = list(parsed.semantic_criteria.industry_keywords)

    for keyword in llm_keywords:
        expansion = expand_industry(keyword)
        expanded_keywords = expansion.get("keywords", [])
        for keyword in expanded_keywords:
            if isinstance(keyword, list):
                all_keywords.extend(str(k) for k in keyword)
            else:
                all_keywords.append(str(keyword))

    if all_keywords:
        parsed.semantic_criteria.industry_keywords = list(set(all_keywords))

    kw_count = len(parsed.semantic_criteria.industry_keywords)
    naics_count = len(parsed.hard_filters.naics_codes or [])
    trace.add("stage_0", "industry", f"{len(llm_keywords)} LLM terms -> {kw_count} keywords, {naics_count} NAICS codes")

    excluded_ind = llm_output.get("excluded_industries") or []
    if excluded_ind:
        excluded_naics = []
        for term in excluded_ind:
            expansion = expand_industry(term)
            codes = expansion.get("naics_prefixes", [])
            for code in codes:
                if isinstance(code, list):
                    excluded_naics.extend(str(c) for c in code)
                else:
                    excluded_naics.append(str(code))

        if excluded_naics:
            parsed.hard_filters.excluded_naics_codes = list(set(excluded_naics))
            # Clear any positive NAICS that overlap with exclusions
            if parsed.hard_filters.naics_codes:
                parsed.hard_filters.naics_codes = [
                    c for c in parsed.hard_filters.naics_codes
                    if c not in parsed.hard_filters.excluded_naics_codes
                ]
                if not parsed.hard_filters.naics_codes:
                    parsed.hard_filters.naics_codes = None

            trace.add("stage_0", "excluded_naics", f"{excluded_ind} -> {len(excluded_naics)} excluded NAICS codes")


def validate_business_model(parsed: ParsedQuery, llm_output: dict):

    trace = parsed.trace
    llm_bm = llm_output.get("business_model_keywords") or []

    regex_parsed = ParsedQuery(original_query=parsed.original_query)
    detect_business_model(parsed.original_query, regex_parsed)
    regex_bm = regex_parsed.semantic_criteria.business_model_keywords

    # Merge: LLM + regex (union)
    all_business_model = set()
    for keyword in llm_bm:
        kw_lower = keyword.lower().strip()
        for model_key in BUSINESS_MODEL_KEYWORDS:
            if kw_lower == model_key.lower() or kw_lower in [
                a.lower() for a in BUSINESS_MODEL_KEYWORDS[model_key]
            ]:
                all_business_model.add(model_key)
                break
        else:
            # LLM keyword doesn't match taxonomy — keep it anyway
            all_business_model.add(keyword)

    all_business_model.update(regex_bm)

    if all_business_model:
        parsed.semantic_criteria.business_model_keywords = list(all_business_model)
        trace.add("stage_0", "biz_model", str(list(all_business_model)))


COMPLEX_SIGNALS = [ "could supply", "could provide", "could serve", "could replace", "could benefit", "could help", "transition to",
    "competing with", "compete with", "rivals of", "alternatives to", "supply chain", "suppliers of", "vendors for", "expanding into",
    "value chain", "upstream", "downstream", "handling cross-border", "cross-border logistics", "whose technology", "whose services"
]

MODERATE_SIGNALS = [ "startup", "startups", "fast-growing", "fast growing", "government contracts", "cross-border", "handling", "providing",
    "solutions", "innovative", "disruptive", "emerging", "leading", "serving"
]


def postvalidate_complexity(query: str, parsed: ParsedQuery, llm_output: dict):
    """
    Deterministic override of LLM-assigned complexity.

    The LLM's classification is a good first guess, but it tends to under-classify (marking things "simple" that need semantic matching).
    This method applies hard rules that can only UPGRADE complexity, never downgrade it.
    """

    trace = parsed.trace
    original = parsed.complexity
    query_lower = query.lower()

    if parsed.complexity != QueryComplexity.COMPLEX:
        for signal in COMPLEX_SIGNALS:
            if signal in query_lower:
                parsed.complexity = QueryComplexity.COMPLEX
                trace.add("stage_0", "complexity_override", f"{original.value} -> complex (signal: '{signal}')")
                return
            
    if parsed.complexity == QueryComplexity.SIMPLE:
        reasons = []

        if parsed.semantic_criteria.business_model_keywords:
            reasons.append(f"business_model={parsed.semantic_criteria.business_model_keywords}")

        for signal in MODERATE_SIGNALS:
            if signal in query_lower:
                reasons.append(f"signal: '{signal}'")
                break

        keyword_count = len(parsed.semantic_criteria.industry_keywords)
        if keyword_count > 8:
            reasons.append(f"high_keyword_count={keyword_count}")

        if reasons:
            parsed.complexity = QueryComplexity.MODERATE
            trace.add("stage_0", "complexity_override", f"simple -> moderate ({'; '.join(reasons)})")
            return

    if parsed.complexity != original:
        trace.add("stage_0", "complexity_override", f"{original.value} -> {parsed.complexity.value}")
    else:
        trace.add("stage_0", "complexity_check", f"LLM classification '{original.value}' confirmed")


def compute_confidence(query: str, parsed: ParsedQuery, llm_output: dict):

    score = 0.5 

    if parsed.hard_filters.locations: score += 0.15
    if parsed.hard_filters.naics_codes: score += 0.15
    llm_had_numerics = any(
        llm_output.get(field) is not None
        for field in ("employee_count_min", "employee_count_max", "revenue_min", "revenue_max", "year_founded_min", "year_founded_max")
    )
    if llm_had_numerics: score += 0.1
    if parsed.semantic_criteria.business_model_keywords: score += 0.05
    if len(query.split()) < 4 and not parsed.hard_filters.locations: score -= 0.1
    if parsed.complexity == QueryComplexity.COMPLEX: score -= 0.1

    parsed.confidence = max(0.0, min(1.0, score))


def build_match_expressions(parsed: ParsedQuery):
    """
    Build AND/OR match expression trees from the extracted keywords.
        > using clay-style double pattern: OR(full phrases) + AND(OR(category words), OR(type words))
    """
    # Ensure all keywords are strings
    keywords = [ str(keyword) for keyword in parsed.semantic_criteria.industry_keywords if not isinstance(keyword, list) ]
    if not keywords: 
        return

    # Layer 1: OR of full keyword phrases
    phrase_operands = [MatchLeaf(kw) for kw in keywords]
    phrase_node = MatchNode( operator=MatchOperator.OR, operands=phrase_operands )

    # Layer 2: Combinatorial AND/OR
    category_words = []
    modifier_words = []
    for keyword in keywords:
        parts = keyword.split()
        for part in parts:
            if len(part) > 3:
                if part in ("energy", "power", "technology", "tech", "software", "service", "services", "solutions", "systems", "platform", 
                            "tools", "management", "analytics", "manufacturing", "production", "processing", "development", "engineering"):
                    modifier_words.append(part)
                else:
                    category_words.append(part)

    category_words = list(set(category_words))
    modifier_words = list(set(modifier_words))

    combinatorial_node = None
    if category_words and modifier_words:
        cat_node = MatchNode( operator=MatchOperator.OR, operands=[MatchLeaf(word) for word in category_words] )
        mod_node = MatchNode( operator=MatchOperator.OR, operands=[MatchLeaf(word) for word in modifier_words] )
        combinatorial_node = MatchNode( operator=MatchOperator.AND, operands=[cat_node, mod_node] )

    # Combine: OR(phrase_match, combinatorial_match)
    if combinatorial_node:
        root = MatchNode( operator=MatchOperator.OR, operands=[phrase_node, combinatorial_node] )
    else:
        root = phrase_node

    parsed.semantic_criteria.match_expressions = [root]
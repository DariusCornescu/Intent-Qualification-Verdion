"""
Deterministic regex patterns and helpers for query parsing.

These are used by the LLM validation pipeline to cross-check LLM extractions against hard regex rules. 
Regex wins on disagreements because it's deterministic — if it finds "500 employees" in the text, 
that's what it means, no matter what the LLM hallucinated.

Also contains keyword-based detectors for public/private status and business model classification.
"""
import re
from code.models import ParsedQuery
from code.taxonomy import BUSINESS_MODEL_KEYWORDS


NUMERIC_PATTERNS = [
    # "more than 1,000 employees" / "over 500 employees"
    (r"(?:more than|over|above|exceeding|greater than|at least|minimum)\s+([\d,]+)\s+employees",
     "employee_count_min"),
    # "fewer than 200 employees" / "under 100 employees"
    (r"(?:fewer than|less than|under|below|at most|maximum|no more than)\s+([\d,]+)\s+employees",
     "employee_count_max"),
    # "1,000+ employees" — only with explicit "+" sign
    (r"([\d,]+)\+\s*employees", "employee_count_min"),
    # "revenue over $50 million"
    (r"revenue\s+(?:over|above|exceeding|greater than|more than|at least)\s+\$?([\d,.]+)\s*(million|billion|m|b|k)?",
     "revenue_min"),
    # "revenue under $10 million"
    (r"revenue\s+(?:under|below|less than|at most)\s+\$?([\d,.]+)\s*(million|billion|m|b|k)?",
     "revenue_max"),
    # "founded after 2018" / "established after 2010"
    (r"(?:founded|established|started|created|launched)\s+(?:after|since|from)\s+(\d{4})",
     "year_founded_min"),
    # "founded before 2020"
    (r"(?:founded|established|started|created|launched)\s+(?:before|prior to|until)\s+(\d{4})",
     "year_founded_max"),
]

REVENUE_MULTIPLIERS = {
    "billion": 1_000_000_000, "b": 1_000_000_000,
    "million": 1_000_000, "m": 1_000_000,
    "k": 1_000, "thousand": 1_000,
}


def parse_number(s: str) -> float:
    return float(s.replace(",", ""))


def extract_numeric_constraints(query: str, parsed: ParsedQuery):

    query_lower = query.lower()
    for pattern, field_name in NUMERIC_PATTERNS:
        match = re.search(pattern, query_lower)
        if not match:
            continue

        value = parse_number(match.group(1))

        # Handle revenue multipliers
        if field_name in ("revenue_min", "revenue_max") and match.lastindex >= 2:
            multiplier_str = match.group(2)
            if multiplier_str and multiplier_str.lower() in REVENUE_MULTIPLIERS:
                value *= REVENUE_MULTIPLIERS[multiplier_str.lower()]

        # Set the appropriate filter field
        if field_name == "employee_count_min":
            parsed.hard_filters.employee_count_min = int(value)
        elif field_name == "employee_count_max":
            parsed.hard_filters.employee_count_max = int(value)
        elif field_name == "revenue_min":
            parsed.hard_filters.revenue_min = value
        elif field_name == "revenue_max":
            parsed.hard_filters.revenue_max = value
        elif field_name == "year_founded_min":
            parsed.hard_filters.year_founded_min = int(value)
        elif field_name == "year_founded_max":
            parsed.hard_filters.year_founded_max = int(value)


def detect_public_private(query: str, parsed: ParsedQuery):

    query_lower = query.lower()
    if re.search(r"\bpublic\b", query_lower):
        # Make sure it's not "public sector" or "republic"
        if not re.search(r"\bpublic\s+sector\b|\brepublic\b", query_lower):
            parsed.hard_filters.is_public = True

    if re.search(r"\bprivate(?:ly)?\b", query_lower):
        parsed.hard_filters.is_public = False

    if re.search(r"\bpublicly\s+traded\b|\bpublicly\s+listed\b|\blisted\s+compan", query_lower):
        parsed.hard_filters.is_public = True


def detect_business_model(query: str, parsed: ParsedQuery):

    query_lower = query.lower()
    detected = []
    for model_key, aliases in BUSINESS_MODEL_KEYWORDS.items():
        for alias in aliases:
            if alias in query_lower:
                detected.append(model_key)
                break

    if detected:
        parsed.semantic_criteria.business_model_keywords = list(set(detected))
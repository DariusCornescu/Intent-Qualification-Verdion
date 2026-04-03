import json
import os
import logging
logger = logging.getLogger(__name__)

_TAXONOMY_DIR = ("taxonomy")

def _load_json(filename: str) -> dict | None:

    filepath = os.path.join(_TAXONOMY_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
        
    return None

_countries_data = _load_json("countries.json")
_industries_data = _load_json("industries.json")
_ambiguity_data = _load_json("ambiguity_phrases.json")
_bmodel_data = _load_json("business_models.json")

logger.info("Loaded countries taxonomy from taxonomy_data/countries.json")
logger.info("Loaded industries taxonomy from taxonomy_data/industries.json")

REGION_EXPANSIONS: dict[str, list[str]] = _countries_data["regions"]
INDUSTRY_TAXONOMY: dict[str, dict] = _industries_data["industries"]
_AMBIGUOUS_ALIASES = {
    "in", "no", "is", "it", "me", "us", "or", "as", "am", "an", "at", 
    "be", "by", "do", "go", "he", "if", "so", "to", "up", "we"
}

COUNTRY_ALIASES: dict[str, str] = {}
ISO2_TO_COUNTRY: dict[str, str] = {}

for _canonical_name, _info in _countries_data["countries"].items():
    for _alias in _info.get("aliases", []):
        _alias_lower = _alias.lower().strip()

        # ISO2 codes go into a separate dict WITHOUT the ambiguity filter ( "us" means United States, not the pronoun )
        if len(_alias_lower) == 2:
            ISO2_TO_COUNTRY[_alias_lower] = _canonical_name.lower()

        # General aliases still filter ambiguous short words to avoid "companies in Germany" matching India ("in" = India's ISO code)
        if _alias_lower not in _AMBIGUOUS_ALIASES and len(_alias_lower) >= 2:
            COUNTRY_ALIASES[_alias_lower] = _canonical_name.lower()

def normalize_location(location: str) -> list[str]:
    """
    Expand a location string into a list of searchable location names.
        > "Scandinavia" -> ["norway", "sweden", "denmark", "finland", "iceland"]
        > "Germany" -> ["germany"]
        > "DE" -> ["germany"]
    """
    loc_lower = location.lower().strip()
    if loc_lower in REGION_EXPANSIONS:
        return REGION_EXPANSIONS[loc_lower]
    if loc_lower in COUNTRY_ALIASES:
        return [COUNTRY_ALIASES[loc_lower]]

    return [loc_lower]

def expand_industry(term: str) -> dict:
    """
        > Look up an industry term and return NAICS codes + expanded keywords.
        > Handles plural/singular variations and partial matches.
    """
    term_lower = term.lower().strip()
    if term_lower in INDUSTRY_TAXONOMY:
        return INDUSTRY_TAXONOMY[term_lower]

    # Singular/plural variations
    variations = [
        term_lower,
        term_lower + "s",           # logistic → logistics
        term_lower + "es",          # process → processes
        term_lower.rstrip("s"),     # logistics → logistic
        term_lower.rstrip("es"),    # processes → process
        term_lower + "ing",         # manufactur → manufacturing
    ]
    for var in variations:
        if var in INDUSTRY_TAXONOMY:
            return INDUSTRY_TAXONOMY[var]

    # Partial match — check if any taxonomy key is contained in the term
    for key, data in INDUSTRY_TAXONOMY.items():
        if key in term_lower or term_lower in key:
            return data

    # Check if term appears in any keyword list
    for key, data in INDUSTRY_TAXONOMY.items():
        for kw in data["keywords"]:
            if term_lower in kw or kw in term_lower:
                return data

    return {"naics_prefixes": [], "keywords": [term_lower]}

AMBIGUITY_PHRASES: list[str] = _ambiguity_data["phrases"]
BUSINESS_MODEL_KEYWORDS: dict[str, list[str]] = _bmodel_data["business_models"]
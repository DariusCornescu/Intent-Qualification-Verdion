from __future__ import annotations
import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class MatchOperator(str, Enum):
    AND = "and"
    OR = "or"


@dataclass
class MatchLeaf:
    """A single term to match against."""
    value: str


@dataclass
class MatchNode:
    """AND/OR node in the match expression tree."""

    operator: MatchOperator
    operands: list[Union[MatchNode, MatchLeaf]]

    def flatten_values(self) -> list[str]:
        """Recursively collect all leaf values."""

        values = []
        for op in self.operands:
            if isinstance(op, MatchLeaf):
                values.append(op.value)
            else:
                values.extend(op.flatten_values())
        return values

    def format_tree(self, indent: int = 0) -> str:
        """Pretty-print the expression tree."""

        pad = "  " * indent
        op_str = self.operator.value.upper()
        if all(isinstance(o, MatchLeaf) for o in self.operands):
            # Compact: all leaves on one line
            vals = [o.value for o in self.operands]
            if len(vals) <= 4:
                return f"{pad}{op_str}({', '.join(vals)})"
            else:
                return f"{pad}{op_str}({', '.join(vals[:4])}... +{len(vals)-4} more)"
        else:
            lines = [f"{pad}{op_str}("]
            for o in self.operands:
                if isinstance(o, MatchLeaf):
                    lines.append(f"{pad}  \"{o.value}\"")
                else:
                    lines.append(o.format_tree(indent + 1))
            lines.append(f"{pad})")
            return "\n".join(lines)


class QueryComplexity(str, Enum):

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class HardFilters:
    """ Fields that are gonna be used to filter most of the companies in stage 1 """
    locations: Optional[list[str]] = None  
    excluded_locations: Optional[list[str]] = None 
    year_founded_min: Optional[int] = None         # >= this year
    year_founded_max: Optional[int] = None         # <= this year
    employee_count_min: Optional[int] = None       # >= this count
    employee_count_max: Optional[int] = None       # <= this count
    revenue_min: Optional[float] = None            # >= this amount (USD)
    revenue_max: Optional[float] = None            # <= this amount (USD)
    is_public: Optional[bool] = None              
    naics_codes: Optional[list[str]] = None        
    excluded_naics_codes: Optional[list[str]] = None 
    geo_lat: Optional[float] = None                # Center latitude
    geo_lng: Optional[float] = None                # Center longitude
    geo_radius_km: Optional[float] = None          # Search radius in km


@dataclass
class SemanticCriteria:
    """ Keywords and match expressions for soft scoring, that are evaluated against description, core_offerings, target_markets, and business_model fields """

    industry_keywords: list[str] = field(default_factory=list)
    product_keywords: list[str] = field(default_factory=list)
    business_model_keywords: list[str] = field(default_factory=list)
    target_market_keywords: list[str] = field(default_factory=list)

    # Rich match expressions (AND/OR trees) for description matching
    match_expressions: list[MatchNode] = field(default_factory=list)

    def all_keywords(self) -> list[str]:
        """Flatten all keywords into a single list for BM25 / embeddings."""

        raw = ( self.industry_keywords + self.product_keywords + self.business_model_keywords + self.target_market_keywords )
        for expr in self.match_expressions:
            raw.extend(expr.flatten_values())

        # Flatten any nested lists and ensure all items are strings
        flat = []
        for item in raw:
            if isinstance(item, list):
                flat.extend(str(i) for i in item)
            else:
                flat.append(str(item))

        return list(set(flat))


@dataclass
class QueryTrace:
    """  Records exactly what happened at each pipeline stage, and it's attached to ParsedQuery so downstream stages can append to it """
    steps: list[dict] = field(default_factory=list)

    def add(self, stage: str, action: str, detail: str = ""):
        self.steps.append({ "stage": stage, "action": action, "detail": detail })

    def print_trace(self):
        print(f"\n  {'─'*56}")
        print(f"  Pipeline trace:")
        print(f"  {'─'*56}")
        for step in self.steps:
            stage = step["stage"]
            action = step["action"]
            detail = step["detail"]
            if detail:
                print(f"  [{stage:12s}] {action}: {detail}")
            else:
                print(f"  [{stage:12s}] {action}")
        print(f"  {'─'*56}")


@dataclass
class ParsedQuery:
    """
    Complete output of the query analysis pipeline. This is consumed by all downstream stages:
       >  Stage 1 reads hard_filters
       >  Stage 2 reads semantic_criteria
       >  Router reads complexity
       >  Confidence gate reads confidence
    """
    original_query: str
    hard_filters: HardFilters = field(default_factory=HardFilters)
    semantic_criteria: SemanticCriteria = field(default_factory=SemanticCriteria)
    complexity: QueryComplexity = QueryComplexity.MODERATE
    confidence: float = 0.0  # 0.0 to 1.0
    naics_was_relaxed: bool = False  # Set by Stage 1 when NAICS filter was too aggressive
    trace: QueryTrace = field(default_factory=QueryTrace)

    def to_dict(self) -> dict:
        """Serialize for LLM validation prompt."""

        result = {
            "original_query": self.original_query,
            "complexity": self.complexity.value,
            "confidence": self.confidence,
            "hard_filters": {},
            "semantic_criteria": {
                "industry_keywords": self.semantic_criteria.industry_keywords,
                "product_keywords": self.semantic_criteria.product_keywords,
                "business_model_keywords": self.semantic_criteria.business_model_keywords,
                "target_market_keywords": self.semantic_criteria.target_market_keywords,
            },
        }
        hard_filters = self.hard_filters
        if hard_filters.locations:
            result["hard_filters"]["locations"] = hard_filters.locations
        if hard_filters.year_founded_min is not None:
            result["hard_filters"]["year_founded_min"] = hard_filters.year_founded_min
        if hard_filters.year_founded_max is not None:
            result["hard_filters"]["year_founded_max"] = hard_filters.year_founded_max
        if hard_filters.employee_count_min is not None:
            result["hard_filters"]["employee_count_min"] = hard_filters.employee_count_min
        if hard_filters.employee_count_max is not None:
            result["hard_filters"]["employee_count_max"] = hard_filters.employee_count_max
        if hard_filters.revenue_min is not None:
            result["hard_filters"]["revenue_min"] = hard_filters.revenue_min
        if hard_filters.revenue_max is not None:
            result["hard_filters"]["revenue_max"] = hard_filters.revenue_max
        if hard_filters.is_public is not None:
            result["hard_filters"]["is_public"] = hard_filters.is_public
        if hard_filters.naics_codes:
            result["hard_filters"]["naics_codes"] = hard_filters.naics_codes
        if hard_filters.geo_lat is not None:
            result["hard_filters"]["geo_lat"] = hard_filters.geo_lat
            result["hard_filters"]["geo_lng"] = hard_filters.geo_lng
            result["hard_filters"]["geo_radius_km"] = hard_filters.geo_radius_km
        return result

    def print_compiled(self):
        """ Print the full compiled query — the structured representation that Stage 0 produced from the raw user query. Shows exactly what downstream stages will work with """

        print(f"\n  ╔{'═'*58}╗")
        print(f"  ║  COMPILED QUERY{' '*42}║")
        print(f"  ╠{'═'*58}╣")
        print(f"  ║  Original: \"{self.original_query[:45]}{'...' if len(self.original_query) > 45 else ''}\"")
        print(f"  ║  Complexity: {self.complexity.value}  |  Confidence: {self.confidence:.2f}")
        print(f"  ╠{'═'*58}╣")

        # Hard filters
        print(f"  ║  HARD FILTERS (Stage 1 consumes these)")
        hard_filters = self.hard_filters
        any_filter = False
        if hard_filters.locations:
            locs = hard_filters.locations
            if len(locs) <= 5:
                print(f"  ║    location IN [{', '.join(locs)}]")
            else:
                print(f"  ║    location IN [{', '.join(locs[:4])}... +{len(locs)-4} more]")
            any_filter = True
        if hard_filters.employee_count_min is not None:
            print(f"  ║    employee_count >= {hard_filters.employee_count_min:,}")
            any_filter = True
        if hard_filters.employee_count_max is not None:
            print(f"  ║    employee_count <= {hard_filters.employee_count_max:,}")
            any_filter = True
        if hard_filters.revenue_min is not None:
            print(f"  ║    revenue >= ${hard_filters.revenue_min:,.0f}")
            any_filter = True
        if hard_filters.revenue_max is not None:
            print(f"  ║    revenue <= ${hard_filters.revenue_max:,.0f}")
            any_filter = True
        if hard_filters.year_founded_min is not None:
            print(f"  ║    year_founded >= {hard_filters.year_founded_min}")
            any_filter = True
        if hard_filters.year_founded_max is not None:
            print(f"  ║    year_founded <= {hard_filters.year_founded_max}")
            any_filter = True
        if hard_filters.is_public is not None:
            print(f"  ║    is_public == {hard_filters.is_public}")
            any_filter = True
        if hard_filters.naics_codes:
            codes = hard_filters.naics_codes
            if len(codes) <= 5:
                print(f"  ║    NAICS prefix IN [{', '.join(codes)}]")
            else:
                print(f"  ║    NAICS prefix IN [{', '.join(codes[:4])}... +{len(codes)-4} more]")
            any_filter = True
        if not any_filter:
            print(f"  ║    (none — all companies pass)")

        print(f"  ╠{'═'*58}╣")

        # Semantic criteria
        print(f"  ║  SEMANTIC CRITERIA (Stage 2 consumes these)")
        sc = self.semantic_criteria
        if sc.industry_keywords:
            kws = sc.industry_keywords
            if len(kws) <= 6:
                print(f"  ║    industry: [{', '.join(kws)}]")
            else:
                print(f"  ║    industry: [{', '.join(kws[:5])}... +{len(kws)-5} more]")
        if sc.business_model_keywords:
            print(f"  ║    biz_model: [{', '.join(sc.business_model_keywords)}]")
        if sc.product_keywords:
            print(f"  ║    products: [{', '.join(sc.product_keywords[:5])}]")
        if sc.target_market_keywords:
            print(f"  ║    markets: [{', '.join(sc.target_market_keywords[:5])}]")

        # Match expression trees
        if sc.match_expressions:
            print(f"  ║")
            print(f"  ║    Match expression tree:")
            for expr in sc.match_expressions:
                tree_str = expr.format_tree(indent=3)
                for line in tree_str.split("\n"):
                    print(f"  ║    {line}")

        if not sc.industry_keywords and not sc.match_expressions:
            print(f"  ║    (none)")

        print(f"  ╚{'═'*58}╝")


@dataclass
class CompanyProfile:
    """Represents a single company from the dataset."""

    operational_name: Optional[str] = None
    website: Optional[str] = None
    year_founded: Optional[int] = None
    address: Optional[str] = None           
    country_code: Optional[str] = None      # ISO2 lowercase: "us", "de", "ro"
    region_name: Optional[str] = None       # e.g. "Bucharest", "California"
    town: Optional[str] = None              # e.g. "Bucharest", "San Francisco"
    latitude: Optional[float] = None        # GPS latitude 
    longitude: Optional[float] = None       # GPS longitude
    employee_count: Optional[int] = None
    revenue: Optional[float] = None
    primary_naics: Optional[dict] = None    # {"code": "488510", "label": "..."}
    secondary_naics: Optional[list[dict]] = None
    description: Optional[str] = None
    business_model: Optional[list[str]] = None
    target_markets: Optional[list[str]] = None
    core_offerings: Optional[list[str]] = None
    is_public: Optional[bool] = None

    @staticmethod
    def _parse_dict_field(raw) -> Optional[dict]:
        """Parse a field that might be a dict or a string repr of a dict."""

        if isinstance(raw, dict): return raw

        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw) # handles both string and dict 
                if isinstance(parsed, dict): 
                    return parsed
                
            except (ValueError, SyntaxError):
                pass
        return None

    @classmethod
    def from_dict(cls, data: dict) -> CompanyProfile:

        raw_address = data.get("address")
        address_str = None
        country_code = None
        region_name = None
        town = None

        addr_dict = cls._parse_dict_field(raw_address)
        latitude = None
        longitude = None
        if addr_dict:
            country_code = addr_dict.get("country_code", "").lower() or None
            region_name = addr_dict.get("region_name")
            town = addr_dict.get("town")
            latitude = addr_dict.get("latitude")
            longitude = addr_dict.get("longitude")
            parts = [p for p in [town, region_name, country_code] if p]
            address_str = ", ".join(parts) if parts else None

        elif isinstance(raw_address, str):
            address_str = raw_address

        # Parse primary_naics (dict or string repr)
        raw_naics = data.get("primary_naics")
        primary_naics = cls._parse_dict_field(raw_naics)

        # Parse secondary_naics (could be list of dicts or list of strings)
        raw_secondary = data.get("secondary_naics")
        secondary_naics = None
        if isinstance(raw_secondary, list):
            secondary_naics = []
            for item in raw_secondary:
                parsed = cls._parse_dict_field(item)
                if parsed:
                    secondary_naics.append(parsed)

        return cls( 
            operational_name=data.get("operational_name"),
            website=data.get("website"),
            year_founded=data.get("year_founded"),
            address=address_str,
            country_code=country_code,
            region_name=region_name,
            town=town,
            latitude=latitude,
            longitude=longitude,
            employee_count=data.get("employee_count"),
            revenue=data.get("revenue"),
            primary_naics=primary_naics,
            secondary_naics=secondary_naics,
            description=data.get("description"),
            business_model=data.get("business_model"),
            target_markets=data.get("target_markets"),
            core_offerings=data.get("core_offerings"),
            is_public=data.get("is_public"),
        )

    def get_all_naics_codes(self) -> list[str]:

        codes = []
        if self.primary_naics and "code" in self.primary_naics:
            codes.append(self.primary_naics["code"])
        if self.secondary_naics:
            for n in self.secondary_naics:
                if "code" in n:
                    codes.append(n["code"])
        return codes

    def get_text_blob(self) -> str:
        """Concatenate all text fields for TF-IDF / embedding."""

        parts = []
        if self.description:
            parts.append(self.description)
        if self.core_offerings:
            parts.append(" ".join(self.core_offerings))
        if self.target_markets:
            parts.append(" ".join(self.target_markets))
        if self.business_model:
            parts.append(" ".join(self.business_model))
        if self.primary_naics and "label" in self.primary_naics:
            parts.append(self.primary_naics["label"])
        return " ".join(parts)


@dataclass
class QualificationResult:

    company: CompanyProfile
    qualified: bool
    score: float                     # 0.0 to 1.0
    stage_reached: str               # "hard_filter", "soft_score", "llm_qualified"
    reasoning: Optional[str] = None  # LLM explanation (if we reach Stage 3)
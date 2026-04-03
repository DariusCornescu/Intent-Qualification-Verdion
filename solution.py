"""
Company Qualification & Ranking System
========================================

This is a multi-stage pipeline that determines which companies truly match a user's query.

Architecture:
  Stage 0: Query Analysis - LLM extracts structured JSON, regex/taxonomy validate & normalize
  Stage 1: Hard Filters - direct field comparisons (location, revenue, employees, NAICS)
  Stage 2: Soft Scoring - BM25 + semantic embeddings + Reciprocal Rank Fusion
  Stage 3: LLM Batch Qualification - only for complex queries, only top-K)

"""

import json
import argparse
import logging
import os
import sys
from typing import Optional
from code.models import CompanyProfile, ParsedQuery, QualificationResult, QueryComplexity
from code.query_analysis import QueryAnalyzer
from code.hard_filters import apply_hard_filters
from code.soft_scoring import score_and_rank, build_scoring_index
from code.llm_qualifier import qualify_with_llm

logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s" )
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

TOP_K = 40                       
MIN_SOFT_SCORE = 0.0    # RRF scores are rank-based (not 0-1), so no absolute threshold. Stage 2 already returns only top-K ranked companies - all of them qualify for Stage 3 review.
LLM_MODEL = "gpt-4o-mini" 


def get_llm_client():
    """Create OpenAI-compatible client if API key is available."""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set. LLM validation and Stage 3 disabled. "
            "Pipeline will run Stages 0-2 only."
        )
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai package not installed. LLM features disabled.")
        return None


class QualificationPipeline:
    """
    Orchestrates the multi-stage qualification pipeline. The query complexity determines how deep we go:
        > SIMPLE   → Stage 0 + Stage 1 only
        > MODERATE → Stage 0 + Stage 1 + Stage 2
        > COMPLEX  → Stage 0 + Stage 1 + Stage 2 + Stage 3
    """

    def __init__(self, companies: list[CompanyProfile], llm_client=None, llm_model: str = LLM_MODEL, top_k: int = TOP_K):

        self.companies = companies
        self.llm_client = llm_client
        self.top_k = top_k
        self.analyzer = QueryAnalyzer(llm_client=llm_client, llm_model=llm_model)
        self.scoring_index = build_scoring_index(companies, cache_dir="cache")

    def process_query(self, query: str) -> list[QualificationResult]:
        """Process a single query. Returns qualified companies."""

        result = self.process_query_verbose(query)
        return result["results"]

    def process_query_verbose(self, query: str) -> dict:
        """ 
        Process a query and return full stage-by-stage data. 
            > returns dict with: parsed, filtered, scored, results, and stage metadata.
        """
        logger.info(f"Processing query: \"{query}\"")
        logger.info(f"Dataset size: {len(self.companies)} companies")

        output = {
            "query": query,
            "n_total": len(self.companies),
            "parsed": None,
            "filtered": [],
            "scored": [],
            "results": [],
            "stages_reached": [],
        }

        # Stage 0: Query Analysis
        parsed = self.analyzer.analyze(query)
        output["parsed"] = parsed
        output["stages_reached"].append("stage_0")

        logger.info(
            f"Stage 0 complete: complexity={parsed.complexity.value}, "
            f"confidence={parsed.confidence:.2f}, "
            f"hard_filters={self._summarize_filters(parsed)}, "
            f"keywords={len(parsed.semantic_criteria.industry_keywords)}"
        )

        # Stage 1: Hard Filters
        filtered = apply_hard_filters(self.companies, parsed)
        output["filtered"] = filtered
        output["stages_reached"].append("stage_1")

        if not filtered:
            logger.warning("No companies passed hard filters.")
            return output

        # SIMPLE queries: done after hard filters
        if parsed.complexity == QueryComplexity.SIMPLE:
            logger.info("Simple query — returning hard filter results directly.")
            output["results"] = [ QualificationResult( company=c, qualified=True, score=1.0, stage_reached="hard_filter" ) for c in filtered ]

            return output

        # Stage 2: Soft Scoring 
        scored = score_and_rank(filtered, parsed, top_k=self.top_k, scoring_index=self.scoring_index)
        output["scored"] = scored
        output["stages_reached"].append("stage_2")

        # MODERATE queries: done after soft scoring
        if parsed.complexity == QueryComplexity.MODERATE or not self.llm_client:
            output["results"] = [ QualificationResult( company=company, qualified=score > MIN_SOFT_SCORE, score=score, stage_reached="soft_score" )
                for company, score in scored
                if score > MIN_SOFT_SCORE
            
            ]
            return output

        # Stage 3: LLM Batch Qualification
        logger.info("Complex query — running LLM batch qualification.")
        output["stages_reached"].append("stage_3")
        results = qualify_with_llm( scored, parsed, self.llm_client, model=LLM_MODEL )
        output["results"] = [r for r in results if r.qualified]
        return output

    def _summarize_filters(self, parsed: ParsedQuery) -> str:
        """Quick summary of active hard filters."""

        active = []
        hard_filters = parsed.hard_filters
        if hard_filters.locations:
            active.append(f"loc={hard_filters.locations[:3]}")
        if hard_filters.year_founded_min:
            active.append(f"year>={hard_filters.year_founded_min}")
        if hard_filters.year_founded_max:
            active.append(f"year<={hard_filters.year_founded_max}")
        if hard_filters.employee_count_min:
            active.append(f"emp>={hard_filters.employee_count_min}")
        if hard_filters.employee_count_max:
            active.append(f"emp<={hard_filters.employee_count_max}")
        if hard_filters.revenue_min:
            active.append(f"rev>={hard_filters.revenue_min}")
        if hard_filters.revenue_max:
            active.append(f"rev<={hard_filters.revenue_max}")
        if hard_filters.is_public is not None:
            active.append(f"public={hard_filters.is_public}")
        if hard_filters.naics_codes:
            active.append(f"naics={hard_filters.naics_codes[:3]}")
        return ", ".join(active) if active else "none"


def format_results( query: str, results: list[QualificationResult] ) -> dict:

    return {
        "query": query,
        "total_qualified": len(results),
        "companies": [
            {
                "name": r.company.operational_name or "Unknown",
                "score": round(r.score, 4),
                "stage": r.stage_reached,
                "qualified": r.qualified,
                "reasoning": r.reasoning,
                "website": r.company.website,
                "address": r.company.address,
                "description": (
                    r.company.description[:450] + "..."
                    if r.company.description and len(r.company.description) > 200
                    else r.company.description
                ),
            }
            for r in results
        ],
    }


def _print_summary(query: str, output: dict, pipeline: QualificationPipeline):
    """Compact CLI summary - one block per stage, no debug noise."""

    parsed = output["parsed"]
    filtered = output["filtered"]
    scored = output["scored"]
    results = output["results"]
    hard_filters = parsed.hard_filters
    sc = parsed.semantic_criteria

    print(f"\n{'━'*70}")
    print(f"  QUERY: \"{query}\"")
    print(f"{'━'*70}")

    # Stage 0
    parts = [f"complexity={parsed.complexity.value}", f"confidence={parsed.confidence:.2f}"]
    if hard_filters.locations:
        parts.append(f"locations={hard_filters.locations[:4]}")
    if hard_filters.naics_codes:
        parts.append(f"NAICS={len(hard_filters.naics_codes)} codes")
    if sc.industry_keywords:
        parts.append(f"keywords={len(sc.industry_keywords)}")
    if hard_filters.geo_lat is not None:
        parts.append(f"geo=({hard_filters.geo_lat:.2f},{hard_filters.geo_lng:.2f}) r={hard_filters.geo_radius_km:.0f}km")
    print(f"  Stage 0  {', '.join(parts)}")

    # Stage 1
    n_total = len(pipeline.companies)
    n_filtered = len(filtered)
    pct = round((1 - n_filtered / n_total) * 100) if n_total else 0
    print(f"  Stage 1  {n_total} → {n_filtered} companies ({pct}% eliminated)")

    # Stage 2
    if scored:
        print(f"  Stage 2  {n_filtered} scored -> top {len(scored)} "
              f"(range: {scored[0][1]:.4f} – {scored[-1][1]:.4f})")

    # Stage 3
    llm_results = [r for r in results if r.stage_reached == "llm_qualified"]
    if llm_results:
        print(f"  Stage 3  {len(scored)} evaluated -> {len(llm_results)} qualified")

    # Top results
    print(f"  {'─'*66}")
    for i, r in enumerate(results[:8]):
        name = (r.company.operational_name or "?")[:32]
        nc = r.company.primary_naics
        code = nc.get("code", "?") if nc else "?"
        cc = r.company.country_code or "??"
        print(f"  {i+1:2d}. [{r.score:.4f}] {name:32s} {cc:2s}  {code}")
    if len(results) > 8:
        print(f"  ... +{len(results) - 8} more")

    total = len(results)
    if total > 0:
        print(f"\n  ✓ {total} companies qualified")
    else:
        print(f"\n  ✗ 0 companies qualified")
    print(f"{'━'*70}")

def main():
    parser = argparse.ArgumentParser( description="Company Qualification & Ranking System" )
    parser.add_argument( "--data", required=True, help="Path to companies file (.json or .jsonl)" )
    parser.add_argument( "--query", help="Single query string to process" )
    parser.add_argument( "--queries", help="Path to queries JSON file (array of strings)" )
    parser.add_argument( "--output", default="results.json", help="Output file path (default: results.json)" )
    parser.add_argument( "--top-k", type=int, default=TOP_K, help=f"Max candidates for Stage 2→3 (default: {TOP_K})" )
    parser.add_argument( "--model", default=LLM_MODEL, help=f"LLM model for extraction and qualification (default: {LLM_MODEL})" )
    args = parser.parse_args()

    logger.info(f"Loading companies from {args.data}")
    with open(args.data, "r") as f:
        if args.data.endswith(".jsonl"):
            raw_companies = [json.loads(line) for line in f if line.strip()]
        else:
            raw_companies = json.load(f)

    companies = [CompanyProfile.from_dict(c) for c in raw_companies]
    logger.info(f"Loaded {len(companies)} companies")

    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries:
        with open(args.queries, "r") as f:
            queries = json.load(f)
    else:
        parser.error("Provide either --query or --queries")

    llm_client = get_llm_client()
    if not llm_client:
        logger.error("Cannot start pipeline without LLM client. Exiting.")
        sys.exit(1)

    pipeline = QualificationPipeline( companies, llm_client=llm_client, llm_model=args.model, top_k=args.top_k )

    all_results = []
    for query in queries:
        output = pipeline.process_query_verbose(query)
        results = output["results"]

        formatted = format_results(query, results)
        all_results.append(formatted)

        _print_summary(query, output, pipeline)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
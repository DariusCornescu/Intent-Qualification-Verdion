"""
Company Qualifier — Web Application
====================================
FastAPI backend that serves the React frontend and exposes the qualification pipeline as a REST API.
    > Usage: python app.py
"""

import json
import os
import time
import logging
import uvicorn
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from code.models import CompanyProfile, ParsedQuery, QualificationResult, QueryComplexity, MatchNode, MatchLeaf
from code.query_analysis import QueryAnalyzer
from code.hard_filters import apply_hard_filters
from code.soft_scoring import score_and_rank, build_scoring_index, ScoringIndex, warm_up_embedding_model
from code.llm_qualifier import qualify_with_llm

logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s" )
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()


TOP_K = 30
MIN_SOFT_SCORE = 0.05
LLM_MODEL = "gpt-4o-mini"
DEFAULT_DATA = "data/companies.jsonl"

app = FastAPI(title="Company Qualifier", version="1.0.0")
app.add_middleware( CORSMiddleware, allow_origins=["*"], allow_methods=["*"],  allow_headers=["*"] )

companies: list[CompanyProfile] = []
analyzer: Optional[QueryAnalyzer] = None
scoring_index: Optional[ScoringIndex] = None
llm_client = None
llm_enabled = True


def get_llm_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def load_companies(data_path: str = DEFAULT_DATA):
    global companies, analyzer, llm_client, scoring_index

    path = Path(data_path)
    if not path.is_absolute():
        path = Path(__file__).parent / path

    raw = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    companies = [CompanyProfile.from_dict(c) for c in raw]
    llm_client = get_llm_client()

    if not llm_client:
        logger.warning(
            "OPENAI_API_KEY not set. The query analyzer requires an LLM client. "
            "Set the OPENAI_API_KEY environment variable and restart."
        )
    else:
        analyzer = QueryAnalyzer(llm_client=llm_client, llm_model=LLM_MODEL)

    # Build pre-computed scoring index (embeddings + BM25)
    # Disk cache: embeddings saved to cache/ dir, skips re-encoding on restart
    cache_dir = str(path.parent / "cache")
    logger.info("Building scoring index (one-time)...")
    scoring_index = build_scoring_index(companies, cache_dir=cache_dir)

    # Pre-load embedding model so first query is fast (~60s model load)
    # Without this, the first query pays the model load cost
    logger.info("Warming up embedding model for query encoding...")
    warm_up_embedding_model()

    llm_status = "ON" if llm_client else "OFF (no API key)"
    logger.info(f"Loaded {len(companies)} companies, LLM: {llm_status}, scoring index: ready")


# Helpers

def _serialize_match_tree(node) -> dict:
    """Serialize a MatchNode/MatchLeaf into a JSON-friendly dict."""
    if isinstance(node, MatchLeaf):
        return {"type": "leaf", "value": node.value}
    elif isinstance(node, MatchNode):
        return {
            "type": "node",
            "operator": node.operator.value.upper(),
            "operands": [_serialize_match_tree(op) for op in node.operands],
        }
    return {}


def _build_compiled_query(parsed: ParsedQuery) -> dict:
    """Build the COMPILED QUERY structure matching the terminal output."""
    hf = parsed.hard_filters
    sc = parsed.semantic_criteria

    # Hard filters
    hard_filters = []
    if hf.locations:
        hard_filters.append({"field": "location", "op": "IN", "value": hf.locations})
    if hf.employee_count_min is not None:
        hard_filters.append({"field": "employee_count", "op": ">=", "value": hf.employee_count_min})
    if hf.employee_count_max is not None:
        hard_filters.append({"field": "employee_count", "op": "<=", "value": hf.employee_count_max})
    if hf.revenue_min is not None:
        hard_filters.append({"field": "revenue", "op": ">=", "value": hf.revenue_min})
    if hf.revenue_max is not None:
        hard_filters.append({"field": "revenue", "op": "<=", "value": hf.revenue_max})
    if hf.year_founded_min is not None:
        hard_filters.append({"field": "year_founded", "op": ">=", "value": hf.year_founded_min})
    if hf.year_founded_max is not None:
        hard_filters.append({"field": "year_founded", "op": "<=", "value": hf.year_founded_max})
    if hf.is_public is not None:
        hard_filters.append({"field": "is_public", "op": "==", "value": hf.is_public})
    if hf.naics_codes:
        hard_filters.append({"field": "NAICS prefix", "op": "IN", "value": hf.naics_codes})
    if hf.excluded_naics_codes:
        hard_filters.append({"field": "NAICS prefix", "op": "NOT IN", "value": hf.excluded_naics_codes})
    if hf.excluded_locations:
        hard_filters.append({"field": "location", "op": "NOT IN", "value": hf.excluded_locations})

    # Semantic criteria
    semantic = {}
    if sc.industry_keywords:
        semantic["industry"] = sc.industry_keywords
    if sc.business_model_keywords:
        semantic["biz_model"] = sc.business_model_keywords
    if sc.product_keywords:
        semantic["products"] = sc.product_keywords
    if sc.target_market_keywords:
        semantic["markets"] = sc.target_market_keywords

    # Match expression trees
    match_trees = []
    for expr in sc.match_expressions:
        match_trees.append(_serialize_match_tree(expr))

    return {
        "original_query": parsed.original_query,
        "complexity": parsed.complexity.value,
        "confidence": round(parsed.confidence, 2),
        "hard_filters": hard_filters,
        "semantic": semantic,
        "match_trees": match_trees,
        "naics_was_relaxed": parsed.naics_was_relaxed,
    }


def _build_pipeline_trace(parsed: ParsedQuery) -> list:
    """Serialize the QueryTrace steps."""
    return [
        {
            "stage": step["stage"],
            "action": step["action"],
            "detail": step.get("detail", ""),
        }
        for step in parsed.trace.steps
    ]


# API Endpoints

@app.get("/api/status")
def get_status():
    return {
        "companies": len(companies),
        "llm_available": llm_client is not None,
        "llm_enabled": llm_enabled and llm_client is not None,
        "llm_model": LLM_MODEL,
        "top_k": TOP_K,
    }


class LLMToggle(BaseModel):
    enabled: bool

class BatchRequest(BaseModel):
    queries: list[str]

@app.post("/api/llm-toggle")
def toggle_llm(body: LLMToggle):
    global llm_enabled, analyzer
    if not body.enabled:
        return JSONResponse(
            status_code=400,
            content={
                "error": "LLM cannot be disabled — the query analyzer requires it.",
                "llm_enabled": True,
                "llm_available": llm_client is not None,
            },
        )
    llm_enabled = body.enabled
    analyzer = QueryAnalyzer(llm_client=llm_client, llm_model=LLM_MODEL)
    logger.info(f"LLM toggled: ON")
    return {
        "llm_enabled": llm_enabled and llm_client is not None,
        "llm_available": llm_client is not None,
    }


@app.get("/api/query")
def run_query(q: str = Query(..., description="Search query")):
    """Run the full qualification pipeline and return results with trace."""

    if not companies or not analyzer:
        return JSONResponse(status_code=503, content={"error": "Pipeline not initialized"})

    t0 = time.perf_counter()

    try:
        return _run_query_inner(q, t0)
    except Exception as e:
        # Catch-all: NEVER return raw 500 text — always return JSON
        logger.exception(f"Unhandled error for query: {q}")
        elapsed = time.perf_counter() - t0
        return JSONResponse(
            status_code=500,
            content={
                "query": q,
                "error": str(e),
                "error_type": type(e).__name__,
                "stages": [],
                "results": [],
                "elapsed": round(elapsed, 3),
            },
        )


def _run_query_inner(q: str, t0: float):
    """Inner query handler — separated so the outer handler catches all exceptions."""

    effective_llm = llm_client if llm_enabled else None
    stages = []

    # Stage 0: Query Analysis
    try:
        parsed = analyzer.analyze(q)
    except Exception as e:
        stages.append({"stage": 0, "label": "Query Analysis", "status": "error", "error": str(e)})
        return {"query": q, "stages": stages, "results": [], "elapsed": time.perf_counter() - t0}

    # Build compiled query (the terminal-style structured output)
    compiled = _build_compiled_query(parsed)

    # Build pipeline trace from QueryTrace
    pipeline_trace = _build_pipeline_trace(parsed)

    stages.append({
        "stage": 0, "label": "Query Analysis", "status": "done",
        "complexity": parsed.complexity.value,
        "confidence": round(parsed.confidence, 2),
    })

    # Stage 1: Hard Filters
    filtered = apply_hard_filters(companies, parsed)

    rate = round((1 - len(filtered) / len(companies)) * 100, 1) if companies else 0
    stages.append({
        "stage": 1, "label": "Hard Filters", "status": "done",
        "input_count": len(companies),
        "output_count": len(filtered),
        "filter_rate": rate,
    })

    if not filtered:
        elapsed = time.perf_counter() - t0
        return {
            "query": q, "compiled": compiled, "pipeline_trace": pipeline_trace,
            "stages": stages, "results": [], "elapsed": round(elapsed, 3),
        }

    # SIMPLE -> done
    if parsed.complexity == QueryComplexity.SIMPLE:
        results = [
            _format_result(QualificationResult(company=c, qualified=True, score=1.0, stage_reached="hard_filter"))
            for c in filtered
        ]
        elapsed = time.perf_counter() - t0
        return {
            "query": q, "compiled": compiled, "pipeline_trace": pipeline_trace,
            "stages": stages, "results": results, "elapsed": round(elapsed, 3),
            "note": "Simple query — hard filters only",
        }

    # Stage 2: Soft Scoring 
    scored = score_and_rank(filtered, parsed, top_k=TOP_K, scoring_index=scoring_index)
    above = [(c, s) for c, s in scored if s > MIN_SOFT_SCORE]

    # Check if LLM verification should be forced
    FORCE_LLM_THRESHOLD = 0.20
    top_score = scored[0][1] if scored else 0.0
    force_llm = (
        parsed.naics_was_relaxed
        and top_score < FORCE_LLM_THRESHOLD
        and effective_llm is not None
    )

    stages.append({
        "stage": 2, "label": "Soft Scoring", "status": "done",
        "input_count": len(filtered),
        "scored_count": len(scored),
        "above_threshold": len(above),
        "top3": [{"name": c.operational_name or "?", "score": round(s, 4)} for c, s in scored[:3]],
        "naics_was_relaxed": parsed.naics_was_relaxed,
        "force_llm": force_llm,
        "top_score": round(top_score, 4),
    })

    # MODERATE or no LLM -> done (unless forced)
    if not force_llm and (parsed.complexity == QueryComplexity.MODERATE or not effective_llm):
        results = [
            _format_result(QualificationResult(company=c, qualified=True, score=s, stage_reached="soft_score"))
            for c, s in scored if s > MIN_SOFT_SCORE
        ]
        elapsed = time.perf_counter() - t0
        extra = {}
        if not effective_llm and parsed.complexity == QueryComplexity.COMPLEX:
            extra["warning"] = "Complex query but LLM disabled — using soft scores only"
        if parsed.naics_was_relaxed and not effective_llm:
            extra["warning"] = (extra.get("warning", "") +
                " NAICS was relaxed but LLM unavailable for verification — results may be inaccurate.").strip()
        return {
            "query": q, "compiled": compiled, "pipeline_trace": pipeline_trace,
            "stages": stages, "results": results, "elapsed": round(elapsed, 3), **extra,
        }

    # Stage 3: LLM Qualification 
    reason = "forced (NAICS relaxed + low scores)" if force_llm else "complex query"
    no_data = False

    try:
        llm_results = qualify_with_llm(scored, parsed, effective_llm, model=LLM_MODEL)
        qualified = [r for r in llm_results if r.qualified]

        stage3_info = {
            "stage": 3, "label": "LLM Qualification", "status": "done",
            "input_count": len(scored),
            "qualified_count": len(qualified),
            "total_evaluated": len(llm_results),
            "forced": force_llm,
            "reason": reason,
        }

        # If LLM rejected everything after forced verification → no matching data
        if force_llm and len(qualified) == 0:
            no_data = True
            stage3_info["no_data"] = True
            stage3_info["no_data_reason"] = (
                "LLM rejected all candidates — dataset likely does not contain "
                "companies matching this query."
            )

        stages.append(stage3_info)
        results = [_format_result(r) for r in qualified]
    except Exception as e:
        stages.append({"stage": 3, "label": "LLM Qualification", "status": "error", "error": str(e)})
        results = [
            _format_result(QualificationResult(company=c, qualified=True, score=s, stage_reached="soft_score"))
            for c, s in scored if s > MIN_SOFT_SCORE
        ]

    elapsed = time.perf_counter() - t0
    response = {
        "query": q, "compiled": compiled, "pipeline_trace": pipeline_trace,
        "stages": stages, "results": results, "elapsed": round(elapsed, 3),
    }
    if no_data:
        response["no_data"] = True
        response["no_data_message"] = (
            f"No matching companies found. The NAICS industry filter was relaxed "
            f"(0 exact matches), and LLM verification confirmed that none of the "
            f"{len(scored)} remaining candidates match the query."
        )
    return response


@app.post("/api/batch")
def run_batch(body: BatchRequest):
    """Run the qualification pipeline on multiple queries sequentially."""

    if not companies or not analyzer:
        return JSONResponse(status_code=503, content={"error": "Pipeline not initialized"})

    if not body.queries:
        return JSONResponse(status_code=400, content={"error": "No queries provided"})

    if len(body.queries) > 50:
        return JSONResponse(status_code=400, content={"error": "Maximum 50 queries per batch"})

    batch_results = []
    t_batch = time.perf_counter()

    for i, q in enumerate(body.queries):
        q = q.strip()
        if not q:
            batch_results.append({"query": q, "error": "Empty query", "results": []})
            continue

        t0 = time.perf_counter()
        try:
            result = _run_query_inner(q, t0)
            if isinstance(result, JSONResponse):
                # Extract body from JSONResponse
                batch_results.append({"query": q, "error": "Internal error", "results": []})
            else:
                batch_results.append(result)
        except Exception as e:
            logger.exception(f"Batch query {i} failed: {q}")
            batch_results.append({
                "query": q,
                "error": str(e),
                "results": [],
                "elapsed": round(time.perf_counter() - t0, 3),
            })

    total_elapsed = round(time.perf_counter() - t_batch, 3)
    return {
        "batch_size": len(body.queries),
        "total_elapsed": total_elapsed,
        "results": batch_results,
    }


def _format_result(r: QualificationResult) -> dict:
    c = r.company
    return {
        "name": c.operational_name or "Unknown",
        "score": round(r.score, 4),
        "stage": r.stage_reached,
        "qualified": r.qualified,
        "reasoning": r.reasoning,
        "website": c.website,
        "address": c.address,
        "country_code": c.country_code,
        "year_founded": c.year_founded,
        "employee_count": c.employee_count,
        "revenue": c.revenue,
        "is_public": c.is_public,
        "naics": f"{c.primary_naics['code']} — {c.primary_naics['label']}" if c.primary_naics else None,
        "description": c.description,
        "business_model": c.business_model,
        "target_markets": c.target_markets,
        "core_offerings": c.core_offerings[:8] if c.core_offerings else None,
    }


# Serve Frontend 

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    frontend_path = Path(__file__).parent / "frontend.html"
    if frontend_path.exists():
        return frontend_path.read_text(encoding="utf-8")
    return "<h1>frontend.html not found</h1>"


# Main 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Company Qualifier Web UI")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to companies JSONL")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    args = parser.parse_args()

    load_companies(args.data)

    print(f"\n  ▸ Company Qualifier running at http://{args.host}:{args.port}")
    print(f"  ▸ {len(companies)} companies loaded, LLM: {'ON' if llm_client else 'OFF (no API key)'}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
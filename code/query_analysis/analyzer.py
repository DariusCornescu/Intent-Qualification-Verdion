"""
Stage 0: Query Analysis Pipeline -- Orchestrator

This coordinates the LLM-first query analysis:
  1. LLM extracts structured JSON (llm_extractor)
  2. Regex validates numerics (validators)
  3. Taxonomy normalizes locations and industries (validators)
  4. Match expression trees built from keywords (validators)
  5. Complexity post-validated with deterministic signals (validators)
  6. Confidence scored based on LLM <-> taxonomy agreement (validators)
  7. Training pair logged for Phase 2 fine-tuning (training_logger)

This file is the orchestrator only -> each step's implementation lives in its own module.
"""

import os
import time
import logging

from code.models import ParsedQuery
from code.query_analysis.llm_extractor import LLMExtractor
from code.query_analysis.validators import ( validate_numerics_with_regex, validate_locations_with_taxonomy, fix_region_from_query,
    expand_industry_with_taxonomy, validate_business_model, build_match_expressions, postvalidate_complexity, compute_confidence )
from code.query_analysis.training_logger import log_training_pair

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Orchestrates the Stage 0 query analysis pipeline (LLM-first).

    Pipeline steps:
      1. LLM extracts structured JSON from raw query
      2. Regex validates numeric fields (catches hallucinations)
      3. Taxonomy normalizes locations -> country codes, industry -> NAICS
      4. Build match expression trees from validated keywords
      5. Post-validate complexity with deterministic signals
      6. Assess confidence based on LLM <-> taxonomy agreement
      7. Log (query -> output) pair for Phase 2 fine-tuning

    Requires an LLM client (OpenAI-compatible). Will raise RuntimeError if no client is available.
    """

    def __init__(self, llm_client=None, llm_model: str = "gpt-4o-mini", training_log_path: str = None):
        self.extractor = LLMExtractor(llm_client, llm_model)
        self.training_log_path = training_log_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "training", "stage0_pairs.jsonl"
        )

    def analyze(self, query: str) -> ParsedQuery:
        """
        Main entry point. Requires an LLM client for query extraction.
        Returns a fully structured ParsedQuery.
        """
        if not self.extractor.llm_client:
            raise RuntimeError(
                "QueryAnalyzer requires an LLM client. "
                "Set OPENAI_API_KEY or pass llm_client to the constructor."
            )
        return self._analyze_llm_first(query)

    def _analyze_llm_first(self, query: str) -> ParsedQuery:
        """
        LLM extracts -> regex/taxonomy validate -> build match trees.
        """
        parsed = ParsedQuery(original_query=query)
        trace = parsed.trace

        trace.add("stage_0", "start", f'query: "{query}"')
        trace.add("stage_0", "path", "LLM-first")

        # Step 1: LLM extracts structured JSON
        t0 = time.perf_counter()
        llm_output = self.extractor.extract(query)
        llm_ms = (time.perf_counter() - t0) * 1000

        if llm_output is None:
            trace.add("stage_0", "llm_extract", f"FAILED ({llm_ms:.0f}ms)")
            raise RuntimeError(
                f'LLM extraction failed for query: "{query}". '
                "Check API key and connectivity."
            )

        cache_hint = " (cached)" if llm_ms < 50 else ""
        trace.add("stage_0", "llm_extract", f"success{cache_hint} ({llm_ms:.0f}ms)")

        # Step 2: Apply LLM output to parsed query
        self.extractor.apply_extraction(parsed, llm_output)

        # Step 3: Validate and normalize with regex/taxonomy
        validate_numerics_with_regex(query, parsed, llm_output)
        validate_locations_with_taxonomy(parsed, llm_output)
        fix_region_from_query(query, parsed)
        expand_industry_with_taxonomy(parsed, llm_output)
        validate_business_model(parsed, llm_output)

        # Step 4: Build match expressions from validated keywords
        build_match_expressions(parsed)
        n_expr = len(parsed.semantic_criteria.match_expressions)
        trace.add("stage_0", "match_tree", f"{n_expr} expression(s)")

        # Step 4b: Post-validate complexity
        postvalidate_complexity(query, parsed, llm_output)

        # Step 5: Compute confidence
        compute_confidence(query, parsed, llm_output)
        trace.add("stage_0", "assessment",
                   f"complexity={parsed.complexity.value}, confidence={parsed.confidence:.2f}")

        # Step 6: Log training pair for Phase 2
        log_training_pair(query, parsed, self.training_log_path)

        return parsed
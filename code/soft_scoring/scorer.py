"""
Stage 2: Soft Scoring - Orchestrator + Reciprocal Rank Fusion ( RRF )

Coordinates the five scoring signals and fuses them using RRF.
Why I chose RRF instead of weighted sum:
  - weighted sum requires normalizing scores to the same scale (BM25 returns 0-25, embeddings return 0-1). How to normalize? Each choice is arbitrary.
  - weighted sum requires tuning weights. 0.30 vs 0.35 - on what basis?
  - RRF operates on RANKS, not scores. Ranks are already normalized. A company that ranks highly across multiple signals naturally bubbles to the top.
  - also for taking this decision I used this paper: Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods - Cormack, Clarke, Buettcher (2009)
"""

import logging
from typing import Optional
from code.models import CompanyProfile, ParsedQuery
from code.soft_scoring.scoring_index import ScoringIndex
from code.soft_scoring.signals import ( compute_bm25_scores, compute_embedding_scores, compute_match_expression_scores, compute_naics_scores, compute_business_model_scores )

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard constant from the original paper

def score_and_rank( companies: list[CompanyProfile], parsed_query: ParsedQuery, top_k: int = 30, scoring_index: Optional[ScoringIndex] = None) -> list[tuple[CompanyProfile, float]]:
    """
    Score all companies against the parsed query and return top-K.
    Here I'm using RRF to combine multiple ranking signals without heuristic weights.

    return: list of (company, score) tuples sorted by RRF score descending.
    """
    if not companies:
        return []

    trace = parsed_query.trace
    n = len(companies)

    # Resolve index for filtered companies
    idx_map = scoring_index.get_indices(companies) if scoring_index else None
    cached = scoring_index is not None and idx_map is not None and len(idx_map) == n
    if cached:
        trace.add("stage_2", "index", f"using pre-computed index ({scoring_index.n} total)")
    else:
        trace.add("stage_2", "index", "no pre-computed index - computing from scratch")

    # Computing all signals
    signals: dict[str, list[float]] = {}

    keyword_scores = compute_bm25_scores(companies, parsed_query, scoring_index=scoring_index, idx_map=idx_map if cached else None)
    if max(keyword_scores) > 0: signals["bm25"] = keyword_scores

    embed_scores = compute_embedding_scores(companies, parsed_query, scoring_index=scoring_index, idx_map=idx_map if cached else None)
    if max(embed_scores) > 0: signals["embedding"] = embed_scores

    match_scores = compute_match_expression_scores(companies, parsed_query)
    if max(match_scores) > 0: signals["match_expr"] = match_scores

    naics_scores = compute_naics_scores(companies, parsed_query)
    if max(naics_scores) > 0: signals["naics"] = naics_scores

    bmodel_scores = compute_business_model_scores(companies, parsed_query)
    if max(bmodel_scores) > 0: signals["bmodel"] = bmodel_scores

    # Log active signals
    signal_summary = ", ".join( f"{name}(max={max(scores):.3f})" for name, scores in signals.items() )
    trace.add("stage_2", "signals", signal_summary if signals else "no signals active")

    if not signals: return [(c, 0.0) for c in companies[:top_k]]

    rrf_scores = _reciprocal_rank_fusion(signals, n)
    trace.add("stage_2", "fusion", f"RRF (k={RRF_K}) on {len(signals)} signals: {list(signals.keys())}")

    # Normalize scores to [0, 1] cause the values produce from RRF are hrader to interpret
    n_signals = len(signals)
    theoretical_max = n_signals * (1.0 / (RRF_K + 1))
    theoretical_min = n_signals * (1.0 / (RRF_K + n))

    denom = theoretical_max - theoretical_min
    if denom > 0:
        scored = [(companies[i], max(0.0, min(1.0, (rrf_scores[i] - theoretical_min) / denom))) for i in range(n)]
    else:
        scored = [(companies[i], 1.0 if rrf_scores[i] > 0 else 0.0) for i in range(n)]

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    trace.add("stage_2", "top_k", f"top {len(top)} (normalized: {top[0][1]:.4f} – {top[-1][1]:.4f})")

    logger.info( f"Soft scoring: {n} companies, {len(signals)} signals -> top {len(top)} (normalized: {top[0][1]:.4f} – {top[-1][1]:.4f})" )
    return top


def _reciprocal_rank_fusion( signals: dict[str, list[float]], n: int ) -> list[float]:
    """
    Combine multiple score lists into a single ranking using RRF.
        > RRF_score(i) = sum over signals of 1 / (k + rank_i)
    """
    rrf = [0.0] * n

    for signal_name, scores in signals.items():
        ranks = _scores_to_ranks(scores)
        for i in range(n):
            rrf[i] += 1.0 / (RRF_K + ranks[i])

    return rrf


def _scores_to_ranks(scores: list[float]) -> list[int]:
    """ Convert a list of scores to 1-based ranks (highest score = rank 1, and ties are getting the same rank ) """

    n = len(scores)
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

    ranks = [0] * n
    current_rank = 1
    for pos, idx in enumerate(sorted_indices):
        if pos > 0 and scores[idx] < scores[sorted_indices[pos - 1]]:
            current_rank = pos + 1
        ranks[idx] = current_rank

    return ranks
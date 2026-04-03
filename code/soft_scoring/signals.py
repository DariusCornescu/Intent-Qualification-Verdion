"""
Scoring signals for Stage 2 — each produces an independent ranked list.

Five signals, each capturing a different aspect of relevance:
  1. BM25 (keyword matching with length normalization)
  2. Semantic embeddings (dense vector similarity, understands synonyms)
  3. Match expressions (AND/OR tree evaluation from Stage 0)
  4. NAICS similarity (hierarchical industry code matching)
  5. Business model alignment (B2B/B2C/SaaS)

These signals are combined by Reciprocal Rank Fusion, each signal being independent 
"""

import logging
import numpy as np
from typing import Optional
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from code.models import ( CompanyProfile, ParsedQuery, MatchNode, MatchLeaf, MatchOperator )
from code.soft_scoring.scoring_index import ( ScoringIndex, _get_embedding_model, _check_bm25 )

logger = logging.getLogger(__name__)


# SIGNAL 1: BM25

def compute_bm25_scores( companies: list[CompanyProfile], parsed_query: ParsedQuery, scoring_index: Optional[ScoringIndex] = None, idx_map: Optional[list[int]] = None ) -> list[float]:
    """ BM25 keyword relevance scoring with document length normalization """

    keywords = parsed_query.semantic_criteria.all_keywords()
    if not keywords: return [0.0] * len(companies)
    query_tokens = " ".join(keywords).lower().split()

    if scoring_index and scoring_index.bm25_index and idx_map:
        try:
            all_scores = scoring_index.bm25_index.get_scores(query_tokens)
            return [float(all_scores[i]) for i in idx_map]
        except Exception:
            pass  #

    company_texts = [c.get_text_blob() or "" for c in companies]

    if _check_bm25():
        return _bm25_score(keywords, company_texts)

    return [0.0] * len(companies)

def _bm25_score(keywords: list[str], company_texts: list[str]) -> list[float]:
    """BM25Okapi scoring — proper keyword matching with length normalization."""

    tokenized = [text.lower().split() for text in company_texts]
    query_tokens = " ".join(keywords).lower().split()

    if not any(tokenized): return [0.0] * len(company_texts)

    try:
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query_tokens)
        return scores.tolist()
    except Exception:
        return [0.0] * len(company_texts)

# SIGNAL 2: Semantic Embeddings
def compute_embedding_scores(companies: list[CompanyProfile], parsed_query: ParsedQuery, scoring_index: Optional[ScoringIndex] = None, idx_map: Optional[list[int]] = None ) -> list[float]:
    """ Semantic similarity using sentence-transformer embeddings """

    model = _get_embedding_model()
    if model is None: return [0.0] * len(companies)

    keywords = parsed_query.semantic_criteria.all_keywords()
    if not keywords: return [0.0] * len(companies)
    query_text = " ".join(keywords)

    if scoring_index and scoring_index.embedding_matrix is not None and idx_map:
        try:
            query_embedding = model.encode([query_text], show_progress_bar=False)
            query_embedding = query_embedding.reshape(1, -1)
            company_embeddings = scoring_index.embedding_matrix[idx_map]
            similarities = cosine_similarity(query_embedding, company_embeddings).flatten()

            return [max(0.0, float(s)) for s in similarities]

        except Exception as e:
            logger.warning(f"Cached embedding lookup failed: {e}, falling back")

    company_texts = [company.get_text_blob() or "" for company in companies]
    non_empty_mask = [bool(text.strip()) for text in company_texts]
    texts_to_encode = [text for text, m in zip(company_texts, non_empty_mask) if m]

    if not texts_to_encode: return [0.0] * len(companies)

    try:

        all_texts = [query_text] + texts_to_encode
        embeddings = model.encode(all_texts, show_progress_bar=False, batch_size=64)

        query_embedding = embeddings[0].reshape(1, -1)
        company_embeddings = embeddings[1:]
        similarities = cosine_similarity(query_embedding, company_embeddings).flatten()

        scores = [0.0] * len(companies)
        sim_idx = 0
        for i, has_text in enumerate(non_empty_mask):
            if has_text:
                scores[i] = max(0.0, float(similarities[sim_idx]))
                sim_idx += 1

        return scores

    except Exception as e:
        logger.warning(f"Embedding scoring failed: {e}")
        return [0.0] * len(companies)

# SIGNAL 3: Match Expression Trees
def compute_match_expression_scores( companies: list[CompanyProfile], parsed_query: ParsedQuery ) -> list[float]:
    """ Evaluating AND/OR match expression trees against each company's text, returning 0.0–1.0 based on how well the text matches the logical structure. """

    expressions = parsed_query.semantic_criteria.match_expressions
    if not expressions: return [0.0] * len(companies)

    scores = []
    for company in companies:
        text = (company.get_text_blob() or "").lower()
        if not text:
            scores.append(0.0)
            continue

        expr_scores = [_evaluate_node(expr, text) for expr in expressions]
        scores.append(max(expr_scores) if expr_scores else 0.0)

    return scores


def _evaluate_node(node, text: str) -> float:
    """
    Recursively evaluate a match expression node.
        > OR: fraction of matching operands (soft OR — rewards breadth).
        > AND: minimum score across operands (strict AND).
    """

    if isinstance(node, MatchLeaf):
        return 1.0 if node.value.lower() in text else 0.0

    if isinstance(node, MatchNode):
        child_scores = [_evaluate_node(child, text) for child in node.operands]
        if not child_scores: return 0.0

        if node.operator == MatchOperator.OR:
            return sum(child_scores) / len(child_scores)
        else:
            return min(child_scores)

    return 0.0


# SIGNAL 4: NAICS Similarity

def compute_naics_scores( companies: list[CompanyProfile], parsed_query: ParsedQuery ) -> list[float]:
    """ Hierarchical NAICS code matching. Full 6-digit match = 1.0. Partial prefix match => proportional score """

    target_codes = parsed_query.hard_filters.naics_codes
    if not target_codes: return [0.0] * len(companies)

    scores = []
    for company in companies:
        company_codes = company.get_all_naics_codes()
        if not company_codes:
            scores.append(0.0)
            continue

        best_score = 0.0
        for code_company in company_codes:
            for target_code in target_codes:
                score = _naics_similarity(code_company, target_code)
                best_score = max(best_score, score)

        scores.append(best_score)

    return scores


def _naics_similarity(code1: str, code2: str) -> float:
    """Hierarchical prefix matching between two NAICS codes."""

    min_len = min(len(code1), len(code2))
    matching = 0
    for i in range(min_len):
        if code1[i] == code2[i]:
            matching += 1
        else:
            break

    if matching == 0: return 0.0

    max_len = max(len(code1), len(code2))
    return matching / max_len


# SIGNAL 5: Business Model Alignment

def compute_business_model_scores( companies: list[CompanyProfile], parsed_query: ParsedQuery ) -> list[float]:
    """Score based on business model keyword overlap (B2B, SaaS, etc.)."""

    target_models = parsed_query.semantic_criteria.business_model_keywords
    if not target_models: return [0.0] * len(companies)
    
    target_set = set(m.lower() for m in target_models)
    scores = []

    for company in companies:
        if not company.business_model:
            scores.append(0.0)
            continue

        company_models = set(m.lower() for m in company.business_model)
        overlap = len(target_set & company_models)
        score = overlap / len(target_set) if target_set else 0.0
        scores.append(score)

    return scores
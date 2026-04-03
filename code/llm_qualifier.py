import json
import logging
from typing import Optional
from code.models import CompanyProfile, ParsedQuery, QualificationResult

logger = logging.getLogger(__name__)
BATCH_SIZE = 8


def qualify_with_llm( candidates: list[tuple[CompanyProfile, float]], parsed_query: ParsedQuery, llm_client, model: str = "gpt-4o-mini") -> list[QualificationResult]:

    results = []
    batches = _create_batches(candidates, BATCH_SIZE)
    logger.info( f"LLM qualification: {len(candidates)} candidates in {len(batches)} batches of ≤{BATCH_SIZE}" )

    for batch_idx, batch in enumerate(batches):
        try:
            batch_results = _qualify_batch( batch, parsed_query, llm_client, model )
            results.extend(batch_results)
        except Exception as e:
            logger.warning( f"Batch {batch_idx} LLM call failed: {e}.Falling back to soft scores.")

            for company, score in batch:
                results.append(QualificationResult( company=company, qualified=score > 0.3, score=score, stage_reached="soft_score_fallback", reasoning="LLM qualification failed, using soft score" ))

    results.sort(key=lambda result: result.score, reverse=True)
    return results


def _create_batches( items: list[tuple[CompanyProfile, float]], batch_size: int ) -> list[list[tuple[CompanyProfile, float]]]:

    batches = []
    for index in range(0, len(items), batch_size):
        batch = items[index:index+batch_size]
        batches.append(batch)

    return batches

def _qualify_batch( batch: list[tuple[CompanyProfile, float]], parsed_query: ParsedQuery, llm_client, model: str) -> list[QualificationResult]:

    prompt = _build_batch_prompt(batch, parsed_query)

    response = llm_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "You are a company qualification expert. "
                "Determine if each company genuinely matches the user's query. "
                "Consider the full intent, not just keyword overlap. "
                "Return ONLY valid JSON."
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0, max_tokens=1000,
        response_format={"type": "json_object"},
    )
    response_text = response.choices[0].message.content
    llm_results = json.loads(response_text)

    return _parse_batch_results(batch, llm_results)


def _build_batch_prompt( batch: list[tuple[CompanyProfile, float]], parsed_query: ParsedQuery ) -> str:

    companies_text = []
    for i, (company, score) in enumerate(batch):
        profile = _format_company_profile(company, i)
        companies_text.append(profile)

    companies_block = "\n\n".join(companies_text)

    return (
        f"User query: \"{parsed_query.original_query}\"\n\n"
        f"For each company below, determine if it genuinely matches "
        f"the user's intent (not just surface-level keyword similarity).\n\n"
        f"{companies_block}\n\n"
        f"Return a JSON object with a \"results\" array. "
        f"Each element should have:\n"
        f"- \"index\": company index (0-based)\n"
        f"- \"qualified\": true/false\n"
        f"- \"confidence\": 0.0-1.0\n"
        f"- \"reasoning\": brief explanation (1-2 sentences)\n\n"
        f"Example: {{\"results\": ["
        f"{{\"index\": 0, \"qualified\": true, \"confidence\": 0.9, "
        f"\"reasoning\": \"Direct match...\"}}]}}"
    )


def _format_company_profile(company: CompanyProfile, index: int) -> str:

    parts = [f"[Company {index}] {company.operational_name or 'Unknown'}"]

    if company.description:
        desc = company.description[:500]
        if len(company.description) > 500:
            desc += "..."
        parts.append(f"  Description: {desc}")

    if company.address:
        parts.append(f"  Location: {company.address}")

    if company.primary_naics:
        label = company.primary_naics.get("label", "")
        code = company.primary_naics.get("code", "")
        parts.append(f"  Industry: {label} ({code})")

    if company.core_offerings:
        parts.append(f"  Products: {', '.join(company.core_offerings[:5])}")

    if company.business_model:
        parts.append(f"  Business model: {', '.join(company.business_model)}")

    if company.target_markets:
        parts.append(f"  Target markets: {', '.join(company.target_markets[:5])}")
        
    if company.employee_count is not None:
        parts.append(f"  Employees: {company.employee_count}")

    return "\n".join(parts)


def _parse_batch_results( batch: list[tuple[CompanyProfile, float]], llm_results: dict ) -> list[QualificationResult]:

    results = []
    llm_items = llm_results.get("results", [])
    llm_by_index = {}
    for item in llm_items:
        idx = item.get("index")
        if idx is not None:
            llm_by_index[idx] = item

    for i, (company, soft_score) in enumerate(batch):
        llm_item = llm_by_index.get(i)

        if llm_item is None:
            # LLM skipped this company - fall back to soft score
            results.append( QualificationResult( company=company, qualified=soft_score > 0.3, score=soft_score, stage_reached="soft_score_fallback", reasoning="LLM did not evaluate this company" ))
            continue

        qualified = llm_item.get("qualified", False)
        llm_confidence = llm_item.get("confidence", 0.5)
        reasoning = llm_item.get("reasoning", "")
        final_score = 0.3 * soft_score + 0.7 * llm_confidence

        results.append(QualificationResult( company=company, qualified=qualified, score=final_score, stage_reached="llm_qualified", reasoning=reasoning ))


    return results
"""
Training data logger for Phase 2 fine-tuning.

Logs every (query -> structured output) pair produced by the LLM-first pipeline. These pairs accumulate as free 
training data - at 500+ examples, they will be used to fine-tune a local model. Each logged entry includes the 
full extraction result and confidence score, so low-confidence pairs can be filtered out during training.
"""

import json
import os
import logging
from datetime import datetime, timezone
from code.models import ParsedQuery

logger = logging.getLogger(__name__)

def log_training_pair(query: str, parsed: ParsedQuery, log_path: str):
    """
    Log (query -> structured output) for Phase 2 fine-tuning. Each line is a JSON object that can be used directly as training data.
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        pair = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": query,
            "output": {
                "locations": parsed.hard_filters.locations,
                "geo_lat": parsed.hard_filters.geo_lat,
                "geo_lng": parsed.hard_filters.geo_lng,
                "geo_radius_km": parsed.hard_filters.geo_radius_km,
                "employee_count_min": parsed.hard_filters.employee_count_min,
                "employee_count_max": parsed.hard_filters.employee_count_max,
                "revenue_min": parsed.hard_filters.revenue_min,
                "revenue_max": parsed.hard_filters.revenue_max,
                "year_founded_min": parsed.hard_filters.year_founded_min,
                "year_founded_max": parsed.hard_filters.year_founded_max,
                "is_public": parsed.hard_filters.is_public,
                "industry_keywords": parsed.semantic_criteria.industry_keywords[:20],
                "business_model_keywords": parsed.semantic_criteria.business_model_keywords,
                "naics_codes": parsed.hard_filters.naics_codes,
                "complexity": parsed.complexity.value,
            },
            "confidence": parsed.confidence,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    except Exception as e:
        logger.debug(f"Training pair logging failed: {e}")
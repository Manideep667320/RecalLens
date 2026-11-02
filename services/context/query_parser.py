"""
Query Context Parser for RecalLens
Extracts Entity, Event, Temporal reference, Relationship, and Target Information from natural language search queries.
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
try:
    from models import QueryContext
except ImportError:
    from services.models import QueryContext
from .temporal_parser import TemporalParser

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")

# Heuristic vocabulary lists for fast deterministic fallback
TARGET_INFO_WORDS = {
    "receipt": ["receipt", "slip", "proof of purchase"],
    "warranty": ["warranty", "guarantee", "protection plan"],
    "invoice": ["invoice", "tax invoice"],
    "bill": ["bill", "utility bill", "electric bill", "water bill"],
    "service_record": ["service record", "repair record", "service slip"],
    "ticket": ["ticket", "boarding pass", "flight ticket"],
    "prescription": ["prescription", "rx", "medical report", "test report"],
    "contract": ["contract", "lease", "agreement", "policy document"]
}

EVENT_WORDS = {
    "repair": ["repair", "fix", "servicing", "service", "maintenance", "repaired"],
    "purchase": ["purchase", "bought", "buy", "ordered", "purchased"],
    "billing": ["bill", "billed", "charged", "payment", "due"],
    "submission": ["submission", "submitted", "apply", "applied", "form submission"],
    "travel": ["travel", "flight", "trip", "hotel", "journey", "visited"],
    "medical": ["visit", "appointment", "checkup", "consultation", "treatment"]
}

ITEM_ENTITIES = [
    "laptop", "smartphone", "phone", "macbook", "computer", "washing machine",
    "refrigerator", "fridge", "tv", "television", "car", "vehicle", "bike",
    "watch", "camera", "tablet", "printer", "device"
]


class QueryContextParser:
    """Parses natural language queries into structured personal context components"""

    def __init__(self, provider=None, default_language: str = "English"):
        self.provider = provider
        self.default_language = default_language
        self.temporal_parser = TemporalParser()

    def parse(self, query: str, ref_date: Optional[datetime] = None) -> QueryContext:
        """
        Extracts structured context from user search query using LLM with deterministic fallback.
        
        Args:
            query: Natural language query (e.g. "Find the warranty for the laptop I bought last year")
            ref_date: Optional reference date for relative temporal resolution
        """
        if not query or not query.strip():
            return QueryContext()

        clean_query = query.strip()
        parsed_context: Optional[QueryContext] = None

        # 1. Try LLM-based extraction if provider is available
        if self.provider is not None:
            parsed_context = self._extract_with_llm(clean_query)

        # 2. Fallback to rule-based parser if LLM extraction returned incomplete result
        if parsed_context is None or not (parsed_context.entity or parsed_context.event or parsed_context.time_reference):
            heuristic_ctx = self._extract_heuristic(clean_query)
            if parsed_context is None:
                parsed_context = heuristic_ctx
            else:
                # Merge heuristic fields into missing LLM fields
                if not parsed_context.entity and heuristic_ctx.entity:
                    parsed_context.entity = heuristic_ctx.entity
                if not parsed_context.event and heuristic_ctx.event:
                    parsed_context.event = heuristic_ctx.event
                if not parsed_context.time_reference and heuristic_ctx.time_reference:
                    parsed_context.time_reference = heuristic_ctx.time_reference
                if not parsed_context.target_information and heuristic_ctx.target_information:
                    parsed_context.target_information = heuristic_ctx.target_information
                if not parsed_context.relationship and heuristic_ctx.relationship:
                    parsed_context.relationship = heuristic_ctx.relationship

        # 3. Resolve temporal constraints into concrete date ranges
        if parsed_context.time_reference:
            date_range = self.temporal_parser.parse(parsed_context.time_reference, ref_date=ref_date)
            parsed_context.resolved_date_start = date_range.start_date
            parsed_context.resolved_date_end = date_range.end_date

        logger.info(
            f"QueryContext parsed: entity='{parsed_context.entity}', event='{parsed_context.event}', "
            f"time='{parsed_context.time_reference}' ({parsed_context.resolved_date_start} to {parsed_context.resolved_date_end}), "
            f"target='{parsed_context.target_information}'"
        )
        return parsed_context

    def _extract_with_llm(self, query: str) -> Optional[QueryContext]:
        """Calls LLM with prompt template to extract structured query context"""
        prompt_path = os.path.join(PROMPTS_DIR, "query_context_extraction.txt")
        if not os.path.exists(prompt_path):
            return None

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()

            prompt = template.format(query=query, language=self.default_language)
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=256
            )

            output = response_data.get("response", "").strip()
            if not output and response_data.get("thinking"):
                output = response_data.get("thinking", "").strip()

            # Parse JSON from output
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return QueryContext(
                    entity=data.get("entity") if data.get("entity") != "null" else None,
                    event=data.get("event") if data.get("event") != "null" else None,
                    time_reference=data.get("time_reference") if data.get("time_reference") != "null" else None,
                    relationship=data.get("relationship") if data.get("relationship") != "null" else None,
                    target_information=data.get("target_information") if data.get("target_information") != "null" else None
                )

        except Exception as e:
            logger.warning(f"LLM query context extraction failed: {e}")

        return None

    def _extract_heuristic(self, query: str) -> QueryContext:
        """Deterministic rule-based extractor for fast query parsing"""
        q_lower = query.lower()

        # Target info
        target_info = None
        for t_type, words in TARGET_INFO_WORDS.items():
            if any(w in q_lower for w in words):
                target_info = t_type
                break

        # Event
        event = None
        for ev_type, words in EVENT_WORDS.items():
            if any(w in q_lower for w in words):
                event = ev_type
                break

        # Entity (Item or Person)
        entity = None
        for item in sorted(ITEM_ENTITIES, key=len, reverse=True):
            if re.search(rf'\b{item}s?\b', q_lower):
                entity = item
                break

        # If no item found, check for proper nouns (e.g. "in Patna", "from Metro Electronics")
        if not entity:
            m_entity = re.search(r'(?:from|with|at|by)\s+([A-Z][a-zA-Z\s]+?)(?=[.,\n]|\s+(?:in|on|at|around)|$)', query)
            if m_entity:
                entity = m_entity.group(1).strip()

        # Time reference
        time_ref = None
        time_patterns = [
            r'\b(?:in|around|during|from)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b',
            r'\b(?:in|around|during)?\s*(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(last year|this year|previous year)\b',
            r'\b(last month|this month)\b',
            r'\b(last\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))\b',
            r'\b(20\d{2})\b',
            r'\b(\d{4}-\d{2}-\d{2})\b'
        ]

        for p in time_patterns:
            m_t = re.search(p, q_lower)
            if m_t:
                time_ref = m_t.group(0).strip()
                break

        # Relationship
        relationship = None
        if entity and event:
            relationship = f"{entity} -> {event}"

        return QueryContext(
            entity=entity,
            event=event,
            time_reference=time_ref,
            relationship=relationship,
            target_information=target_info
        )

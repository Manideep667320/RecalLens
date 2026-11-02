"""
Query processing with LLM-based expansion and contextual understanding for RecalLens
"""

import re
import json
import logging
import os
from typing import Dict, Any, List, Optional
import requests
from models import ExpandedQuery, QueryContext
from providers import BaseProvider
from context import QueryContextParser

logger = logging.getLogger(__name__)

# Constants
MAX_SENTENCE_LENGTH = 200  # Maximum sentence length before truncation in highlights

# Load prompt templates
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory"""
    with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


class QueryProcessor:
    """Processes and expands search queries using LLM with entity/event/temporal understanding"""

    def __init__(self, provider: BaseProvider, default_language: str = "English"):
        self.provider = provider
        self.default_language = default_language
        self.context_parser = QueryContextParser(provider=provider, default_language=default_language)

    def expand_query(self, query: str, enable_expansion: bool = True) -> ExpandedQuery:
        """
        Expand the search query using LLM and parse personal context (entities, events, dates, relations).
        """
        # 1. Parse personal context (entities, events, temporal references)
        query_ctx: QueryContext = self.context_parser.parse(query)
        extracted_filters: Dict[str, Any] = {}

        # If date range was resolved from relative text (e.g. "January 2024", "last year"), inject date filter
        if query_ctx.resolved_date_start or query_ctx.resolved_date_end:
            extracted_filters['created_date'] = {}
            if query_ctx.resolved_date_start:
                extracted_filters['created_date']['gte'] = query_ctx.resolved_date_start
            if query_ctx.resolved_date_end:
                extracted_filters['created_date']['lte'] = query_ctx.resolved_date_end

        if not enable_expansion:
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                extracted_filters=extracted_filters,
                keywords=[],
                context=query_ctx
            )

        try:
            # 2. Build prompt for lexical/semantic query expansion
            prompt = self._build_query_expansion_prompt(query)

            # Call provider's LLM
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.1,
                max_tokens=512
            )

            llm_output = response_data.get('response', '').strip()
            thinking_output = response_data.get('thinking', '').strip()

            if not llm_output and thinking_output:
                llm_output = thinking_output

            # Parse LLM output
            expanded = self._parse_llm_output(query, llm_output)
            expanded.context = query_ctx
            expanded.extracted_filters.update(extracted_filters)

            if expanded.expanded_query != query:
                logger.info(f"Query expanded: '{query}' -> '{expanded.expanded_query}'")

            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query with parsed context")
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                extracted_filters=extracted_filters,
                keywords=[],
                context=query_ctx
            )

    def _build_query_expansion_prompt(self, query: str) -> str:
        """Build prompt for LLM query expansion"""
        template = load_prompt("query_expansion.txt")
        return template.format(query=query, language=self.default_language)

    def _parse_llm_output(self, original_query: str, llm_output: str) -> ExpandedQuery:
        """Parse LLM JSON output into ExpandedQuery"""
        try:
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                expanded_query = parsed.get('expanded_query', original_query)

                return ExpandedQuery(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    extracted_filters={},
                    keywords=[]
                )

        except Exception as e:
            logger.warning(f"Failed to parse LLM output: {e}")

        # Fallback: use original query
        return ExpandedQuery(
            original_query=original_query,
            expanded_query=original_query,
            extracted_filters={},
            keywords=[]
        )

    def extract_highlights(self, text: str, query: str, max_highlights: int = 3) -> List[str]:
        """
        Extract text snippets that match the query for highlighting.
        """
        if not text or not query:
            return []

        highlights = []
        query_terms = query.lower().split()

        # Split text into sentences
        sentences = re.split(r'[.!?]\s+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                if len(sentence) > MAX_SENTENCE_LENGTH:
                    for term in query_terms:
                        if term in sentence_lower:
                            pos = sentence_lower.index(term)
                            start = max(0, pos - 100)
                            end = min(len(sentence), pos + 100)
                            snippet = sentence[start:end]
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(sentence):
                                snippet = snippet + "..."
                            highlights.append(snippet)
                            break
                else:
                    highlights.append(sentence)

                if len(highlights) >= max_highlights:
                    break

        return highlights

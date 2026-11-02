"""
Cross-encoder and Context-Aware Reranking for RecalLens
Implements multi-signal scoring:
    S(d, q) = w_s * S_s + w_e * S_e + w_v * S_v + w_t * S_t + w_r * S_r
"""

import logging
import math
import os
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re
try:
    from models import SearchResult, QueryContext
    from providers import BaseProvider, EmbeddingTaskType
    from retrieval.context_scorer import ContextScorer
    from storage import get_postgres_client
except ImportError:
    from services.models import SearchResult, QueryContext
    from services.providers import BaseProvider, EmbeddingTaskType
    from services.retrieval.context_scorer import ContextScorer
    from services.storage import get_postgres_client

logger = logging.getLogger(__name__)

# Constants
RERANK_SCORE_VARIANCE_THRESHOLD = 0.01  # Minimum variance indicating reranking is working

# Load prompt templates
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory"""
    with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()


class Reranker:
    """Reranks search results using cross-encoder and context-aware scoring"""

    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.use_cross_encoder = self._check_cross_encoder_available()
        self.rerank_mode = None
        self.failed_llm_calls = 0
        self.context_scorer = ContextScorer()
        self.postgres_client = get_postgres_client()

        # Configurable max_tokens for different operations (from env vars)
        self.dedicated_max_tokens = int(os.getenv('RERANK_DEDICATED_MAX_TOKENS', '10'))
        self.llm_max_tokens = int(os.getenv('RERANK_LLM_MAX_TOKENS', '500'))
        self.explain_max_tokens = int(os.getenv('RERANK_EXPLAIN_MAX_TOKENS', '1500'))
        self.max_workers = int(os.getenv('RERANK_MAX_WORKERS', '20'))

        logger.info(
            f"Reranker initialized with ContextScorer. max_workers={self.max_workers}, "
            f"max_tokens: dedicated={self.dedicated_max_tokens}, llm={self.llm_max_tokens}, explain={self.explain_max_tokens}"
        )

    def _check_cross_encoder_available(self) -> bool:
        """Check if cross-encoder model is available"""
        try:
            models = self.provider.list_models()
            model_names = [name.lower() for name in models]
            available = any('rerank' in name for name in model_names)
            if available:
                logger.info("Dedicated cross-encoder/rerank model available")
            else:
                logger.info("No dedicated rerank model found, using LLM-based scoring")
            return available
        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365,
        query_context: Optional[QueryContext] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using context-aware multi-signal scoring S(d, q).
        """
        if not results:
            return []

        try:
            reranked = self._rerank_with_cross_encoder(
                query, results, top_k, recency_weight, recency_decay_days, query_context
            )
            logger.info(f"Reranked {len(results)} results to top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results with context scoring")
            for result in results[:top_k]:
                final_s, _, meta = self.context_scorer.score_document(result, query_context, result.score)
                result.final_score = final_s
                result.rerank_score = final_s * 10.0
                result.evidence = meta.get("evidence")
            return results[:top_k]

    def rerank_with_progress(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365,
        query_context: Optional[QueryContext] = None
    ):
        """
        Generator yielding results as they are reranked in parallel.
        """
        if not results:
            return

        import time
        start_time = time.time()
        max_workers = min(self.max_workers, len(results))
        logger.info(f"Starting parallel progressive reranking of {len(results)} results with {max_workers} workers")

        completion_count = 0
        failures = 0
        successes = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(
                    self._score_single_result,
                    query,
                    result,
                    recency_weight,
                    recency_decay_days,
                    query_context
                ): idx for idx, result in enumerate(results)
            }

            for future in as_completed(future_to_index):
                result, score, success = future.result()
                if success and score is not None:
                    successes += 1
                else:
                    failures += 1

                completion_count += 1
                yield result

        self.failed_llm_calls += failures
        self.failed_llm_calls = max(0, self.failed_llm_calls - successes)

        elapsed = time.time() - start_time
        logger.info(f"Parallel reranking completed: {completion_count} results in {elapsed:.2f}s")

    def _score_single_result(
        self,
        query: str,
        result: SearchResult,
        recency_weight: float,
        recency_decay_days: int,
        query_context: Optional[QueryContext] = None
    ) -> Tuple[SearchResult, Optional[float], bool]:
        """
        Calculates S(d, q) = w_s*S_s + w_e*S_e + w_v*S_v + w_t*S_t + w_r*S_r
        """
        import time
        import threading
        start_time = time.time()
        thread_id = threading.current_thread().name

        try:
            doc_id_val = result.recallens_id or result.paperless_id
            logger.info(f"[{thread_id}] Scoring doc {doc_id_val}")

            # 1. Semantic score S_s via cross-encoder / LLM
            context = self._build_result_context(result)
            semantic_score = self._score_with_reranker(query, context)
            if semantic_score is None:
                # Use vector similarity score normalized
                semantic_score = max(0.1, min(1.0, result.score))

            # 2. Fetch stored context details from PostgreSQL if available
            ctx_details: Dict[str, Any] = {}
            if self.postgres_client and self.postgres_client.is_connected:
                try:
                    ctx_details = self.postgres_client.get_document_context_details(str(doc_id_val))
                except Exception:
                    pass

            # 3. Compute multi-signal score S(d, q)
            final_score, breakdown, metadata = self.context_scorer.score_document(
                result=result,
                query_context=query_context,
                semantic_score=semantic_score,
                context_details=ctx_details
            )

            # 4. Populate result attributes
            result.context_score = final_score
            result.final_score = final_score
            result.rerank_score = final_score * 10.0  # 0-10 scale for UI
            result.matched_entities = metadata.get("matched_entities", [])
            result.matched_event = metadata.get("matched_event")
            result.matched_relationship = metadata.get("matched_relationship")
            result.evidence = metadata.get("evidence")

            # 5. Apply recency boost if requested
            if recency_weight > 0:
                result.recency_boost = self._calculate_recency_boost(result, recency_weight, recency_decay_days)
                result.final_score = min(1.0, result.final_score * (1.0 + result.recency_boost))
                result.rerank_score = result.final_score * 10.0
            else:
                result.recency_boost = 0.0

            elapsed = time.time() - start_time
            logger.info(
                f"[{thread_id}] Doc {doc_id_val}: S={final_score:.4f} "
                f"(S_s={breakdown['semantic']}, S_e={breakdown['entity']}, S_v={breakdown['event']}, "
                f"S_t={breakdown['temporal']}, S_r={breakdown['relationship']}) in {elapsed:.2f}s"
            )

            return (result, result.final_score, True)

        except Exception as e:
            logger.warning(f"Failed to rerank result {result.chunk_id}: {e}")
            result.final_score = result.score
            result.rerank_score = result.score * 10.0
            result.recency_boost = 0.0
            return (result, result.score, False)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365,
        query_context: Optional[QueryContext] = None
    ) -> List[SearchResult]:
        """Parallel multi-signal reranking"""
        import time
        start_time = time.time()

        scored_results = []
        max_workers = min(self.max_workers, len(results))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_result = {
                executor.submit(
                    self._score_single_result,
                    query,
                    result,
                    recency_weight,
                    recency_decay_days,
                    query_context
                ): result for result in results
            }

            for future in as_completed(future_to_result):
                res, _, _ = future.result()
                scored_results.append(res)

        # Sort by final_score descending
        scored_results.sort(key=lambda x: (x.final_score if x.final_score is not None else x.score), reverse=True)
        return scored_results[:top_k]

    def _score_with_reranker(self, query: str, context: str) -> Optional[float]:
        """Score query-context relevance using dedicated model or LLM"""
        if self.use_cross_encoder:
            return self._score_with_dedicated_reranker(query, context)
        return self._score_relevance_with_llm(query, context)

    def _score_with_dedicated_reranker(self, query: str, context: str) -> Optional[float]:
        """Use dedicated classification reranker model"""
        try:
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            template = load_prompt("reranker_dedicated.txt")
            prompt = template.format(
                instruction=instruction,
                query=query,
                document=context[:512]
            )

            data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.dedicated_max_tokens,
                logprobs=True,
                top_logprobs=10
            )

            if "logprobs" in data and data["logprobs"]:
                first_token = data["logprobs"][0]
                token_probs = {}
                if "top_logprobs" in first_token:
                    for t_info in first_token["top_logprobs"]:
                        t_text = t_info.get("token", "").lower().strip()
                        token_probs[t_text] = t_info.get("logprob", -100)

                yes_logprob = token_probs.get("yes", token_probs.get("yes.", -100))
                no_logprob = token_probs.get("no", token_probs.get("no.", -100))

                if yes_logprob != -100 or no_logprob != -100:
                    yes_prob = math.exp(yes_logprob)
                    no_prob = math.exp(no_logprob)
                    total = yes_prob + no_prob
                    if total > 0:
                        return yes_prob / total

        except Exception as e:
            logger.warning(f"Dedicated reranker scoring failed: {e}")

        return None

    def _score_relevance_with_llm(self, query: str, context: str) -> Optional[float]:
        """Score query-document relevance via text generation"""
        context_excerpt = context[:400]
        template = load_prompt("reranker_llm_scoring.txt")
        prompt = template.format(query=query, document=context_excerpt)

        try:
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.llm_max_tokens
            )
            output = response_data.get('response', '').strip()
            match = re.search(r'(\d+(?:\.\d+)?)', output)
            if match:
                raw_score = float(match.group())
                return max(0.0, min(1.0, raw_score / 10.0))

        except Exception as e:
            logger.warning(f"LLM relevance scoring failed: {e}")

        return None

    def explain_score(
        self,
        query: str,
        result_context: str,
        matched_entities: Optional[List[str]] = None,
        matched_event: Optional[str] = None,
        matched_relationship: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable explanation combining contextual matches and LLM rationale.
        """
        context_parts = []
        if matched_entities:
            context_parts.append(f"Matched Entities: {', '.join(matched_entities)}")
        if matched_event:
            context_parts.append(f"Matched Event: {matched_event}")
        if matched_relationship:
            context_parts.append(f"Matched Relationship: {matched_relationship}")

        ctx_summary = " | ".join(context_parts)

        try:
            template = load_prompt("explain_relevance.txt")
            prompt = template.format(query=query, document=result_context[:400])
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.explain_max_tokens
            )
            output = response_data.get('response', '').strip()
            if ctx_summary:
                return f"{output}\n[Context Proof: {ctx_summary}]"
            return output

        except Exception as e:
            if ctx_summary:
                return f"Match Proof: {ctx_summary}"
            return f"Error generating explanation: {str(e)}"

    def _calculate_recency_boost(
        self,
        result: SearchResult,
        recency_weight: float,
        recency_decay_days: int
    ) -> float:
        """Calculates exponential recency decay boost"""
        if not result.created_date or recency_weight == 0:
            return 0.0

        try:
            if isinstance(result.created_date, str):
                c_dt = datetime.fromisoformat(result.created_date.replace('Z', '+00:00'))
            else:
                c_dt = datetime.fromtimestamp(result.created_date, tz=timezone.utc)

            if c_dt.tzinfo is None:
                c_dt = c_dt.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - c_dt).days
            decay = 0.5 ** (age_days / recency_decay_days)
            return recency_weight * decay
        except Exception:
            return 0.0

    def _build_result_context(self, result: SearchResult) -> str:
        """Builds context string from search result for reranking"""
        parts = [f"Title: {result.title}"]
        if result.summary:
            parts.append(f"Summary: {result.summary}")
        if result.tags:
            parts.append(f"Tags: {', '.join(result.tags)}")
        if result.correspondent_name:
            parts.append(f"From: {result.correspondent_name}")
        if result.content:
            parts.append(f"Content: {result.content[:500]}")
        return " | ".join(parts)

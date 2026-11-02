"""
Reciprocal Rank Fusion (RRF) for RecalLens
Combines unstructured vector candidates with structured entity/event/metadata candidates.
"""

import logging
from typing import List, Dict, Any, Optional, Union
try:
    from models import SearchResult
except ImportError:
    from services.models import SearchResult

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) to merge and calibrate candidate rankings:
        RRF_score(d) = sum_{m in M} ( w_m / (k + rank_m(d)) )
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: Smoothing constant to control the influence of high vs low ranks (default: 60).
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, List[SearchResult]],
        weights: Optional[Dict[str, float]] = None,
        top_k: int = 50
    ) -> List[SearchResult]:
        """
        Fuses multiple ranked lists of SearchResult candidates.

        Args:
            ranked_lists: Dict mapping ranker name (e.g. 'vector', 'structured', 'keyword')
                          to a list of SearchResult objects ordered by rank.
            weights: Optional dict mapping ranker name to relative weight (defaults to 1.0 each).
            top_k: Maximum number of merged results to return.

        Returns:
            Deduplicated List[SearchResult] sorted by fused RRF score.
        """
        if not ranked_lists:
            return []

        if isinstance(ranked_lists, list):
            ranked_dict = {f"ranker_{i}": lst for i, lst in enumerate(ranked_lists)}
            if isinstance(weights, list):
                weights = {f"ranker_{i}": w for i, w in enumerate(weights)}
            else:
                weights = weights or {}
            ranked_lists = ranked_dict
        else:
            weights = weights or {}

        rrf_scores: Dict[int, float] = {}          # recallens_id -> fused score
        doc_registry: Dict[int, SearchResult] = {} # recallens_id -> best SearchResult instance
        ranker_appearances: Dict[int, List[str]] = {}

        # 1. Calculate reciprocal rank scores
        for ranker_name, results in ranked_lists.items():
            if not results:
                continue

            weight = weights.get(ranker_name, 1.0)

            for rank_idx, result in enumerate(results):
                doc_id = result.recallens_id or result.paperless_id
                rank = rank_idx + 1  # 1-based rank

                score_addition = weight / (self.k + rank)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score_addition

                if doc_id not in ranker_appearances:
                    ranker_appearances[doc_id] = []
                ranker_appearances[doc_id].append(ranker_name)

                # Keep the candidate with highest individual score or more complete metadata
                if doc_id not in doc_registry:
                    doc_registry[doc_id] = result
                else:
                    existing = doc_registry[doc_id]
                    # Retain any populated context or higher raw score
                    if result.score > existing.score:
                        existing.score = result.score
                    if result.matched_entities and not existing.matched_entities:
                        existing.matched_entities = result.matched_entities
                    if result.matched_event and not existing.matched_event:
                        existing.matched_event = result.matched_event
                    if result.matched_relationship and not existing.matched_relationship:
                        existing.matched_relationship = result.matched_relationship
                    if result.page_number and result.page_number > 1 and existing.page_number == 1:
                        existing.page_number = result.page_number
                    if result.evidence and not existing.evidence:
                        existing.evidence = result.evidence

        if not rrf_scores:
            return []

        # 2. Normalize RRF scores into 0.0 - 1.0 scale
        max_score = max(rrf_scores.values())
        min_score = min(rrf_scores.values())
        score_range = max_score - min_score if max_score > min_score else 1.0

        fused_results: List[SearchResult] = []
        for doc_id, raw_rrf in rrf_scores.items():
            res = doc_registry[doc_id]
            # Normalize to 0.1 - 1.0 range
            normalized = 0.1 + 0.9 * ((raw_rrf - min_score) / score_range)
            res.score = round(normalized, 4)
            fused_results.append(res)

        # 3. Sort by fused score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)
        logger.info(
            f"RRF fused {len(fused_results)} unique documents from {list(ranked_lists.keys())} "
            f"(top score: {fused_results[0].score:.4f})"
        )

        return fused_results[:top_k]

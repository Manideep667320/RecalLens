"""
Context-Aware Multi-Signal Scorer for RecalLens
Implements the core research formula:
    S(d, q) = w_s * S_s + w_e * S_e + w_v * S_v + w_t * S_t + w_r * S_r
"""

import os
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone

try:
    from models import SearchResult, QueryContext, EntityRecord
except ImportError:
    from services.models import SearchResult, QueryContext, EntityRecord

logger = logging.getLogger(__name__)

# Default weights from RecalLens research design (can be tuned or set via env vars)
DEFAULT_WEIGHT_SEMANTIC = float(os.getenv("WEIGHT_SEMANTIC", "0.35"))
DEFAULT_WEIGHT_ENTITY = float(os.getenv("WEIGHT_ENTITY", "0.25"))
DEFAULT_WEIGHT_EVENT = float(os.getenv("WEIGHT_EVENT", "0.15"))
DEFAULT_WEIGHT_TEMPORAL = float(os.getenv("WEIGHT_TEMPORAL", "0.15"))
DEFAULT_WEIGHT_RELATIONSHIP = float(os.getenv("WEIGHT_RELATIONSHIP", "0.10"))

# Event compatibility clusters (e.g. warranty queries are compatible with purchase documents)
COMPATIBLE_EVENTS = {
    "warranty": ["purchase", "billing", "service_record"],
    "purchase": ["warranty", "billing", "receipt", "invoice"],
    "repair": ["service_record", "maintenance", "submission"],
    "billing": ["purchase", "invoice", "payment"],
    "invoice": ["billing", "purchase", "payment"],
    "submission": ["form", "claim", "application"]
}


class ContextScorer:
    """
    Computes individual contextual signals and blends them into a final calibrated score.
    """

    def __init__(
        self,
        w_s: float = DEFAULT_WEIGHT_SEMANTIC,
        w_e: float = DEFAULT_WEIGHT_ENTITY,
        w_v: float = DEFAULT_WEIGHT_EVENT,
        w_t: float = DEFAULT_WEIGHT_TEMPORAL,
        w_r: float = DEFAULT_WEIGHT_RELATIONSHIP,
        **kwargs
    ):
        w_s = kwargs.get("w_semantic", w_s)
        w_e = kwargs.get("w_entity", w_e)
        w_v = kwargs.get("w_event", w_v)
        w_t = kwargs.get("w_temporal", w_t)
        w_r = kwargs.get("w_relationship", w_r)

        total_w = w_s + w_e + w_v + w_t + w_r
        if total_w <= 0:
            total_w = 1.0

        # Normalize weights to sum strictly to 1.0
        self.w_s = w_s / total_w
        self.w_e = w_e / total_w
        self.w_v = w_v / total_w
        self.w_t = w_t / total_w
        self.w_r = w_r / total_w

        logger.info(
            f"ContextScorer initialized with weights: semantic={self.w_s:.2f}, "
            f"entity={self.w_e:.2f}, event={self.w_v:.2f}, temporal={self.w_t:.2f}, relationship={self.w_r:.2f}"
        )

    def score_document(
        self,
        result: SearchResult,
        query_context: Optional[QueryContext],
        semantic_score: float,
        context_details: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, float], Dict[str, Any]]:
        """
        Calculates S(d, q) and returns (final_score, breakdown_dict, match_metadata).
        """
        s_s = max(0.0, min(1.0, float(semantic_score)))

        if not query_context:
            return s_s, {"semantic": s_s, "entity": 0.0, "event": 0.0, "temporal": 0.0, "relationship": 0.0, "final": s_s}, {}

        context_details = context_details or {}

        # 1. Entity Match (S_e)
        s_e, matched_entities = self._score_entity(result, query_context, context_details)

        # 2. Event Match (S_v)
        s_v, matched_event = self._score_event(result, query_context, context_details)

        # 3. Temporal Match (S_t)
        s_t, matched_date = self._score_temporal(result, query_context)

        # 4. Relationship Match (S_r)
        s_r, matched_rel = self._score_relationship(result, query_context, context_details)

        # 5. Weighted multi-signal score S(d, q)
        final_score = (
            self.w_s * s_s +
            self.w_e * s_e +
            self.w_v * s_v +
            self.w_t * s_t +
            self.w_r * s_r
        )
        final_score = max(0.0, min(1.0, round(final_score, 4)))

        breakdown = {
            "semantic": round(s_s, 4),
            "entity": round(s_e, 4),
            "event": round(s_v, 4),
            "temporal": round(s_t, 4),
            "relationship": round(s_r, 4),
            "final": final_score
        }

        # Build evidence rationale
        evidence_parts = []
        if matched_entities:
            evidence_parts.append(f"Entity: {', '.join(matched_entities)}")
        if matched_event:
            evidence_parts.append(f"Event: {matched_event}")
        if matched_date:
            evidence_parts.append(f"Date: {matched_date}")
        if matched_rel:
            evidence_parts.append(f"Relation: {matched_rel}")

        evidence_str = " | ".join(evidence_parts) if evidence_parts else "Semantic match"

        metadata = {
            "matched_entities": matched_entities,
            "matched_event": matched_event,
            "matched_relationship": matched_rel,
            "evidence": evidence_str
        }

        return final_score, breakdown, metadata

    def score_entity_match(
        self,
        doc: SearchResult,
        query_context: QueryContext,
        doc_entities: Optional[List[Any]] = None
    ) -> Tuple[float, List[str]]:
        """Public helper to compute entity match score and matched entity names"""
        context_details: Dict[str, Any] = {}
        if doc_entities is not None:
            formatted_ents = []
            for e in doc_entities:
                if isinstance(e, EntityRecord):
                    cat = e.category or e.entity_type or "entity"
                    formatted_ents.append({"name": e.name, "category": cat})
                elif isinstance(e, dict):
                    formatted_ents.append(e)
                else:
                    formatted_ents.append({"name": str(e), "category": "entity"})
            context_details["entities"] = formatted_ents
        return self._score_entity(doc, query_context, context_details)

    def score_event_compatibility(
        self,
        doc_or_event: Union[SearchResult, str],
        query_context: QueryContext
    ) -> Tuple[float, Optional[str]]:
        """Public helper to compute event match score and matched event string"""
        if isinstance(doc_or_event, str):
            doc = SearchResult(
                recallens_id=0, paperless_id=0, chunk_id="", chunk_index=0,
                title="", content="", score=0.0,
                document_type=doc_or_event
            )
            ctx_details = {"events": [{"type": doc_or_event}]}
            return self._score_event(doc, query_context, ctx_details)
        return self._score_event(doc_or_event, query_context, {})

    def score_temporal_proximity(
        self,
        doc: SearchResult,
        query_context: QueryContext
    ) -> float:
        """Public helper to compute temporal match proximity score [0.0 - 1.0]"""
        score, _ = self._score_temporal(doc, query_context)
        return score

    def score_relationship_match(
        self,
        doc_or_relationships: Union[SearchResult, List[str]],
        query_context: QueryContext
    ) -> Tuple[float, Optional[str]]:
        """Public helper to compute relationship match score and matched triple"""
        if isinstance(doc_or_relationships, list):
            doc = SearchResult(
                recallens_id=0, paperless_id=0, chunk_id="", chunk_index=0,
                title="", content="", score=0.0
            )
            ctx_details = {"relationships": doc_or_relationships}
            return self._score_relationship(doc, query_context, ctx_details)
        return self._score_relationship(doc_or_relationships, query_context, {})

    def compute_total_score(
        self,
        semantic_score: float,
        doc: SearchResult,
        query_context: QueryContext,
        doc_entities: Optional[List[Any]] = None,
        doc_event: Optional[str] = None,
        doc_relationships: Optional[List[str]] = None
    ) -> Tuple[float, Dict[str, float], str]:
        """Public helper to compute the complete multi-signal score and evidence string"""
        ctx_details: Dict[str, Any] = {}
        if doc_entities:
            formatted_ents = []
            for e in doc_entities:
                if isinstance(e, EntityRecord):
                    cat = e.category or e.entity_type or "entity"
                    formatted_ents.append({"name": e.name, "category": cat})
                elif isinstance(e, dict):
                    formatted_ents.append(e)
                else:
                    formatted_ents.append({"name": str(e), "category": "entity"})
            ctx_details["entities"] = formatted_ents
        if doc_event:
            ctx_details["events"] = [{"type": doc_event}]
        if doc_relationships:
            ctx_details["relationships"] = doc_relationships

        final_score, breakdown, metadata = self.score_document(
            result=doc,
            query_context=query_context,
            semantic_score=semantic_score,
            context_details=ctx_details
        )
        return final_score, breakdown, metadata.get("evidence", "")

    def _score_entity(
        self,
        result: SearchResult,
        query_context: QueryContext,
        context_details: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """Scores entity match S_e between query and document"""
        q_entities = query_context.get_entities() if hasattr(query_context, "get_entities") else []
        if not q_entities and query_context.entity:
            q_entities = [query_context.entity]

        if not q_entities:
            return 0.5, []  # Neutral score when query has no entity constraint

        # Build document entity representation
        doc_entity_names = set()

        # From Postgres details
        for ent in context_details.get("entities", []):
            ename = ent.get("name") if isinstance(ent, dict) else getattr(ent, "name", str(ent))
            if ename:
                doc_entity_names.add(ename.lower().strip())

        # From result object directly
        for ent in getattr(result, "matched_entities", []):
            doc_entity_names.add(ent.lower().strip())

        # Check content and metadata
        title_lower = result.title.lower() if result.title else ""
        content_lower = result.content.lower() if result.content else ""
        tags_lower = [t.lower() for t in (result.tags or [])]
        corr_lower = (result.correspondent_name or "").lower()

        matched = []
        for q_ent in q_entities:
            q_lower = q_ent.lower().strip()
            # 1. Exact or substring match in stored entities
            found = False
            for d_name in doc_entity_names:
                if q_lower in d_name or d_name in q_lower:
                    matched.append(q_ent)
                    found = True
                    break

            if found:
                continue

            # 2. Check title, correspondent, tags, and content
            if q_lower in title_lower or q_lower in corr_lower:
                matched.append(q_ent)
            elif any(q_lower in tag for tag in tags_lower):
                matched.append(q_ent)
            elif q_lower in content_lower:
                matched.append(q_ent)
            else:
                # 3. Partial token overlap (e.g. "Apple" in "Apple Store")
                q_tokens = set(q_lower.split())
                doc_tokens = set(f"{title_lower} {content_lower[:300]}".split())
                if len(q_tokens.intersection(doc_tokens)) >= max(1, len(q_tokens) // 2):
                    matched.append(q_ent)

        matched_unique = list(dict.fromkeys(matched))
        match_ratio = len(matched_unique) / len(q_entities)
        return round(match_ratio, 3), matched_unique

    def _score_event(
        self,
        result: SearchResult,
        query_context: QueryContext,
        context_details: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """Scores event compatibility S_v between query and document"""
        target_event = query_context.event or query_context.target_information
        if not target_event:
            return 0.5, None

        q_evt = target_event.lower().strip()
        doc_type = (result.document_type or "").lower()
        doc_title = (result.title or "").lower()
        doc_content = (result.content or "")[:300].lower()

        # Check stored events from Postgres or result
        stored_events = [
            (ev.get("type", "").lower() if isinstance(ev, dict) else str(ev).lower())
            for ev in context_details.get("events", [])
        ]
        if getattr(result, "matched_event", None):
            stored_events.append(result.matched_event.lower())

        # 1. Exact match
        if (q_evt in stored_events or q_evt in doc_type or
            q_evt in doc_title or q_evt in doc_content):
            return 1.0, q_evt

        # 2. Compatibility mapping (e.g. warranty queries matched with purchase documents)
        compat = COMPATIBLE_EVENTS.get(q_evt, [])
        for c in compat:
            if (c in stored_events or c in doc_type or
                c in doc_title or c in doc_content):
                return 0.75, c

        return 0.0, None

    def _score_temporal(
        self,
        result: SearchResult,
        query_context: QueryContext
    ) -> Tuple[float, Optional[str]]:
        """Scores temporal match S_t based on query time range and document date"""
        start_bound = query_context.resolved_date_start or getattr(query_context, "target_date", None)
        end_bound = query_context.resolved_date_end or getattr(query_context, "target_date", None)

        if not start_bound and not end_bound:
            return 0.5, None

        doc_date_str = str(result.created_date or result.modified_date or "")[:10]
        if not doc_date_str or len(doc_date_str) < 10:
            return 0.3, None

        try:
            doc_dt = datetime.fromisoformat(doc_date_str)
            s_dt = datetime.fromisoformat(start_bound) if start_bound else doc_dt
            e_dt = datetime.fromisoformat(end_bound) if end_bound else doc_dt

            # If inside interval [start, end]
            if s_dt <= doc_dt <= e_dt:
                return 1.0, doc_date_str

            # Outside interval: compute exponential distance decay
            days_diff = min(abs((doc_dt - s_dt).days), abs((doc_dt - e_dt).days))
            if days_diff == 0:
                return 1.0, doc_date_str

            decay = math.exp(-days_diff / 90.0)
            return round(max(0.05, decay), 3), doc_date_str

        except Exception:
            return 0.3, None

    def _score_relationship(
        self,
        result: SearchResult,
        query_context: QueryContext,
        context_details: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """Scores relationship match S_r against stored semantic triples"""
        if not query_context.relationship:
            return 0.5, None

        q_rel = query_context.relationship.lower().strip()
        stored_rels = [
            str(r).lower() for r in context_details.get("relationships", [])
        ]
        if getattr(result, "matched_relationship", None):
            stored_rels.append(result.matched_relationship.lower())

        # Check in content if no stored relationships
        if not stored_rels and result.content:
            stored_rels.append(result.content.lower())

        # 1. Exact relationship match
        for r in stored_rels:
            if q_rel in r or r in q_rel:
                return 1.0, r

        # 2. Check predicate or subject-object overlap
        q_parts = [p.strip() for p in q_rel.split("->")]
        for r in stored_rels:
            r_parts = [p.strip() for p in r.split("->")]
            common = set(q_parts).intersection(set(r_parts))
            if len(common) >= 2:
                return 0.8, r
            elif len(common) == 1:
                return 0.4, r

        return 0.0, None

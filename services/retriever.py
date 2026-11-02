"""
Multi-stage retrieval from Qdrant and PostgreSQL with Reciprocal Rank Fusion (RRF)
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)
try:
    from models import SearchResult, ExpandedQuery
    from providers import BaseProvider, EmbeddingTaskType
    from retrieval.fusion import ReciprocalRankFusion
    from storage import get_postgres_client
except ImportError:
    from services.models import SearchResult, ExpandedQuery
    from services.providers import BaseProvider, EmbeddingTaskType
    from services.retrieval.fusion import ReciprocalRankFusion
    from services.storage import get_postgres_client

logger = logging.getLogger(__name__)


def stable_hash(text: str) -> int:
    """Generate stable positive integer hash using MD5"""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


class Retriever:
    """Handles hybrid document retrieval combining Qdrant vector search with PostgreSQL structured context"""

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        provider: BaseProvider
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.provider = provider
        self.fusion = ReciprocalRankFusion(k=60)
        self.postgres_client = get_postgres_client()

    def retrieve(
        self,
        expanded_query: ExpandedQuery,
        top_k: int = 50,
        hybrid_alpha: float = 0.5,
        user_id: str = "USER_001"
    ) -> List[SearchResult]:
        """
        Retrieve documents using multi-source hybrid search:
        1. Semantic vector search (Qdrant)
        2. Structured entity/event/temporal joins (PostgreSQL)
        3. Reciprocal Rank Fusion (RRF) candidate merging
        """
        try:
            # 1. Unstructured candidate retrieval (Qdrant vector / keyword)
            query_embedding = self._get_query_embedding(expanded_query.expanded_query)
            qdrant_filter = self._build_filter(expanded_query.extracted_filters)

            if hybrid_alpha > 0.0 and query_embedding:
                unstructured_results = self._vector_search(query_embedding, qdrant_filter, top_k)
            else:
                unstructured_results = self._keyword_search(expanded_query.expanded_query, qdrant_filter, top_k)

            logger.info(f"Retrieved {len(unstructured_results)} candidates from Qdrant vector search")

            # 2. Structured context retrieval (PostgreSQL)
            structured_results: List[SearchResult] = []
            q_ctx = getattr(expanded_query, 'context', None)

            if q_ctx and self.postgres_client and self.postgres_client.is_connected:
                pg_candidates = self.postgres_client.query_contextual_candidates(
                    user_id=user_id,
                    entity=q_ctx.entity,
                    event=q_ctx.event,
                    date_start=q_ctx.resolved_date_start,
                    date_end=q_ctx.resolved_date_end,
                    relationship=q_ctx.relationship,
                    limit=top_k
                )

                for idx, c in enumerate(pg_candidates):
                    doc_id_str = c["document_id"]
                    numeric_id = int(doc_id_str) if doc_id_str.isdigit() else stable_hash(doc_id_str)
                    s_res = SearchResult(
                        recallens_id=numeric_id,
                        paperless_id=numeric_id,
                        chunk_id=f"{doc_id_str}_chunk_0",
                        chunk_index=0,
                        title=c.get("title", ""),
                        content=f"Document {c.get('title')}. Type: {c.get('document_type')}. Event date: {c.get('event_date')}.",
                        document_type=c.get("document_type"),
                        created_date=c.get("document_date") or c.get("event_date"),
                        modified_date=c.get("event_date"),
                        score=1.0 - (idx * 0.015),  # Rank-ordered confidence
                        tags=[c.get("document_type", "")] if c.get("document_type") else []
                    )
                    structured_results.append(s_res)

                logger.info(f"Retrieved {len(structured_results)} candidates from PostgreSQL structured joins")

            # 3. Fuse candidates using Reciprocal Rank Fusion (RRF)
            if structured_results:
                ranked_lists = {
                    "vector": unstructured_results,
                    "structured": structured_results
                }
                weights = {"vector": 1.0, "structured": 1.1}
                results = self.fusion.fuse(ranked_lists, weights=weights, top_k=top_k)
            else:
                results = unstructured_results

            # 4. Enrich results with stored context details (entities, events, relationships)
            if self.postgres_client and self.postgres_client.is_connected:
                for r in results:
                    doc_id_val = str(r.recallens_id or r.paperless_id)
                    details = self.postgres_client.get_document_context_details(doc_id_val)
                    if details:
                        if not r.matched_entities and details.get("entities"):
                            r.matched_entities = [e["name"] for e in details["entities"]]
                        if not r.matched_event and details.get("events"):
                            r.matched_event = details["events"][0]["type"]
                        if not r.matched_relationship and details.get("relationships"):
                            r.matched_relationship = details["relationships"][0]
                        if details.get("chunks"):
                            r.page_number = details["chunks"][0].get("page", 1)

            logger.info(f"Final hybrid retrieval returned {len(results)} merged documents")
            return results[:top_k]

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for search query"""
        try:
            embedding = self.provider.get_embedding(
                text=query,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY
            )
            return embedding if embedding else None
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    def _vector_search(
        self,
        query_embedding: List[float],
        qdrant_filter: Optional[Filter],
        top_k: int
    ) -> List[SearchResult]:
        """Perform vector similarity search in Qdrant with deduplication by document ID"""
        try:
            logger.info(f"Searching Qdrant with filter: {qdrant_filter}, limit={top_k}")
            try:
                search_result = self.qdrant_client.query_points_groups(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    group_by="recallens_id",
                    group_size=1,
                    with_payload=True
                )
            except Exception:
                search_result = self.qdrant_client.query_points_groups(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=qdrant_filter,
                    limit=top_k,
                    group_by="paperless_id",
                    group_size=1,
                    with_payload=True
                )

            results = []
            for group in search_result.groups:
                if group.hits:
                    point = group.hits[0]
                    result = self._point_to_search_result(point)
                    if result:
                        results.append(result)

            return results
        except Exception as e:
            logger.error(f"Vector search with grouping failed: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        qdrant_filter: Optional[Filter],
        top_k: int
    ) -> List[SearchResult]:
        """Perform keyword search via Qdrant scroll and token matching"""
        try:
            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            results = []
            query_terms = set(query.lower().split())

            for point in points:
                result = self._point_to_search_result(point)
                if result:
                    content_lower = result.content.lower()
                    title_lower = result.title.lower()
                    matches = sum(1 for term in query_terms if term in content_lower or term in title_lower)
                    result.score = matches / len(query_terms) if query_terms else 0.0
                    results.append(result)

            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _build_filter(self, extracted_filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from extracted filters"""
        if not extracted_filters:
            return None

        conditions = []

        # Tags filter
        if 'tags' in extracted_filters:
            tags = extracted_filters['tags']
            tag_logic = extracted_filters.get('tag_logic', 'any')
            if isinstance(tags, list) and tags:
                if tag_logic == 'all':
                    for tag in tags:
                        conditions.append(FieldCondition(key="tags", match=MatchAny(any=[tag])))
                else:
                    conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))

        # Correspondent filter
        if 'correspondent_name' in extracted_filters:
            conditions.append(FieldCondition(key="correspondent_name", match=MatchValue(value=extracted_filters['correspondent_name'])))

        # Date range filter
        if 'created_date' in extracted_filters:
            date_range = extracted_filters['created_date']
            if isinstance(date_range, dict):
                from datetime import datetime
                range_dict = {}

                if 'gte' in date_range:
                    val = date_range['gte']
                    if isinstance(val, str):
                        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        range_dict['gte'] = int(dt.timestamp())
                    else:
                        range_dict['gte'] = val

                if 'lte' in date_range:
                    val = date_range['lte']
                    if isinstance(val, str):
                        if len(val) == 10:
                            val = val + 'T23:59:59'
                        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        range_dict['lte'] = int(dt.timestamp())
                    else:
                        range_dict['lte'] = val

                if range_dict:
                    conditions.append(FieldCondition(key="created_date", range=Range(**range_dict)))

        if not conditions:
            return None

        return Filter(must=conditions)

    def _timestamp_to_iso(self, timestamp) -> Optional[str]:
        """Convert Unix timestamp to ISO date string"""
        if timestamp is None:
            return None
        if isinstance(timestamp, str):
            return timestamp
        if isinstance(timestamp, int):
            from datetime import datetime
            try:
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            except Exception:
                return None
        return None

    def _point_to_search_result(self, point) -> Optional[SearchResult]:
        """Convert Qdrant point to SearchResult"""
        try:
            payload = point.payload
            score = point.score if hasattr(point, 'score') else 0.0

            created_date = self._timestamp_to_iso(payload.get('created_date'))
            modified_date = self._timestamp_to_iso(payload.get('modified_date'))
            ctx_payload = payload.get('context', {})
            document_type = payload.get('document_type') or (ctx_payload.get('document_type') if isinstance(ctx_payload, dict) else None)
            page_number = payload.get('page_number', 1)

            num_id = payload.get('recallens_id', payload.get('paperless_id', 0))
            return SearchResult(
                recallens_id=num_id,
                paperless_id=num_id,
                chunk_id=payload.get('id', str(point.id)),
                chunk_index=payload.get('chunk_index', 0),
                title=payload.get('title', ''),
                content=payload.get('content', ''),
                summary=payload.get('summary'),
                tags=payload.get('tags', []),
                correspondent_name=payload.get('correspondent_name'),
                document_type=document_type,
                page_number=page_number,
                created_date=created_date,
                modified_date=modified_date,
                modified=modified_date,
                score=score,
                highlights=[]
            )
        except Exception as e:
            logger.warning(f"Failed to convert point to result: {e}")
            return None

"""
Comparative Retrieval Modes for RecalLens Benchmark Evaluation
Implements:
- E1: BM25 / Exact Lexical Keyword Retrieval
- E2: Dense Vector Semantic Retrieval
- E3: Hybrid Retrieval Baseline (Dense Vector + BM25 Lexical, without context reasoning)
- E4: RecalLens Context-Driven Retrieval (Entities + Events + Temporal + Triples + RRF)
- Ablation Studies: A1 (No Entity), A2 (No Event), A3 (No Temporal), A4 (No Relationship), A5 (No RRF)
"""

import math
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter

try:
    from models import SearchResult, QueryContext, EntityRecord
    from retrieval.fusion import ReciprocalRankFusion
    from retrieval.context_scorer import ContextScorer
    from context.temporal_parser import TemporalParser
    from context.entity_extractor import EntityExtractor
    from context.event_extractor import EventExtractor
except ImportError:
    from services.models import SearchResult, QueryContext, EntityRecord
    from services.retrieval.fusion import ReciprocalRankFusion
    from services.retrieval.context_scorer import ContextScorer
    from services.context.temporal_parser import TemporalParser
    from services.context.entity_extractor import EntityExtractor
    from services.context.event_extractor import EventExtractor


def tokenize(text: str) -> List[str]:
    """Basic lowercased alphanumeric tokenizer"""
    return re.findall(r'\b[a-zA-Z0-9_-]+\b', (text or "").lower())


class BM25Engine:
    """
    Standard Okapi BM25 implementation for lexical keyword baseline (E1).
    """

    def __init__(self, corpus: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)

        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Counter] = {}
        self.doc_freqs: Counter = Counter()

        total_length = 0
        for doc in corpus:
            doc_id = doc["document_id"]
            # Combine title + text + document_type + location for fair BM25 representation
            combined_text = f"{doc.get('title', '')} {doc.get('text', '')} {doc.get('document_type', '')} {doc.get('location', '')}"
            tokens = tokenize(combined_text)
            length = len(tokens)
            self.doc_lengths[doc_id] = length
            total_length += length

            tf = Counter(tokens)
            self.doc_term_freqs[doc_id] = tf
            for term in tf:
                self.doc_freqs[term] += 1

        self.avg_doc_len = (total_length / float(self.n_docs)) if self.n_docs > 0 else 1.0

    def score(self, query_text: str, user_id: Optional[str] = None, top_k: int = 50) -> List[Tuple[str, float]]:
        """Scores documents for query_text and returns [(doc_id, score), ...] sorted descending"""
        query_terms = tokenize(query_text)
        if not query_terms:
            return []

        scores: Dict[str, float] = {}

        for doc in self.corpus:
            doc_id = doc["document_id"]
            if user_id and doc.get("user_id") != user_id:
                continue

            doc_len = self.doc_lengths.get(doc_id, 0)
            tf_dict = self.doc_term_freqs.get(doc_id, Counter())
            doc_score = 0.0

            for term in query_terms:
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue

                df = self.doc_freqs.get(term, 0)
                # Standard Robertson-Spärck Jones IDF with smoothing
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                doc_score += idf * (numerator / denominator)

            if doc_score > 0.0:
                scores[doc_id] = doc_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class VectorSimulator:
    """
    Simulates dense semantic vector retrieval (E2) using word-level sub-token cosine overlap
    and semantic context alignment.
    """

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus

    def score(self, query_text: str, user_id: Optional[str] = None, top_k: int = 50) -> List[Tuple[str, float]]:
        q_tokens = set(tokenize(query_text))
        if not q_tokens:
            return []

        scores: Dict[str, float] = {}
        for doc in self.corpus:
            doc_id = doc["document_id"]
            if user_id and doc.get("user_id") != user_id:
                continue

            d_text = f"{doc.get('title', '')} {doc.get('text', '')}"
            d_tokens = set(tokenize(d_text))
            if not d_tokens:
                continue

            # Jaccard / Cosine overlap simulation
            intersection = len(q_tokens.intersection(d_tokens))
            union = len(q_tokens.union(d_tokens))
            sim = (intersection / math.sqrt(len(q_tokens) * min(len(d_tokens), 100))) if union > 0 else 0.0

            # Semantic document type boost
            doc_type_tokens = set(tokenize(doc.get("document_type", "")))
            if q_tokens.intersection(doc_type_tokens):
                sim += 0.2

            if sim > 0:
                scores[doc_id] = min(1.0, round(sim, 4))

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


class BenchmarkSearchEngine:
    """
    Executes benchmark queries across the 4 primary research experiments and 5 ablation conditions.
    """

    def __init__(self, corpus: List[Dict[str, Any]]):
        self.corpus = corpus
        self.doc_map: Dict[str, Dict[str, Any]] = {d["document_id"]: d for d in corpus}
        self.bm25 = BM25Engine(corpus)
        self.vector_sim = VectorSimulator(corpus)
        self.temporal_parser = TemporalParser()
        self.entity_extractor = EntityExtractor()
        self.event_extractor = EventExtractor()
        self.rrf = ReciprocalRankFusion(k=60)

    def _to_search_result(self, doc_id: str, score: float) -> SearchResult:
        doc = self.doc_map.get(doc_id, {})
        int_id = int(doc_id.replace("DOC_", "").lstrip("0") or "1")
        return SearchResult(
            recallens_id=int_id,
            paperless_id=int_id,
            chunk_id=f"{doc_id}_chunk_0",
            chunk_index=0,
            title=doc.get("title", ""),
            content=doc.get("text", "")[:500],
            score=score,
            document_type=doc.get("document_type"),
            created_date=doc.get("date")
        )

    # -------------------------------------------------------------
    # E1: BM25 / Keyword Retrieval
    # -------------------------------------------------------------
    def run_e1_keyword(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        q_text = query.get("query_text", "")
        u_id = query.get("user_id")
        ranked = self.bm25.score(q_text, user_id=u_id, top_k=top_k)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    # -------------------------------------------------------------
    # E2: Dense Vector Retrieval
    # -------------------------------------------------------------
    def run_e2_vector(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        q_text = query.get("query_text", "")
        u_id = query.get("user_id")
        ranked = self.vector_sim.score(q_text, user_id=u_id, top_k=top_k)
        return [doc_id for doc_id, _ in ranked[:top_k]]

    # -------------------------------------------------------------
    # E3: Hybrid Baseline (Vector + Keyword, no context reasoning)
    # -------------------------------------------------------------
    def run_e3_hybrid_baseline(self, query: Dict[str, Any], alpha: float = 0.5, top_k: int = 5) -> List[str]:
        q_text = query.get("query_text", "")
        u_id = query.get("user_id")

        bm25_ranked = dict(self.bm25.score(q_text, user_id=u_id, top_k=50))
        vec_ranked = dict(self.vector_sim.score(q_text, user_id=u_id, top_k=50))

        # Max-normalize BM25 scores
        max_bm25 = max(bm25_ranked.values()) if bm25_ranked else 1.0
        if max_bm25 <= 0: max_bm25 = 1.0

        all_doc_ids = set(bm25_ranked.keys()).union(set(vec_ranked.keys()))
        hybrid_scores = {}
        for did in all_doc_ids:
            s_vec = vec_ranked.get(did, 0.0)
            s_bm25 = bm25_ranked.get(did, 0.0) / max_bm25
            hybrid_scores[did] = alpha * s_vec + (1.0 - alpha) * s_bm25

        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return [did for did, _ in sorted_results[:top_k]]

    # -------------------------------------------------------------
    # E4: Full RecalLens Context-Driven Retrieval
    # -------------------------------------------------------------
    def run_e4_recallens(
        self,
        query: Dict[str, Any],
        top_k: int = 5,
        w_s: float = 0.35,
        w_e: float = 0.25,
        w_v: float = 0.15,
        w_t: float = 0.15,
        w_r: float = 0.10,
        use_rrf: bool = True
    ) -> List[str]:
        q_text = query.get("query_text", "")
        u_id = query.get("user_id")

        # 1. Parse or load Query Context
        q_ctx = self._build_query_context(query)

        # 2. Dual-channel candidate retrieval
        # Channel A: Vector candidates
        vec_candidates = [
            self._to_search_result(did, score)
            for did, score in self.vector_sim.score(q_text, user_id=u_id, top_k=40)
        ]

        # Channel B: Structured context candidates (matching entity/event/dates)
        structured_candidates = self._find_structured_candidates(q_ctx, user_id=u_id, top_k=40)

        # 3. Fuse candidates
        if use_rrf:
            fused_candidates = self.rrf.fuse(
                ranked_lists={"vector": vec_candidates, "structured": structured_candidates},
                weights={"vector": 1.0, "structured": 1.2},
                top_k=50
            )
        else:
            fused_candidates = vec_candidates[:50]

        # 4. Multi-Signal Scoring with ContextScorer
        scorer = ContextScorer(w_s=w_s, w_e=w_e, w_v=w_v, w_t=w_t, w_r=w_r)
        scored_docs: List[Tuple[str, float]] = []

        for cand in fused_candidates:
            # Map integer recallens_id/paperless_id back to doc_id
            active_id = cand.recallens_id or cand.paperless_id or 1
            doc_id = f"DOC_{active_id:05d}"
            doc_raw = self.doc_map.get(doc_id, {})

            # Prepare structured context details
            ctx_details = {
                "entities": [
                    {"name": v, "category": k}
                    for k, v in doc_raw.get("entities", {}).items() if v
                ],
                "events": [{"type": doc_raw.get("event", "")}] if doc_raw.get("event") else [],
                "relationships": doc_raw.get("relationships", [])
            }

            final_score, _, _ = scorer.score_document(
                result=cand,
                query_context=q_ctx,
                semantic_score=cand.score,
                context_details=ctx_details
            )
            scored_docs.append((doc_id, final_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [did for did, _ in scored_docs[:top_k]]

    # -------------------------------------------------------------
    # Ablation Studies (A1 - A5)
    # -------------------------------------------------------------
    def run_a1_no_entity(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        """A1: Ablate entity matching (w_e = 0.0)"""
        return self.run_e4_recallens(query, top_k=top_k, w_s=0.45, w_e=0.0, w_v=0.20, w_t=0.20, w_r=0.15)

    def run_a2_no_event(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        """A2: Ablate event compatibility (w_v = 0.0)"""
        return self.run_e4_recallens(query, top_k=top_k, w_s=0.40, w_e=0.30, w_v=0.0, w_t=0.20, w_r=0.10)

    def run_a3_no_temporal(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        """A3: Ablate temporal reasoning (w_t = 0.0)"""
        return self.run_e4_recallens(query, top_k=top_k, w_s=0.40, w_e=0.30, w_v=0.20, w_t=0.0, w_r=0.10)

    def run_a4_no_relationship(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        """A4: Ablate semantic relationship matching (w_r = 0.0)"""
        return self.run_e4_recallens(query, top_k=top_k, w_s=0.40, w_e=0.30, w_v=0.15, w_t=0.15, w_r=0.0)

    def run_a5_no_rrf(self, query: Dict[str, Any], top_k: int = 5) -> List[str]:
        """A5: Ablate RRF candidate fusion (Vector candidates only before context scoring)"""
        return self.run_e4_recallens(query, top_k=top_k, use_rrf=False)

    # -------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------
    def _build_query_context(self, query: Dict[str, Any]) -> QueryContext:
        """Constructs QueryContext from query annotations or parsing heuristics"""
        q_text = query.get("query_text", "")
        # Resolve time bounds using TemporalParser
        time_ref = query.get("time_reference") or q_text
        t_res = self.temporal_parser.parse(time_ref)

        return QueryContext(
            entity=query.get("entity") or None,
            event=query.get("event") or None,
            time_reference=query.get("time_reference") or None,
            relationship=query.get("relationship") or None,
            target_information=query.get("event") or None,
            resolved_date_start=t_res.start_date if t_res else None,
            resolved_date_end=t_res.end_date if t_res else None,
            item=query.get("entity") or None
        )

    def _find_structured_candidates(
        self,
        query_context: QueryContext,
        user_id: Optional[str] = None,
        top_k: int = 40
    ) -> List[SearchResult]:
        """Finds candidate documents that match structured entities, events, or dates"""
        candidates: List[Tuple[str, float]] = []
        q_entities = [e.lower() for e in query_context.get_entities()]
        q_event = (query_context.event or "").lower()

        for doc in self.corpus:
            if user_id and doc.get("user_id") != user_id:
                continue

            doc_id = doc["document_id"]
            match_points = 0.0

            # Check entities
            doc_ents = doc.get("entities", {})
            doc_ent_vals = [str(v).lower() for v in doc_ents.values() if v]
            for q_e in q_entities:
                if any(q_e in dev or dev in q_e for dev in doc_ent_vals):
                    match_points += 1.0

            # Check event
            doc_ev = (doc.get("event") or "").lower()
            if q_event and (q_event in doc_ev or doc_ev in q_event):
                match_points += 1.0

            # Check date containment
            d_date = doc.get("date", "")
            if d_date and query_context.resolved_date_start and query_context.resolved_date_end:
                if query_context.resolved_date_start <= d_date[:10] <= query_context.resolved_date_end:
                    match_points += 1.0

            if match_points > 0:
                candidates.append((doc_id, match_points))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [self._to_search_result(did, score) for did, score in candidates[:top_k]]

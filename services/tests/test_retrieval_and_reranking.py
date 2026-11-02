"""
Unit tests for RecalLens Phase 3:
- Reciprocal Rank Fusion (RRF)
- Context-Aware Multi-Signal Scorer (ContextScorer)
- Reranker integration and evidence generation
"""

import unittest
from datetime import datetime, timezone
from services.models import SearchResult, QueryContext, EntityRecord
from services.retrieval.fusion import ReciprocalRankFusion
from services.retrieval.context_scorer import ContextScorer
from services.reranker import Reranker


class TestReciprocalRankFusion(unittest.TestCase):
    def setUp(self):
        self.rrf = ReciprocalRankFusion(k=60)

    def _make_result(self, recallens_id: int, score: float = 0.5) -> SearchResult:
        return SearchResult(
            recallens_id=recallens_id,
            paperless_id=recallens_id,
            chunk_id=f"doc_{recallens_id}_chunk_0",
            chunk_index=0,
            title=f"Doc {recallens_id}",
            content=f"Content for document {recallens_id}",
            score=score
        )

    def test_single_list_rrf(self):
        ranked = [self._make_result(1), self._make_result(2), self._make_result(3)]
        fused = self.rrf.fuse([ranked], weights=[1.0])

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0].recallens_id, 1)
        self.assertEqual(fused[0].paperless_id, 1)
        self.assertEqual(fused[1].recallens_id, 2)
        self.assertEqual(fused[1].paperless_id, 2)
        self.assertEqual(fused[2].recallens_id, 3)
        self.assertEqual(fused[2].paperless_id, 3)
        # Normalized score of top item is 1.0
        self.assertAlmostEqual(fused[0].score, 1.0, places=4)

    def test_multi_list_fusion_overlap(self):
        # List A (e.g. vector): [1, 2, 3]
        # List B (e.g. postgres): [2, 4, 1]
        list_a = [self._make_result(1), self._make_result(2), self._make_result(3)]
        list_b = [self._make_result(2), self._make_result(4), self._make_result(1)]

        fused = self.rrf.fuse([list_a, list_b], weights=[1.0, 1.0])

        # Doc 2 is rank 1 in list B and rank 2 in list A:
        # Doc 1 is rank 1 in list A and rank 3 in list B:
        # RRF(2) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        # RRF(1) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.016393 + 0.015873 = 0.032266
        # Therefore doc 2 should strictly rank first!
        self.assertEqual(fused[0].recallens_id, 2)
        self.assertEqual(fused[0].paperless_id, 2)
        self.assertEqual(fused[1].recallens_id, 1)
        self.assertEqual(fused[1].paperless_id, 1)
        # Verify all 4 unique docs are returned
        self.assertEqual(len(fused), 4)

    def test_weighted_fusion(self):
        list_a = [self._make_result(10), self._make_result(20)]
        list_b = [self._make_result(20), self._make_result(10)]

        # Heavily weight list B (weight 5.0 vs 1.0)
        fused = self.rrf.fuse([list_a, list_b], weights=[1.0, 5.0])
        # In list B, 20 is rank 1 -> weighted score dominates
        self.assertEqual(fused[0].recallens_id, 20)
        self.assertEqual(fused[0].paperless_id, 20)
        self.assertEqual(fused[1].recallens_id, 10)
        self.assertEqual(fused[1].paperless_id, 10)


class TestContextScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ContextScorer(
            w_semantic=0.35,
            w_entity=0.25,
            w_event=0.15,
            w_temporal=0.15,
            w_relationship=0.10
        )

    def test_entity_matching(self):
        q_ctx = QueryContext(
            person="Alice Smith",
            item="MacBook Pro",
            organization="Apple Store",
            location="New York"
        )
        doc = SearchResult(
            recallens_id=1,
            paperless_id=1,
            chunk_id="c1",
            chunk_index=0,
            title="Apple Receipt",
            content="Purchased MacBook Pro by Alice Smith at Apple Store NYC",
            score=0.8
        )
        doc_entities = [
            EntityRecord(name="Alice Smith", category="person"),
            EntityRecord(name="MacBook Pro", category="item"),
            EntityRecord(name="Apple Store", category="organization")
        ]

        score, matches = self.scorer.score_entity_match(doc, q_ctx, doc_entities)
        # 3 out of 4 non-empty query context entity fields matched
        self.assertAlmostEqual(score, 3.0 / 4.0, places=3)
        self.assertIn("Alice Smith", matches)
        self.assertIn("MacBook Pro", matches)
        self.assertIn("Apple Store", matches)

    def test_event_matching(self):
        q_ctx = QueryContext(event="purchase")

        score_exact, event_match = self.scorer.score_event_compatibility("purchase", q_ctx)
        self.assertEqual(score_exact, 1.0)
        self.assertEqual(event_match, "purchase")

        score_mismatch, _ = self.scorer.score_event_compatibility("repair", q_ctx)
        self.assertEqual(score_mismatch, 0.0)

        # Partial event compatibility (e.g. invoice/billing)
        q_billing = QueryContext(event="billing")
        score_partial, _ = self.scorer.score_event_compatibility("invoice", q_billing)
        self.assertGreater(score_partial, 0.0)

    def test_temporal_scoring(self):
        # Query specifies target date 2024-06-15
        q_ctx = QueryContext(target_date="2024-06-15")

        # Doc on same day -> score 1.0
        doc_same_day = SearchResult(
            recallens_id=1, paperless_id=1, chunk_id="c1", chunk_index=0,
            title="Bill", content="Bill", score=0.5,
            created_date="2024-06-15T12:00:00Z"
        )
        score_same = self.scorer.score_temporal_proximity(doc_same_day, q_ctx)
        self.assertAlmostEqual(score_same, 1.0, places=2)

        # Doc within 30 days -> high score (>0.75)
        doc_close = SearchResult(
            recallens_id=2, paperless_id=2, chunk_id="c2", chunk_index=0,
            title="Bill", content="Bill", score=0.5,
            created_date="2024-06-30T12:00:00Z"
        )
        score_close = self.scorer.score_temporal_proximity(doc_close, q_ctx)
        self.assertGreater(score_close, 0.75)

        # Doc 1 year away -> low score (<0.3)
        doc_far = SearchResult(
            recallens_id=3, paperless_id=3, chunk_id="c3", chunk_index=0,
            title="Bill", content="Bill", score=0.5,
            created_date="2023-06-15T12:00:00Z"
        )
        score_far = self.scorer.score_temporal_proximity(doc_far, q_ctx)
        self.assertLess(score_far, 0.3)

    def test_relationship_matching(self):
        q_ctx = QueryContext(
            relationship="MacBook -> covered_by -> AppleCare"
        )
        doc_relationships = [
            "MacBook -> covered_by -> AppleCare",
            "AppleCare -> valid_until -> 2026-05-01"
        ]
        score, matched = self.scorer.score_relationship_match(doc_relationships, q_ctx)
        self.assertEqual(score, 1.0)
        self.assertIn("applecare", matched.lower())

    def test_multi_signal_score_formula(self):
        """
        Verify S(d, q) = w_s S_s + w_e S_e + w_v S_v + w_t S_t + w_r S_r
        """
        q_ctx = QueryContext(
            item="iPhone 15",
            event="purchase",
            target_date="2024-01-10",
            relationship="iPhone 15 -> purchased_at -> Best Buy"
        )
        doc = SearchResult(
            recallens_id=10,
            paperless_id=10,
            chunk_id="c10",
            chunk_index=0,
            title="Best Buy Receipt",
            content="iPhone 15 purchased at Best Buy",
            score=0.8,
            created_date="2024-01-10T09:00:00Z"
        )
        doc_entities = [EntityRecord(name="iPhone 15", category="item")]
        doc_event = "purchase"
        doc_relationships = ["iPhone 15 -> purchased_at -> Best Buy"]

        total_score, breakdown, evidence = self.scorer.compute_total_score(
            semantic_score=1.0,  # S_s = 1.0
            doc=doc,
            query_context=q_ctx,
            doc_entities=doc_entities,
            doc_event=doc_event,
            doc_relationships=doc_relationships
        )

        # S_s = 1.0 (w=0.35) -> 0.35
        # S_e = 1.0 (w=0.25) -> 0.25
        # S_v = 1.0 (w=0.15) -> 0.15
        # S_t = 1.0 (w=0.15) -> 0.15
        # S_r = 1.0 (w=0.10) -> 0.10
        # Total = 1.0
        self.assertAlmostEqual(total_score, 1.0, places=2)
        self.assertIn("iphone 15", evidence.lower())
        self.assertIn("purchase", evidence.lower())
        self.assertIn("best buy", evidence.lower())


class TestRerankerIntegration(unittest.TestCase):
    def test_context_elevates_relevant_document(self):
        """
        Verify that a document with high contextual match (entity, event, date)
        beats an unrelated document with higher semantic keyword score.
        """
        # Mock provider to avoid network calls during unit test
        class DummyProvider:
            llm_model = "test-model"
            def generate_text(self, *args, **kwargs):
                return {"response": "5"}
            def list_models(self):
                return []

        reranker = Reranker(provider=DummyProvider())
        reranker.use_cross_encoder = False

        q_ctx = QueryContext(
            item="Sony Headphones",
            event="repair",
            target_date="2024-03-01"
        )

        # Doc 1: Keyword match only (e.g. general Sony ad), but wrong event, no target entity
        doc1 = SearchResult(
            recallens_id=1,
            paperless_id=1,
            chunk_id="c1",
            chunk_index=0,
            title="Sony Catalog 2022",
            content="Headphones sound quality catalog",
            score=0.9,  # Higher initial score
            created_date="2022-01-01"
        )

        # Doc 2: Perfect contextual match for Sony Headphones repair service in March 2024
        doc2 = SearchResult(
            recallens_id=2,
            paperless_id=2,
            chunk_id="c2",
            chunk_index=0,
            title="Service Invoice",
            content="Sony Headphones repair invoice service center",
            score=0.6,  # Lower initial score
            created_date="2024-03-01",
            matched_entities=["Sony Headphones"],
            matched_event="repair"
        )

        results = [doc1, doc2]
        reranked = reranker.rerank(
            query="Find the repair receipt for my Sony Headphones from March 2024",
            results=results,
            top_k=2,
            query_context=q_ctx
        )

        # Doc 2 should be boosted to the top because of entity, event, and temporal match
        self.assertEqual(reranked[0].recallens_id, 2)
        self.assertEqual(reranked[0].paperless_id, 2)
        self.assertGreater(reranked[0].context_score, reranked[1].context_score)


if __name__ == '__main__':
    unittest.main()

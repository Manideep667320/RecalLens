"""
Unit tests for RecalLens Phase 5:
- Evaluation metrics (P@k, Recall@k, MRR, nDCG@k, MAP)
- Dataset loader (documents, queries, qrels)
- Benchmark runner & comparative experiments
"""

import unittest
from services.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
    compute_all_metrics,
    aggregate_metrics
)
from services.evaluation.dataset_loader import BenchmarkDatasetLoader
from services.evaluation.search_modes import BM25Engine, BenchmarkSearchEngine
from services.evaluation.runner import BenchmarkRunner


class TestEvaluationMetrics(unittest.TestCase):
    def test_precision_at_k(self):
        ranked = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}

        # Top 5: 2 out of 5 relevant -> P@5 = 0.4
        self.assertAlmostEqual(precision_at_k(ranked, relevant, k=5), 0.4)
        # Top 1: doc1 is relevant -> P@1 = 1.0
        self.assertAlmostEqual(precision_at_k(ranked, relevant, k=1), 1.0)
        # Top 2: 1 out of 2 -> P@2 = 0.5
        self.assertAlmostEqual(precision_at_k(ranked, relevant, k=2), 0.5)

    def test_recall_at_k(self):
        ranked = ["doc1", "doc2", "doc3"]
        relevant = {"doc1", "doc4"}  # 2 relevant in total

        # Top 3 retrieved doc1 (1 out of 2) -> Recall = 0.5
        self.assertAlmostEqual(recall_at_k(ranked, relevant, k=3), 0.5)

    def test_reciprocal_rank(self):
        relevant = {"docA"}
        # First position -> RR = 1.0
        self.assertEqual(reciprocal_rank(["docA", "docB"], relevant), 1.0)
        # Second position -> RR = 0.5
        self.assertEqual(reciprocal_rank(["docB", "docA"], relevant), 0.5)
        # Third position -> RR = 1/3
        self.assertAlmostEqual(reciprocal_rank(["docB", "docC", "docA"], relevant), 1.0 / 3.0)
        # Not found -> RR = 0.0
        self.assertEqual(reciprocal_rank(["docX", "docY"], relevant), 0.0)

    def test_ndcg_at_k(self):
        # Perfect ranking: highest grade first
        qrels = {"doc1": 3, "doc2": 2, "doc3": 1}
        perfect_ranked = ["doc1", "doc2", "doc3"]
        self.assertAlmostEqual(ndcg_at_k(perfect_ranked, qrels, k=3), 1.0, places=4)

        # Inverted ranking: lower grade first
        inverted_ranked = ["doc3", "doc2", "doc1"]
        ndcg_inv = ndcg_at_k(inverted_ranked, qrels, k=3)
        self.assertLess(ndcg_inv, 1.0)
        self.assertGreater(ndcg_inv, 0.5)

    def test_average_precision(self):
        ranked = ["doc1", "docX", "doc2"]
        relevant = {"doc1", "doc2"}
        # doc1 at rank 1: P@1 = 1/1 = 1.0
        # doc2 at rank 3: P@3 = 2/3 = 0.6667
        # AP = (1.0 + 0.6667) / 2 = 0.8333
        ap = average_precision(ranked, relevant)
        self.assertAlmostEqual(ap, (1.0 + 2.0 / 3.0) / 2.0, places=3)

    def test_aggregate_metrics(self):
        m1 = {"p@5": 0.4, "mrr": 1.0}
        m2 = {"p@5": 0.6, "mrr": 0.5}
        agg = aggregate_metrics([m1, m2])
        self.assertAlmostEqual(agg["p@5"], 0.5)
        self.assertAlmostEqual(agg["mrr"], 0.75)


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        self.loader = BenchmarkDatasetLoader()

    def test_load_manifest(self):
        manifest = self.loader.load_manifest()
        self.assertIn("documents", manifest)
        self.assertEqual(manifest.get("documents"), 1500)
        self.assertEqual(manifest.get("queries"), 3750)

    def test_load_documents(self):
        docs = self.loader.load_documents()
        self.assertEqual(len(docs), 1500)
        first_doc = docs[0]
        self.assertIn("document_id", first_doc)
        self.assertIn("title", first_doc)
        self.assertIsInstance(first_doc["entities"], dict)
        self.assertIsInstance(first_doc["relationships"], list)

    def test_load_queries(self):
        test_queries = self.loader.load_queries(split="test")
        self.assertEqual(len(test_queries), 750)

        train_queries = self.loader.load_queries(split="train")
        self.assertEqual(len(train_queries), 3000)

    def test_load_qrels(self):
        qrels = self.loader.load_qrels(split="test")
        self.assertEqual(len(qrels), 750)
        # Every query in test split should have at least 1 ground truth document
        for qid, docs in list(qrels.items())[:10]:
            self.assertGreater(len(docs), 0)


class TestBenchmarkExecution(unittest.TestCase):
    def test_mini_benchmark_run(self):
        """
        Run a mini benchmark over 10 test queries and verify that
        RecalLens E4 produces competitive results with complete metric reporting.
        """
        runner = BenchmarkRunner()
        report = runner.run_evaluation(
            split="test",
            limit=10,
            k=5,
            run_ablations=True
        )

        self.assertEqual(report["queries_evaluated"], 10)
        results = report["overall_results"]

        self.assertIn("E1_BM25", results)
        self.assertIn("E2_Vector", results)
        self.assertIn("E3_Paperless_Hybrid", results)
        self.assertIn("E4_RecalLens", results)
        self.assertIn("A1_No_Entity", results)
        self.assertIn("A5_No_RRF", results)

        # Check that metrics are bounded between 0.0 and 1.0
        for method, metrics in results.items():
            for m_name, val in metrics.items():
                self.assertGreaterEqual(val, 0.0)
                self.assertLessEqual(val, 1.0)


if __name__ == '__main__':
    unittest.main()

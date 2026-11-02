"""
Experiment Runner & Benchmark Evaluator for RecalLens
Executes comparative experiments E1-E4 and ablation studies A1-A5.
Computes P@1, P@5, Recall@5, MRR, nDCG@5, and MAP.
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional

# Ensure services directory is in sys.path
services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if services_dir not in sys.path:
    sys.path.insert(0, services_dir)

try:
    from evaluation.metrics import compute_all_metrics, aggregate_metrics
    from evaluation.dataset_loader import BenchmarkDatasetLoader
    from evaluation.search_modes import BenchmarkSearchEngine
except ImportError:
    from services.evaluation.metrics import compute_all_metrics, aggregate_metrics
    from services.evaluation.dataset_loader import BenchmarkDatasetLoader
    from services.evaluation.search_modes import BenchmarkSearchEngine

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Executes automated evaluation on the RecalLens benchmark dataset.
    """

    def __init__(self, dataset_dir: Optional[str] = None):
        self.loader = BenchmarkDatasetLoader(dataset_dir) if dataset_dir else BenchmarkDatasetLoader()
        self.corpus = self.loader.load_documents()
        self.engine = BenchmarkSearchEngine(self.corpus)

    def run_evaluation(
        self,
        split: str = "test",
        limit: Optional[int] = 100,
        k: int = 5,
        run_ablations: bool = True,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs full comparative benchmark across E1, E2, E3, E4 and optional ablations A1-A5.
        """
        queries = self.loader.load_queries(split=split, limit=limit)
        qrels_all = self.loader.load_qrels(split=split)

        logger.info(f"Loaded {len(queries)} evaluation queries from split '{split}' (limit={limit})")
        logger.info(f"Loaded corpus of {len(self.corpus)} personal documents")

        # Methods to evaluate
        methods = {
            "E1_BM25": self.engine.run_e1_keyword,
            "E2_Vector": self.engine.run_e2_vector,
            "E3_Paperless_Hybrid": self.engine.run_e3_hybrid_baseline,
            "E4_RecalLens": self.engine.run_e4_recallens,
        }

        if run_ablations:
            methods.update({
                "A1_No_Entity": self.engine.run_a1_no_entity,
                "A2_No_Event": self.engine.run_a2_no_event,
                "A3_No_Temporal": self.engine.run_a3_no_temporal,
                "A4_No_Relationship": self.engine.run_a4_no_relationship,
                "A5_No_RRF": self.engine.run_a5_no_rrf
            })

        # Metric collectors per method: method -> list of metric dicts
        metrics_by_method: Dict[str, List[Dict[str, float]]] = {m: [] for m in methods}
        # Breakdown by difficulty
        metrics_by_difficulty: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        # Breakdown by query type
        metrics_by_type: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

        start_time = time.time()

        for idx, q in enumerate(queries):
            qid = q["query_id"]
            qrels = qrels_all.get(qid, {})
            if not qrels:
                continue

            diff = q.get("difficulty", "Medium")
            qtype = q.get("query_type", "semantic")

            if diff not in metrics_by_difficulty:
                metrics_by_difficulty[diff] = {m: [] for m in methods}
            if qtype not in metrics_by_type:
                metrics_by_type[qtype] = {m: [] for m in methods}

            # Evaluate each retrieval method
            for m_name, m_func in methods.items():
                ranked_doc_ids = m_func(q, top_k=k)
                m_scores = compute_all_metrics(ranked_doc_ids, qrels, k=k)

                metrics_by_method[m_name].append(m_scores)
                metrics_by_difficulty[diff][m_name].append(m_scores)
                metrics_by_type[qtype][m_name].append(m_scores)

        elapsed_seconds = round(time.time() - start_time, 2)

        # Aggregate overall results
        overall_results: Dict[str, Dict[str, float]] = {}
        for m_name, m_list in metrics_by_method.items():
            overall_results[m_name] = aggregate_metrics(m_list)

        # Aggregate difficulty results
        difficulty_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for diff, m_dict in metrics_by_difficulty.items():
            difficulty_results[diff] = {}
            for m_name, m_list in m_dict.items():
                difficulty_results[diff][m_name] = aggregate_metrics(m_list)

        # Aggregate query type results
        type_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for qtype, m_dict in metrics_by_type.items():
            type_results[qtype] = {}
            for m_name, m_list in m_dict.items():
                type_results[qtype][m_name] = aggregate_metrics(m_list)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": split,
            "queries_evaluated": len(queries),
            "total_corpus_docs": len(self.corpus),
            "elapsed_seconds": elapsed_seconds,
            "overall_results": overall_results,
            "by_difficulty": difficulty_results,
            "by_query_type": type_results
        }

        # Save to file if path specified
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved evaluation benchmark report to {output_file}")

        return report

    @staticmethod
    def format_table(results: Dict[str, Dict[str, float]], k: int = 5) -> str:
        """
        Formats evaluation metrics into an ASCII / Markdown table.
        """
        headers = ["Model / System", "P@1", f"P@{k}", f"Recall@{k}", "MRR", f"nDCG@{k}", "MAP"]
        col_w = [25, 8, 8, 10, 8, 9, 8]

        header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_w)) + " |"
        separator_line = "|-" + "-|-".join("-" * w for w in col_w) + "-|"

        rows = [header_line, separator_line]

        for method_name, scores in results.items():
            p1 = f"{scores.get('p@1', 0.0):.4f}"
            pk = f"{scores.get(f'p@{k}', 0.0):.4f}"
            rec = f"{scores.get(f'recall@{k}', 0.0):.4f}"
            mrr = f"{scores.get('mrr', 0.0):.4f}"
            ndcg = f"{scores.get(f'ndcg@{k}', 0.0):.4f}"
            map_score = f"{scores.get('map', 0.0):.4f}"

            vals = [method_name, p1, pk, rec, mrr, ndcg, map_score]
            row_str = "| " + " | ".join(v.ljust(w) for v, w in zip(vals, col_w)) + " |"
            rows.append(row_str)

        return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="RecalLens Benchmark Evaluation Runner")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate ('test' or 'train')")
    parser.add_argument("--limit", type=int, default=100, help="Number of queries to evaluate (default 100; use 0 for all)")
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff k (default 5)")
    parser.add_argument("--no-ablations", action="store_true", help="Disable ablation studies A1-A5")
    parser.add_argument("--output", type=str, default="services/evaluation/results/benchmark_report.json", help="Output JSON path")

    args = parser.parse_args()

    runner = BenchmarkRunner()
    limit = None if args.limit <= 0 else args.limit

    print(f"\n=======================================================")
    print(f"RecalLens Benchmark Evaluation (Split: {args.split}, Queries: {limit or 'ALL'})")
    print(f"=======================================================\n")

    report = runner.run_evaluation(
        split=args.split,
        limit=limit,
        k=args.k,
        run_ablations=not args.no_ablations,
        output_file=args.output
    )

    print("\n--- OVERALL BENCHMARK RESULTS ---")
    table = runner.format_table(report["overall_results"], k=args.k)
    print(table)

    print(f"\nEvaluation completed in {report['elapsed_seconds']}s.")
    print(f"Detailed JSON results written to: {args.output}\n")


if __name__ == "__main__":
    main()

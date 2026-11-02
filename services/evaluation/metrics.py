"""
Standard Information Retrieval & Ranking Metrics for RecalLens Evaluation
Implements: Precision@k, Recall@k, MRR, nDCG@k, and MAP
"""

import math
from typing import List, Dict, Any, Union, Set, Optional


def precision_at_k(ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 5) -> float:
    """
    Precision@k: Fraction of top-k retrieved documents that are relevant.
    """
    if k <= 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_in_top_k / float(k)


def recall_at_k(ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 5) -> float:
    """
    Recall@k: Fraction of all relevant documents that are retrieved in top-k.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = ranked_doc_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return relevant_in_top_k / float(len(relevant_doc_ids))


def reciprocal_rank(ranked_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """
    Reciprocal Rank (RR): 1 / (rank of first relevant document).
    Returns 0.0 if no relevant document appears in the ranking.
    """
    for rank_idx, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (rank_idx + 1)
    return 0.0


def dcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 5) -> float:
    """
    Discounted Cumulative Gain at rank k with exponential gain:
        DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        rel_grade = qrels.get(doc_id, 0)
        gain = (2.0 ** rel_grade) - 1.0
        discount = math.log2(i + 2)  # i=0 -> log2(2) = 1.0
        dcg += gain / discount
    return dcg


def ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 5) -> float:
    """
    Normalized Discounted Cumulative Gain at rank k:
        nDCG@k = DCG@k / IDCG@k
    """
    actual_dcg = dcg_at_k(ranked_doc_ids, qrels, k)

    # Sort all known relevant grades in descending order for Ideal DCG
    ideal_grades = sorted([grade for grade in qrels.values() if grade > 0], reverse=True)[:k]
    if not ideal_grades:
        return 0.0

    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        gain = (2.0 ** grade) - 1.0
        discount = math.log2(i + 2)
        idcg += gain / discount

    if idcg <= 0.0:
        return 0.0
    return min(1.0, actual_dcg / idcg)


def average_precision(ranked_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """
    Average Precision (AP) for a single query.
    """
    if not relevant_doc_ids:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_doc_ids:
            hit_count += 1
            precision_sum += hit_count / float(i + 1)

    return precision_sum / float(len(relevant_doc_ids))


def compute_all_metrics(
    ranked_doc_ids: List[str],
    qrels: Dict[str, int],
    k: int = 5
) -> Dict[str, float]:
    """
    Computes all standard metrics for a single query evaluation.
    """
    # Documents with relevance grade > 0 are considered relevant
    relevant_set = {doc_id for doc_id, grade in qrels.items() if grade > 0}

    return {
        "p@1": precision_at_k(ranked_doc_ids, relevant_set, k=1),
        f"p@{k}": precision_at_k(ranked_doc_ids, relevant_set, k=k),
        f"recall@{k}": recall_at_k(ranked_doc_ids, relevant_set, k=k),
        "mrr": reciprocal_rank(ranked_doc_ids, relevant_set),
        f"ndcg@{k}": ndcg_at_k(ranked_doc_ids, qrels, k=k),
        "map": average_precision(ranked_doc_ids, relevant_set)
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates a list of metric dicts across queries into mean scores rounded to 4 decimals.
    """
    if not metrics_list:
        return {}

    totals: Dict[str, float] = {}
    for item in metrics_list:
        for metric_name, val in item.items():
            totals[metric_name] = totals.get(metric_name, 0.0) + float(val)

    n = float(len(metrics_list))
    return {k: round(v / n, 4) for k, v in totals.items()}

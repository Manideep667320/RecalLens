"""
Evaluation and benchmark modules for RecalLens
"""

import sys
import os

services_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if services_dir not in sys.path:
    sys.path.insert(0, services_dir)

from .metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    average_precision,
    compute_all_metrics,
    aggregate_metrics
)
from .dataset_loader import BenchmarkDatasetLoader
from .search_modes import BM25Engine, VectorSimulator, BenchmarkSearchEngine
from .runner import BenchmarkRunner

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "average_precision",
    "compute_all_metrics",
    "aggregate_metrics",
    "BenchmarkDatasetLoader",
    "BM25Engine",
    "VectorSimulator",
    "BenchmarkSearchEngine",
    "BenchmarkRunner"
]

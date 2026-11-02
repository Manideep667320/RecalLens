"""
Dataset Loader for RecalLens / MemoraX Evaluation Benchmark
Loads documents, queries, and ground truth relevance labels from dataset/ directory.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dataset")


class BenchmarkDatasetLoader:
    """
    Loads and indexes the 1,500 documents, 3,750 queries, 3,750 relevance labels,
    and 100 synthetic users for benchmark evaluation.
    """

    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.manifest_path = os.path.join(dataset_dir, "manifest.json")
        self.documents_path = os.path.join(dataset_dir, "documents.csv")
        self.queries_path = os.path.join(dataset_dir, "queries.csv")
        self.relevance_path = os.path.join(dataset_dir, "relevance.csv")
        self.users_path = os.path.join(dataset_dir, "synthetic_users.csv")

        self._manifest: Optional[Dict[str, Any]] = None
        self._documents: Optional[pd.DataFrame] = None
        self._queries: Optional[pd.DataFrame] = None
        self._relevance: Optional[pd.DataFrame] = None
        self._users: Optional[pd.DataFrame] = None

    def load_manifest(self) -> Dict[str, Any]:
        """Loads dataset manifest metadata"""
        if self._manifest is None:
            if os.path.exists(self.manifest_path):
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self._manifest = json.load(f)
            else:
                self._manifest = {}
        return self._manifest

    def load_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Loads document records as clean Python dicts.
        If user_id is provided, filters documents belonging to that user.
        """
        if self._documents is None:
            if not os.path.exists(self.documents_path):
                raise FileNotFoundError(f"Documents file not found at {self.documents_path}")
            self._documents = pd.read_csv(self.documents_path).fillna("")

        df = self._documents
        if user_id:
            df = df[df["user_id"] == user_id]

        records = []
        for _, row in df.iterrows():
            # Parse entities JSON safely
            ent_str = str(row.get("entities", ""))
            entities_dict = {}
            if ent_str.strip().startswith("{"):
                try:
                    entities_dict = json.loads(ent_str)
                except Exception:
                    pass

            # Parse relationships JSON safely
            rel_str = str(row.get("relationships", ""))
            relationships_list = []
            if rel_str.strip().startswith("["):
                try:
                    relationships_list = json.loads(rel_str)
                except Exception:
                    pass

            doc_record = {
                "document_id": str(row.get("document_id", "")),
                "user_id": str(row.get("user_id", "")),
                "document_type": str(row.get("document_type", "Document")),
                "title": str(row.get("title", "")),
                "date": str(row.get("date", "")),
                "text": str(row.get("text", "")),
                "entities": entities_dict,
                "event": str(row.get("event", "")),
                "relationships": relationships_list,
                "location": str(row.get("location", "")),
                "amount": float(row["amount"]) if row.get("amount") and str(row["amount"]).replace(".", "", 1).isdigit() else None,
                "page_count": int(row.get("page_count", 1)) if str(row.get("page_count", 1)).isdigit() else 1,
                "source_dataset": str(row.get("source_dataset", "")),
                "source_document_id": str(row.get("source_document_id", ""))
            }
            records.append(doc_record)

        return records

    def load_queries(
        self,
        split: Optional[str] = "test",
        query_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Loads queries with optional filtering by split ('train', 'test'),
        query_type ('semantic', 'contextual_memory', 'memory_context'), or difficulty.
        """
        if self._queries is None:
            if not os.path.exists(self.queries_path):
                raise FileNotFoundError(f"Queries file not found at {self.queries_path}")
            self._queries = pd.read_csv(self.queries_path).fillna("")

        df = self._queries
        if split:
            df = df[df["split"].str.lower() == split.lower()]
        if query_type:
            df = df[df["query_type"].str.lower() == query_type.lower()]
        if difficulty:
            df = df[df["difficulty"].str.lower() == difficulty.lower()]
        if user_id:
            df = df[df["user_id"] == user_id]

        if limit and limit > 0:
            df = df.head(limit)

        queries = []
        for _, row in df.iterrows():
            queries.append({
                "query_id": str(row.get("query_id", "")),
                "user_id": str(row.get("user_id", "")),
                "query_text": str(row.get("query_text", "")),
                "query_type": str(row.get("query_type", "")),
                "entity": str(row.get("entity", "")),
                "event": str(row.get("event", "")),
                "time_reference": str(row.get("time_reference", "")),
                "relationship": str(row.get("relationship", "")),
                "difficulty": str(row.get("difficulty", "Medium")),
                "split": str(row.get("split", "test"))
            })
        return queries

    def load_qrels(self, split: Optional[str] = "test") -> Dict[str, Dict[str, int]]:
        """
        Loads ground truth relevance assessments (qrels) as:
            { query_id: { doc_id: relevance_grade, ... }, ... }
        """
        if self._relevance is None:
            if not os.path.exists(self.relevance_path):
                raise FileNotFoundError(f"Relevance file not found at {self.relevance_path}")
            self._relevance = pd.read_csv(self.relevance_path).fillna("")

        df = self._relevance
        if split:
            df = df[df["split"].str.lower() == split.lower()]

        qrels: Dict[str, Dict[str, int]] = {}
        for _, row in df.iterrows():
            qid = str(row.get("query_id", ""))
            did = str(row.get("document_id", ""))
            rel = int(row.get("relevance", 2))
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel

        return qrels

    def load_evidence_map(self) -> Dict[Tuple[str, str], str]:
        """Loads mapping (query_id, document_id) -> ground truth evidence explanation"""
        if self._relevance is None:
            self.load_qrels()

        evidence_map = {}
        for _, row in self._relevance.iterrows():
            qid = str(row.get("query_id", ""))
            did = str(row.get("document_id", ""))
            ev = str(row.get("evidence", ""))
            evidence_map[(qid, did)] = ev
        return evidence_map

    def get_summary(self) -> Dict[str, Any]:
        """Returns high-level statistics of the benchmark dataset"""
        docs = self.load_documents()
        queries_train = self.load_queries(split="train")
        queries_test = self.load_queries(split="test")
        qrels_test = self.load_qrels(split="test")

        user_ids = set(d["user_id"] for d in docs)

        return {
            "total_documents": len(docs),
            "total_users": len(user_ids),
            "queries_train": len(queries_train),
            "queries_test": len(queries_test),
            "total_queries": len(queries_train) + len(queries_test),
            "labeled_qrels_test": sum(len(v) for v in qrels_test.values())
        }

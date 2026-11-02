"""
Data models for RecalLens: Context-Driven Personal Information Retrieval
Using Event, Entity and Temporal Reasoning
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class EntityRecord(BaseModel):
    """An entity mentioned in a document or query"""
    name: str
    entity_type: str = "item"  # 'person', 'item', 'organization', 'location', 'product'
    category: Optional[str] = None
    mention_text: Optional[str] = None


class RelationshipRecord(BaseModel):
    """A semantic relationship between entities"""
    subject: str
    predicate: str
    object: str
    evidence_text: Optional[str] = None

    @classmethod
    def from_string(cls, rel_str: str) -> "RelationshipRecord":
        """Parse 'Subject -> Predicate -> Object' string representation"""
        parts = [p.strip() for p in rel_str.split("->")]
        if len(parts) >= 3:
            return cls(subject=parts[0], predicate=parts[1], object=parts[2])
        elif len(parts) == 2:
            return cls(subject=parts[0], predicate="associated_with", object=parts[1])
        return cls(subject=rel_str, predicate="related_to", object="")


class EventRecord(BaseModel):
    """An event extracted from a document or query"""
    event_type: str  # 'purchase', 'repair', 'billing', 'submission', 'travel', etc.
    event_date: Optional[str] = None  # ISO format string YYYY-MM-DD
    description: Optional[str] = None
    role: Optional[str] = None


class ExtractedContext(BaseModel):
    """Structured context extracted from personal documents"""
    entities: Dict[str, Any] = Field(default_factory=dict)  # {"person": "...", "item": "..."}
    entity_list: List[EntityRecord] = Field(default_factory=list)
    event: Optional[str] = None
    event_date: Optional[str] = None
    relationships: List[str] = Field(default_factory=list)  # ["A -> verb -> B"]
    relationship_records: List[RelationshipRecord] = Field(default_factory=list)
    document_type: Optional[str] = None
    amount: Optional[Union[float, str]] = None
    location: Optional[str] = None
    evidence: Optional[str] = None


class QueryContext(BaseModel):
    """Context extracted from user natural language search queries"""
    entity: Optional[str] = None
    event: Optional[str] = None
    time_reference: Optional[str] = None
    relationship: Optional[str] = None
    target_information: Optional[str] = None
    resolved_date_start: Optional[str] = None
    resolved_date_end: Optional[str] = None
    # Fine-grained entity/date slots (for dataset & query compatibility)
    person: Optional[str] = None
    item: Optional[str] = None
    organization: Optional[str] = None
    location: Optional[str] = None
    target_date: Optional[str] = None

    def get_entities(self) -> List[str]:
        """Returns non-empty entity constraints"""
        res = []
        if self.entity:
            res.append(self.entity)
        for val in [self.person, self.item, self.organization, self.location]:
            if val and val not in res:
                res.append(val)
        return res


class DocumentChunk(BaseModel):
    """A chunk of a document for indexing"""
    id: str
    recallens_id: Optional[int] = None
    paperless_id: Optional[int] = None
    chunk_index: int
    chunk_type: str  # 'metadata', 'content'
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None
    created_date: Optional[int] = None  # Unix timestamp
    modified_date: Optional[int] = None  # Unix timestamp
    added_date: Optional[int] = None  # Unix timestamp
    archive_serial_number: Optional[int] = None
    original_filename: Optional[str] = None
    page_number: Optional[int] = 1
    total_chunks: int = 1
    context: Optional[ExtractedContext] = None

    def __init__(self, **data):
        if "recallens_id" not in data and "paperless_id" in data:
            data["recallens_id"] = data["paperless_id"]
        elif "paperless_id" not in data and "recallens_id" in data:
            data["paperless_id"] = data["recallens_id"]
        elif "recallens_id" not in data and "paperless_id" not in data:
            data["recallens_id"] = 0
            data["paperless_id"] = 0
        super().__init__(**data)


class SearchQuery(BaseModel):
    """Search query from user with personal context options"""
    query: str
    user_id: Optional[str] = "USER_001"
    top_k: Optional[int] = 10
    retrieval_top_k: Optional[int] = None  # Number of results to retrieve before reranking
    filters: Optional[Dict[str, Any]] = None
    enable_reranking: Optional[bool] = True
    enable_query_expansion: Optional[bool] = True
    hybrid_alpha: Optional[float] = 0.5  # 0=keyword, 1=semantic
    recency_weight: Optional[float] = 0.0  # 0.0=no boost, 1.0=full boost
    recency_decay_days: Optional[int] = 365  # Days for 50% decay
    rerank_influence: Optional[float] = 0.2  # How much reranking adjusts score
    query_context: Optional[QueryContext] = None


class ExplainRequest(BaseModel):
    """Request to explain a rerank score"""
    query: str
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None
    matched_entities: List[str] = Field(default_factory=list)
    matched_event: Optional[str] = None
    matched_relationship: Optional[str] = None


class ExplainResponse(BaseModel):
    """Response with explanation"""
    explanation: str


class ExpandedQuery(BaseModel):
    """Query after LLM expansion and contextual parsing"""
    original_query: str
    expanded_query: str
    extracted_filters: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)
    context: Optional[QueryContext] = None


class SearchResult(BaseModel):
    """A single search result with contextual match attributes"""
    recallens_id: Optional[int] = None
    paperless_id: Optional[int] = None
    chunk_id: str
    chunk_index: int
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None
    document_type: Optional[str] = None
    page_number: Optional[int] = 1
    created_date: Optional[Union[str, int]] = None  # ISO datetime string or Unix timestamp
    modified_date: Optional[Union[str, int]] = None  # ISO datetime string or Unix timestamp
    modified: Optional[Union[str, int]] = None  # Deprecated - kept for backward compatibility
    score: float
    rerank_score: Optional[float] = None
    rerank_reason: Optional[str] = None
    recency_boost: Optional[float] = 0.0
    final_score: Optional[float] = None
    highlights: List[str] = Field(default_factory=list)

    # Context-driven fields
    matched_entities: List[str] = Field(default_factory=list)
    matched_event: Optional[str] = None
    matched_relationship: Optional[str] = None
    evidence: Optional[str] = None
    context_score: Optional[float] = None

    def __init__(self, **data):
        if "recallens_id" not in data and "paperless_id" in data:
            data["recallens_id"] = data["paperless_id"]
        elif "paperless_id" not in data and "recallens_id" in data:
            data["paperless_id"] = data["recallens_id"]
        elif "recallens_id" not in data and "paperless_id" not in data:
            data["recallens_id"] = 0
            data["paperless_id"] = 0
        super().__init__(**data)


class SearchResponse(BaseModel):
    """Complete search response including parsed query context"""
    query: str
    expanded_query: Optional[str] = None
    query_context: Optional[QueryContext] = None
    results: List[SearchResult]
    total_results: int
    retrieval_time_ms: float
    rerank_time_ms: Optional[float] = None
    total_time_ms: float


class RecalLensDocument(BaseModel):
    """RecalLens document metadata"""
    id: int
    title: str
    content: Optional[str] = None
    tags: List[Any] = Field(default_factory=list)
    correspondent: Optional[Any] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    added: Optional[str] = None
    archive_serial_number: Optional[int] = None
    original_file_name: Optional[str] = None


# Backward-compatibility alias
PaperlessDocument = RecalLensDocument


class DatasetDocumentRecord(BaseModel):
    """A document record from the MemoraX / RecalLens benchmark dataset"""
    document_id: str
    user_id: str
    document_type: str
    title: str
    source_type: Optional[str] = None
    date: Optional[str] = None
    text: str
    entities: Optional[Union[str, Dict[str, Any]]] = None
    event: Optional[str] = None
    relationships: Optional[Union[str, List[str]]] = None
    location: Optional[str] = None
    amount: Optional[Union[float, str]] = None
    page_count: Optional[int] = 1
    source_dataset: Optional[str] = None
    source_document_id: Optional[str] = None


class IndexingStats(BaseModel):
    """Statistics about indexing operation"""
    documents_processed: int
    chunks_created: int
    time_taken_seconds: float
    errors: List[str] = Field(default_factory=list)

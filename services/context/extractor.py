"""
Unified Document Context Extractor for RecalLens
Coordinates Entity, Event, Temporal, and Relationship extraction for document indexing.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List
try:
    from models import ExtractedContext, EntityRecord, RelationshipRecord
except ImportError:
    from services.models import ExtractedContext, EntityRecord, RelationshipRecord
from .entity_extractor import EntityExtractor
from .event_extractor import EventExtractor
from .relationship_extractor import RelationshipExtractor
from .temporal_parser import TemporalParser

logger = logging.getLogger(__name__)

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


class ContextExtractor:
    """Unified context extraction pipeline for incoming documents"""

    def __init__(self, provider=None):
        self.provider = provider
        self.entity_extractor = EntityExtractor(provider=provider)
        self.event_extractor = EventExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.temporal_parser = TemporalParser()

    def extract_document_context(
        self,
        text: str,
        title: str = "",
        raw_entities: Optional[Any] = None,
        raw_event: Optional[str] = None,
        raw_relationships: Optional[Any] = None,
        raw_date: Optional[str] = None,
        document_type: Optional[str] = None,
        location: Optional[str] = None,
        amount: Optional[Any] = None
    ) -> ExtractedContext:
        """
        Extracts structured personal context from a document.
        Supports both pre-annotated dataset records and raw OCR text from RecalLens/Paperless.
        """
        # 1. Parse Entities
        entities: Dict[str, str] = {}
        if raw_entities:
            entities = self.entity_extractor.extract_from_json_or_text(raw_entities)
        else:
            entities = self.entity_extractor.extract_heuristic(text, title=title)

        entity_records = self.entity_extractor.to_records(entities)

        # 2. Extract Event & Date
        event = raw_event
        event_date = raw_date
        if not event or not event_date:
            inferred_event, inferred_date = self.event_extractor.extract_event(text, title=title, metadata_date=raw_date)
            event = event or inferred_event
            event_date = event_date or inferred_date

        # 3. Parse / Derive Relationships
        relationships: List[str] = []
        if raw_relationships:
            relationships = self.relationship_extractor.parse_relationships_from_input(raw_relationships)
        else:
            relationships = self.relationship_extractor.derive_relationships(entities, event=event)

        relationship_records = self.relationship_extractor.to_records(relationships)

        # 4. Extract Amount & Location if not given
        if not location and entities.get("location"):
            location = entities.get("location")
        if not location:
            m_loc = re.search(r'(?:location|city|in|at)\s*[:=]?\s*([A-Z][a-z]+)', text)
            if m_loc:
                location = m_loc.group(1)

        if not amount:
            m_amt = re.search(r'(?:amount|total|rs\.?|\$)\s*[:=]?\s*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
            if m_amt:
                amount = m_amt.group(1)

        # 5. Extract 1-line Evidence
        evidence = f"Document type={document_type or 'record'}; Event={event or 'none'}; Date={event_date or 'none'}; Entities={list(entities.values())}"

        return ExtractedContext(
            entities=entities,
            entity_list=entity_records,
            event=event,
            event_date=event_date,
            relationships=relationships,
            relationship_records=relationship_records,
            document_type=document_type,
            amount=str(amount) if amount is not None else None,
            location=location,
            evidence=evidence
        )

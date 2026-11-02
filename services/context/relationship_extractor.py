"""
Relationship Extraction Engine for RecalLens
Extracts and canonicalizes Subject -> Predicate -> Object semantic triples from documents and queries.
"""

import json
import logging
from typing import List, Dict, Any, Optional
try:
    from models import RelationshipRecord
except ImportError:
    from services.models import RelationshipRecord

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """Extracts and formats entity-event relationships into semantic triples"""

    def parse_relationships_from_input(self, raw_input: Any) -> List[str]:
        """
        Parses pre-formatted relationship arrays from dataset or JSON strings.
        Example: ["Aryan Maharaj -> repair -> laptop", "laptop -> associated_with -> Metro Electronics"]
        """
        if not raw_input:
            return []

        if isinstance(raw_input, list):
            return [str(r).strip() for r in raw_input if str(r).strip()]

        if isinstance(raw_input, str):
            text = raw_input.strip()
            # If JSON array: '["A -> b -> C"]'
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [str(r).strip() for r in parsed if str(r).strip()]
                except Exception:
                    pass

            # If single relationship string: 'A -> b -> C'
            if "->" in text:
                return [text]

        return []

    def derive_relationships(
        self,
        entities: Dict[str, str],
        event: Optional[str] = None,
        default_user_name: str = "User"
    ) -> List[str]:
        """
        Derives semantic triples when structured relationships are missing:
        - Person -> Event -> Item (e.g. Aryan -> repair -> laptop)
        - Item -> associated_with -> Organization (e.g. laptop -> associated_with -> Metro Electronics)
        - Item -> located_at -> Location
        """
        triples: List[str] = []

        person = entities.get("person") or default_user_name
        item = entities.get("item")
        org = entities.get("organization")
        loc = entities.get("location")
        evt = event or "associated_with"

        if item:
            # Person -> Event -> Item
            triples.append(f"{person} -> {evt} -> {item}")

            # Item -> associated_with -> Organization
            if org:
                triples.append(f"{item} -> associated_with -> {org}")

            # Item -> located_at -> Location
            if loc:
                triples.append(f"{item} -> located_at -> {loc}")
        elif org:
            triples.append(f"{person} -> {evt} -> {org}")

        return triples

    def to_records(self, rel_strings: List[str], evidence_text: Optional[str] = None) -> List[RelationshipRecord]:
        """Converts triple strings into typed RelationshipRecord models"""
        records = []
        for r_str in rel_strings:
            rec = RelationshipRecord.from_string(r_str)
            if evidence_text:
                rec.evidence_text = evidence_text
            records.append(rec)
        return records

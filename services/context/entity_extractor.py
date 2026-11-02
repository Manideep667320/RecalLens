"""
Entity Extraction Engine for RecalLens
Extracts and normalizes persons, items/devices, organizations/merchants, and locations from personal records.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
try:
    from models import EntityRecord
except ImportError:
    from services.models import EntityRecord

logger = logging.getLogger(__name__)

# Common personal item keywords for heuristic detection
COMMON_ITEMS = [
    "laptop", "smartphone", "phone", "tablet", "computer", "macbook",
    "washing machine", "refrigerator", "fridge", "microwave", "television", "tv",
    "printer", "scanner", "camera", "headphone", "earbuds", "watch", "smartwatch",
    "vehicle", "car", "motorcycle", "bike", "bicycle", "ac", "air conditioner",
    "heater", "water purifier", "vacuum", "router", "modem"
]

COMMON_ORG_SUFFIXES = [
    "electronics", "services", "care", "technologies", "tech", "corp", "corporation",
    "inc", "ltd", "limited", "pvt", "solutions", "motors", "retail", "store",
    "hospital", "clinic", "lab", "bank", "insurance", "agency"
]


class EntityExtractor:
    """Extracts typed entities (person, item, organization, location) from personal documents"""

    def __init__(self, provider=None):
        self.provider = provider

    def extract_from_json_or_text(self, raw_input: Any) -> Dict[str, str]:
        """
        Parses pre-structured JSON strings (from dataset) or falls back to text extraction.
        Returns a dict: {"person": "...", "item": "...", "organization": "...", "location": "..."}
        """
        if not raw_input:
            return {}

        # 1. If already a dictionary
        if isinstance(raw_input, dict):
            return {k: str(v).strip() for k, v in raw_input.items() if v}

        # 2. If it's a JSON string
        if isinstance(raw_input, str):
            text = raw_input.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return {k: str(v).strip() for k, v in parsed.items() if v}
                except Exception:
                    pass

            # 3. Fallback to heuristic text extraction
            return self.extract_heuristic(text)

        return {}

    def extract_heuristic(self, text: str, title: str = "") -> Dict[str, str]:
        """Rule-based heuristic entity extractor when LLM is offline or for rapid indexing"""
        entities: Dict[str, str] = {}
        combined = f"{title} {text}".lower()

        # Item detection
        for item in sorted(COMMON_ITEMS, key=len, reverse=True):
            if re.search(rf'\b{item}s?\b', combined):
                entities["item"] = item
                break

        # Person detection from patterns like "for Aryan Maharaj", "Patient: John Doe", "Customer: Jane"
        person_match = re.search(r'(?:for|patient|customer|client|name|user)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', f"{title} {text}")
        if person_match:
            entities["person"] = person_match.group(1).strip()

        # Organization detection from patterns like "Metro Electronics", "QuickCare", "Organization: X"
        org_match = re.search(r'(?:organization|merchant|store|company|vendor|provider)[:\s]+([A-Za-z0-9\s&]+?)(?=[.,\n]|$)', text, re.IGNORECASE)
        if org_match:
            entities["organization"] = org_match.group(1).strip()
        else:
            for suffix in COMMON_ORG_SUFFIXES:
                m = re.search(rf'\b([A-Z][a-zA-Z0-9\s]+\s+{suffix})\b', text, re.IGNORECASE)
                if m:
                    entities["organization"] = m.group(1).strip()
                    break

        # Location detection
        loc_match = re.search(r'(?:location|city|place|at|in)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text)
        if loc_match:
            entities["location"] = loc_match.group(1).strip()

        return entities

    def to_records(self, entities_dict: Dict[str, Any]) -> List[EntityRecord]:
        """Convert an entity mapping into a list of typed EntityRecord instances"""
        records = []
        for etype, ename in entities_dict.items():
            if ename and isinstance(ename, str) and ename.strip():
                records.append(EntityRecord(
                    name=ename.strip(),
                    entity_type=etype.strip(),
                    mention_text=ename.strip()
                ))
        return records

"""
Event Extraction Engine for RecalLens
Classifies personal events (purchase, repair, billing, submission, travel, medical) and isolates event dates.
"""

import re
from typing import Optional, Tuple
from datetime import datetime

EVENT_KEYWORDS = {
    "repair": ["repair", "service record", "servicing", "maintenance", "fix", "replacement", "diagnostics", "overhaul"],
    "purchase": ["purchase", "bought", "receipt", "order", "sales invoice", "store purchase", "checkout", "bought from"],
    "billing": ["billing", "bill", "invoice", "statement", "utility", "charge", "payment due", "amount payable"],
    "submission": ["submission", "submitted", "form", "application", "claim submission", "declaration", "registration form"],
    "travel": ["travel", "flight", "boarding pass", "hotel", "itinerary", "booking", "reservation", "train ticket"],
    "warranty": ["warranty", "guarantee", "protection plan", "coverage", "extended warranty"],
    "medical": ["medical", "doctor", "hospital", "prescription", "clinic", "diagnosis", "health report", "vaccine", "lab test"],
    "contract": ["contract", "agreement", "lease", "rental", "policy", "terms of service", "insurance policy"]
}


class EventExtractor:
    """Extracts personal event types and event dates from document text"""

    def extract_event(self, text: str, title: str = "", metadata_date: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Classifies event type and extracts the primary event date.
        
        Returns:
            (event_type, event_date_iso)
        """
        combined = f"{title} {text}".lower()

        # 1. Classify event type
        detected_event = None
        # Check title first (higher confidence)
        title_lower = title.lower()
        for etype, keywords in EVENT_KEYWORDS.items():
            if any(k in title_lower for k in keywords):
                detected_event = etype
                break

        # Check full text if not found in title
        if not detected_event:
            for etype, keywords in EVENT_KEYWORDS.items():
                if any(k in combined for k in keywords):
                    detected_event = etype
                    break

        # Default fallback
        if not detected_event:
            detected_event = "general_record"

        # 2. Extract event date (YYYY-MM-DD pattern)
        event_date = None
        date_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',                             # 2024-01-01
            r'(?:event|service|purchase|bill|dated?)\s*[:=]\s*(\d{4}-\d{2}-\d{2})',
            r'\b(\d{2}/\d{2}/\d{4})\b'                              # 01/15/2024
        ]

        for p in date_patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                raw_d = m.group(1)
                if "/" in raw_d:
                    try:
                        dt = datetime.strptime(raw_d, "%m/%d/%Y")
                        event_date = dt.strftime("%Y-%m-%d")
                    except Exception:
                        pass
                else:
                    event_date = raw_d
                break

        # Fallback to metadata date if text date wasn't found
        if not event_date and metadata_date:
            event_date = str(metadata_date)[:10]

        return detected_event, event_date

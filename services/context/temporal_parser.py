"""
Temporal Reasoning Engine for RecalLens
Resolves explicit, relative, and fuzzy temporal expressions into concrete Unix timestamps and ISO date ranges.
"""

import re
import calendar
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass


@dataclass
class DateRangeResult:
    """Represents a resolved temporal range with ISO strings and Unix timestamps"""
    start_date: Optional[str] = None       # "YYYY-MM-DD"
    end_date: Optional[str] = None         # "YYYY-MM-DD"
    start_ts: Optional[int] = None         # Unix timestamp (seconds)
    end_ts: Optional[int] = None           # Unix timestamp (seconds)
    confidence: float = 0.0                # 0.0 to 1.0
    is_exact: bool = False                 # True if specific date, False if interval
    raw_expression: Optional[str] = None


MONTH_NAMES = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

SEASONS = {
    "spring": (3, 5),
    "summer": (6, 8),
    "autumn": (9, 11),
    "fall": (9, 11),
    "winter": (12, 2)
}


class TemporalParser:
    """Parses natural language and structured date expressions for personal document retrieval"""

    def __init__(self, default_ref_year: int = 2024):
        self.default_ref_year = default_ref_year

    def parse(self, text: Optional[str], ref_date: Optional[datetime] = None) -> DateRangeResult:
        """
        Parse temporal reference into start and end date bounds.
        
        Args:
            text: Date string or natural expression (e.g. "January 2024", "last year", "2023-05-12")
            ref_date: Reference datetime for relative expressions (defaults to 2024-12-31 or current)
        """
        if not text or not str(text).strip():
            return DateRangeResult()

        raw = str(text).strip()
        expr = raw.lower()

        if ref_date is None:
            # Default to reference year benchmark baseline (2024)
            ref_date = datetime(self.default_ref_year, 12, 31, tzinfo=timezone.utc)
        elif ref_date.tzinfo is None:
            ref_date = ref_date.replace(tzinfo=timezone.utc)

        current_year = ref_date.year

        # 1. Exact ISO date: YYYY-MM-DD
        m_iso = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', expr)
        if m_iso:
            y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            return self._build_range(datetime(y, m, d), datetime(y, m, d, 23, 59, 59), confidence=1.0, is_exact=True, raw=raw)

        # 2. Date intervals: "between X and Y" or "from X to Y"
        m_range = re.search(r'(?:between|from)\s+([a-z]+|\d{4})\s+(?:and|to)\s+([a-z]+|\d{4})', expr)
        if m_range:
            part1, part2 = m_range.group(1), m_range.group(2)
            m1 = MONTH_NAMES.get(part1)
            m2 = MONTH_NAMES.get(part2)
            if m1 and m2:
                y = current_year
                # Check if an explicit year was also mentioned
                m_yr = re.search(r'\b(20\d{2})\b', expr)
                if m_yr:
                    y = int(m_yr.group(1))
                _, last_day = calendar.monthrange(y, m2)
                start = datetime(y, m1, 1)
                end = datetime(y, m2, last_day, 23, 59, 59)
                return self._build_range(start, end, confidence=0.9, is_exact=False, raw=raw)

        # 3. Seasons: "summer 2024", "winter 2023"
        for sname, (sm, em) in SEASONS.items():
            if sname in expr:
                m_sy = re.search(r'\b(20\d{2})\b', expr)
                y = int(m_sy.group(1)) if m_sy else current_year
                start = datetime(y, sm, 1)
                end_y = y + 1 if em < sm else y
                _, last_day = calendar.monthrange(end_y, em)
                end = datetime(end_y, em, last_day, 23, 59, 59)
                return self._build_range(start, end, confidence=0.85, is_exact=False, raw=raw)

        # 4. Month and Year: e.g. "January 2024", "Jan 2024"
        for mname, mnum in MONTH_NAMES.items():
            pattern = rf'\b{mname}\s+(\d{{4}})\b'
            m = re.search(pattern, expr)
            if m:
                y = int(m.group(1))
                _, last_day = calendar.monthrange(y, mnum)
                start = datetime(y, mnum, 1)
                end = datetime(y, mnum, last_day, 23, 59, 59)
                return self._build_range(start, end, confidence=0.95, is_exact=False, raw=raw)

        # 5. Relative year expressions: "last year", "previous year", "this year"
        if "last year" in expr or "previous year" in expr or "past year" in expr:
            y = current_year - 1
            start = datetime(y, 1, 1)
            end = datetime(y, 12, 31, 23, 59, 59)
            return self._build_range(start, end, confidence=0.9, is_exact=False, raw=raw)

        if "this year" in expr or "current year" in expr:
            start = datetime(current_year, 1, 1)
            end = datetime(current_year, 12, 31, 23, 59, 59)
            return self._build_range(start, end, confidence=0.9, is_exact=False, raw=raw)

        # 6. Relative month: "last march", "last january"
        for mname, mnum in MONTH_NAMES.items():
            if f"last {mname}" in expr or f"past {mname}" in expr:
                y = current_year - 1 if ref_date.month <= mnum else current_year
                _, last_day = calendar.monthrange(y, mnum)
                start = datetime(y, mnum, 1)
                end = datetime(y, mnum, last_day, 23, 59, 59)
                return self._build_range(start, end, confidence=0.85, is_exact=False, raw=raw)

        # 7. Standalone month: "around january", "in march", "january"
        for mname, mnum in MONTH_NAMES.items():
            if re.search(rf'\b(around|in|during|for)?\s*{mname}\b', expr):
                y = current_year
                _, last_day = calendar.monthrange(y, mnum)
                start = datetime(y, mnum, 1)
                end = datetime(y, mnum, last_day, 23, 59, 59)
                return self._build_range(start, end, confidence=0.8, is_exact=False, raw=raw)

        # 8. Standalone year: e.g. "in 2023", "2024"
        m_year = re.search(r'\b(20\d{2}|19\d{2})\b', expr)
        if m_year:
            y = int(m_year.group(1))
            start = datetime(y, 1, 1)
            end = datetime(y, 12, 31, 23, 59, 59)
            return self._build_range(start, end, confidence=0.9, is_exact=False, raw=raw)

        return DateRangeResult(raw_expression=raw, confidence=0.0)

    def _build_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        confidence: float,
        is_exact: bool,
        raw: str
    ) -> DateRangeResult:
        """Helper to generate ISO and Unix timestamp representation"""
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

        return DateRangeResult(
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
            start_ts=int(start_dt.timestamp()),
            end_ts=int(end_dt.timestamp()),
            confidence=confidence,
            is_exact=is_exact,
            raw_expression=raw
        )

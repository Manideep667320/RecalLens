"""
Unit Tests for RecalLens Context Extraction Engine (Phase 2)
Tests Temporal Reasoning, Entity Extraction, Event Classification,
Relationship Modeling, and Query Context Parsing against MemoraX dataset patterns.
"""

import os
import sys
import unittest
from datetime import datetime, timezone

# Ensure services directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from context.temporal_parser import TemporalParser, DateRangeResult
from context.entity_extractor import EntityExtractor
from context.event_extractor import EventExtractor
from context.relationship_extractor import RelationshipExtractor
from context.query_parser import QueryContextParser
from context.extractor import ContextExtractor
from models import RelationshipRecord, EntityRecord, QueryContext, ExtractedContext


class TestTemporalParser(unittest.TestCase):
    """Test temporal resolution across explicit, relative, and interval expressions"""

    def setUp(self):
        self.parser = TemporalParser(default_ref_year=2024)
        self.ref_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    def test_exact_iso_date(self):
        res = self.parser.parse("2024-01-15", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2024-01-15")
        self.assertEqual(res.end_date, "2024-01-15")
        self.assertTrue(res.is_exact)
        self.assertEqual(res.confidence, 1.0)

    def test_month_and_year(self):
        res = self.parser.parse("January 2024", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2024-01-01")
        self.assertEqual(res.end_date, "2024-01-31")
        self.assertFalse(res.is_exact)
        self.assertGreaterEqual(res.confidence, 0.9)

    def test_relative_last_year(self):
        res = self.parser.parse("last year", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2023-01-01")
        self.assertEqual(res.end_date, "2023-12-31")
        self.assertFalse(res.is_exact)

    def test_around_month(self):
        res = self.parser.parse("around January", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2024-01-01")
        self.assertEqual(res.end_date, "2024-01-31")

    def test_month_range(self):
        res = self.parser.parse("between January and March", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2024-01-01")
        self.assertEqual(res.end_date, "2024-03-31")

    def test_seasons(self):
        res = self.parser.parse("summer 2024", ref_date=self.ref_date)
        self.assertEqual(res.start_date, "2024-06-01")
        self.assertEqual(res.end_date, "2024-08-31")


class TestEntityExtractor(unittest.TestCase):
    """Test entity extraction from JSON strings and heuristic document text"""

    def setUp(self):
        self.extractor = EntityExtractor()

    def test_parse_json_dict_from_dataset(self):
        raw_json = '{"person": "Aryan Maharaj", "item": "laptop", "organization": "Metro Electronics", "location": "Rajahmundry"}'
        entities = self.extractor.extract_from_json_or_text(raw_json)
        self.assertEqual(entities.get("person"), "Aryan Maharaj")
        self.assertEqual(entities.get("item"), "laptop")
        self.assertEqual(entities.get("organization"), "Metro Electronics")
        self.assertEqual(entities.get("location"), "Rajahmundry")

    def test_to_records(self):
        raw_dict = {"item": "smartphone", "organization": "Apple Store"}
        records = self.extractor.to_records(raw_dict)
        self.assertEqual(len(records), 2)
        names = {r.name for r in records}
        self.assertIn("smartphone", names)
        self.assertIn("Apple Store", names)

    def test_heuristic_extraction(self):
        text = "Personal service record for Harsh Chahal. Item: washing machine. Organization: QuickCare. Date: 2025-01-10."
        entities = self.extractor.extract_heuristic(text, title="Washing Machine Repair")
        self.assertEqual(entities.get("item"), "washing machine")
        self.assertIn("Harsh Chahal", entities.get("person", ""))


class TestEventExtractor(unittest.TestCase):
    """Test event classification and date isolation"""

    def setUp(self):
        self.extractor = EventExtractor()

    def test_repair_event(self):
        title = "Laptop Service Record"
        text = "Event: repair. Date: 2024-01-01. Organization: Metro Electronics."
        event, date = self.extractor.extract_event(text, title=title)
        self.assertEqual(event, "repair")
        self.assertEqual(date, "2024-01-01")

    def test_purchase_event(self):
        title = "Best Buy Store Purchase"
        text = "Receipt for item purchase on 2023-11-20."
        event, date = self.extractor.extract_event(text, title=title)
        self.assertEqual(event, "purchase")
        self.assertEqual(date, "2023-11-20")

    def test_warranty_event(self):
        title = "AppleCare Protection Plan"
        text = "Warranty certificate valid from 2024-05-01."
        event, _ = self.extractor.extract_event(text, title=title)
        self.assertEqual(event, "warranty")


class TestRelationshipExtractor(unittest.TestCase):
    """Test relationship parsing and derivation"""

    def setUp(self):
        self.extractor = RelationshipExtractor()

    def test_parse_triples_from_dataset(self):
        raw = '["Aryan Maharaj -> repair -> laptop", "laptop -> associated_with -> Metro Electronics"]'
        triples = self.extractor.parse_relationships_from_input(raw)
        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0], "Aryan Maharaj -> repair -> laptop")

        records = self.extractor.to_records(triples)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].subject, "Aryan Maharaj")
        self.assertEqual(records[0].predicate, "repair")
        self.assertEqual(records[0].object, "laptop")

    def test_derive_relationships(self):
        entities = {"person": "John Doe", "item": "car", "organization": "Honda Care", "location": "Austin"}
        triples = self.extractor.derive_relationships(entities, event="repair")
        self.assertIn("John Doe -> repair -> car", triples)
        self.assertIn("car -> associated_with -> Honda Care", triples)
        self.assertIn("car -> located_at -> Austin", triples)


class TestQueryContextParser(unittest.TestCase):
    """Test query context parsing across representative benchmark queries"""

    def setUp(self):
        self.parser = QueryContextParser(provider=None)
        self.ref_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    def test_semantic_query(self):
        q = "Show me the document related to my laptop repair."
        ctx = self.parser.parse(q, ref_date=self.ref_date)
        self.assertEqual(ctx.entity, "laptop")
        self.assertEqual(ctx.event, "repair")

    def test_contextual_query_with_time_and_location(self):
        q = "I don't remember the file name. Find the record for the laptop repair around January in Rajahmundry."
        ctx = self.parser.parse(q, ref_date=self.ref_date)
        self.assertEqual(ctx.entity, "laptop")
        self.assertEqual(ctx.event, "repair")
        self.assertIsNotNone(ctx.time_reference)
        self.assertEqual(ctx.resolved_date_start, "2024-01-01")
        self.assertEqual(ctx.resolved_date_end, "2024-01-31")

    def test_warranty_query(self):
        q = "Find the warranty for the laptop I bought last year."
        ctx = self.parser.parse(q, ref_date=self.ref_date)
        self.assertEqual(ctx.entity, "laptop")
        self.assertEqual(ctx.target_information, "warranty")
        self.assertEqual(ctx.event, "purchase")
        self.assertEqual(ctx.resolved_date_start, "2023-01-01")
        self.assertEqual(ctx.resolved_date_end, "2023-12-31")


class TestContextExtractorUnified(unittest.TestCase):
    """Test top-level ContextExtractor integration"""

    def setUp(self):
        self.extractor = ContextExtractor(provider=None)

    def test_extract_from_dataset_fields(self):
        text = "Personal service record for Aryan Maharaj. Item: laptop. Organization: Metro Electronics. Event: repair. Date: 2024-01-01."
        entities_json = '{"person": "Aryan Maharaj", "item": "laptop", "organization": "Metro Electronics", "location": "Rajahmundry"}'
        rel_json = '["Aryan Maharaj -> repair -> laptop", "laptop -> associated_with -> Metro Electronics"]'

        ctx = self.extractor.extract_document_context(
            text=text,
            title="Laptop Service Record",
            raw_entities=entities_json,
            raw_event="repair",
            raw_relationships=rel_json,
            raw_date="2024-01-01",
            document_type="Service Record"
        )

        self.assertIsInstance(ctx, ExtractedContext)
        self.assertEqual(ctx.event, "repair")
        self.assertEqual(ctx.event_date, "2024-01-01")
        self.assertEqual(ctx.entities.get("item"), "laptop")
        self.assertEqual(len(ctx.relationships), 2)
        self.assertEqual(len(ctx.relationship_records), 2)
        self.assertEqual(ctx.relationship_records[0].predicate, "repair")


if __name__ == "__main__":
    unittest.main()

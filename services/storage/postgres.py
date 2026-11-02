"""
PostgreSQL Storage Layer for RecalLens
Handles relational storage for Entities, Events, Relationships, Documents, and Chunks.
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Environment configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "recallens")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

SCHEMA_FILE = Path(__file__).parent / "schema.sql"


class PostgresStorage:
    """Manages structured context persistence in PostgreSQL"""

    def __init__(
        self,
        host: str = POSTGRES_HOST,
        port: int = POSTGRES_PORT,
        dbname: str = POSTGRES_DB,
        user: str = POSTGRES_USER,
        password: str = POSTGRES_PASSWORD
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self._engine = None
        self._connected = False

    def connect(self, retries: int = 5, backoff_seconds: float = 2.0) -> bool:
        """Establish database connection with retry backoff"""
        try:
            from sqlalchemy import create_engine, text
            url = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.dbname}"
            self._engine = create_engine(
                url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True
            )

            for attempt in range(1, retries + 1):
                try:
                    with self._engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    self._connected = True
                    logger.info(f"Connected to PostgreSQL at {self.host}:{self.port}/{self.dbname}")
                    return True
                except Exception as e:
                    logger.warning(f"PostgreSQL connection attempt {attempt}/{retries} failed: {e}")
                    if attempt < retries:
                        time.sleep(backoff_seconds * attempt)

            logger.error("Could not connect to PostgreSQL after retries. Storage running in offline mode.")
            self._connected = False
            return False

        except ImportError:
            logger.warning("psycopg2 / sqlalchemy not installed. PostgreSQL storage running in offline mode.")
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def init_db(self) -> bool:
        """Execute DDL schema to create tables and indexes"""
        if not self._connected and not self.connect():
            logger.warning("PostgreSQL not connected; skipping schema initialization.")
            return False

        try:
            from sqlalchemy import text
            if not SCHEMA_FILE.exists():
                logger.error(f"Schema file not found at {SCHEMA_FILE}")
                return False

            with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
                schema_sql = f.read()

            with self._engine.begin() as conn:
                for statement in schema_sql.split(";"):
                    stmt = statement.strip()
                    if stmt:
                        conn.execute(text(stmt))

            logger.info("RecalLens PostgreSQL schema initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL schema: {e}")
            return False

    def upsert_user(self, user_id: str, name: Optional[str] = None, city: Optional[str] = None, country: Optional[str] = None):
        """Insert or update user record"""
        if not self._connected:
            return
        try:
            from sqlalchemy import text
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO users (user_id, name, city, country)
                        VALUES (:uid, :name, :city, :country)
                        ON CONFLICT (user_id) DO UPDATE SET
                            name = COALESCE(EXCLUDED.name, users.name),
                            city = COALESCE(EXCLUDED.city, users.city),
                            country = COALESCE(EXCLUDED.country, users.country);
                    """),
                    {"uid": user_id, "name": name, "city": city, "country": country}
                )
        except Exception as e:
            logger.warning(f"Failed to upsert user {user_id}: {e}")

    def save_document_context(
        self,
        document_id: str,
        user_id: str,
        title: str,
        document_type: Optional[str] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        document_date: Optional[str] = None,
        event_date: Optional[str] = None,
        location: Optional[str] = None,
        amount: Optional[str] = None,
        page_count: int = 1,
        entities: Optional[Dict[str, Any]] = None,
        event: Optional[str] = None,
        relationships: Optional[List[str]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None
    ):
        """Save a complete document with its entities, events, relationships, and chunks"""
        if not self._connected:
            return

        try:
            from sqlalchemy import text
            self.upsert_user(user_id)

            # Parse dates
            doc_dt = None
            if document_date:
                try:
                    doc_dt = datetime.fromisoformat(str(document_date).replace("Z", "+00:00"))
                except Exception:
                    pass

            evt_dt = None
            if event_date:
                try:
                    evt_dt = datetime.fromisoformat(str(event_date).replace("Z", "+00:00"))
                except Exception:
                    pass

            with self._engine.begin() as conn:
                # 1. Upsert document
                conn.execute(
                    text("""
                        INSERT INTO documents (
                            document_id, user_id, title, document_type, source_type,
                            source_id, document_date, event_date, location, amount, page_count
                        ) VALUES (
                            :doc_id, :uid, :title, :dtype, :stype,
                            :sid, :ddate, :edate, :loc, :amt, :pcount
                        )
                        ON CONFLICT (document_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            document_type = EXCLUDED.document_type,
                            document_date = EXCLUDED.document_date,
                            event_date = EXCLUDED.event_date,
                            location = EXCLUDED.location,
                            amount = EXCLUDED.amount;
                    """),
                    {
                        "doc_id": document_id,
                        "uid": user_id,
                        "title": title,
                        "dtype": document_type,
                        "stype": source_type,
                        "sid": source_id,
                        "ddate": doc_dt,
                        "edate": evt_dt or doc_dt,
                        "loc": location,
                        "amt": str(amount) if amount else None,
                        "pcount": page_count
                    }
                )

                # 2. Chunks
                if chunks:
                    for chunk in chunks:
                        conn.execute(
                            text("""
                                INSERT INTO chunks (chunk_id, document_id, page_number, chunk_index, chunk_text, start_offset, end_offset)
                                VALUES (:cid, :did, :pnum, :cidx, :ctxt, :soff, :eoff)
                                ON CONFLICT (chunk_id) DO UPDATE SET
                                    chunk_text = EXCLUDED.chunk_text;
                            """),
                            {
                                "cid": chunk.get("id"),
                                "did": document_id,
                                "pnum": chunk.get("page_number", 1),
                                "cidx": chunk.get("chunk_index", 0),
                                "ctxt": chunk.get("content", ""),
                                "soff": chunk.get("start_offset"),
                                "eoff": chunk.get("end_offset")
                            }
                        )

                # 3. Entities
                if entities:
                    for etype, ename in entities.items():
                        if ename and isinstance(ename, str):
                            res = conn.execute(
                                text("""
                                    INSERT INTO entities (user_id, entity_name, entity_type)
                                    VALUES (:uid, :ename, :etype)
                                    ON CONFLICT (user_id, entity_name, entity_type) DO UPDATE SET entity_name = EXCLUDED.entity_name
                                    RETURNING entity_id;
                                """),
                                {"uid": user_id, "ename": ename.strip(), "etype": etype.strip()}
                            ).fetchone()
                            eid = res[0] if res else None
                            if eid:
                                conn.execute(
                                    text("""
                                        INSERT INTO document_entities (document_id, entity_id, mention_text)
                                        VALUES (:did, :eid, :mtext);
                                    """),
                                    {"did": document_id, "eid": eid, "mtext": ename.strip()}
                                )

                # 4. Events
                if event:
                    eres = conn.execute(
                        text("""
                            INSERT INTO events (user_id, event_type, event_date, description)
                            VALUES (:uid, :etype, :edate, :desc)
                            RETURNING event_id;
                        """),
                        {"uid": user_id, "etype": event.strip(), "edate": evt_dt or doc_dt, "desc": f"Event {event} on {title}"}
                    ).fetchone()
                    ev_id = eres[0] if eres else None
                    if ev_id:
                        conn.execute(
                            text("""
                                INSERT INTO event_documents (event_id, document_id, role)
                                VALUES (:evid, :did, :role);
                            """),
                            {"evid": ev_id, "did": document_id, "role": document_type or "primary"}
                        )

                # 5. Relationships
                if relationships:
                    for rel in relationships:
                        if isinstance(rel, str) and "->" in rel:
                            parts = [p.strip() for p in rel.split("->")]
                            subj = parts[0] if len(parts) > 0 else ""
                            pred = parts[1] if len(parts) > 1 else "related_to"
                            obj = parts[2] if len(parts) > 2 else ""
                            if subj and obj:
                                conn.execute(
                                    text("""
                                        INSERT INTO relationships (user_id, subject_name, predicate, object_name, document_id, evidence_text)
                                        VALUES (:uid, :sub, :pred, :obj, :did, :ev);
                                    """),
                                    {"uid": user_id, "sub": subj, "pred": pred, "obj": obj, "did": document_id, "ev": rel}
                                )

            logger.debug(f"Saved document context for {document_id} in PostgreSQL")

        except Exception as e:
            logger.error(f"Failed to save document context for {document_id}: {e}")

    def query_contextual_candidates(
        self,
        user_id: str,
        entity: Optional[str] = None,
        event: Optional[str] = None,
        date_start: Optional[str] = None,
        date_end: Optional[str] = None,
        relationship: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Structured candidate retrieval via PostgreSQL joins across entities, events, dates, and relationships.
        Returns matching document IDs with match scores and evidence.
        """
        if not self._connected:
            return []

        try:
            from sqlalchemy import text
            conditions = ["d.user_id = :uid"]
            params: Dict[str, Any] = {"uid": user_id, "limit": limit}

            # Entity matching
            if entity:
                conditions.append("""
                    (EXISTS (
                        SELECT 1 FROM document_entities de
                        JOIN entities e ON de.entity_id = e.entity_id
                        WHERE de.document_id = d.document_id
                        AND LOWER(e.entity_name) LIKE :entity
                    ) OR LOWER(d.title) LIKE :entity)
                """)
                params["entity"] = f"%{entity.lower().strip()}%"

            # Event matching
            if event:
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM event_documents ed
                        JOIN events ev ON ed.event_id = ev.event_id
                        WHERE ed.document_id = d.document_id
                        AND LOWER(ev.event_type) = :event
                    )
                """)
                params["event"] = event.lower().strip()

            # Date range matching
            if date_start:
                try:
                    dt_s = datetime.fromisoformat(str(date_start).replace("Z", "+00:00"))
                    conditions.append("d.event_date >= :date_start")
                    params["date_start"] = dt_s
                except Exception:
                    pass

            if date_end:
                try:
                    dt_e = datetime.fromisoformat(str(date_end).replace("Z", "+00:00"))
                    conditions.append("d.event_date <= :date_end")
                    params["date_end"] = dt_e
                except Exception:
                    pass

            # Relationship matching
            if relationship:
                conditions.append("""
                    EXISTS (
                        SELECT 1 FROM relationships r
                        WHERE r.document_id = d.document_id
                        AND (LOWER(r.subject_name) LIKE :rel OR LOWER(r.object_name) LIKE :rel OR LOWER(r.predicate) LIKE :rel)
                    )
                """)
                params["rel"] = f"%{relationship.lower().strip()}%"

            where_clause = " AND ".join(conditions)
            sql = f"""
                SELECT
                    d.document_id,
                    d.title,
                    d.document_type,
                    d.document_date,
                    d.event_date,
                    d.location,
                    d.amount
                FROM documents d
                WHERE {where_clause}
                ORDER BY d.event_date DESC NULLS LAST
                LIMIT :limit;
            """

            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()

            candidates = []
            for r in rows:
                candidates.append({
                    "document_id": r[0],
                    "title": r[1],
                    "document_type": r[2],
                    "document_date": str(r[3]) if r[3] else None,
                    "event_date": str(r[4]) if r[4] else None,
                    "location": r[5],
                    "amount": r[6]
                })

            logger.info(f"Structured context query returned {len(candidates)} candidates for user {user_id}")
            return candidates

        except Exception as e:
            logger.error(f"Structured context query failed: {e}")
            return []

    def get_document_context_details(self, document_id: str) -> Dict[str, Any]:
        """Fetch all stored entities, events, relationships, and chunks for a document"""
        if not self._connected:
            return {}

        try:
            from sqlalchemy import text
            details: Dict[str, Any] = {
                "entities": [],
                "events": [],
                "relationships": [],
                "chunks": []
            }

            with self._engine.connect() as conn:
                # Entities
                erows = conn.execute(
                    text("""
                        SELECT e.entity_name, e.entity_type
                        FROM document_entities de
                        JOIN entities e ON de.entity_id = e.entity_id
                        WHERE de.document_id = :did;
                    """),
                    {"did": document_id}
                ).fetchall()
                details["entities"] = [{"name": r[0], "type": r[1]} for r in erows]

                # Events
                evrows = conn.execute(
                    text("""
                        SELECT ev.event_type, ev.event_date, ed.role
                        FROM event_documents ed
                        JOIN events ev ON ed.event_id = ev.event_id
                        WHERE ed.document_id = :did;
                    """),
                    {"did": document_id}
                ).fetchall()
                details["events"] = [{"type": r[0], "date": str(r[1]) if r[1] else None, "role": r[2]} for r in evrows]

                # Relationships
                rrows = conn.execute(
                    text("""
                        SELECT subject_name, predicate, object_name, evidence_text
                        FROM relationships
                        WHERE document_id = :did;
                    """),
                    {"did": document_id}
                ).fetchall()
                details["relationships"] = [f"{r[0]} -> {r[1]} -> {r[2]}" for r in rrows]

                # Chunks
                crows = conn.execute(
                    text("""
                        SELECT chunk_id, page_number, chunk_index, chunk_text
                        FROM chunks
                        WHERE document_id = :did
                        ORDER BY chunk_index ASC;
                    """),
                    {"did": document_id}
                ).fetchall()
                details["chunks"] = [{"id": r[0], "page": r[1], "index": r[2], "text": r[3]} for r in crows]

            return details

        except Exception as e:
            logger.error(f"Failed to fetch context details for doc {document_id}: {e}")
            return {}

    def log_retrieval(
        self,
        query_id: str,
        user_id: str,
        query_text: str,
        method: str,
        document_id: str,
        rank: int,
        score: float
    ):
        """Log a retrieval event for benchmark experiments (E1-E4)"""
        if not self._connected:
            return
        try:
            from sqlalchemy import text
            with self._engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO retrieval_logs (query_id, user_id, query_text, retrieval_method, document_id, rank, score)
                        VALUES (:qid, :uid, :qtxt, :method, :did, :rank, :score);
                    """),
                    {
                        "qid": query_id,
                        "uid": user_id,
                        "qtxt": query_text,
                        "method": method,
                        "did": document_id,
                        "rank": rank,
                        "score": score
                    }
                )
        except Exception as e:
            logger.debug(f"Failed to log retrieval: {e}")


# Singleton instance
_client_instance: Optional[PostgresStorage] = None


def get_postgres_client() -> PostgresStorage:
    """Get or create singleton PostgreSQL client instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = PostgresStorage()
    return _client_instance


def init_db() -> bool:
    """Helper to initialize database and tables"""
    client = get_postgres_client()
    return client.init_db()

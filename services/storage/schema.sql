-- =============================================================================
-- RecalLens Database Schema (PostgreSQL 16)
-- Context-Driven Personal Information Retrieval
-- =============================================================================

-- Users and contexts
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255),
    city VARCHAR(128),
    country VARCHAR(128),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    document_type VARCHAR(64),
    source_type VARCHAR(64),
    source_id VARCHAR(128),
    document_date TIMESTAMP WITH TIME ZONE,
    event_date TIMESTAMP WITH TIME ZONE,
    location VARCHAR(128),
    amount VARCHAR(64),
    page_count INTEGER DEFAULT 1,
    status VARCHAR(32) DEFAULT 'indexed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_docs_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_docs_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_docs_date ON documents(document_date);
CREATE INDEX IF NOT EXISTS idx_docs_event_date ON documents(event_date);

-- Page-level document chunks for evidence grounding
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(128) PRIMARY KEY,
    document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    page_number INTEGER DEFAULT 1,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

-- Named Entities
CREATE TABLE IF NOT EXISTS entities (
    entity_id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    entity_name VARCHAR(255) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    CONSTRAINT uq_user_entity UNIQUE (user_id, entity_name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_entities_lookup ON entities(user_id, LOWER(entity_name));

-- Document Entity occurrences
CREATE TABLE IF NOT EXISTS document_entities (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    entity_id INTEGER REFERENCES entities(entity_id) ON DELETE CASCADE,
    mention_text VARCHAR(255),
    chunk_id VARCHAR(128) REFERENCES chunks(chunk_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_doc_entities ON document_entities(document_id, entity_id);

-- Events
CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    event_type VARCHAR(64) NOT NULL,
    event_date TIMESTAMP WITH TIME ZONE,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_events_user_type ON events(user_id, event_type);

-- Event to Document connections
CREATE TABLE IF NOT EXISTS event_documents (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES events(event_id) ON DELETE CASCADE,
    document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    role VARCHAR(64)
);

-- Semantic Relationships
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id SERIAL PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    subject_name VARCHAR(255) NOT NULL,
    predicate VARCHAR(64) NOT NULL,
    object_name VARCHAR(255) NOT NULL,
    document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    evidence_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_rel_user ON relationships(user_id);
CREATE INDEX IF NOT EXISTS idx_rel_search ON relationships(LOWER(subject_name), predicate, LOWER(object_name));

-- Benchmark queries table
CREATE TABLE IF NOT EXISTS queries (
    query_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) REFERENCES users(user_id) ON DELETE CASCADE,
    query_text TEXT NOT NULL,
    query_type VARCHAR(64),
    entity VARCHAR(255),
    event VARCHAR(64),
    time_reference VARCHAR(128),
    relationship TEXT,
    difficulty VARCHAR(32),
    split VARCHAR(32),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Relevance ground truth table
CREATE TABLE IF NOT EXISTS relevance (
    id SERIAL PRIMARY KEY,
    query_id VARCHAR(64) REFERENCES queries(query_id) ON DELETE CASCADE,
    document_id VARCHAR(64) REFERENCES documents(document_id) ON DELETE CASCADE,
    relevance INTEGER NOT NULL,
    relevance_type VARCHAR(64),
    evidence TEXT,
    split VARCHAR(32)
);

CREATE INDEX IF NOT EXISTS idx_rel_query ON relevance(query_id);

-- Real-time retrieval logs for research evaluation
CREATE TABLE IF NOT EXISTS retrieval_logs (
    log_id SERIAL PRIMARY KEY,
    query_id VARCHAR(64),
    user_id VARCHAR(64),
    query_text TEXT,
    retrieval_method VARCHAR(32),
    document_id VARCHAR(64),
    rank INTEGER,
    score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

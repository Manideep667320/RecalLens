"""
RecalLens Dataset Ingestion CLI
Indexes documents from dataset/documents.csv into Qdrant & PostgreSQL.
"""

import os
import sys
import argparse
import logging
import time
import csv

# Ensure services directory is in sys.path
services_dir = os.path.dirname(os.path.abspath(__file__))
if services_dir not in sys.path:
    sys.path.insert(0, services_dir)

from providers import create_provider
from indexer import DocumentIndexer
from models import DatasetDocumentRecord
from storage import get_postgres_client

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("index_dataset")


def _init_environment():
    root_dir = os.path.dirname(services_dir)
    env_paths = [os.path.join(root_dir, '.env'), os.path.join(services_dir, '.env'), '.env']
    for p in env_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k not in os.environ and v:
                            os.environ[k] = v
            break

    is_docker = os.path.exists('/.dockerenv') or os.getenv('RUNNING_IN_DOCKER', '').lower() == 'true'
    if not is_docker:
        if os.getenv('POSTGRES_HOST') == 'postgres':
            os.environ['POSTGRES_HOST'] = 'localhost'
        q_url = os.getenv('QDRANT_URL', '')
        if 'qdrant:6333' in q_url:
            os.environ['QDRANT_URL'] = q_url.replace('qdrant:6333', 'localhost:6333')
        if not os.getenv('QDRANT_URL'):
            os.environ['QDRANT_URL'] = 'http://localhost:6333'
        ol_url = os.getenv('OLLAMA_URL', '')
        if 'host.docker.internal:11434' in ol_url:
            os.environ['OLLAMA_URL'] = ol_url.replace('host.docker.internal:11434', 'localhost:11434')


def index_documents(csv_path: str, limit: int = 100):
    _init_environment()

    if not os.path.exists(csv_path):
        logger.error(f"Dataset CSV not found at {csv_path}")
        sys.exit(1)

    logger.info(f"Loading documents from {csv_path}...")
    rows = []
    with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if limit > 0:
        rows = rows[:limit]
    total_docs = len(rows)
    logger.info(f"Preparing to index {total_docs} personal documents...")

    # Initialize Provider
    provider_type = os.getenv('PROVIDER', 'ollama')
    is_docker = os.path.exists('/.dockerenv') or os.getenv('RUNNING_IN_DOCKER', '').lower() == 'true'
    ollama_url = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434' if is_docker else 'http://localhost:11434')
    qdrant_url = os.getenv('QDRANT_URL', 'http://qdrant:6333' if is_docker else 'http://localhost:6333')
    collection_name = os.getenv('QDRANT_COLLECTION', 'documents')
    embedding_model = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
    llm_model = os.getenv('LLM_MODEL', 'phi:latest')
    rerank_model = os.getenv('RERANK_MODEL', 'phi:latest')

    provider = create_provider(
        provider_type=provider_type,
        ollama_url=ollama_url,
        embedding_model=embedding_model,
        llm_model=llm_model,
        rerank_model=rerank_model,
        google_api_key=os.getenv('GOOGLE_API_KEY')
    )

    logger.info(f"Provider: {provider_type} (Embedding: {embedding_model}, Qdrant: {qdrant_url})")

    indexer = DocumentIndexer(
        paperless_url=os.getenv('RECALLENS_URL', 'http://localhost:8000'),
        paperless_api_key=os.getenv('RECALLENS_API_KEY', 'local-dev-key'),
        qdrant_url=qdrant_url,
        collection_name=collection_name,
        provider=provider
    )

    # Pre-flight check: Verify Qdrant connection immediately
    logger.info(f"Checking Qdrant connection at {qdrant_url}...")
    try:
        indexer.qdrant_client.get_collections()
        logger.info(f"✅ Qdrant connection verified at {qdrant_url}")
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ CANNOT CONNECT TO QDRANT at {qdrant_url}")
        logger.error("The Qdrant container was created before port 6333 was exposed.")
        logger.error("Please run this command to recreate Qdrant with port 6333 open:")
        logger.error("    docker-compose up -d --force-recreate qdrant")
        logger.error("=" * 60)
        sys.exit(1)

    start_time = time.time()
    indexed_count = 0
    total_chunks = 0

    for idx, row in enumerate(rows):
        try:
            record = DatasetDocumentRecord(
                document_id=str(row.get('document_id', f'DOC_{idx:05d}')),
                user_id=str(row.get('user_id', 'USER_001')),
                document_type=str(row.get('document_type', 'Document')),
                title=str(row.get('title', 'Untitled Document')),
                source_type=str(row.get('source_type', 'personal')),
                date=str(row.get('date', '')) if row.get('date') else None,
                text=str(row.get('text', '')),
                entities=str(row.get('entities', '')) if row.get('entities') else None,
                event=str(row.get('event', '')) if row.get('event') else None,
                relationships=str(row.get('relationships', '')) if row.get('relationships') else None,
                location=str(row.get('location', '')) if row.get('location') else None,
                amount=str(row.get('amount', '')) if row.get('amount') else None,
                page_count=int(row.get('page_count', 1)) if row.get('page_count') else 1
            )

            chunks = indexer.index_dataset_record(record)
            indexed_count += 1
            total_chunks += chunks
            if (idx + 1) % 10 == 0 or (idx + 1) == total_docs:
                logger.info(f"Progress: [{idx + 1}/{total_docs}] documents indexed ({total_chunks} chunks)...")
        except Exception as e:
            logger.warning(f"Error indexing document {row.get('document_id')}: {e}")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"✅ Ingestion Complete! Successfully indexed {indexed_count}/{total_docs} documents ({total_chunks} vector chunks)")
    logger.info(f"Time taken: {elapsed:.2f}s (~{elapsed / max(1, indexed_count):.2f}s per document)")
    logger.info("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RecalLens Dataset Ingestion")
    parser.add_argument('--limit', type=int, default=100, help="Number of documents to index (0 = all documents, default = 100)")
    parser.add_argument('--csv', type=str, default=os.path.join(os.path.dirname(services_dir), 'dataset', 'documents.csv'), help="Path to documents.csv")
    args = parser.parse_args()

    index_documents(args.csv, limit=args.limit)

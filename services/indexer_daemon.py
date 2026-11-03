"""
Standalone indexer daemon for RecalLens documents
Runs independently from the web UI
"""

import os
import sys
import time
import signal
import logging
import schedule
from indexer import DocumentIndexer

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper to automatically load .env and adapt hostnames when running locally
def _init_environment():
    services_dir = os.path.dirname(os.path.abspath(__file__))
    env_paths = [
        os.path.join(os.path.dirname(services_dir), '.env'),
        os.path.join(services_dir, '.env'),
        '.env'
    ]
    for p in env_paths:
        if os.path.exists(p):
            try:
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
            except Exception:
                pass

    is_docker = os.path.exists('/.dockerenv') or os.getenv('RUNNING_IN_DOCKER', '').lower() == 'true'
    if not is_docker:
        if os.getenv('POSTGRES_HOST') == 'postgres':
            os.environ['POSTGRES_HOST'] = 'localhost'
        q_url = os.getenv('QDRANT_URL', '')
        if 'qdrant:6333' in q_url:
            os.environ['QDRANT_URL'] = q_url.replace('qdrant:6333', 'localhost:6333')
        ol_url = os.getenv('OLLAMA_URL', '')
        if 'host.docker.internal:11434' in ol_url:
            os.environ['OLLAMA_URL'] = ol_url.replace('host.docker.internal:11434', 'localhost:11434')

_init_environment()
RECALLENS_URL = os.getenv('RECALLENS_URL', os.getenv('PAPERLESS_URL', 'http://paperless:8000'))
PAPERLESS_URL = RECALLENS_URL
RECALLENS_API_KEY = (os.getenv('RECALLENS_API_KEY') or os.getenv('PAPERLESS_API_KEY') or '').strip()
PAPERLESS_API_KEY = RECALLENS_API_KEY

# Validate required configuration
if not RECALLENS_API_KEY or RECALLENS_API_KEY.startswith('your_'):
    logger.info("RECALLENS_API_KEY not configured. Using 'local-dev-key' for standalone mode.")
    RECALLENS_API_KEY = 'local-dev-key'
    PAPERLESS_API_KEY = RECALLENS_API_KEY




is_docker_env = os.path.exists('/.dockerenv') or os.getenv('RUNNING_IN_DOCKER', '').lower() == 'true'
QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333' if is_docker_env else 'http://localhost:6333')
if not is_docker_env and 'qdrant:6333' in QDRANT_URL:
    QDRANT_URL = QDRANT_URL.replace('qdrant:6333', 'localhost:6333')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'documents')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434' if is_docker_env else 'http://localhost:11434')
if not is_docker_env and 'host.docker.internal' in OLLAMA_URL:
    OLLAMA_URL = OLLAMA_URL.replace('host.docker.internal', 'localhost')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
POLL_INTERVAL_MINUTES = int(os.getenv('POLL_INTERVAL_MINUTES', '5'))
REQUIRE_REVIEWED = os.getenv('REQUIRE_REVIEWED', 'true').lower() == 'true'
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def run_indexing_poll(indexer: DocumentIndexer):
    """Run indexing poll (called by scheduler)"""
    try:
        logger.info("Starting scheduled indexing poll...")
        stats = indexer.poll_and_index()
        logger.info(
            f"Poll complete: {stats.documents_processed} docs, "
            f"{stats.chunks_created} chunks in {stats.time_taken_seconds:.2f}s"
        )
        if stats.errors:
            logger.warning(f"Encountered {len(stats.errors)} errors during indexing")
    except Exception as e:
        logger.error(f"Indexing poll failed: {e}")


def main():
    """Main indexer daemon"""
    logger.info("=" * 60)
    logger.info("RecalLens Indexer Daemon starting...")
    logger.info(f"RecalLens / Document Store URL: {RECALLENS_URL}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Poll interval: {POLL_INTERVAL_MINUTES} minutes")
    logger.info(f"Require reviewed: {REQUIRE_REVIEWED}")
    logger.info("=" * 60)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize indexer
    try:
        logger.info("Initializing indexer...")

        # Initialize PostgreSQL storage
        try:
            from storage import init_db
            init_db()
        except Exception as e:
            logger.warning(f"PostgreSQL initialization deferred: {e}")

        # Create provider
        from providers import create_provider
        provider = create_provider()

        indexer = DocumentIndexer(
            paperless_url=PAPERLESS_URL,
            paperless_api_key=PAPERLESS_API_KEY,
            qdrant_url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            provider=provider,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            require_reviewed=REQUIRE_REVIEWED
        )

        logger.info("Indexer initialized successfully (collection will be created on first document)")

    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
    

    # Schedule indexing polls
    logger.info(f"Scheduling indexing every {POLL_INTERVAL_MINUTES} minutes")
    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(
        lambda: run_indexing_poll(indexer)
    )

    # Run initial indexing immediately
    logger.info("Running initial indexing poll...")
    run_indexing_poll(indexer)

    # Main loop
    logger.info("Indexer daemon ready, entering main loop")
    while not shutdown_requested:
        schedule.run_pending()
        time.sleep(10)  # Check every 10 seconds

    logger.info("Indexer daemon shutting down")
    sys.exit(0)


if __name__ == "__main__":
    main()

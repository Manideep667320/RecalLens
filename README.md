# RecalLens: Context-Driven Personal Information Retrieval
### Using Event, Entity, and Temporal Reasoning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20Alpine-blue.svg)](https://www.postgresql.org/)
[![Tests](https://img.shields.io/badge/Tests-44%20Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**RecalLens** is an advanced context-driven personal information retrieval system that extends [Paperless Search](https://github.com/xehonk/paperless-search) and [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx). 

Instead of relying solely on generic keyword matching or embedding cosine similarity, RecalLens grounds retrieval in the way human memory actually recalls personal records: **who was involved (Entities), what happened (Events), when it occurred (Temporal Logic), and how items connect (Relationships)**.

---

## Table of Contents

- [Why RecalLens? The "Semantic Trap" in Personal Search](#why-recallens-the-semantic-trap-in-personal-search)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Benchmark & Evaluation Results](#benchmark--evaluation-results)
- [How to Run the Project](#how-to-run-the-project)
  - [Prerequisites](#prerequisites)
  - [Method 1: Docker Compose (Full Stack)](#method-1-docker-compose-full-stack)
  - [Method 2: Local Python Execution (Development)](#method-2-local-python-execution-development)
  - [Method 3: Running the Benchmark Evaluation Suite](#method-3-running-the-benchmark-evaluation-suite)
  - [Method 4: Running Unit Tests](#method-4-running-unit-tests)
- [Configuration Reference](#configuration-reference)
- [Directory Structure](#directory-structure)
- [License & Acknowledgements](#license--acknowledgements)

---

## Why RecalLens? The "Semantic Trap" in Personal Search

When searching personal document archives (tuition fees, gadget receipts, utility bills, rental agreements, doctor prescriptions), users typically ask episodic questions:
> *"I don't remember the file name. Find the record for the laptop repair around January in Rajahmundry."*

Standard search systems fail on these queries:
1. **Lexical Keyword Search (BM25):** Fails because conversational queries contain natural language filler that does not match OCR-scanned receipt texts.
2. **Dense Vector Search:** Falls into the **"Semantic Trap"**—to an embedding model, multiple laptop receipts or recurring utility bills look almost identical in vector space. Without temporal bounds and entity constraints, vector search retrieves the wrong year or the wrong merchant.
3. **RecalLens Solution:** Extracts structured semantic slots, retrieves candidates through dual channels (dense vectors + PostgreSQL structured queries), fuses them via **Reciprocal Rank Fusion (RRF)**, and reranks candidates using a **multi-signal scoring formula**.

---

## Key Features

- **Dual-Store Architecture:**
  - **Qdrant:** Dense semantic embeddings for passage recall.
  - **PostgreSQL:** Relational schema storing extracted entities, events, temporal dates, and semantic triples (`Subject -> Predicate -> Object`).
- **Context Extraction Engine:**
  - `TemporalParser`: Resolves fuzzy, natural temporal expressions (*"last fall"*, *"around January"*, *"between March and May 2024"*) into strict ISO calendar intervals $[t_{\text{start}}, t_{\text{end}}]$.
  - `EntityExtractor`: Extracts devices, merchants, individuals, and locations.
  - `EventExtractor`: Identifies personal events (`purchase`, `repair`, `billing`, `submission`, `travel`, `warranty`, `medical`, `contract`).
  - `RelationshipExtractor`: Canonicalizes relational knowledge triples.
- **Reciprocal Rank Fusion (RRF with $k=60$):**
  Combines unstructured vector candidates with structured database candidates to guarantee that documents with OCR noise or rare entity names are never lost from candidate sets.
- **Multi-Signal Context Scoring:**
  $$S(d, q) = w_s S_{\text{semantic}} + w_e S_{\text{entity}} + w_v S_{\text{event}} + w_t S_{\text{temporal}} + w_r S_{\text{relationship}}$$
  Calibrated defaults: $w_s=0.35, w_e=0.25, w_v=0.15, w_t=0.15, w_r=0.10$.
- **Human-Centered UI/UX:**
  - Double-Bezel (*Doppelrand*) cards with clean typography (**Plus Jakarta Sans** & **JetBrains Mono**).
  - 1-click **Quick Example Chips** for students and everyday life (*🎓 Tuition Receipt*, *💻 Laptop Warranty*, *🏠 Rental Lease*, *🩺 Medical Bill*).
  - **Live Context Reasoning Banner** showing users what entities, events, and dates were understood.
  - **Grounded Proof Ribbons** on every result card stating exactly *why* a document matched.
  - Interactive **AI Explanation Drawer** and collapsible **Score Details Breakdown**.
- **Automated Research Benchmark Suite:**
  Evaluates P@1, P@5, Recall@5, MRR, nDCG@5, and MAP with difficulty breakdowns and component ablation studies (A1–A5).

---

## System Architecture

```
User Query: "Find the warranty for my laptop I bought last year"
                           │
                           ▼
          ┌──────────────────────────────────┐
          │     Query Context Parser         │
          │  • Entity: "laptop"              │
          │  • Event: "purchase" / "warranty"│
          │  • Time: 2024-01-01 → 2024-12-31 │
          └────────────────┬─────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌───────────────────┐       ┌───────────────────┐
   │   Vector Search   │       │ PostgreSQL Filter │
   │ (Qdrant Candidates│       │(Structured Context│
   │   Top-40 Pool)    │       │   Top-40 Pool)    │
   └─────────┬─────────┘       └─────────┬─────────┘
             └─────────────┬─────────────┘
                           ▼
          ┌──────────────────────────────────┐
          │   Reciprocal Rank Fusion (RRF)   │
          │   k = 60, Weights: 1.0 vs 1.2    │
          └────────────────┬─────────────────┘
                           ▼
          ┌──────────────────────────────────┐
          │   Multi-Signal Context Scorer    │
          │ S(d,q) = 0.35*Ss + 0.25*Se +     │
          │          0.15*Sv + 0.15*St +     │
          │          0.10*Sr                 │
          └────────────────┬─────────────────┘
                           ▼
          ┌──────────────────────────────────┐
          │  Double-Bezel Results UI         │
          │  • Context Understanding Banner  │
          │  • Grounded Evidence Match Proof │
          │  • One-Click PDF View & Download │
          └──────────────────────────────────┘
```

---

## Benchmark & Evaluation Results

Evaluated across the **1,500 personal documents** and **750 held-out test queries** of the MemoraX benchmark dataset (split strictly by synthetic user context to prevent data leakage):

| System / Model Architecture | Precision@1 | Precision@5 | Recall@5 | MRR | nDCG@5 | MAP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E1: BM25 (Keyword Baseline)** | 0.1600 | 0.1557 | 0.7787 | 0.3631 | 0.4652 | 0.3631 |
| **E2: Dense Vector Retrieval** | 0.2427 | 0.1571 | 0.7853 | 0.4356 | 0.5225 | 0.4356 |
| **E3: Hybrid Baseline (Dense + Lexical)** | 0.1573 | 0.1557 | 0.7787 | 0.3617 | 0.4642 | 0.3617 |
| **E4: RecalLens (Full System)** | **1.0000** | **0.2000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| *A1: No Entity Matching ($w_e=0$)* | 0.9987 | 0.2000 | 1.0000 | 0.9993 | 0.9995 | 0.9993 |
| *A2: No Event Compatibility ($w_v=0$)* | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| *A3: No Temporal Reasoning ($w_t=0$)* | **0.6853** | **0.2000** | **1.0000** | **0.8416** | **0.8830** | **0.8416** |
| *A4: No Relationship Matching ($w_r=0$)* | 1.0000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| *A5: No RRF Fusion (Vector candidates only)*| 1.0000 | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Note: For single targeted document queries ($|\text{Relevant}|=1$), maximum theoretical $\text{Precision@5} = \frac{1}{5} = 0.2000$ and $\text{Recall@5} = 1.0000$.*

### Key Analytical Takeaways
- **Contextual Memory Queries (+1053% P@1):** On hard conversational queries (*"I don't remember the file name..."*), baseline hybrid search placed the target document first only **8.67% of the time**. RecalLens achieved **100% P@1**.
- **Temporal Reasoning is Crucial (Ablation A3):** Disabling temporal reasoning causes Top-1 accuracy to collapse by **31.47%** (1.0000 $\rightarrow$ 0.6853), proving that temporal interval resolution is the decisive factor in personal document retrieval.

---

## How to Run the Project

### Prerequisites

- **Docker & Docker Compose** (v2+)
- **Python 3.10+** (if running locally without Docker)
- Optional: RecalLens / Paperless-ngx instance (or use the built-in dataset mode)
- Optional: Google Gemini API key or local [Ollama](https://ollama.ai/) instance

---

### Method 1: Docker Compose (Full Stack)

This is the recommended method to run all services (Web UI, background indexer daemon, Qdrant vector database, and PostgreSQL database).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/RecalLens.git
   cd RecalLens
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   Open `.env` and set your preferred AI provider:
   ```env
   # Choose provider: 'google' or 'ollama'
   PROVIDER=google
   GOOGLE_API_KEY=your_gemini_api_key_here

   # PostgreSQL Configuration
   POSTGRES_DB=recallens
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=postgres
   ```

3. **Start all services:**
   ```bash
   docker-compose up -d --build
   ```

4. **Access the Web Interface:**
   - Open your browser at: **`http://localhost:7702`**
   - Click any of the **Quick Example Chips** (e.g. *🎓 Tuition Receipt*, *💻 Laptop Warranty*) to perform instant contextual searches.

5. **Stop services:**
   ```bash
   docker-compose down
   ```

---

### Method 2: Local Python Execution (Development)

To run the FastAPI search application directly on your host machine:

1. **Create and activate a virtual environment:**
   ```bash
   # Windows PowerShell:
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r services/requirements.txt
   pip install pandas
   ```

3. **Start the database containers (PostgreSQL & Qdrant):**
   ```bash
   docker-compose up -d postgres qdrant
   ```

4. **Set environment variables:**
   ```bash
   # Windows PowerShell:
   $env:POSTGRES_HOST="localhost"
   $env:POSTGRES_PORT="5432"
   $env:POSTGRES_DB="recallens"
   $env:POSTGRES_USER="postgres"
   $env:POSTGRES_PASSWORD="postgres"
   $env:QDRANT_URL="http://localhost:6333"
   $env:RECALLENS_API_KEY="test-key"  # (or PAPERLESS_API_KEY)
   $env:PROVIDER="ollama"  # or "google"

   # Linux / macOS:
   export POSTGRES_HOST=localhost
   export POSTGRES_PORT=5432
   export POSTGRES_DB=recallens
   export POSTGRES_USER=postgres
   export POSTGRES_PASSWORD=postgres
   export QDRANT_URL=http://localhost:6333
   export RECALLENS_API_KEY=test-key  # (or PAPERLESS_API_KEY)
   export PROVIDER=ollama
   ```

5. **Launch the RecalLens Web Server:**
   ```bash
   python services/main.py
   ```
   The UI will be accessible at: **`http://localhost:8080`**

---

### Method 3: Running the Benchmark Evaluation Suite

RecalLens includes an automated research evaluation runner that benchmarks the system on the MemoraX dataset:

1. **Run evaluation across all 750 test queries:**
   ```bash
   python services/evaluation/runner.py --split test --limit 0 --k 5
   ```

2. **Run a fast sample evaluation (100 queries):**
   ```bash
   python services/evaluation/runner.py --split test --limit 100 --k 5
   ```

3. **Inspect the detailed benchmark results:**
   Results, per-query breakdowns, and ablation tables are automatically saved to:
   - `services/evaluation/results/benchmark_report.json`

---

### Method 4: Running Unit Tests

To run the complete automated test suite (verifying storage, temporal parsing, entity extraction, RRF fusion, context scoring, and IR metrics):

```bash
python -m unittest discover -s services/tests
```

**Expected output:**
```
............................................
----------------------------------------------------------------------
Ran 44 tests in 0.433s

OK
```

---

## Configuration Reference

Key settings configurable in `.env` or system environment variables:

| Variable | Default | Description |
| :--- | :---: | :--- |
| `PROVIDER` | `ollama` | AI provider: `google` or `ollama` |
| `GOOGLE_API_KEY` | - | Google Gemini API key (if using Google provider) |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Bi-encoder text embedding model |
| `RERANK_MODEL` | `bge-reranker-v2-m3` | Dedicated cross-encoder model |
| `POSTGRES_HOST` | `postgres` | PostgreSQL hostname |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `recallens` | Database name |
| `POSTGRES_USER` | `postgres` | Database user |
| `POSTGRES_PASSWORD` | `postgres` | Database password |
| `WEIGHT_SEMANTIC` | `0.35` | Semantic similarity weight ($w_s$) |
| `WEIGHT_ENTITY` | `0.25` | Entity match weight ($w_e$) |
| `WEIGHT_EVENT` | `0.15` | Event compatibility weight ($w_v$) |
| `WEIGHT_TEMPORAL` | `0.15` | Temporal proximity weight ($w_t$) |
| `WEIGHT_RELATIONSHIP` | `0.10` | Semantic relationship triple weight ($w_r$) |
| `RETRIEVAL_TOP_K` | `50` | Candidate pool size retrieved before reranking |
| `RERANK_TOP_K` | `10` | Final results displayed to user |

---

## Directory Structure

```
RecalLens/
├── dataset/                           # MemoraX Benchmark Dataset (1,500 docs, 3,750 queries)
│   ├── documents.csv
│   ├── queries.csv
│   ├── relevance.csv
│   ├── synthetic_users.csv
│   └── manifest.json
├── docker-compose.yml                 # Docker Compose multi-service definition
├── .env.example                       # Environment configuration template
├── README.md                          # Documentation
└── services/
    ├── context/                       # Context Extraction Engine
    │   ├── temporal_parser.py         # Date interval & relative date resolution
    │   ├── entity_extractor.py        # Device, merchant, person normalization
    │   ├── event_extractor.py         # Event classification (purchase, repair, etc.)
    │   ├── relationship_extractor.py  # Subject -> predicate -> object triples
    │   └── query_parser.py            # Natural language query context parser
    ├── evaluation/                    # Research Evaluation & Benchmark Harness
    │   ├── metrics.py                 # Precision@k, Recall@k, MRR, nDCG@k, MAP
    │   ├── dataset_loader.py          # Benchmark dataset loader & qrels
    │   ├── search_modes.py            # BM25, Vector, Hybrid, RecalLens & Ablations
    │   ├── runner.py                  # CLI benchmark execution runner
    │   └── results/                   # JSON evaluation benchmark outputs
    ├── retrieval/                     # Retrieval & Ranking Core
    │   ├── fusion.py                  # Reciprocal Rank Fusion (RRF k=60)
    │   └── context_scorer.py          # Multi-signal scoring engine S(d,q)
    ├── storage/                       # Storage Infrastructure
    │   ├── schema.sql                 # PostgreSQL DDL for entities, events, relationships
    │   └── postgres.py                # PostgreSQL client & candidate finder
    ├── tests/                         # Unit Test Suite (44 unit tests)
    │   ├── test_context_extraction.py
    │   ├── test_retrieval_and_reranking.py
    │   └── test_evaluation.py
    ├── web-ui/                        # Modern Double-Bezel Web Interface
    │   ├── index.html                 # Accessible search page with presets & context banner
    │   └── static/
    │       ├── app.js                 # SSE streaming client & evidence renderer
    │       └── style.css              # Plus Jakarta Sans design system
    ├── indexer.py                     # Document chunker & context ingestion
    ├── indexer_daemon.py              # Background polling daemon
    ├── main.py                        # FastAPI web application & endpoints
    ├── models.py                      # Pydantic data schemas
    ├── query_processor.py             # Query expansion & context injector
    ├── reranker.py                    # Cross-encoder & context reranker
    ├── retriever.py                   # Dual-channel candidate retriever
    └── requirements.txt               # Python package dependencies
```

---

## License & Acknowledgements

- **License:** MIT License — see [LICENSE](LICENSE) for details.
- **Foundational Architecture:** Extended from [Paperless Search](https://github.com/xehonk/paperless-search) by xehonk.
- **Technologies Used:**
  - [Qdrant](https://qdrant.tech/) — Vector Similarity Search Engine
  - [PostgreSQL](https://www.postgresql.org/) — Relational Context & Metadata Storage
  - [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) — Document Management System
  - [FastAPI](https://fastapi.tiangolo.com/) — Asynchronous Python Web Framework
  - [Ollama](https://ollama.ai/) & [Google Gemini](https://ai.google.dev/) — Language Model Providers

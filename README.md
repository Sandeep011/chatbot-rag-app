# Chatbot RAG App

Retrieval-Augmented Generation (RAG) backend built with FastAPI, PostgreSQL + pgvector, and SentenceTransformer embeddings.

This "Chat with your Docs" project lets you ask questions about your documents and get grounded answers with citations. It ingests files, splits them into manageable chunks, creates dense vector embeddings, and stores them in Postgres with the pgvector extension. At query time, it embeds the question, retrieves relevant passages via vector search, and composes an answer that quotes and links back to the sources.

This system is LLM agnostic and embedding model agnostic i.e., one can plug in OpenAI, local models, or other providers as needed, provided the embedding dimensions match the database schema. It is container-friendly, and designed for repeatable ingestion so that the knowledge base stays fresh as documents change. Chunking strategy (size/overlap), metadata, and indexing can be tuned for different document types, from long technical specs and handbooks to contracts, and research papers.

# Overview

This project implements the backend for a document-aware chatbot using semantic search over embedded PDF chunks. 

It provides REST APIs for:
1) Document ingestion (/ingest) – extract text, chunk pages, generate embeddings, and store them in Postgres/pgvector.
2) Semantic retrieval (/search) – query relevant chunks using cosine similarity with SentenceTransformer embeddings.

Built with modular components:
1) FastAPI – API interface
2) pgvector + psycopg2 – vector storage and search
3) SentenceTransformer – embedding generation
4) Loguru – structured logging

## Features

1) PDF/text ingestion with deterministic chunking
2) Embeddings stored in Postgres via pgvector
3) Dockerized local stack
4) kNN retrieval with metadata filtering
5) Simple API for asking questions (coming soon)

## Repo structure

```bash
backend/
├── app/              # FastAPI application (main entry point)
│   ├── app.py        # API endpoints: /ingest, /search, /health
│   └── __init__.py
├── db/               # Postgres + pgvector helpers
│   ├── db.py         # get_conn, upsert_document, delete_and_insert_chunks
│   └── __init__.py
├── chunker/          # PDF text processing utilities
│   ├── chunker.py    # clean_text, text_to_chunks, page_to_chunks
│   └── __init__.py
├── embeddings/       # Embedding model loading & inference
│   ├── embeddings.py # get_model, embed_passage, embed_query
│   └── __init__.py
├── retrieve/         # Semantic search engine
│   ├── retrieve.py   # run_search, run_diagnostics
│   └── __init__.py
└── ingest/           # ingestion script for PDFs
    └── ingest.py
```

## Architecture

The RAG model broadly has 3 steps:
1) Ingestion: parse -> chunk -> embed -> write to Postgres
2) Retrieve: query -> vector search (pgvector) -> top-k chunks
3) Generate: LLM answer conditioned on retrieved chunks (with citations)

PDF → Chunker → Embeddings → pgvector DB → Semantic Search → JSON Respons

## Tech stack

1) Python 3.10+
2) Postgres 15+ with pgvector 0.7+ (HNSW requires pgvector ≥ 0.7)
3) Any embedding model (e5-small-v2, text-embedding-3-small, bge, etc.)
4) FastAPI + Uvicorn (future version)

## Environment variables

Create backend/.env from backend/.env.example

Keep .env out of git; commit only .env.example.


# Quickstart

## 1) Start the database (docker)

```bash
# From repo root
docker compose -f docker/docker-compose.yml up -d

# Confirm Postgres is up
docker ps
```

![Docer service running](helper_images/docker.png)

## 2) Create db and enable pgvector

```bash
# Shell into the container (psql)
docker exec -it pgvector-db psql -U postgres
```

then run queries to create db and extension 

```sql
CREATE DATABASE ragdb;
\c ragdb
CREATE EXTENSION IF NOT EXISTS vector;
```

## 3) Install backend dependencies

```bash
# having created the file structure as mentioned above
cd backend
python -m venv .venv && source .venv/bin/activate   # create a virtual environment
pip install -r requirements.txt
cp .env.example .env
```

## 4) Initialize schema

Either a or b

a) run the queries in psql. run in psql connected to ragdb (example schema)

```sql
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  path TEXT,
  title TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INT,
  text TEXT,
  embedding vector(1536),   -- set to your model’s dimensions
  meta JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
```

b) create a helper, say init_schema() with above queries, in db.py, then

```bash
python -c "import db; db.init_schema()"
```

Documents table schema
![Documents table schema](helper_images/documents_schema.png)

Chunks table schema
![Chunks table schema](helper_images/chunks_schema.png)

## 5) Ingest sample file

```bash
cd backend
python -m backend.ingest.ingest --pdf_path ../sample/file_sample.pdf --doc-title "Sample file"
```

Ingest pdf
![Ingestion](helper_images/ingestion.png)

## 6) Retrieve chunks similar to query text

```bash
python -m backend.retrieve.retrieve --query "test" --k 5
```

Retrieved chunks
![Retrieval](helper_images/retrieval.png)

## 7) Running end-to-end via API

### Start the application

```bash
uvicorn backend.app:app
```

![Start the application](helper_images/start_the_application.png)

### Ingest

```bash
curl -s -X POST http://127.0.0.1:8000/ingest -F "file=@sample/<file_name>.pdf" -f "title=<title for the file>"
```

Ingest API
![Ingest API](helper_images/ingest_api.png)

document DB updated from ingestion
![document DB updated from ingestion](helper_images/document_db_updated_from_ingest_api.png)

chunks DB updated from ingestion
![chunks DB updated from ingestion](helper_images/chunks_db_updated_from_ingest_api.png)

### Search

```bash
 curl -s -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{"query":"<query>", "top_k":5, "min_score":0.0}'
```

Search API
![Search API](helper_images/search_api.png)

Result chunks from search
![Result chunks from search](helper_images/result_chunks_from_search_api.png)

## 8) Next steps

Add a response generation layer using an LLM API. This will allow asking questions (instead of querying words) and getting answers (instead of getting similar chunks).


# Usage tips

1) Keep chunk size 200-800 chars; overlap 10–20% is typical.
2) Match vector(N) to your embedding dimensions.
3) Add an HNSW index after loading some data: 
    CREATE INDEX ... USING HNSW ...;
    ANALYZE chunks;

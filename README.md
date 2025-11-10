# Chat with Docs — Retrieval-Augmented Generation (RAG) Chatbot

Ask questions about your PDF documents and get structured, fact-grounded answers generated through retrieval-augmented generation (RAG).

## 🧭 Project Overview

Chatbot-RAG-App is a Retrieval-Augmented Generation (RAG) system that lets users query PDF documents in natural language.
It retrieves relevant text from a vector database and generates concise, cited answers using Azure OpenAI GPT models.
The project demonstrates a complete pipeline—from document ingestion and embedding to intelligent, context-aware response generation.

---

## 🔄 User Flow

1. **Upload a PDF** via the `/ingest` API.  
   → The file is chunked, embedded using `intfloat/e5-small-v2`, and stored in PostgreSQL with `pgvector`.  
2. **Ask a question** through `/search` or `/answer`.  
   → Relevant chunks are retrieved based on cosine similarity.  
3. **Get answers** using Azure OpenAI (`GPT-4o-mini`) for natural-language, cited responses.  
   → If the LLM is unavailable, the system gracefully falls back to extractive summarization.

**Result:** Context-aware, reliable, and structured answers from your documents, locally or in the cloud.

---

## 🧱 Architecture

**Components**
- **FastAPI backend** (`backend/app/app.py`) exposing `/ingest`, `/search`, `/answer`, `/health`
- **Embeddings** with SentenceTransformers Hugging Face `intfloat/e5-small-v2`
- **Database**: PostgreSQL + `pgvector` (stores chunk embeddings)
- **LLM layer**: Azure OpenAI (Chat Completions in JSON mode) with safe fallback
- **PDF parsing**: `pypdf` (`PdfReader`)
- **Containers**: Docker + Docker Compose, deployable to Azure Container Apps

**Data Flow**
1. **Ingest** → `/ingest`: Parse PDF → chunk pages → embed chunks → upsert into `documents` + `chunks` tables.
2. **Retrieve** → `/search`: Embed query → **pgvector** cosine similarity matching → return ranked chunks.
3. **Answer** → `/answer`: 
   - Try **Azure OpenAI** first (structured JSON: `answer`, `answer_bullets`).
   - If LLM not configured/unavailable → **extractive fallback** builds bullets from top chunks.
4. **Health** → `/health`: Confirms model load & DB connectivity.

---

## 📁 Repository Structure

```bash
chatbot-rag-app/
├── backend/
│   ├── .dockerignore
│   ├── .env.example                 # Sample env vars
│   ├── Dockerfile                   # Backend image
│   ├── requirements.txt
│   ├── requirements.lock.txt
│   ├── app/
│   │   ├── app.py                   # FastAPI routes: /health, /ingest, /search, /answer
│   │   └── __init__.py
│   ├── chunker/
│   │   ├── chunker.py               # PDF chunking logic
│   │   └── __init__.py
│   ├── db/
│   │   ├── db.py                    # Postgres + pgvector connection/helpers
│   │   ├── test_db.py
│   │   └── __init__.py
│   ├── embeddings/
│   │   ├── embeddings.py            # e5-small-v2 embedding functions
│   │   └── __init__.py
│   ├── ingest/
│   │   ├── ingest.py                # /ingest handler
│   │   └── __init__.py
│   ├── retrieve/
│   │   ├── retrieve.py              # /search handler (vector similarity)
│   │   └── __init__.py
│   └── services/
│       ├── llm.py                   # Azure OpenAI (LLM answer mode)
│       └── __init__.py
│
├── docker/
│   ├── docker-compose.yml           # Local stack: backend + postgres (pgvector)
│   └── init/
│       ├── init.sql                 # Schema + pgvector enablement
│       └── once-suser.sql
│
├── helper_images/                   # Diagrams & screenshots for README
├── sample/                          # Example PDFs (for quick tests)
├── .gitignore
└── README.md
```

---

## 🛠️ Tech stack
-----------------------------------------------------------------------------------------------
|       Category       |                                Tools                                 |
|----------------------|----------------------------------------------------------------------|
|     **Backend**      |                         FastAPI(Python 3.11)                         |
|     **Database**     |                         PostgreSQL + pgvector                        |
|    **Embeddings**    |                   Hugging Face (intfloat/e5-small-v2)                |
|       **LLM**        |                        Azure OpenAI GPT-4o-mini                      |
| **Containerization** |                        Docker / Docker Compose                       |
|      **Cloud**       | Azure Container Apps + Azure Database for PostgreSQL Flexible Server |
-----------------------------------------------------------------------------------------------

---

## ⚡ Quickstart

Review the requirements.txt, .env, .yml, and docker files to ensure you have the necessary credentials, parameters, and tools to perform the 
steps that follow.

### 1) Clone this repository

```bash
git clone https://github.com/Sandeep011/chatbot-rag-app.git
cd chatbot-rag-app
```

Tip: If you’ve forked this repository for portfolio purposes, replace the URL with your forked repo.

### 2) Configure environment variables

Create a `.env` file in the repo root:

```bash
POSTGRES_USER=user
POSTGRES_PASSWORD=pass
POSTGRES_DB=ragdb
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}

# Azure OpenAI settings
AZURE_OPENAI_API_KEY=<your_azure_openai_key>
AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
LLM_MODEL=gpt-4o-mini

# Embedding model
EMBED_MODEL=intfloat/e5-small-v2
```

1) All configuration, database, embeddings, and LLM, are now handled via environment variables for clean portability across local, Docker, and Azure environments.
2) `.env` should be git-ignored to prevent sharing secrets.

### 3) Run Locally Using Docker Compose

Start both FastAPI backend and Postgres (pgvector) services:

```bash
docker compose up --build
```

Once containers start, you should see:

```nginx
Application startup complete.
Uvicorn running on http://0.0.0.0:8000
```

This confirms:
FastAPI app is live
Embedding model intfloat/e5-small-v2 successfully loaded
Database connected (with pgvector enabled)

### 4) Verify Health Check

Test your setup:

```bash
curl -fsS http://localhost:8000/health | jq
```

Expected output:

```json
{"status":"OK","model":"intfloat/e5-small-v2"}
```

![Health checkup](helper_images/verify_health.png)

### 5) Cloud Deployment (Azure)

Deploy to Azure using the prebuilt Docker image and your Azure environment setup:

```bash
# Build locally
docker build -t <app-name>:<tag> backend/

# Push to Docker Hub
docker tag <app-name>:<tag> <docker-username>/<app-name>:<cloud-tag>
docker push <docker-username>/<app-name>:<cloud-tag>

# Update Azure Container App
az containerapp update -g <resource-group> -n <app-name> \
  --image "docker.io/<docker-username>/<app-name>:<cloud-tag>"
```

Verify cloud health:

```bash
curl -fsS https://<APP_FQDN>/health
```

Note: <APP_FQDN> is your Azure Container App FQDN

Expected:

```json
{"status":"OK","model":"intfloat/e5-small-v2"}
```

This confirms your Azure Container App is live and connected to Azure PostgreSQL Flexible Server

![ACA dashboard](helper_images/azure_container_app_dashboard.png)

### 6) Full Flow

Once running, test your pipeline end-to-end:

```bash
# Ingest a PDF
curl -fS -X POST "https://<APP_FQDN>/ingest" \
  -F "file=@/path/to/file/system.pdf" -F "doc_id=test-1"

# Search across ingested documents
curl -fS -X POST "https://<APP_FQDN>/search" \
  -H "Content-Type: application/json" \
  -d '{"query":"Tell me about system design."}'
```

![Search gets top chunks](helper_images/search_api_azure.png)

```bash
# Get LLM-generated answer (with fallback)
curl -fS -X POST "https://<APP_FQDN>/answer" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is system design?"}'
```

![LLM generated answer](helper_images/answer_api_azure.png)

The ```/answer``` endpoint will:
1) Attempt LLM-based answer generation using Azure GPT-4o-mini
2) Automatically fallback to extractive summarization if the LLM is unreachable
3) Return structured JSON with answer, bullets, citations, and timing info

---

## 🌱 Planned Enhancements

1) Front-end chat interface using a lightweight React UI.
2) Multi-document ingestion.
3) A lightweight /delete endpoint for document cleanup
4) Command-line ingestion script ```python scripts/ingest.py <path>```
5) Async ingestion pipeline for large PDF uploads.
6) Cloud-native logging and metrics, e.g., query latency, retrieval depth, fallback counts.

**Strech goals**:  
7) Persistent conversation memory so users can ask follow-up questions.  
8) Custom LLM routing, allowing users to switch between local summarization and Azure GPT-4o-mini on demand.

---

## 💡 Usage Tips
1) Model caching: Pre-download intfloat/e5-small-v2 in your Dockerfile layer to avoid repeated startup delays (~300MB model)
2) Use unique doc_id values per file to prevent silent overwrites.
3) It confirms model load and DB connectivity before ingestion — saves debugging time.
4) Ask factual questions from your PDF to verify grounding and relevance scoring.
5) Read the “used_model” field:
```json
"used_model": {"embedding": "intfloat/e5-small-v2", "llm": "gpt-4o-mini"}
```
If "llm": null, the system is in fallback mode.  
6) Very large prompts can exceed token limits and trigger truncation on Azure; concise questions yield clearer results.  
7) Use ```az containerapp logs show``` to confirm live queries; the backend logs include chunk count and LLM call duration.  
8) If repo is public, automated Azure pulls are allowed; otherwise, use an ACR authentication step.

---

## ⚠️ Common Pitfalls
1) Health check passes locally, fails on Azure - Missing ```DATABASE_URL``` or ```AZURE_OPENAI_API_KEY``` in container environment.
2) Fallback triggers even when API key is valid - deployment name, endpoint mismatch or typo
3) Docker image rebuilds are slow - Model re-download each build
Keep ```RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"``` above ```COPY``` in Dockerfile.

---
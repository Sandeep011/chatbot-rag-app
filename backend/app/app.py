# v0

import os
from io import BytesIO
import hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, Form, File, HTTPException
# from fastapi.response import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

# External libs
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import psycopg2
# from pgvector.psycopg2 import register_vertor

# Local modules
from backend.db import get_conn, upsert_document, delete_and_insert_chunks
from backend.chunker import page_to_chunks
from backend.retrieve import run_search
from backend.embeddings import get_model, embed_passage, embed_query, embedding_dim

load_dotenv()


app = FastAPI(title="ChatBot RAG API", version="1.0")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(int(os.getenv("TOP_K", "8")), ge=1, le=200)
    min_score: float = Field(float(os.getenv("MIN_COSINE_SIM", "0.0")), ge=0.0, le=1.0)
    doc_id: Optional[str] = Field(None, description="Limit to a document UUID")
    preview_chars: int = Field(220, ge=0, le=5000)
    debug: bool = False

class SearchHit(BaseModel):
    document_id: str
    title: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    score: float = Field(..., ge=0.0, le=1.0)
    chunk_text: Optional[str] = None

class SearchResponse(BaseModel):
    hits: List[SearchHit]
    used_model: str
    timings_ms: Dict[str, Any] = Field(deafult_factory=dict)

class IngestResponse(BaseModel):
    document_id: str
    chunks_inserted: int
    title: Optional[str] = None

# -----------------Startup-----------------

@app.on_event("startup")
def startup():
    # warm the model and verify DB connectivity
    model_name = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")
    dim = embedding_dim()
    logger.info(f"Embedding model ready: {model_name} (dim={dim})")
    conn = None
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        logger.info("DB connectivity OK and pgvector registered.")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Startup DB check failed")
        raise
    finally:
        if conn:
            conn.close()


# -----------------Startup-----------------

@app.get("/health")
def health():
    return {"status": "OK", "model": os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")}

@app.post("/ingest", response_model=IngestResponse)
def ingest_pdf(file: UploadFile = File(..., description="PDF file"), 
    title: Optional[str] = Form(None)):

    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Parse PDF
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = list(enumerate(reader.pages, start=1))
    except Exception as e:
        logger.exception("Failed to parse PDF")
        raise HTTPException(status_code=400, detail=f"PDF parse error: {e}")
    
    file_title = title or (file.filename or "Untitled PDF")

    # Build chunk list (single-batch embed)
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for page_num, page in pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if not page_text.strip() or len(page_text) < 30:
            continue
        for ch in page_to_chunks(page_text):
            texts.append(ch)
            metas.append({
                "page": page_num,
                "title": file_title,
                "file_type": "pdf",
                "filename": file.filename,
            })
        
    if not texts:
        raise HTTPException(status_code=400, detail="No readable text found in the PDF.")
    
    # Embed
    try:
        vectors = embed_passage(texts)  # normalized E5 passages
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")
    
    # Write to DB
    conn = None
    try:
        checksum = hashlib.sha256(pdf_bytes).hexdigest()

        conn = get_conn()
        document_id = upsert_document(
            conn,
            title=file_title,
            source_path=f"upload:{file.filename or 'memory'}",
            file_checksum=checksum,
        )

        rows = []
        for idx, (txt, meta, emb) in enumerate(zip(texts, metas, vectors)):
            rows.append((document_id, meta["page"], idx, txt, emb, meta))

        delete_and_insert_chunks(conn, document_id, rows)
        conn.commit()
    except HTTPException:
        raise
    except Exception as e:
        if conn:
            conn.rollback()
            logger.exception("DB error during ingest")
            raise HTTPException(status_code=500, detail=f"DB error: {e}")
    finally:
        if conn:
            conn.close()

    return IngestResponse(document_id=str(document_id), chunks_inserted=len(rows), title=file_title)

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Embed query
    try:
        qvec = embed_query(req.query)
        logger.debug(f"qvec type={type(qvec)}, len={len(qvec) if hasattr(qvec,'__len__') else 'NA'} head={qvec[:4] if isinstance(qvec, list) else qvec}")
    except Exception as e:
        logger.exception("Query embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding error: {e}")

    # DB query
    conn = None
    try:
        conn = get_conn()
        rows = run_search(
            conn=conn,
            query_vec=qvec,
            document_id=req.doc_id,
            min_score=req.min_score,
            top_k=req.top_k,
        )
    except Exception as e:
        if conn:
            conn.rollback()
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=f"Search error: {e}")
    finally:
        if conn:
            conn.close()

    hits: List[SearchHit] = []
    for r in rows:
        text = (r.get("chunk_text") or "")
        if req.preview_chars and len(text) > req.preview_chars:
            text = text[:req.preview_chars] + "..."
        hits.append(
            SearchHit(
                document_id=str(r.get("document_id")),
                title=r.get("title"),
                page_number=r.get("page_number"),
                chunk_index=r.get("chunk_index"),
                score=float(r.get("score")),
                chunk_text=text,
            )
        )

    return SearchResponse(
        hits=hits,
        used_model=os.getenv("EMBED_MODEL", "intfloat/e5-small-v2"),
        timings_ms={},
    )

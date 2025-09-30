
import argparse
import os
from  typing import List, Dict, Any
from  loguru import logger
from dotenv import load_dotenv

import psycopg2
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Local imports
from backend.db import get_conn

load_dotenv()

def load_model() -> SentenceTransformer:
    model_name = os.getenv("EMBED_MODEL","intfloat/e5-small-v2")
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def embed_query(model, query):
    vector = model.encode([f"query: {query}"], normalize_embeddings=True)
    vector_list = vector[0].tolist()
    return vector_list


def run_search(conn, query_vec, document_id, min_score, top_k = 8):
    # title_contains
    params = []
    query = []

    query.append("""
    With q as (SELECT %s::vector AS v)
    SELECT 
        d.id AS document_id,
        d.title,
        c.page_number,
        c.chunk_index,
        c.chunk_text,
        (1 - (c.embedding <=> q.v)) AS score
    FROM chunks c
    JOIN documents d on d.id = c.document_id, q
    """)
    params.append(query_vec)

    logger.info("where conditions, if any")

    where = []
    if document_id:
        where.append(" d.id = %s")
        params.append(document_id)
    
    # if  title_contains:
    #     where.append("d.title ILIKE %s")
    #     params.append(f"%{title_contains}")
    
    if where:
        query.append("WHERE" + "AND ".join(where))
    
    logger.info("order and limit on query")
    query.append("ORDER BY c.embedding <=> q.v")
    query.append("LIMIT %s")
    params.append(top_k)

    full_query = " ".join(query)

    logger.info("connecting to db")
    with conn.cursor() as cur:
        cur.execute(full_query, params)
        logger.info("query execution  finished")
        cols, rows = [], []
        for c in cur.description:
            cols.append(c.name)
        for r in cur.fetchall():
            rows.append(dict(zip(cols, r)))
    if  min_score is not None:
        for r in rows:
            if r["score"] >= min_score:
                rows.append(r)
    return rows


def pretty_print(results: List[Dict[str, Any]], show_text_chars: int = 220):
    if not results:
        print("No results.")
        return
    
    for  i, r in enumerate(results, 1):
        if len(r["chunk_text"]) > show_text_chars:
            preview = (r["chunk_text"][:show_text_chars] + "...")
        else:
            preview = r["chunk_text"]
        print(f"[{i}] {r['title']}  |  page {r.get('page_number') or '-'}  |  score {r['score']:.4f}")
        print(f"    chunk#{r['chunk_index']}  |  doc_id={r['document_id']}")
        print(f"    {preview}\n")


def main():
    parser = argparse.ArgumentParser(description="Query pgvector for similar shunks")
    parser.add_argument("--query", required=True, help="Question (search  text)")
    parser.add_argument("--k", type=int, default=8, help="top-k results")
    parser.add_argument("--min_score", type=float, default=0.9, help="keep results above this similarity score")
    parser.add_argument("--doc_id", help="Limit to a specific  document UUID")
    # parser.add_argument("--title_contains", help="Limit to  documents with title ILIKE this substring")
    parser.add_argument("--chars", type=int, default=220, help="previeww characters to display")
    args = parser.parse_args()

    model = load_model()
    qvec = embed_query(model, args.query)

    conn = get_conn()
    register_vector(conn)
    # conn.autocommit = True

    try:
        results = run_search(
            conn=conn,
            query_vec=qvec,
            top_k=args.k,
            min_score=args.min_score,
            document_id=args.doc_id,
            # title_contains=args.title_contains,
        )
        pretty_print(results, show_text_chars=args.chars)
    except Exception as e:
        logger.exception(f"Exception: {e}")
        raise
    finally:
        conn.close()

if __name__=="__main__":
    main()

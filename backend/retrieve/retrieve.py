
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
    params = []
    query = []

    query.append("WITH q AS (SELECT %s::vector AS v)")
    params.append(query_vec)
    query.append(
    """
    SELECT * FROM (
        SELECT
        d.id AS document_id,
        d.title,
        c.page_number,
        c.chunk_index,
        c.chunk_text,
        (1 - (c.embedding <=> q.v)) AS score, 
            ROW_NUMBER() OVER (
                PARTITION BY d.id, c.chunk_index, c.page_number
                ORDER BY (1 - (c.embedding <=> q.v)) DESC
            ) AS rn
        FROM chunks c
        JOIN documents d ON d.id = c.document_id, q) ranked
    """
    )

    logger.info("where conditions, if any")
    where = []
    if document_id:
        where.append(" ranked.document_id = %s")
        params.append(document_id)
    
    if min_score and min_score > 0:
        where.append(" ranked.score >= %s")
        params.append(min_score)
    
    where.append(" ranked.rn = 1")
    if where:
        query.append("WHERE" + " AND ".join(where))
    
    logger.info("order and limit on query")
    query.append("ORDER BY ranked.score DESC")
    query.append(" LIMIT %s")
    params.append(top_k)

    full_query = " ".join(query)

    logger.info("connecting to db")
    
    rows = []
    with conn.cursor() as cur:
        cur.execute(full_query, params)
        logger.info("query execution finished")
        cols = [c.name for c in cur.description]

        for r in cur.fetchall():
            rows.append(dict(zip(cols, r)))

    return rows


def pretty_print(results: List[Dict[str, Any]], show_text_chars: int = 220, debug = False):
    if not results:
        print("No results.")
        return
    
    for  i, r in enumerate(results, 1):
        preview = (r.get("chunk_text") or "")
        if len(preview) > show_text_chars:
            preview = preview[:show_text_chars] + "..."
        line = (f"[{i}] {r.get('title')} | page {r.get('page_number')} "
                f"| idx {r.get('chunk_index')} | score {r.get('score'):.6f}")
        if debug and 'cos_dist' in  r:
            line += f" | cos_dist {r.get('cos_dist'):.6f}"
        print(line)
        print(preview + ("\n" if preview.endswith("\n") else "\n"))


def run_diagnostics(conn, qvec):
    with conn.cursor() as cur:
        logger.info("Checking vector dimension and variability")

        # 1) How many chunks do we have?
        cur.execute("SELECT COUNT(*) FROM chunks")
        logger.info(f"chunks.count = {cur.fetchone()[0]}")

        # 2) Are stored embeddings normalized? (dot with self ~= 1.0 if normalized)
        cur.execute("""
            SELECT
                AVG(- (embedding <#> embedding)) AS avg_self_dot,
                MIN(- (embedding <#> embedding)) AS min_self_dot,
                MAX(- (embedding <#> embedding)) AS max_self_dot
            FROM chunks
        """)
        avg_d, min_d, max_d = cur.fetchone()
        logger.info(f"self_dot avg/min/max = {avg_d:.6f}/{min_d:.6f}/{max_d:.6f}")

        # 3) Cosine distance spread to the query
        cur.execute("""
            WITH q AS (SELECT %s::vector AS v)
            SELECT
                MIN(c.embedding <=> q.v) AS min_cos,
                MAX(c.embedding <=> q.v) AS max_cos,
                AVG(c.embedding <=> q.v) AS avg_cos,
                STDDEV_POP(c.embedding <=> q.v) AS std_cos
            FROM chunks c, q
        """, (qvec,))
        min_cos, max_cos, avg_cos, std_cos = cur.fetchone()
        logger.info(f"cos_dist to query min/max/avg/std = "
                    f"{min_cos:.9f}/{max_cos:.9f}/{avg_cos:.9f}/{(std_cos or 0):.9f}")

        # 4) Sample a few cosine distances (rounded) to visually confirm variation
        cur.execute("""
            WITH q AS (SELECT %s::vector AS v)
            SELECT ROUND((c.embedding <=> q.v)::numeric, 6) AS cos_dist
            FROM chunks c, q
            ORDER BY cos_dist
            LIMIT 5
        """, (qvec,))
        samples = [row[0] for row in cur.fetchall()]
        logger.info(f"sample cos_dist (rounded): {samples}")

        # 5) Check for obviously broken vectors: near-zero norms
        cur.execute("""
            SELECT COUNT(*)
            FROM chunks
            WHERE (- (embedding <#> embedding)) < 1e-6
        """)
        zeros = cur.fetchone()[0]
        logger.info(f"near-zero-norm vectors = {zeros}")

        # 6) (Optional) peek at a text rendering of one vector to sanity check content
        cur.execute("SELECT left(embedding::text, 100) FROM chunks LIMIT 1")
        sample_text = cur.fetchone()
        if sample_text:
            logger.info(f"embedding text preview (first 100 chars): {sample_text[0]}")



def main():
    parser = argparse.ArgumentParser(description="Query pgvector for similar shunks")
    parser.add_argument("--query", required=True, help="Question (search text)")
    parser.add_argument("--k", type=int, default=os.getenv("TOP_K"), help="top-k results")
    parser.add_argument("--min_score", type=float, default=os.getenv("MIN_COSINE_SIM"), help="keep results above this similarity score")
    parser.add_argument("--doc_id", help="Limit to a specific  document UUID")
    parser.add_argument("--chars", type=int, default=220, help="previeww characters to display")
    parser.add_argument("--debug", action="store_true", help="print raw cosine distancce as well")
    args = parser.parse_args()

    model = load_model()
    qvec = embed_query(model, args.query)

    conn = get_conn()

    if args.debug:
        run_diagnostics(conn, qvec)

    try:
        results = run_search(
            conn=conn,
            query_vec=qvec,
            top_k=args.k,
            min_score=args.min_score,
            document_id=args.doc_id,
        )
        pretty_print(results, show_text_chars=args.chars, debug=args.debug)
    except Exception as e:
        logger.exception(f"Exception: {e}")
        raise
    finally:
        conn.close()

if __name__=="__main__":
    main()

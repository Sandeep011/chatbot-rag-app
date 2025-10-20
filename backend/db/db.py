# v0

import os
import psycopg2
from psycopg2.extras import Json
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    """
    form a connection between python and postgres.
    One connection per execution.
    """
    url = os.getenv("DATABASE_URL")
    if url:
        conn = psycopg2.connect(url)
    else:
        conn = psycopg2.connect(
            host=os.getenv("PG_HOST", "postgres"),
            port=int(os.getenv("PG_PORT", "5432")),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD"),
            dbname=os.getenv("PG_DATABASE"),
        )

    register_vector(conn) # allows us to pass python lists which will be interpreted as vector columns
    return conn


def upsert_document(conn, title: str, source_path: str, file_checksum: str) -> int:
    """
    when inserting a new document, if the document does not exist, create a new row for the document.
    If it does, update the title and source path provided in the new request.

    args:
        title: document title
        source_path: location of the document file
        file_checksum: a fingerprint of a file. a char update to a document changes the checksum.

    return:
        id: unique document_id (UUID) created when row is created.
    """
    query = """
    INSERT INTO documents (title, source_path, file_checksum)
    VALUES (%s, %s, %s)
    ON CONFLICT(file_checksum)
    DO UPDATE 
        SET title = EXCLUDED.title,
            source_path = EXCLUDED.source_path
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(query, (title, source_path, file_checksum))
        uuid = cur.fetchone()[0]
    return uuid


def delete_and_insert_chunks(conn, document_id: str, rows: tuple) -> int:
    """
    For a given document (document_id), if its chunks are present, delete them.
    Insert the new chunks.

    args:
        document_id: uuid of document with the new chucks
        rows: tuple with value (document_id, page_number, chunk_index, chunk_text, embedding, metadata)
            for each chunk
    
    return:
        num_rows: number of rows inserted
    """

    with conn.cursor() as cur:
        # delete rows of chunks for given document_id if they exists.
        delete_query = """
        DELETE FROM chunks where document_id = %s;
        """
        cur.execute(delete_query, (document_id,))

        # insert the rows provided in the current request
        all_rows = []
        for doc_id, page_num, chunk_idx, chunk_txt, embedding, metadata in rows:
            all_rows.append((doc_id, page_num, chunk_idx, chunk_txt, embedding, Json(metadata)))
        
        insert_query = """
        INSERT INTO chunks (document_id, page_number, chunk_index, chunk_text, embedding, metadata)
        VALUES (%s,%s,%s,%s,%s,%s);
        """
        cur.executemany(insert_query, all_rows)
    
    return len(rows)

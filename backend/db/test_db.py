# -------------------------------------------------------------------
"""
THE TEST METHOD IS COPIED FORM CHATGPT. NEED TO UNDERSTAND IT, AND REMOVE
UNNECESSARY BITS, AND ADD BITS TESTS IF NECESSARY.

THIS FILE IS SOLELY TO TEST THE db.py FILE
"""
# -------------------------------------------------------------------



# test_db.py (run with `python test_db.py`)
from db import get_conn, upsert_document, replace_chunks

def test():
    conn = get_conn()
    try:
        conn.autocommit = False

        # 1. Insert a document
        doc_id = upsert_document(conn, "Test Doc", "/tmp/test.pdf", "checksum123")
        print("Doc ID:", doc_id)

        # 2. Insert some chunks
        rows = [
            (doc_id, 1, 0, "hello world", [0.1] * 768, {"source": "test"}),
            (doc_id, 1, 1, "goodbye world", [0.2] * 768, {"source": "test"}),
        ]
        count = replace_chunks(conn, doc_id, rows)
        print("Inserted chunks:", count)

        # 3. Query back
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s;", (doc_id,))
            print("Chunks in DB:", cur.fetchone()[0])

        conn.commit()
    except Exception as e:
        conn.rollback()
        print("Test failed:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    test()

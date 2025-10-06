
import argparse
import hashlib
from pathlib import Path

from tqdm import tqdm
from loguru import logger
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Local imports
from backend.db import get_conn, upsert_document, delete_and_insert_chunks
from backend.chunker import page_to_chunks
from  backend.embeddings import get_model, embed_passage


logger.add("logs/ingest.log", rotation="1 MB", enqueue=True)

def generate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    file_bytes = path.read_bytes()
    h.update(file_bytes)
    return  h.hexdigest()


def read_pdf_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    for i, page in  enumerate(pages, start = 1):
        try:
            page_text = page.extract_text() or ""
        except:
            page_text = ""
        yield i, page_text


def main():
    parser = argparse.ArgumentParser(description="Ingest a single PDF into postgres+pgvector")
    parser.add_argument("--pdf_path", required=True, help="Path to the PDF file")
    parser.add_argument("--title", help="pdf file name (optional)")
    parser.add_argument("--model", default="intfloat/e5-small-v2", help="Sentence transformer model")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    
    title = args.title
    checksum = generate_sha256(pdf_path)

    logger.info("Loading embedding model")
    model = SentenceTransformer(args.model)

    logger.info("Connecting to Postgres")
    conn = get_conn()

    try:
        document_id = upsert_document(conn, title=title, source_path=str(pdf_path), file_checksum=checksum)

        rows = []
        chunk_index = 0

        pages = list(read_pdf_pages(pdf_path))
        if not pages:
            logger.warning("No pages found in PDF")
            raise
        for page_num, page_text in tqdm(pages, desc="Pages"):
            if not page_text.strip() or len(page_text) < 30:
                continue
            
            chunks = page_to_chunks(page_text)

            if not chunks:
                continue

            to_embed = []
            for c in chunks:
                to_embed.append(f"passage: {c}")
            embeddings = embed_passage(to_embed)

            chunk_emb = zip(chunks, embeddings)
            for chunk_text, emb in  chunk_emb:
                rows.append(
                    (
                        document_id,
                        page_num,
                        chunk_index,
                        chunk_text,
                        emb.tolist(),
                        {"file_type": "pdf", "file": str(pdf_path), "title": title, "page": page_num}
                    )
                )
                chunk_index += 1
        
        logger.info(f"Deleting old chunks, if any, and inserting new chunks for document {document_id} with {len(rows)} rows")
        delete_and_insert_chunks(conn, document_id, rows)
        conn.commit()
        logger.success(f"Ingestion complete: {len(rows)} chunks stored for document {document_id}")
    except Exception as e:
        conn.rollback()
        logger.exception(f"Ingestion  failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
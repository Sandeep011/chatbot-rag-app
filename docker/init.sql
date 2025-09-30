-- extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- tables
CREATE TABLE IF NOT EXISTS documents (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  title           text NOT NULL,
  source_path     text,
  file_checksum   text UNIQUE,
  created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id            bigserial PRIMARY KEY,
  document_id   uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  page_number   integer,
  chunk_index   integer NOT NULL,
  chunk_text    text NOT NULL,
  metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding     vector(384) NOT NULL,
  UNIQUE (document_id, chunk_index)
);

-- indexes
CREATE INDEX IF NOT EXISTS idx_chunks_doc_idx
  ON chunks (document_id, chunk_index);

-- HNSW cosine for 384-dim vectors
CREATE INDEX IF NOT EXISTS chunks_embedding_cos_idx
  ON chunks USING hnsw (embedding vector_cosine_ops);

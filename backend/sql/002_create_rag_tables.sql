CREATE TABLE IF NOT EXISTS rag_documents (
    id BIGSERIAL PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    file_name TEXT NOT NULL,
    category TEXT NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT ck_rag_documents_status
        CHECK (status IN ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED'))
);

CREATE INDEX IF NOT EXISTS ix_rag_documents_checksum
    ON rag_documents (checksum);

CREATE INDEX IF NOT EXISTS ix_rag_documents_category
    ON rag_documents (category);

CREATE TABLE IF NOT EXISTS rag_document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL
        REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    embedding vector(768) NOT NULL,
    CONSTRAINT uq_rag_document_chunks_position
        UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_rag_document_chunks_document_id
    ON rag_document_chunks (document_id);

CREATE INDEX IF NOT EXISTS ix_rag_document_chunks_embedding_hnsw
    ON rag_document_chunks
    USING hnsw (embedding vector_cosine_ops);

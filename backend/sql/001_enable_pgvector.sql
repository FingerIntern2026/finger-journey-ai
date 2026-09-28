-- Enable pgvector for the current PostgreSQL database.
-- The pgvector server extension must be installed before running this script.
CREATE EXTENSION IF NOT EXISTS vector;

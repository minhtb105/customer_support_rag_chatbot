-- Initialize PostgreSQL database with pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_memory_facts_user_type ON memory_facts(user_id, fact_type);
CREATE INDEX IF NOT EXISTS idx_memory_facts_session ON memory_facts(session_id);
CREATE INDEX IF NOT EXISTS idx_memory_facts_created ON memory_facts(created_at);
CREATE INDEX IF NOT EXISTS idx_session_summaries_user ON session_summaries(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_stats_user ON memory_stats(user_id);

-- Create function for vector cosine similarity (if not using pgvector's built-in)
-- Note: pgvector provides built-in operators, but this is for reference
-- CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector) RETURNS float AS $$
-- BEGIN
--     RETURN (a <=> b);
-- END;
-- $$ LANGUAGE plpgsql IMMUTABLE;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
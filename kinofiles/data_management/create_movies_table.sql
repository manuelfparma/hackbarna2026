-- SQL Script to create the 'movies' table for the RAG dataset in Supabase

CREATE TABLE IF NOT EXISTS movies (
    id BIGINT PRIMARY KEY,
    name TEXT NOT NULL,
    date INTEGER,
    tagline TEXT,
    description TEXT,
    minute INTEGER,
    duration_category TEXT,
    rating DOUBLE PRECISION,
    genres TEXT[] DEFAULT '{}',
    themes TEXT[] DEFAULT '{}',
    studios TEXT[] DEFAULT '{}',
    languages TEXT[] DEFAULT '{}',
    actors TEXT[] DEFAULT '{}',
    directors TEXT[] DEFAULT '{}'
);

-- Create GIN indexes on array columns for fast filtering using @> or ANY
CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies USING gin (genres);
CREATE INDEX IF NOT EXISTS idx_movies_themes ON movies USING gin (themes);
CREATE INDEX IF NOT EXISTS idx_movies_studios ON movies USING gin (studios);
CREATE INDEX IF NOT EXISTS idx_movies_languages ON movies USING gin (languages);
CREATE INDEX IF NOT EXISTS idx_movies_actors ON movies USING gin (actors);
CREATE INDEX IF NOT EXISTS idx_movies_directors ON movies USING gin (directors);

-- Example queries using these indexes:
-- Fast lookup using array containment operator (@>)
-- SELECT * FROM movies WHERE genres @> ARRAY['Comedy'];
-- Fast lookup using ANY
-- SELECT * FROM movies WHERE 'Margot Robbie' = ANY(actors);




-- Install vector extension

create extension if not exists vector with schema extensions;

-- Create themes_embeddings table

create table if not exists public.themes_embeddings (
    id bigint generated always as identity primary key,
    theme text not null unique,
    embedding extensions.vector(1024) not null,
    created_at timestamptz not null default now()
);

-- Create HNSW index on embedding column

create index if not exists themes_embeddings_embedding_hnsw
    on public.themes_embeddings
    using hnsw (embedding vector_cosine_ops);



-- Create match_themes function

create or replace function match_themes(
  query_embedding extensions.vector(1024),
  match_count int default 10
)
returns table(theme text, similarity float)
language sql stable
as $$
  select
    te.theme,
    1 - (te.embedding <=> query_embedding) as similarity
  from public.themes_embeddings as te
  order by te.embedding <=> query_embedding
  limit match_count;
$$;

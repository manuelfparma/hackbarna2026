-- SQL script to create movies and embedding tables for the RAG dataset in Supabase

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
    directors TEXT[] DEFAULT '{}',
    poster TEXT
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



-- Create descriptions_embeddings table

create table if not exists public.descriptions_embeddings (
    movie_id bigint primary key references public.movies(id),
    description text not null,
    embedding extensions.vector(1024) not null,
    created_at timestamptz not null default now()
);

-- Create HNSW index on embedding column

create index if not exists descriptions_embeddings_embedding_hnsw
    on public.descriptions_embeddings
    using hnsw (embedding vector_cosine_ops);



-- Create match_descriptions function

create or replace function match_descriptions(
  query_embedding extensions.vector(1024),
  match_count int default 10
)
returns table(movie_id bigint, name text, description text, similarity float)
language sql stable
as $$
  select
    de.movie_id,
    m.name,
    de.description,
    1 - (de.embedding <=> query_embedding) as similarity
  from public.descriptions_embeddings as de
  join public.movies as m on m.id = de.movie_id
  order by de.embedding <=> query_embedding
  limit match_count;
$$;

-- #######################
-- FUZZY SEARCH FOR MOVIES
-- #######################

-- 1. Enable extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS citext;

-- 2. Convert the existing 'name' column to case-insensitive text (preserves original casing)
ALTER TABLE movies ALTER COLUMN name TYPE citext;

-- 2.5. Create an IMMUTABLE wrapper for array_to_string
-- PostgreSQL requires GENERATED ALWAYS expressions to be strictly IMMUTABLE.
-- array_to_string is technically STABLE, so we wrap it for TEXT arrays.
CREATE OR REPLACE FUNCTION array_to_string_immutable(arr TEXT[], sep TEXT)
RETURNS TEXT
IMMUTABLE PARALLEL SAFE
LANGUAGE sql
AS $$
    SELECT array_to_string(arr, sep);
$$;

-- 3. Add generated text columns for array fields (excluding 'themes')
ALTER TABLE movies 
  ADD COLUMN genres_text TEXT GENERATED ALWAYS AS (array_to_string_immutable(genres, ' ')) STORED,
  ADD COLUMN studios_text TEXT GENERATED ALWAYS AS (array_to_string_immutable(studios, ' ')) STORED,
  ADD COLUMN languages_text TEXT GENERATED ALWAYS AS (array_to_string_immutable(languages, ' ')) STORED,
  ADD COLUMN actors_text TEXT GENERATED ALWAYS AS (array_to_string_immutable(actors, ' ')) STORED,
  ADD COLUMN directors_text TEXT GENERATED ALWAYS AS (array_to_string_immutable(directors, ' ')) STORED;

-- 4. Create Trigram GIN indexes for fast fuzzy search
CREATE INDEX IF NOT EXISTS idx_movies_name_trgm ON movies USING GIN (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_movies_genres_trgm ON movies USING GIN (genres_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_movies_studios_trgm ON movies USING GIN (studios_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_movies_languages_trgm ON movies USING GIN (languages_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_movies_actors_trgm ON movies USING GIN (actors_text gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_movies_directors_trgm ON movies USING GIN (directors_text gin_trgm_ops);

-- #######################
-- MOCK STREAMING AVAILABILITY
-- #######################

-- Array of service slugs (netflix, hbo, prime, appletv); populated by
-- data_management/migrate_streaming.py with deterministic mock data.
ALTER TABLE movies ADD COLUMN IF NOT EXISTS streaming TEXT[] DEFAULT '{}';
ALTER TABLE movies
  ADD COLUMN IF NOT EXISTS streaming_text TEXT
    GENERATED ALWAYS AS (array_to_string_immutable(streaming, ' ')) STORED;
CREATE INDEX IF NOT EXISTS idx_movies_streaming ON movies USING gin (streaming);

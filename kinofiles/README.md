# Kino Files: Project Documentation

This directory contains the entire source code for the Kino Files application. Kino Files is a full-stack AI application designed to provide an intelligent, hallucination-free movie recommendation experience for both individuals and groups.

---

## 1. Setup & Data Management

This section details the scripts required to fetch the dataset, process it into a rich, RAG-friendly format, and populate your Supabase database.

### Prerequisites
1. **Kaggle Credentials:** Create an API token from your Kaggle settings and place `kaggle.json` in `~/.kaggle/` (or set `KAGGLE_USERNAME` / `KAGGLE_KEY`).
2. **Supabase Credentials:** Ensure you have a `.env` file containing `SUPABASE_URL` and `SUPABASE_KEY`.

### Running the Pipeline
Run the following commands using `uv` from the `kinofiles` directory:

**1. Create the Database Schema:**
Execute the SQL in `data_management/create_schema.sql` in your Supabase SQL Editor. This sets up the movies table, GIN indexes, vector embedding tables, and RPC functions.

**2. Clean and Build the Dataset:**
```bash
uv run python data_management/data_cleaning.py
```
*Downloads the top 1000 popular films since 2000, aggregates metadata, and exports to a flat JSONL file.*

**3. Upload to Supabase:**
```bash
uv run python data_management/upload_supabase.py
```
*Uploads the records to your Supabase `movies` table using upsert logic.*

---

## 2. System Overview & Capabilities

Kino Files operates on a core conversational loop: **discover a movie, refine suggestions, ask questions, and make a choice.**

### Operating Modes
*   **Group Mediator (`agent.orchestrator`):** Asks multiple users for their preferences, merges them into a single group brief, and conducts a numbered voting round to settle on a movie.
*   **Standalone Agent (`agent.recomedation_agent`):** A single-user conversational agent for free-form discovery, refining, and factual lookups.

### Core Capabilities
*   **Semantic Mood Search:** ("Something melancholy about family") Uses vector search.
*   **Plot Identification:** ("An animated movie about a rat in Paris") Uses description matching to guess the movie.
*   **Similar-To Search:** ("Like The Dark Knight") Finds plot neighbors of a seed film.
*   **Direct Filters:** ("A 90s French comedy") Strict SQL filtering on genres, languages, and decades.
*   **Factual Lookups:** ("Who directed Inception?") Answers directly from the catalog row without losing your active movie shortlist.
*   *(Note: Watchlists, two-film comparisons, and explicit content exclusions are not supported).*

---

## 3. Algorithm & Retrieval Decisions

Kino Files uses a strict Retrieval-Augmented Generation (RAG) pipeline. We separate retrieval from generation to eliminate LLM hallucinations.

### Dual Vector Embeddings (Mistral 1024D + pgvector)
*   **`themes_embeddings`:** We embed unique theme strings (e.g., "Epic heroes"). Used for general mood queries, keeping matching fast and semantically accurate.
*   **`descriptions_embeddings`:** We embed the specific plot description of every movie. Used for "similar to X" searches or when the user provides specific plot terms.

### Hybrid Ranking Logic
For theme-based searches, we use a custom ranker:
1.  **Match:** Find top themes via `pgvector` cosine similarity.
2.  **Filter:** Keep only themes within a `0.10` similarity margin of the very best match (dropping unrelated tail results).
3.  **Weight:** Apply reciprocal rank weighting (`1, 1/2, 1/3...`) so hitting the closest theme heavily outweighs hitting several weaker ones (preventing heavily-tagged blockbusters from always winning).
4.  **Tie-Breaker:** Sort matched films by rating, then alphabetically.

### The Narrator (Nebius Qwen)
Capabilities strictly return raw structured data (titles, themes). The `ReplyComposer` (Nebius LLM at `temp=0.4`) takes this data and synthesizes a natural response of up to three sentences. **Rules:** Never invent films, never read database IDs/scores aloud, and never list titles in the text (the UI renders them separately).

---

## 4. Agent Workflows (LangGraph)

### Group Orchestrator Flow
`welcome -> collect_preferences -> mediate -> group_vote -> refine (retry)`
The orchestrator extracts participant names, classifies one request per person, and merges the criteria. It retrieves options and initiates a numbered vote. Ties or zero-votes lead to a refinement round (capped at three rounds).

### Standalone Flow
`classify -> capability -> reply -> turn (interrupt) -> select / exit`
The agent persists state across the conversation. Hard constraints (genres) act as strict SQL `AND` filters. Soft constraints (moods, themes) are fed into the vector query. The classifier handles `add`, `replace`, `reset`, or `keep` logic based on user input (e.g., "also" vs "instead"). 

### Running the Agents
```bash
# Run Group Mediation
uv run python -m agent.orchestrator

# Run Solo Chat
uv run python -m agent.recomedation_agent
```

---

## 5. Frontend & Voice Integration

The user-facing layer of Kino Files is located in the `frontend/` directory and is built to deliver a seamless, living-room-ready experience.

*   **Full-Stack with Reflex:** We used [Reflex](https://reflex.dev/) to build the entire web application. This allowed us to write the frontend entirely in Python, deeply integrating it with our LangGraph backend while providing a highly interactive, reactive UI. The interface separates narrative chat from the active movie shortlist, allowing users to easily make selections via a numbered UI (ideal for TV remotes).
*   **Voice Execution via SLNG:** Typing on a TV is a poor experience, so Kino Files is voice-enabled. We integrated the **SLNG** platform as our audio execution layer. SLNG bridges the Reflex frontend and the agent backend by routing user audio input to Deepgram Nova-3 (Speech-to-Text) and synthesizing the agent's textual responses back into natural voice using Deepgram Aura-2 (Text-to-Speech).

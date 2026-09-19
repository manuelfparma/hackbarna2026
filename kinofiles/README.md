# Kinofiles

Kinofiles is a full-stack AI application designed to provide an intelligent movie recommendation experience. 

The project is composed of three main pillars:
1. **Data Management Pipeline:** Scripts to clean, process, and upload the Letterboxd Kaggle dataset into a Supabase database for Retrieval-Augmented Generation (RAG).
2. **LLM Agent:** The core conversational intelligence layer powered by LangGraph.
3. **Frontend:** An interactive user interface built with Reflex to seamlessly interact with the agent.

---

## 1. Data Management Pipeline

This section details the scripts required to fetch the dataset, process it into a rich, RAG-friendly JSONL format, and populate your Supabase database.

### Prerequisites

**Kaggle Credentials (For Downloading Data)**
You will need a Kaggle account and an API token to download the source dataset.
1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Under the **API** section, click **Create New Token**.
3. Place the downloaded `kaggle.json` file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<User>\.kaggle\` (Windows).
*(Alternatively, you can set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.)*

**Supabase Credentials (For Uploading Data)**
1. Ensure you have a `.env` file in the `kinofiles` directory containing:
   ```env
   SUPABASE_URL="your-supabase-url"
   SUPABASE_KEY="your-api-key"
   ```

### Setting Up the Database
Before uploading the data, you must initialize the database table.
1. Open the **SQL Editor** in your Supabase dashboard.
2. Copy and execute the exact SQL query found in `data_management/create_movies_table.sql`.
   *This script sets up the table schema and configures powerful GIN indexes so array columns (like genres, themes, and actors) can be queried at lightning speed.*

### Running the Pipeline
We use `uv` to manage dependencies. Run the following commands from this directory (`kinofiles`):

**Step 1: Clean and Build the Dataset**
```bash
uv run python data_management/data_cleaning.py
```
* **Downloads exactly what you need:** Bypasses 20GB+ of posters to fetch just the metadata.
* **Pre-filters for Popularity:** Restricts the dataset to the top 1000 most popular films released since 2000.
* **Aggregates and Cleans:** Consolidates 6 different CSVs into a single flat file, categorizes movie duration, filters for Primary Languages, and isolates the top 5 main actors.
* **Exports:** Outputs the final processed records to `data/top_1000_movies.jsonl`.

**Step 2: Upload to Supabase**
```bash
uv run python data_management/upload_supabase.py
```
* Authenticates with Supabase using your `.env` secrets.
* Reads the generated `.jsonl` file and uploads the records to your `movies` table in chunks of 250.
* Uses `upsert` logic, meaning you can safely stop and rerun the script at any time without triggering duplicate errors.

---

## 2. LLM Agent
*(Agent configuration and execution instructions go here)*

---

## 3. Frontend
*(Frontend configuration and execution instructions go here)*

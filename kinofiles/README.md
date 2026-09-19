# Kinofiles Dataset Builder

This project contains the data cleaning script required to fetch the Letterboxd dataset from Kaggle and process it into a rich, flat, RAG-friendly JSONL format.

## Prerequisites

You will need a Kaggle account and an API token to download the source dataset.
1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Under the **API** section, click **Create New Token**.
3. This will download a `kaggle.json` file. Place this file in `~/.kaggle/` (Linux/Mac) or `C:\Users\<User>\.kaggle\` (Windows).

*(Alternatively, you can set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.)*

## Generating the Data

We use `uv` to manage dependencies. You can generate the final dataset in exactly one line.

Run the following command from this directory (`kinofiles`):

```bash
uv run python data_cleaning.py
```

### What this script does:
1. **Downloads exactly what you need**: Avoids downloading the 20GB+ of posters and only downloads the 7 core metadata CSVs.
2. **Pre-filters for Popularity**: Filters the top 1000 most popular movies released since the year 2000.
3. **Aggregates and Cleans**: Groups metadata from 6 different CSVs into a single flat file, extracting only the Primary Language and the top 5 main actors.
4. **Exports**: Outputs the final processed data to `data/top_1000_movies.jsonl`.

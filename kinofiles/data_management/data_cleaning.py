import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data(output_dir="data"):
    """
    Downloads only the necessary CSV files from the Kaggle dataset.
    Requires Kaggle credentials (e.g., ~/.kaggle/kaggle.json).
    """
    try:
        api = KaggleApi()
        api.authenticate()
    except OSError as e:
        print("Error authenticating with Kaggle API.")
        print("Please ensure you have your Kaggle credentials set up:")
        print("1. Create an API token from your Kaggle account settings.")
        print("2. Place the kaggle.json file in ~/.kaggle/ (Linux/Mac) or C:\\Users\\<User>\\.kaggle\\ (Windows).")
        print("Or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        raise e

    dataset = "gsimonx37/letterboxd"
    files_to_download = [
        "movies.csv",
        "genres.csv",
        "themes.csv",
        "crew.csv",
        "actors.csv",
        "studios.csv",
        "languages.csv",
        "posters.csv"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file in files_to_download:
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            print(f"{file} already exists, skipping download.")
            continue
            
        print(f"Downloading {file}...")
        api.dataset_download_file(dataset, file_name=file, path=output_dir)
        
        # Kaggle might download single files as a .zip, extract if needed
        zip_path = os.path.join(output_dir, f"{file}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(zip_path)

def build_movie_dataset(data_dir="data"):
    """
    Reads the downloaded CSVs, pre-filters the top 1000 popular movies since 2000, 
    joins them, and returns a rich movie metadata DataFrame.
    """
    print("Loading and pre-filtering movies...")
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    
    # Convert date to numeric to allow filtering
    movies['date'] = pd.to_numeric(movies['date'], errors='coerce')
    movies = movies.dropna(subset=['date'])
    
    # Filter for movies from 2000 onwards
    movies = movies[movies['date'] >= 2000]
    
    # The movies.csv file is implicitly sorted by popularity on Letterboxd!
    # Therefore, taking the first 1000 rows gives us the most popular movies.
    movies = movies.head(1000)
    top_ids = movies['id'].unique()
    
    print("Loading and filtering supplementary datasets...")
    # Load and immediately filter the other datasets to save memory and processing time
    genres = pd.read_csv(os.path.join(data_dir, "genres.csv"))
    genres = genres[genres['id'].isin(top_ids)]
    
    themes = pd.read_csv(os.path.join(data_dir, "themes.csv"))
    themes = themes[themes['id'].isin(top_ids)]
    
    actors = pd.read_csv(os.path.join(data_dir, "actors.csv"))
    actors = actors[actors['id'].isin(top_ids)]
    
    crew = pd.read_csv(os.path.join(data_dir, "crew.csv"))
    crew = crew[crew['id'].isin(top_ids)]
    
    studios = pd.read_csv(os.path.join(data_dir, "studios.csv"))
    studios = studios[studios['id'].isin(top_ids)]
    
    languages = pd.read_csv(os.path.join(data_dir, "languages.csv"))
    languages = languages[languages['id'].isin(top_ids)]
    # Filter for only the primary language (can be labeled as 'Language' or 'Primary language')
    languages = languages[languages['type'].isin(['Language', 'Primary language'])]
    
    posters = pd.read_csv(os.path.join(data_dir, "posters.csv"))
    posters = posters[posters['id'].isin(top_ids)]
    posters = posters.rename(columns={'link': 'poster'})
    
    print("Aggregating metadata...")
    # Get directors from crew
    directors = crew[crew['role'] == 'Director']
    
    # Aggregate to list format for each movie id
    genres_agg = genres.groupby('id')['genre'].apply(list).reset_index(name='genres')
    themes_agg = themes.groupby('id')['theme'].apply(list).reset_index(name='themes')
    studios_agg = studios.groupby('id')['studio'].apply(list).reset_index(name='studios')
    langs_agg = languages.groupby('id')['language'].apply(list).reset_index(name='languages')
    
    # Take MAIN 5 ACTORS ONLY
    top_actors = actors.groupby('id').head(5).groupby('id')['name'].apply(list).reset_index(name='actors')
    
    directors_agg = directors.groupby('id')['name'].apply(list).reset_index(name='directors')
    
    print("Merging data...")
    merged = movies
    for df in [genres_agg, themes_agg, studios_agg, langs_agg, top_actors, directors_agg, posters]:
        merged = merged.merge(df, on='id', how='left')
        
    # Convert floats to integers for cleaner JSON output
    merged['date'] = merged['date'].astype(int)
    merged['minute'] = pd.to_numeric(merged['minute'], errors='coerce').fillna(0).astype(int)
    
    # Categorize duration
    def categorize_duration(mins):
        if mins < 100:
            return "Short"
        elif mins < 130:
            return "Standard"
        else:
            return "Long"
            
    merged['duration_category'] = merged['minute'].apply(categorize_duration)
    
    # Select only the final list of columns specified by the user
    final_columns = [
        'id', 'name', 'date', 'tagline', 'description', 'minute', 'duration_category', 'rating',
        'genres', 'themes', 'studios', 'languages', 'actors', 'directors', 'poster'
    ]
    merged = merged[final_columns]
        
    return merged

def export_top_movies(df, output_path="data/top_1000_movies.jsonl"):
    """
    Exports the movies DataFrame to a JSONL file suitable for a RAG system.
    """
    print(f"Exporting to {output_path}...")
    df.to_json(output_path, orient="records", lines=True)
    print("Done!")

if __name__ == "__main__":
    download_data()
    movie_dataset = build_movie_dataset()
    export_top_movies(movie_dataset)

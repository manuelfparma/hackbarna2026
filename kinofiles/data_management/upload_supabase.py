import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

def upload_movies(filepath="data/top_1000_movies.jsonl", table_name="movies"):
    # Load environment variables from .env file
    load_dotenv()
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key or key == "your-api-key-here":
        print("Error: SUPABASE_URL or SUPABASE_KEY is missing or invalid.")
        print("Please update your .env file with the correct credentials.")
        return
        
    # Initialize the Supabase client
    print(f"Connecting to Supabase at {url}...")
    supabase: Client = create_client(url, key)
    
    # Check if the JSONL file exists
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'.")
        print("Please make sure you have run 'data_cleaning.py' first.")
        return
        
    print(f"Reading '{filepath}'...")
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
            
    print(f"Loaded {len(records)} movies. Uploading to Supabase table '{table_name}'...")
    
    # Uploading in chunks to avoid payload limits
    chunk_size = 250
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        print(f"Uploading batch {i + 1} to {i + len(chunk)}...")
        try:
            # Using upsert so it updates existing rows based on the 'id' primary key instead of failing
            response = supabase.table(table_name).upsert(chunk).execute()
        except Exception as e:
            print(f"An error occurred during upload: {e}")
            return
            
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_movies()

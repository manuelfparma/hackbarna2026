import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client

def migrate_posters_db(filepath="data/top_1000_movies.jsonl", table_name="movies"):
    # Load environment variables from .env file
    # We resolve from the kinofiles directory as done in other scripts
    load_dotenv()
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key or key == "your-api-key-here":
        print("Error: SUPABASE_URL or SUPABASE_KEY is missing or invalid.")
        print("Please update your .env file with the correct credentials.")
        return
        
    print(f"Connecting to Supabase at {url}...")
    supabase: Client = create_client(url, key)
    
    print("\n" + "="*60)
    print("IMPORTANT: Ensure you have added the 'poster' column!")
    print("If you haven't, run this in the Supabase SQL Editor:")
    print("ALTER TABLE movies ADD COLUMN IF NOT EXISTS poster TEXT;")
    print("="*60 + "\n")
    
    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'.")
        print("Please make sure you have run 'data_cleaning.py' first.")
        return
        
    print(f"Reading '{filepath}' to get IDs and posters...")
    updates = []
    with open(filepath, 'r') as f:
        for line in f:
            row = json.loads(line.strip())
            # We specifically extract ONLY id and poster
            if "id" in row and "poster" in row:
                updates.append({
                    "id": row["id"],
                    "poster": row["poster"]
                })
            
    print(f"Found {len(updates)} movies with posters. Starting update of Supabase table '{table_name}'...")
    print("Performing explicit single-column updates to guarantee no other columns are affected.")
    
    success_count = 0
    error_count = 0
    
    for item in updates:
        try:
            # Explicitly update only the poster column for the matching ID
            supabase.table(table_name).update({"poster": item["poster"]}).eq("id", item["id"]).execute()
            success_count += 1
            if success_count % 100 == 0:
                print(f"Updated {success_count}/{len(updates)} movies...")
        except Exception as e:
            print(f"Error updating movie ID {item['id']}: {e}")
            error_count += 1
            if error_count == 1:
                print("Hint: Did you forget to add the 'poster' column in Supabase?")
                # We stop on the first error since it's likely a schema issue
                break
            
    print(f"\nMigration completed! Successfully updated {success_count} rows. Errors: {error_count}")

if __name__ == "__main__":
    migrate_posters_db()

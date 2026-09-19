"""Direct request subagent: queries Supabase.

Searches for movies by a specific attribute (e.g. director, genre)
using the entity extracted by the Classifier.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

class DirectRequestHandler:
    def __init__(self, llm=None):
        self.llm = llm  # Kept for signature compatibility if needed
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            self.supabase: Client = create_client(url, key)
        else:
            self.supabase = None

    def handle(self, column: str | None, value: str | None) -> str:
        """Query Supabase using the extracted column and value."""
        if not column or not value:
            return "Sorry, I couldn't figure out exactly what actor, director, genre, or studio you are looking for."

        if not self.supabase:
            return "Database connection is not configured."

        try:
            # Search 'name' directly, otherwise use the generated '_text' column
            search_column = column if column == "name" else f"{column}_text"
            
            # Fuzzy match using ILIKE
            res = self.supabase.table("movies").select("id, name").ilike(search_column, f"%{value}%").limit(5).execute()
            movies = res.data
            
            if not movies:
                return f"I couldn't find any movies for {column}: '{value}'."
            
            names = [f"[{m['id']}] {m['name']}" for m in movies]
            return f"Here are some movies I found for {column} '{value}':\n" + "\n".join(f"- {name}" for name in names)
        except Exception as e:
            return f"Error querying database: {str(e)}"

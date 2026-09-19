"""Intent classifier subagent: a single LLM call.

First layer of the router agent: decides which node should handle the
incoming message and extracts relevant search fields if it's a direct request.
"""

import json

PROMPT = """You are the router for a movie assistant. Classify the message into exactly one category:

- theme_recommendation: the user wants suggestions based on mood, vibe, topic, or thematic content (e.g. "melancholy family stories", "coming of age", "movies about grief", "something with found family"). Prefer this over recommendation whenever the ask is about themes rather than a generic "recommend a movie".
- recommendation: the user wants movie suggestions but did not describe themes, mood, or topic (e.g. "recommend me a film", "what's good to watch").
- feedback: the user is reacting to recommendations already given (likes, dislikes, or asks to change them).
- direct_request: the user asks for movies by a specific attribute (e.g. director, actors, genres, studios, themes, languages).
- social: greetings, thanks, small talk, or anything unrelated to movies.

Message: {request}

Return a JSON object with the following keys:
- "intent": exactly one of "theme_recommendation", "recommendation", "feedback", "direct_request", "social"
- "column": if intent is "direct_request", the attribute to filter by (one of "directors", "actors", "genres", "studios", "themes", "languages"). Otherwise, null.
- "value": if intent is "direct_request", the exact value to search for. Otherwise, null.

Example 1:
Message: What movies did Christopher Nolan direct?
Answer: {{"intent": "direct_request", "column": "directors", "value": "Christopher Nolan"}}

Example 2:
Message: Show me some Comedy movies.
Answer: {{"intent": "direct_request", "column": "genres", "value": "Comedy"}}

Example 3:
Message: I feel like watching something funny.
Answer: {{"intent": "theme_recommendation", "column": null, "value": null}}

Answer:"""

INTENTS = {
    "theme_recommendation",
    "recommendation",
    "feedback",
    "direct_request",
    "social",
}


class Classifier:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, request: str) -> dict:
        """Return a dict with intent, column, and value."""
        answer = self.llm.invoke(PROMPT.format(request=request))
        content = answer.content.strip()
        
        # Strip markdown block if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            data = json.loads(content)
            intent = data.get("intent", "").strip().lower()
            if intent not in INTENTS:
                intent = "social"
            return {
                "intent": intent,
                "column": data.get("column"),
                "value": data.get("value")
            }
        except json.JSONDecodeError:
            return {"intent": "social", "column": None, "value": None}

"""Intent classifier subagent (MOCK): a single LLM call.

First layer of the router agent: decides which node should handle the
incoming message.
"""

PROMPT = """You are the router for a movie assistant. Classify the message \
into exactly one category:

- theme_recommendation: the user wants suggestions based on mood, vibe, \
topic, or thematic content (e.g. "melancholy family stories", "coming of \
age", "movies about grief", "something with found family"). Prefer this \
over recommendation whenever the ask is about themes rather than a generic \
"recommend a movie".
- recommendation: the user wants movie suggestions but did not describe \
themes, mood, or topic (e.g. "recommend me a film", "what's good to watch").
- feedback: the user is reacting to recommendations already given (likes, \
dislikes, or asks to change them).
- direct_request: the user asks a specific factual question about a movie \
(director, cast, release year, plot, "what are the themes of Inception", \
etc.).
- social: greetings, thanks, small talk, or anything unrelated to movies.

Message: {request}

Answer with exactly one word: theme_recommendation, recommendation, \
feedback, direct_request or social."""

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

    def classify(self, request: str) -> str:
        """Return one of: theme_recommendation, recommendation, feedback, direct_request, social."""
        answer = self.llm.invoke(PROMPT.format(request=request))
        intent = answer.content.strip().lower().replace(" ", "_")
        return intent if intent in INTENTS else "social"

"""Intent classifier subagent (MOCK): a single LLM call.

First layer of the router agent: decides which node should handle the
incoming message.
"""

PROMPT = """You are the router for a movie assistant. Classify the message \
into exactly one category:

- recommendation: the user wants movie suggestions.
- feedback: the user is reacting to recommendations already given (likes, \
dislikes, or asks to change them).
- direct_request: the user asks a specific factual question about a movie \
(director, cast, release year, plot, etc.).
- social: greetings, thanks, small talk, or anything unrelated to movies.

Message: {request}

Answer with exactly one word: recommendation, feedback, direct_request or \
social."""

INTENTS = {"recommendation", "feedback", "direct_request", "social"}


class Classifier:
    def __init__(self, llm):
        self.llm = llm

    def classify(self, request: str) -> str:
        """Return one of: recommendation, feedback, direct_request, social."""
        answer = self.llm.invoke(PROMPT.format(request=request))
        intent = answer.content.strip().lower()
        return intent if intent in INTENTS else "social"

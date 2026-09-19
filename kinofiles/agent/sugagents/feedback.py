"""Feedback subagent (MOCK): a single LLM call.

Handles a user's reaction to previous recommendations, turning free-text
feedback into a short acknowledgement.
"""

PROMPT = """You are a movie recommender collecting feedback.

Previous recommendations: {movies}
User feedback: {request}

Write a short, friendly reply (1-2 sentences) acknowledging the feedback and \
confirming you'll refine the recommendations."""


class FeedbackHandler:
    def __init__(self, llm):
        self.llm = llm

    def handle(self, request: str, movies: list[str]) -> str:
        """Return a short acknowledgement reply for the given feedback."""
        titles = ", ".join(movies) or "none yet"
        answer = self.llm.invoke(PROMPT.format(movies=titles, request=request))
        return answer.content.strip()

"""Package feedback as data for the shared conversational reply layer."""


class FeedbackHandler:
    def handle(self, request: str, movies: list[str]) -> dict:
        """Return feedback context without making a second capability LLM call."""
        return {
            "kind": "feedback",
            "feedback": request,
            "previous_titles": movies,
            "titles": [],
            "error": None,
        }

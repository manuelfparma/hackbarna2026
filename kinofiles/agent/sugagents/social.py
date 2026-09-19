"""Social subagent (MOCK): a single LLM call.

Handles greetings, small talk, and anything unrelated to movies, keeping the
assistant in character without derailing into recommendations.
"""

PROMPT = """You are a friendly movie recommendation assistant.

Reply to the following message in a warm, brief way (1-2 sentences), and if \
appropriate, gently invite the user to ask for a movie recommendation.

Message: {request}"""


class SocialHandler:
    def __init__(self, llm):
        self.llm = llm

    def handle(self, request: str) -> str:
        """Return a friendly reply for social/small-talk messages."""
        answer = self.llm.invoke(PROMPT.format(request=request))
        return answer.content.strip()

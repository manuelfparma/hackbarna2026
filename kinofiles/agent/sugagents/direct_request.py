"""Direct request subagent (MOCK): a single LLM call.

Answers specific, factual questions about movies (e.g. director, cast,
release year) rather than producing recommendations.
"""

PROMPT = """You are a movie knowledge assistant.

Answer the following question directly and concisely, in 1-3 sentences. \
If you don't know the answer, say so honestly.

Question: {request}"""


class DirectRequestHandler:
    def __init__(self, llm):
        self.llm = llm

    def handle(self, request: str) -> str:
        """Return a direct answer to a factual movie question."""
        answer = self.llm.invoke(PROMPT.format(request=request))
        return answer.content.strip()

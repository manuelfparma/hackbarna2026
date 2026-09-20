"""Cost meters for a benchmarked turn: chat tokens, embedding calls, latency.

Measured by snapshot rather than by start/stop flags: the scorer embeds
queries too, and a flag left on would bill those to the system under test.
Every meter exposes a cheap immutable reading, and a run records the
difference around the call it is timing.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler


@dataclass(frozen=True)
class Usage:
    """Token counts, as reported by the provider. Never estimated."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    errors: int = 0

    def __sub__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens - other.prompt_tokens,
            self.completion_tokens - other.completion_tokens,
            self.total_tokens - other.total_tokens,
            self.calls - other.calls,
            self.errors - other.errors,
        )

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.calls,
            "llm_errors": self.errors,
        }


class TokenMeter(BaseCallbackHandler):
    """LangChain callback that accumulates provider-reported token usage.

    Attached to the chat models themselves, so it sees every call the graph
    makes — classifier, reply composer, and anything a node adds later —
    without the nodes knowing they are being measured.

    It also counts failures. `ReplyComposer` swallows its exceptions and
    returns canned copy, so without this a dead provider looks like a cheap,
    fast agent instead of a broken one.
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0
        self.errors = 0

    def snapshot(self) -> Usage:
        return Usage(
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.calls,
            self.errors,
        )

    def on_llm_error(self, error, **kwargs) -> None:
        self.errors += 1

    def on_llm_end(self, response, **kwargs) -> None:
        self.calls += 1
        usage = self._usage_from_generations(response)
        if usage is None:
            usage = self._usage_from_llm_output(response)
        if usage is None:
            return
        prompt, completion, total = usage
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total or (prompt + completion)

    @staticmethod
    def _usage_from_generations(response) -> tuple[int, int, int] | None:
        for batch in getattr(response, "generations", []) or []:
            for generation in batch:
                message = getattr(generation, "message", None)
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    return (
                        meta.get("input_tokens", 0),
                        meta.get("output_tokens", 0),
                        meta.get("total_tokens", 0),
                    )
        return None

    @staticmethod
    def _usage_from_llm_output(response) -> tuple[int, int, int] | None:
        output = getattr(response, "llm_output", None) or {}
        usage = output.get("token_usage") or output.get("usage") or {}
        if not usage:
            return None
        return (
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )


@dataclass(frozen=True)
class EmbeddingUsage:
    """Embedding traffic. Mistral reports no token count for `mistral-embed`,
    so this records what is observable: how many calls, over how much text."""

    calls: int = 0
    characters: int = 0

    def __sub__(self, other: "EmbeddingUsage") -> "EmbeddingUsage":
        return EmbeddingUsage(self.calls - other.calls, self.characters - other.characters)

    def as_dict(self) -> dict:
        return {"embedding_calls": self.calls, "embedded_characters": self.characters}


class EmbeddingMeter:
    """Counts `MistralAIEmbeddings` traffic by wrapping the class methods.

    Patching the class rather than an instance is what makes this work at all:
    `ThemeRecommender` builds its own embedder deep inside the graph and never
    offers a seam to inject one.
    """

    def __init__(self):
        self.calls = 0
        self.characters = 0
        self._installed = False

    def snapshot(self) -> EmbeddingUsage:
        return EmbeddingUsage(self.calls, self.characters)

    def install(self) -> None:
        if self._installed:
            return
        from langchain_mistralai import MistralAIEmbeddings

        meter = self
        original_query = MistralAIEmbeddings.embed_query
        original_documents = MistralAIEmbeddings.embed_documents

        def embed_query(self, text):
            meter.calls += 1
            meter.characters += len(text or "")
            return original_query(self, text)

        def embed_documents(self, texts):
            meter.calls += 1
            meter.characters += sum(len(text or "") for text in texts)
            return original_documents(self, texts)

        MistralAIEmbeddings.embed_query = embed_query
        MistralAIEmbeddings.embed_documents = embed_documents
        self._installed = True


@dataclass
class TurnCost:
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)
    embeddings: EmbeddingUsage = field(default_factory=EmbeddingUsage)

    def as_dict(self) -> dict:
        return {"seconds": round(self.seconds, 3), **self.usage.as_dict(), **self.embeddings.as_dict()}


@contextmanager
def measure(tokens: TokenMeter, embeddings: EmbeddingMeter):
    """Record one turn's cost as the delta across the meters."""
    cost = TurnCost()
    before_tokens = tokens.snapshot()
    before_embeddings = embeddings.snapshot()
    start = time.perf_counter()
    try:
        yield cost
    finally:
        cost.seconds = time.perf_counter() - start
        cost.usage = tokens.snapshot() - before_tokens
        cost.embeddings = embeddings.snapshot() - before_embeddings

"""Shared chat-LLM factory.

Every node takes an LLM injected from the orchestrator, so this is the single
place where the chat provider is chosen. Nebius AI Studio speaks the OpenAI
API, so `ChatOpenAI` pointed at its base URL is all it takes.

Note this covers the *chat* model only. Theme search still embeds with
`mistral-embed` (`data_management/embeddings_pipeline.py`) because the vectors
already stored in Supabase live in that model's space — swapping the embedder
would mean re-embedding every theme.

Reads from the environment:
- `NEBIUS_API_KEY`  (required)
- `NEBIUS_MODEL`    (optional, defaults to `DEFAULT_MODEL`)
- `NEBIUS_ENDPOINT` (optional, defaults to `DEFAULT_ENDPOINT`)
"""

import os

from langchain_openai import ChatOpenAI

DEFAULT_ENDPOINT = "https://api.studio.nebius.com/v1/"
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def build_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """Return the chat model every node runs on."""
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set — add it to .env (see .env.template)."
        )
    return ChatOpenAI(
        model=model or os.environ.get("NEBIUS_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        api_key=api_key,
        base_url=os.environ.get("NEBIUS_ENDPOINT", DEFAULT_ENDPOINT),
    )

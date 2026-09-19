"""Shared chat-LLM factories.

Query classification uses Mistral (`ChatMistralAI`). The other chat nodes
(recommendations, feedback, social, direct request) use Nebius AI Studio,
which speaks the OpenAI API.

Theme search still embeds with `mistral-embed` because the vectors already
stored in Supabase live in that model's space — swapping the embedder
would mean re-embedding every theme.

Reads from the environment:
- `NEBIUS_API_KEY`     (required for non-classifier chat)
- `NEBIUS_MODEL`       (optional, defaults to `DEFAULT_NEBIUS_MODEL`)
- `NEBIUS_ENDPOINT`    (optional, defaults to `DEFAULT_NEBIUS_ENDPOINT`)
- `MISTRAL_API_KEY`    (required for classification and embeddings)
- `MISTRAL_MODEL`      (optional, defaults to `DEFAULT_MISTRAL_MODEL`)
- `MISTRAL_ENDPOINT`   (optional)
"""

import os

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

DEFAULT_NEBIUS_ENDPOINT = "https://api.studio.nebius.com/v1/"
DEFAULT_NEBIUS_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DEFAULT_MISTRAL_MODEL = "ministral-8b-2512"


def build_llm(model: str | None = None, temperature: float = 0) -> ChatOpenAI:
    """Return the Nebius chat model used by non-classifier nodes."""
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set — add it to .env (see .env.template)."
        )
    return ChatOpenAI(
        model=model or os.environ.get("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL),
        temperature=temperature,
        api_key=api_key,
        base_url=os.environ.get("NEBIUS_ENDPOINT", DEFAULT_NEBIUS_ENDPOINT),
    )


def build_mistral_llm(
    model: str | None = None, temperature: float = 0
) -> ChatMistralAI:
    """Return the Mistral chat model used for query classification."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set — add it to .env (see .env.template)."
        )
    kwargs = {
        "model": model or os.environ.get("MISTRAL_MODEL", DEFAULT_MISTRAL_MODEL),
        "temperature": temperature,
        "mistral_api_key": api_key,
    }
    endpoint = os.environ.get("MISTRAL_ENDPOINT")
    if endpoint:
        kwargs["endpoint"] = endpoint
    return ChatMistralAI(**kwargs)

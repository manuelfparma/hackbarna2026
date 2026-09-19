"""Shape of every turn the agents hand back to the caller.

Lives under `io/` because this is a presentation contract, not a domain one:
`text` is the part meant to be read aloud, `options` are picked from on screen
and must never be narrated — a list of titles makes for terrible speech.
Keeping them apart lets the I/O layer speak one and render the other without
having to guess where the sentence ends.

Both the orchestrator and the recommendation subagent raise interrupts, so
neither can own this helper without the other importing it in a circle.
"""


def prompt(text: str, options: list[str] | None = None) -> dict:
    """Build the payload an `interrupt()` hands back to the caller."""
    return {"text": text, "options": options or []}

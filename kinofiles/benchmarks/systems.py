"""The two systems under test, behind one `answer(query) -> Answer` interface.

- `agent`: the full RecommendationAgent graph (classify -> retrieve -> reply).
- `mistral_simple`: one prompt, one Mistral call, no retrieval — the baseline
  the agent has to justify its extra tokens against.

Both share one metered pair of models, so cost is measured the same way for
each of them.
"""

import json
import re
import time
from dataclasses import dataclass, field

from agent.llm import build_llm, build_mistral_llm
from agent.nodes.criteria import normalize_criteria
from agent.recomedation_agent import RecommendationAgent

from benchmarks.instrumentation import EmbeddingMeter, TokenMeter, TurnCost, measure

SHORTLIST_SIZE = 8
MAX_ATTEMPTS = 3
RETRY_BACKOFF = 4.0

BASELINE_PROMPT = """You are a movie recommendation assistant.

Reply with JSON only, no prose around it:
{{"reply": "<at most three short sentences>", "titles": ["Title (Year)", ...]}}

Give at most {k} titles, ordered best first, that answer the user's message.
Use an empty list when the message does not call for recommendations.

User message: {query}"""


@dataclass
class Answer:
    titles: list[str] = field(default_factory=list)
    reply: str = ""
    error: str | None = None
    cost: TurnCost = field(default_factory=TurnCost)

    def as_dict(self) -> dict:
        return {
            "reply": self.reply,
            "titles": self.titles,
            "error": self.error,
            **self.cost.as_dict(),
        }


class Meters:
    """One metered pair of models shared by every arm of a run."""

    def __init__(self):
        self.tokens = TokenMeter()
        self.embeddings = EmbeddingMeter()
        self.embeddings.install()

    def attach(self, llm):
        llm.callbacks = [self.tokens]
        return llm


def _retry(call, label: str):
    """Rate limits are the norm on these tiers, so back off and try again.

    Tokens burned by a failed attempt still count: that is a real cost of
    running the system, and hiding it would flatter whichever arm is flakier.
    """
    last = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as error:  # provider errors are not a shared type
            last = error
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
    raise RuntimeError(f"{label} failed after {MAX_ATTEMPTS} attempts: {last}")


def parse_titles(text: str, limit: int) -> tuple[list[str], str]:
    """Pull (titles, reply) out of a baseline answer.

    Models drift out of strict JSON, and a run that dropped those turns would
    quietly record the baseline on its good days only — so fall back to
    reading a numbered or bulleted list.
    """
    blob = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", blob, re.S)
    if fenced:
        blob = fenced.group(1).strip()
    brace = re.search(r"\{.*\}", blob, re.S)
    if brace:
        try:
            payload = json.loads(brace.group(0))
            titles = [str(title).strip() for title in payload.get("titles", []) if title]
            return titles[:limit], str(payload.get("reply", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    listed = re.findall(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", text, re.M)
    titles = [re.sub(r"\s*[-–—:].*$", "", item).strip(" *\"'") for item in listed]
    return [title for title in titles if title][:limit], text.strip()


class AgentSystem:
    """The RecommendationAgent graph, one fresh thread per query.

    Each query starts its own thread on purpose: this benchmark sends one turn
    at a time, and a shared thread would let criteria from query 3 steer
    query 4.
    """

    name = "agent"

    def __init__(self, meters: Meters):
        self._meters = meters
        self._agent = RecommendationAgent(
            llm=meters.attach(build_llm()),
            classifier_llm=meters.attach(build_mistral_llm()),
            reply_llm=meters.attach(build_llm(temperature=0.4)),
        )
        self._turn = 0

    def answer(self, query: str) -> Answer:
        self._turn += 1
        thread = {"configurable": {"thread_id": f"bench-{self._turn}"}}
        payload = {
            "request": query,
            "feedback": [],
            "movies": [],
            "search_criteria": normalize_criteria({}),
            "result": {},
            "history": [],
            "show_options": False,
        }
        with measure(self._meters.tokens, self._meters.embeddings) as cost:
            try:
                event = _retry(lambda: self._agent.graph.invoke(payload, thread), self.name)
            except RuntimeError as error:
                return Answer(error=str(error), cost=cost)
        interrupt = event.get("__interrupt__")
        if not interrupt:
            # The graph only ends without an interrupt when a turn selected a
            # title, which a single-turn benchmark never does.
            return Answer(error="graph ended without asking for the next turn", cost=cost)
        value = interrupt[0].value
        return Answer(
            titles=list(value.get("options") or [])[:SHORTLIST_SIZE],
            reply=value.get("text", ""),
            cost=cost,
        )


class MistralSimpleSystem:
    """One Mistral call, no retrieval — the baseline."""

    name = "mistral_simple"

    def __init__(self, meters: Meters):
        self._meters = meters
        self._llm = meters.attach(build_mistral_llm(temperature=0))

    def answer(self, query: str) -> Answer:
        prompt = BASELINE_PROMPT.format(k=SHORTLIST_SIZE, query=query)
        with measure(self._meters.tokens, self._meters.embeddings) as cost:
            try:
                response = _retry(lambda: self._llm.invoke(prompt), self.name)
            except RuntimeError as error:
                return Answer(error=str(error), cost=cost)
        titles, reply = parse_titles(response.content, SHORTLIST_SIZE)
        return Answer(titles=titles, reply=reply, cost=cost)


REGISTRY = {AgentSystem.name: AgentSystem, MistralSimpleSystem.name: MistralSimpleSystem}


def build_systems(names: list[str], meters: Meters) -> list:
    systems = []
    for name in names:
        if name not in REGISTRY:
            raise SystemExit(f"unknown system {name!r} — choose from {sorted(REGISTRY)}")
        systems.append(REGISTRY[name](meters))
    return systems

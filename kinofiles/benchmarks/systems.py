"""The two arms: the same three-turn conversation, with and without the agent.

- `agent`: the real flow, `agent/orchestrator.py`, driven through its
  interrupts exactly as a person would drive it from the terminal.
- `catalog_prompt`: the same three user turns answered by one Mistral call
  per turn, with all 1000 films pasted into the prompt and the conversation
  so far carried along.

Both end each turn with a shortlist of titles that exist in the catalog, so
what separates them is not access to the films but how they get to them: the
agent classifies, merges criteria and retrieves; the baseline re-reads the
catalog from scratch every turn. That difference is what the token counts are
there to price.
"""

import json
import re
import time
from dataclasses import dataclass, field

from agent.llm import build_llm, build_mistral_llm

from benchmarks.catalog import catalog_lines, load_movies
from benchmarks.instrumentation import EmbeddingMeter, TokenMeter, TurnCost, measure

SHORTLIST_SIZE = 8
MAX_ATTEMPTS = 3
RETRY_BACKOFF = 4.0

# Anything that is not a digit reads as an abstention at the vote prompt, so
# a refinement typed there falls through to `refine`, which then asks what to
# change. That is the path the captured session took.
VOTE_PROMPT = "which one speaks to you"

BASELINE_PROMPT = """You are a movie recommendation assistant for a catalog.

Recommend only films from the catalog below — nothing else can be played, so
a title that is not on the list is worse than no title at all.

Every catalog line reads `Title (Year) | genres`. Return only the `Title`
part, spelled exactly as the line spells it.

The user is refining their request across several turns. Everything they have
asked for so far still applies.

Reply with JSON only, no prose around it:
{{"titles": ["Title", ...]}}

Give exactly {k} titles, best first.

Catalog ({n} films):
{catalog}

Conversation so far:
{history}

Latest message from the user: {question}"""


@dataclass
class Turn:
    """One user message and what a system answered."""

    question: str
    titles: list[str] = field(default_factory=list)
    error: str | None = None
    cost: TurnCost = field(default_factory=TurnCost)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
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

    def patch_orchestrator(self, all_mistral: bool = True) -> None:
        """Meter the models the orchestrator builds for itself.

        `Classifier` wraps its model in `with_structured_output` and the
        orchestrator calls the factories directly, so there is no instance to
        attach to after the fact. Replacing the factories in that module's
        namespace catches every model before it is wrapped — classifier,
        reply composer and anything a node adds later.

        `all_mistral` also points the conversational factory at Mistral, so
        the agent runs on the same model as the baseline. Tokens counted
        across two providers are not the same unit — different tokenizers,
        different prices — and comparing them would measure the tokenizer as
        much as the system. Pass False to run the agent as it ships, with
        Nebius composing the replies.
        """
        import agent.orchestrator as orchestrator

        meters = self
        conversational = build_mistral_llm if all_mistral else build_llm
        orchestrator.build_llm = lambda *a, **kw: meters.attach(conversational(*a, **kw))
        orchestrator.build_mistral_llm = lambda *a, **kw: meters.attach(
            build_mistral_llm(*a, **kw)
        )


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


def parse_titles(text: str, limit: int) -> list[str]:
    """Pull the titles out of a baseline answer.

    Models drift out of strict JSON, and a run that dropped those turns would
    record the baseline on its good days only — so fall back to reading a
    numbered or bulleted list.
    """
    blob = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", blob, re.S)
    if fenced:
        blob = fenced.group(1).strip()
    brace = re.search(r"\{.*\}", blob, re.S)
    if brace:
        try:
            payload = json.loads(brace.group(0))
            titles = [_clean(title) for title in payload.get("titles", []) if title]
            return [title for title in titles if title][:limit]
        except (json.JSONDecodeError, AttributeError):
            pass
    listed = re.findall(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", text, re.M)
    titles = [_clean(re.sub(r"\s*[-–—:].*$", "", item)) for item in listed]
    return [title for title in titles if title][:limit]


def _clean(title: str) -> str:
    """Strip the year and genre tail, if a title came back wearing them."""
    text = str(title).split("|")[0].strip(" *\"'")
    return re.sub(r"\s*\(\d{4}\)\s*$", "", text).strip()


class AgentSystem:
    """The orchestrator, driven through its interrupts like a terminal user.

    The script sends the user's words and an abstention at each vote, which is
    what the captured session did: refinements typed at the vote prompt are
    read as "nobody picked one", and the flow asks what to change.
    """

    name = "agent"

    def __init__(self, meters: Meters, questions: list[str]):
        from agent.orchestrator import OrquestratorAgent
        from langgraph.types import Command

        self._meters = meters
        self._questions = questions
        self._Command = Command
        self._agent = OrquestratorAgent()
        self.setup_cost = TurnCost()

    def run(self) -> list[Turn]:
        """Replay the conversation, one Turn per shortlist the flow produces.

        Cost is attributed to the invocation that produced each shortlist, so
        a round carries its classifier, retrieval and reply calls together.
        Naming the participants happens before any of that and is recorded
        apart, as `setup`.
        """
        Command = self._Command
        config = {"configurable": {"thread_id": "flow"}}
        turns: list[Turn] = []

        with measure(self._meters.tokens, self._meters.embeddings) as cost:
            event = _retry(lambda: self._agent.graph.invoke({}, config), self.name)
            # welcome asks for names; the captured session ran with one person.
            event = _retry(
                lambda: self._agent.graph.invoke(Command(resume="You"), config), self.name
            )
        self.setup_cost = cost

        for question in self._questions:
            answer = question
            while True:
                with measure(self._meters.tokens, self._meters.embeddings) as cost:
                    try:
                        event = _retry(
                            lambda a=answer: self._agent.graph.invoke(
                                Command(resume=a), config
                            ),
                            self.name,
                        )
                    except RuntimeError as error:
                        turns.append(Turn(question, error=str(error), cost=cost))
                        return turns
                payload = (event.get("__interrupt__") or [None])[0]
                if payload is None:
                    turns.append(
                        Turn(question, error="flow ended before the next shortlist", cost=cost)
                    )
                    return turns
                value = payload.value
                options = list(value.get("options") or [])
                if options and VOTE_PROMPT in (value.get("text") or "").lower():
                    turns.append(Turn(question, titles=options[:SHORTLIST_SIZE], cost=cost))
                    break
                # A prompt with no shortlist is `refine` asking what to
                # change, so the same words go in again. The round therefore
                # spends the user's next message twice: once at the vote
                # prompt, where it counts as an abstention, and once at
                # `refine`, where it is actually read. That is what the
                # captured session did, and both calls are billed to the
                # round they belong to.
                answer = question
        return turns


class CatalogPromptSystem:
    """The same conversation without the agent: one call, whole catalog.

    It is handed its own previous shortlists rather than the agent's. Each arm
    carries its own mistakes forward, which is the honest way to compare two
    systems across turns.
    """

    name = "catalog_prompt"

    def __init__(self, meters: Meters, questions: list[str]):
        self._meters = meters
        self._questions = questions
        self._llm = meters.attach(build_mistral_llm(temperature=0))
        rows = load_movies()
        self._catalog = catalog_lines(rows)
        self._n_films = len(rows)
        self.setup_cost = TurnCost()

    def run(self) -> list[Turn]:
        turns: list[Turn] = []
        history: list[str] = []
        for question in self._questions:
            prompt = BASELINE_PROMPT.format(
                k=SHORTLIST_SIZE,
                n=self._n_films,
                catalog=self._catalog,
                history="\n".join(history) or "(nothing yet)",
                question=question,
            )
            with measure(self._meters.tokens, self._meters.embeddings) as cost:
                try:
                    response = _retry(lambda: self._llm.invoke(prompt), self.name)
                except RuntimeError as error:
                    turns.append(Turn(question, error=str(error), cost=cost))
                    return turns
            titles = parse_titles(response.content, SHORTLIST_SIZE)
            turns.append(Turn(question, titles=titles, cost=cost))
            history.append(f"user: {question}")
            history.append(f"you suggested: {', '.join(titles) or '(nothing)'}")
        return turns


REGISTRY = {system.name: system for system in (AgentSystem, CatalogPromptSystem)}


def build_systems(names: list[str], meters: Meters, questions: list[str]) -> list:
    systems = []
    for name in names:
        if name not in REGISTRY:
            raise SystemExit(f"unknown system {name!r} — choose from {sorted(REGISTRY)}")
        systems.append(REGISTRY[name](meters, questions))
    return systems

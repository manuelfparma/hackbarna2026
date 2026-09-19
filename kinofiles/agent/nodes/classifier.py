"""Intent classifier: structured LLM output with named entities."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

Intent = Literal[
    "theme_recommendation",
    "recommendation",
    "feedback",
    "direct_request",
    "social",
]

MovieRole = Literal["seed", "liked", "seen", "wanted", "asked_about"]
CriteriaAction = Literal["add", "replace", "reset", "keep"]

# Catalog array columns only. Free-text `themes` (moods) are not filter values.
FILTER_COLUMNS = (
    "directors",
    "actors",
    "genres",
    "studios",
    "languages",
)

SYSTEM_PROMPT = """You are the router for a movie assistant.
Classify the user message into exactly one intent, and extract every named
entity that could be used later for catalog lookup or similar-to search.

Intents:
- theme_recommendation: suggestions based on mood, vibe, topic, or thematic
  content (e.g. "melancholy family stories", "coming of age", "movies about grief").
  Prefer this over recommendation when the ask is about themes rather than a
  generic "recommend a movie".
- recommendation: movie suggestions with no theme language, including
  similar-to a named film ("a movie like Super Troopers") and open browse
  ("what should I watch").
- feedback: reacting to recommendations already given (likes, dislikes, change them).
- direct_request: movies by a catalog attribute (director, actor, genre, studio,
  language) or a fact about a named film.
- social: greetings, thanks, small talk, or anything unrelated to movies.

Criteria action:
- add: the user adds constraints using cues such as "also", "and", "with", or
  "it should have". Example: "Also, it should have some action."
- replace: the user changes direction using "instead" or "rather", or proposes
  an unqualified new direction such as "What about a thriller?"
- reset: the user explicitly says to start over, forget prior preferences, or
  asks for something completely different.
- keep: greetings, thanks, factual questions, and turns with no search change.

Entity rules:
- Copy names as the user said them. Do not invent titles or people.
- movies: each mentioned film. Actual titles only — never a person's or
  studio's name. Use role seed for "like X" / similar-to, liked / seen for
  taste, wanted for "I want to watch X", asked_about for plot/quality
  questions. Include year only if the user said it.
- genres: catalog-style labels when possible (Comedy, Horror, Action, Drama,
  Science Fiction, Thriller, Romance, Animation, Documentary, Fantasy).
  Extract genre words on every intent, including recommendation and
  feedback. Words that are not really genres ("heist", "space") belong in
  themes instead.
- genres, actors, directors, studios, languages: only constraints the user
  explicitly stated. Never infer them from a mentioned film — "a movie like
  The Dark Knight" is not an Action/Thriller request.
- themes: free-text mood or topic phrases ("funny", "found family", "grief").
- Leave lists empty when nothing was mentioned. Never guess."""


class MovieMention(BaseModel):
    """A film name mentioned in the utterance, not yet linked to the catalog."""

    title: str = Field(description="Title as spoken, without @ids.")
    year: int | None = Field(
        default=None,
        description="Release year if the user said it, otherwise null.",
    )
    role: MovieRole = Field(
        default="seed",
        description="How the title is used in this turn.",
    )


class Entities(BaseModel):
    """Named entities for retrieval. All lists may be empty."""

    movies: list[MovieMention] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    studios: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    years: list[int] = Field(
        default_factory=list,
        description="Standalone years (not a movie's year field).",
    )
    time_periods: list[str] = Field(
        default_factory=list,
        description="e.g. 90s, classic, recent.",
    )
    audience: list[str] = Field(
        default_factory=list,
        description="e.g. kids, friends, family.",
    )


class Classification(BaseModel):
    intent: Intent
    entities: Entities = Field(default_factory=Entities)
    criteria_action: CriteriaAction = Field(
        default="add",
        description="How this turn changes the accumulated movie-search criteria.",
    )

    def as_filter(self) -> tuple[str | None, str | None]:
        """First catalog attribute for the existing direct_request node.

        Attributes win over `name`: a mentioned title AND-ing with a real
        filter ("movies by Nolan" -> name ILIKE %Nolan% AND directors ILIKE
        %Nolan%) produces impossible queries. `name` is the fallback for a
        title-only ask ("the movie Heat").
        """
        for column in FILTER_COLUMNS:
            values = getattr(self.entities, column)
            if values:
                return column, values[0]
        if self.entities.movies:
            return "name", self.entities.movies[0].title
        return None, None


class Classifier:
    def __init__(self, llm):
        self.llm = llm.with_structured_output(Classification)

    def classify(self, request: str) -> Classification:
        """Return intent plus extracted entities."""
        try:
            result = self.llm.invoke(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=request),
                ]
            )
        except Exception:
            return Classification(intent="social", criteria_action="keep")

        if not isinstance(result, Classification):
            return Classification(intent="social", criteria_action="keep")
        return result

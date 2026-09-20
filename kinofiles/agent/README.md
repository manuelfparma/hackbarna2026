# Agent

KinoFiles has two LangGraph entry points that share classification, criteria,
retrieval, and reply helpers. The group-mediator workflow integrated from main
is the default orchestrator and web experience; the single-user recommendation
graph remains available separately.

## Group orchestrator

`OrquestratorAgent` in `orchestrator.py` owns this graph:

```text
welcome -> collect_preferences (each participant) -> mediate
                                                     |
                                                     v
                 goodbye <- group_vote (each participant)
                                |
                                v
                              refine -> mediate
```

- `welcome` extracts 1–4 participant names, with a text-splitting fallback.
- `collect_preferences` classifies one request per person and stores independent
  preference text and normalized criteria.
- `mediate` merges those criteria, retrieves options, and provides per-person
  context and compromise notes to the shared reply composer.
- `group_vote` accepts a numbered vote per participant. A unique highest-voted
  film wins; ties or no votes lead to refinement.
- `refine` updates the group brief and retries, up to the existing three-round
  limit. At the limit, the upstream workflow picks the first available suggestion
  or a generic fallback label if no movies are available.
- `goodbye` returns the group winner and ends the graph.

The group state is independent of the single-user state. The orchestrator no
longer embeds a `RecommendationAgent` subgraph or exposes `pending_state()`.
Participant-aware interrupts carry the current person's name for the frontend.

The shared group merger retains the upstream genre policy: use the intersection
of specified genre sets when nonempty, otherwise their union. Existing list
criteria are unioned. Scalar lower bounds use the maximum and upper bounds the
minimum, so the newer year/runtime/rating constraints are not silently lost.
Retrieval still applies the resulting genre list as an AND filter; the union
fallback does not implement genre OR by itself.

Classification failures retry the current group stage without advancing the
participant or refinement round. Classified field-removal actions pass
`clear_fields` to the shared criteria merger. The group flow is not the solo
free-form chat loop: voting accepts numbers, and solo control/fact/selection
handlers are not automatically invoked by the group graph.

## Standalone recommendation agent

`RecommendationAgent` in `recomedation_agent.py` retains the single-user loop:

```text
classify -> capability -> reply -> turn (interrupt) -> classify
                                      |
                                      +-> selection / exit -> END
```

It supports movie discovery, plot identification, preferences/refinements,
additional suggestions, single-film facts, streaming lookups, and deterministic
search resets/filter removal. Capabilities return structured data and the reply
layer owns prose. `movies` and `show_options` remain separate so factual/social
detours can hide the list without destroying it.

Explicit controls are parsed before classification. New sessions use
`initial_state()`. The classifier receives recent context; the reply layer gets
up to six history turns. See `../SYSTEM_OVERVIEW.md` for the full solo capability
reference and the intentionally removed features.

## Shared implementation

- `nodes/classifier.py`: structured extraction, reference handling, safe failure reporting.
- `nodes/criteria.py`: normalization, per-turn merge, grouped clearing, and group aggregation.
- `nodes/filters.py`: common hard-filter semantics.
- `nodes/catalog.py`: named-title resolution.
- `nodes/direct_request.py`: catalog retrieval and single-film facts.
- `nodes/theme_recommender.py`, `nodes/description_search.py`: semantic retrieval.
- `nodes/reply.py`: generated narration and deterministic fallbacks, with group context support.
- `io/turn.py`: interrupt payload with text, options, participant, and criteria.

Recommendation results include description mappings for the selected titles,
retaining the data contract introduced by the group-mediator changes. The web API
exposes participants/votes and accepts a voice selection; those frontend and
voice integrations are preserved from main.

## Run and verify

From `kinofiles/`:

```bash
uv run python -m agent.orchestrator
uv run python -m agent.recomedation_agent
```

The first command starts group mediation; the second starts solo chat. Both need
the configured external-service credentials for real interaction. Each graph
uses in-memory checkpoints keyed by `thread_id`; process restarts lose sessions.

Offline verification:

```bash
uv run python -m unittest discover tests
uv run python -m unittest tests.test_group_integration
```

No live provider calls are required by the regression tests. The removed live
conversation and benchmark scripts should not be recreated as part of ordinary
verification.

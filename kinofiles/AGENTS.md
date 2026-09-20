# KinoFiles development

Run commands from `kinofiles/`.

- Offline regression suite: `uv run python -m unittest discover tests`.
- Focused controls/scope regressions: `uv run python -m unittest tests.test_controls tests.test_simplified_scope`.
- Manual terminal interaction: `uv run python -m agent.orchestrator` (uses external services).
- Lint: `uvx --from ruff==0.12.12 ruff check agent tests --ignore E402`.

The separate live conversation/LLM test runners and ReDial replay harness have been removed. Use the offline suite for regression checks.

The user has requested offline verification rather than more benchmark runs. Do not run database migrations or mock backfills against a shared database without explicit approval.

The current dataset builder deliberately filters release years to 2000 onward before selecting 1,000 titles. Correct filters for older decades may return no results; do not relax user constraints to hide missing catalog coverage.

Description search uses the existing `mistral-embed` space. A small live calibration found Ratatouille plot similarity 0.824 (short phrase 0.789), versus mood-only queries around 0.638–0.656. The initial confidence gate is 0.68, combined with explicit description/theme routing; this is not a calibrated probability or a guarantee of title identity.

All retrieval paths share filter semantics in `agent/nodes/filters.py`. After integrating the group-mediator changes from main, `OrquestratorAgent` owns an independent group graph (participants, per-person preferences, mediation, voting, refinement). It no longer embeds `RecommendationAgent` or inherits its state. Preserve that group workflow. `RecommendationAgent` remains the standalone single-user graph; initialize its sessions with `initial_state()` so exclusions, focus, and pagination remain session-scoped. Group integration regressions are in `tests/test_group_integration.py`. Welcome can start a one-person group named `You` from a movie request when no names are extracted; retain that first request and retry classifier failures at welcome. Group requests are semicolon-joined; `build_theme_query()` splits and deduplicates their segments against normalized criteria. Preserve main's merge/route diagnostics and log candidate-filter counts in the shared description-search helper rather than duplicating its filtering.

In the standalone recommendation flow, explicit search reset/removal commands are parsed before classification in `agent/nodes/criteria_commands.py`. The group flow uses classified actions for preference collection/refinement and passes `clear_fields` to the shared merger. Clear date/runtime source fields and derived bounds together; also invalidate cached search text, focus, shortlist, feedback, and history so removed constraints cannot return. Search resets preserve seen-film memory.

Classifier invocation/schema failures raise `ClassificationError` and log only safe error metadata; never convert them to social messages. Social, control, help, unsupported, and error replies are deterministic so they cannot narrate unperformed searches or ungrounded movie facts.

The supported scope intentionally excludes session watchlists, two-film comparison, catalog analytics, studio filtering, semantic content exclusions, and audience criteria. General help remains static; recognized unsupported operations leave the brief unchanged. Do not extract, persist, or append a dedicated audience field to retrieval queries; legacy audience keys are ignored by normalization. Viewer mentions must not infer Family/Animation genres or Horror exclusions. Original request/history text is not scrubbed. Preserve explicit genre/people exclusions and single-film facts. Existing studio metadata remains in the database; no schema/data purge is required. See `SYSTEM_OVERVIEW.md` for current behavior; `UPGRADE_PLAN.md` is historical and does not authorize reintroducing removed features.

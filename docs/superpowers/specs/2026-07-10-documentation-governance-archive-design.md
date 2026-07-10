# Documentation Governance And Archive Design

Date: 2026-07-10

## Goal

Make v4.3 unmistakably the current CPMP mainline while preserving older
algorithm reports as auditable historical evidence.

## Source Of Truth

The current release identity is defined by these active documents:

- `README.md`
- `docs/index.md`
- `docs/guides/current-best-algorithm.md`
- `docs/guides/current-mainline-algorithm.md`
- `docs/guides/v43-mainline-algorithm.md`
- `docs/guides/algorithm-profile-status.md`

They must all name
`qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish`
as the promoted default and report Open120 `120/120`, BF `721/721`, CV
`840/840`, and combined `1681/1681` with `105530` moves.

## Lifecycle Model

- **Current:** v4.3 decisions, reproduction, usage, data format, development,
  project structure, and the QUBO beam core used by v4.3.
- **Reference:** paper baselines and model notes that remain useful but do not
  define the default solver.
- **Evidence archive:** `docs/experiments/` keeps original paths because its
  1113 Markdown files are heavily cross-referenced.
- **Physical archive:** superseded guides, old reports, old brainstorming
  briefs, legacy notes, and completed implementation plans/specs move under
  `docs/archive/`.

## Archive Safety

No historical result is deleted. Moving files must use `git mv`. All active
links are updated, archived documents receive a historical banner, and a
repository-wide relative Markdown link check must pass before commit.

## Mainline Emphasis

The first decision section in the root README, documentation index, experiments
README, reports README, and archive README must point back to v4.3. Historical
documents may describe the conclusion valid at their original date, but must
not appear in active navigation as a current recommendation.

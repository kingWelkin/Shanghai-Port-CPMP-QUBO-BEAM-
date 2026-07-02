# Single-Frontier FRG Union Design

## Status

Approved for experiment on 2026-06-22.

## Goal

Evaluate FRG child generation as part of the existing v5 diversity-union beam and QUBO selector without widening the beam, adding a fallback solver, or adding case-specific gates.

## Evidence

The current frontier profile uses `frg_frontier_child_append_only=True`. It runs the normal beam/QUBO selection first and appends every accepted FRG child afterward. On the paper CV `10-10` residuals, this widens the frontier from roughly 1200 states to 1344 states, but does not make FRG children compete in the selector. The additional breadth helped smaller `5-4` and `6-6` cases, while it consumed the search budget on larger cases.

The existing frontier trace is recorded before appended FRG children are labeled as search survivors. It cannot distinguish a generated child, an appended child, and a child that becomes part of the next frontier.

## Design

### One Candidate Pool

Create a new ablation profile derived from `scheduler_state_v5_qubo_diversity_union_rollout_graft_polish`.

1. Generate legal beam children as today.
2. Generate FRG descendants from the ranked frontier parents before `_select_next_frontier`.
3. Merge those descendants into the same `candidates` mapping used by the existing ranker and diversity QUBO.
4. Keep the selected frontier at the configured beam width. No child is appended after selection.

The proposed profile uses the existing fixed frontier parameters (`parent_width=16`, `child_limit=2`, `max_steps=3`) with `frg_frontier_child_append_only=False`. It has no adaptive expansion schedule.

### Selection and Diversity

FRG candidates use the existing candidate rank, transition signals, and diversity-union QUBO objective. This experiment adds no quota, no origin bonus, and no FRG-only rescue path. A candidate survives only when it wins through the same quality and diversity selection used by ordinary children.

An origin marker is retained for telemetry only. A later experiment may add an origin-aware pair term only if this clean merged-pool ablation proves that useful FRG candidates are generated but consistently excluded.

### Trace Integrity

Record candidate origin before selection and carry it into frontier trace rows. The trace must distinguish `core`, QUBO-selected candidates, tail fill, and `frg_frontier_child` survivors. It must also expose generated, merged, and selected FRG-child counts per layer or in aggregate.

## Non-Goals

- No BS-FRG fallback or second search path.
- No increase above the configured beam width.
- No small-stack, family, or instance-specific gate.
- No learning algorithm.
- No default-profile promotion without paired benchmark evidence.

## Validation

1. Focused tests verify that merged FRG children enter the ordinary selector, the beam width remains fixed, and trace origin survives selection.
2. Smoke benchmark on the paper CV `10-10` residuals verifies the trace and does not claim a performance win.
3. Paired paper CV sample2 run at 20 seconds per case compares against both the union graft and adaptive frontier records.
4. Paired open120 run at 10 seconds per case compares against the current light-frontier mainline.

## Promotion Rule

The profile remains an ablation unless it keeps open120 at 120/120 with no move or penalty regression from the current light frontier, and keeps paper CV sample2 at least at the union-graft quality floor (39/42 solved and penalty no worse than 2819). A stronger promotion requires 40/42 or better on paper CV while retaining the open120 result.

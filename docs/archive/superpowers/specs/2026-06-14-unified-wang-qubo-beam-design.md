# Unified Wang-QUBO Beam Design

Date: 2026-06-14

## Purpose

Build a single algorithm that learns from Wang-Jin-Lim fixed-prefix search and the current compact QUBO scheduler without composing independent solvers in sequence.

The target algorithm is:

```text
scheduler_state_v7_wang_qubo_unified
```

It should keep one search loop, one parent-chain representation, and one scoring objective. Wang-style fixed prefixes and fix-pair structure become state features and candidate-generation rules inside the compact beam, while QUBO remains the candidate selector for each layer.

The primary metric remains full-case quality: valid count first, then full-quality penalty, solved move total, reference gap sum, and runtime.

## Problem

The current evidence says Wang and QUBO are complementary:

- Wang BS-B is fast and reaches feasibility on open 5x5/7x5 cases, but produces long paths.
- Compact QUBO-family profiles produce shorter paths, but individual profiles still miss hard cases.
- The staged portfolio reaches strong quality, but it is not a clean algorithmic contribution because it runs multiple solver profiles and selects among them.

The next step should not add another fallback chain. It should unify the useful mechanisms:

- Wang contributes fixed-prefix progress, current-group focus, and fix-pair placement structure.
- Compact beam contributes legal top-to-top state search, parent-chain reconstruction, and broad candidate retention.
- QUBO contributes subset selection under a single energy model.

## Non-Goals

- Do not implement `v5 -> features -> rescue -> polish -> Wang` as the new algorithm.
- Do not call `WangBsBabyMovesSolver` or `WangBsGiantMovesSolver` from the unified search loop.
- Do not splice externally generated Wang move sequences into compact parent chains.
- Do not make the existing default profile change behavior.
- Do not optimize only for solved-case move totals while hiding invalid cases.
- Do not commit large scratch benchmark CSVs as curated artifacts unless separately requested.

## Unified State

Each beam item remains a compact byte-state code plus parent metadata, but the unified scheduler derives Wang state features for ranking and QUBO construction:

```text
UnifiedState =
  compact_code
  n_value
  depth
  parent_code
  fixed_prefix_by_stack
  current_group
  fixed_prefix_total
  fix_pair_opportunities
  target_gap
  parent_diversity_key
```

The stored parent chain remains the source of truth. Wang fields are recomputed or cached from the compact code and must not replace legal move reconstruction.

## Candidate Generation

Each layer builds one candidate pool from two sources:

1. **Ordinary compact successors**
   - Existing legal top-to-top compact expansion remains available.
   - This preserves robustness when Wang's current-group view is too narrow.

2. **Wang-guided successors**
   - Derive the current group from the highest group not fully fixed.
   - Score candidate moves by whether they increase fixed prefix, create or realize fix-pair opportunity, reduce target gap, or safely clear blockers.
   - Add these candidates into the same pool as ordinary successors.

The algorithm must not run a separate Wang solver. It only uses Wang-style logic to generate and score candidates in the shared pool.

## QUBO Energy

Lower energy is better. The first implementation should use a linear-plus-diversity QUBO over candidate states:

```text
E(i) =
    a * n_value(i)
  + b * depth(i)
  - c * fixed_prefix_gain(i)
  - d * fix_pair_gain(i)
  + e * target_gap(i)
  + f * unsafe_or_blocker_risk(i)
  + g * revisit_or_cycle_risk(i)

E(i, j) =
    h * same_parent_penalty(i, j)
  + k * same_signature_penalty(i, j)
  - m * complementary_group_bonus(i, j)
```

The existing group-QUBO and Wang feature helpers can supply the first version of these terms, but the unified profile should expose them as one coherent parameter set rather than separate Wang rescue, Wang feature, and post-polish knobs.

## Search Loop

The unified loop is:

```text
beam = initial compact state
best_incumbent = none

for depth in search_depth:
    expand ordinary legal successors
    add Wang-guided successors from the same states
    derive UnifiedState features for every candidate
    select core candidates by deterministic safety/rank filters
    use QUBO to select supplement candidates with Wang-QUBO energy
    merge core + QUBO supplement into next beam
    update fixed-prefix statistics and incumbent
    if incumbent exists:
        keep searching only while the active frontier can plausibly improve moves
```

Path shortening should happen inside this loop by preferring shorter incumbent-improving states. There should not be a separate post-polish solver in the first unified profile. A later profile may add local compression only if it uses the same energy terms and preserves the same parent-chain contract.

## Profile And Runner Surface

Add one compact profile:

```text
scheduler_state_v7_wang_qubo_unified
```

Add one open-runner alias:

```text
qubo_scheduler_state_v7_wang_qubo_unified
```

The profile should keep the existing defaults unchanged. It should include explicit metadata fields for:

- unified candidate counts
- Wang-guided candidate counts
- fixed-prefix gains selected by QUBO
- fix-pair gains selected by QUBO
- unified QUBO calls, variables, and time
- incumbent source within the unified loop

## Benchmark Plan

Use layered evidence:

1. Unit tests for feature extraction and energy ordering.
2. Tiny synthetic cases proving fixed-prefix and fix-pair candidates are selected by the unified QUBO.
3. Smoke runs on `5x5_cv1_test` with 2 and 20 cases.
4. Open 120 comparison against:
   - `qubo_scheduler_state_v5`
   - `qubo_scheduler_state_v5_wang_features`
   - `qubo_scheduler_state_v5_wang_rescue`
   - `qubo_scheduler_state_v5_wang_rescue_polish`
   - `qubo_scheduler_state_v7_wang_qubo_unified`
   - Wang BS-B as a reference, not as part of the algorithm.

Report:

- valid and invalid counts
- full-quality penalty
- solved moves
- reference gap sum
- average runtime
- QUBO calls and QUBO time
- selected fixed-prefix and fix-pair contributions

## Acceptance Criteria

- The unified profile is a single search algorithm, not a sequential portfolio.
- It does not call Wang baseline solvers internally.
- It preserves compact parent-chain move reconstruction.
- Existing default profiles preserve behavior.
- Focused unit tests pass.
- Open-runner variant wiring is tested.
- On open 120, it must be judged by full-case quality, not only by solved-case quality.

## Initial Success Bar

The first implementation does not need to beat the oracle portfolio. It should be considered useful if it reaches one of these thresholds on the open 120 benchmark:

- at least `105/120` valid with better full-quality penalty than `scheduler_state_v5_wang_features`, or
- at least `118/120` valid with lower solved moves than `scheduler_state_v5_wang_rescue`, or
- a clear case-level improvement on one of the remaining hard failures without regressing the easier 5x5 families.

If none of these thresholds are met, keep the implementation as an experimental profile and do not promote it.

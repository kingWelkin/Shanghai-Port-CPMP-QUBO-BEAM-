# Wang Feature Rescue Polish Design

Date: 2026-06-14

## Purpose

Improve full-case quality by learning from Wang-Jin-Lim fixed-prefix search without replacing the compact parent-chain QUBO scheduler. The work is split into three measurable probes:

1. Promote Wang fixed-prefix and fix-pair signals into generic QUBO linear features.
2. Polish paths found by Wang rescue so feasibility gains do not come with unnecessarily long paths.
3. Diagnose the remaining QUBO-family failures, especially `5x5_cv1_test_4` and `5x5_cv1_test_6`.

The primary metric remains full-case quality: valid count first, then full-quality penalty and solved move totals.

## Non-Goals

- Do not make Wang rescue unconditional for internal Shanggang cases.
- Do not splice arbitrary Wang move sequences into compact beam paths.
- Do not promote a profile based only on solved-case move totals.
- Do not commit large scratch CSV outputs unless they are curated baseline artifacts.

## Approach A: Generic Wang QUBO Features

Add a reusable feature layer for QUBO linear scoring:

```text
score = beam_score
      - a * fixed_prefix
      - b * fix_pair
      + c * risk
      + d * same_parent_penalty
```

Lower scores remain better. Existing beam rank and `N/depth` terms stay the base signal. Wang-derived terms only bias selection when enabled by explicit weights.

Implementation targets:

- Add a helper that computes Wang feature values from a compact candidate row.
- Use existing compact-code utilities and current Wang helpers where possible.
- Wire the feature helper into the generic frontier/group QUBO linear score.
- Add profile `scheduler_state_v5_wang_features` with conservative nonzero weights.
- Keep existing Wang pair-QUBO supplement profiles available as ablation controls.

Expected tests:

- Fixed-prefix weight lowers the linear score of a better fixed-prefix candidate.
- Fix-pair weight prefers a candidate that matches Wang target-placement structure.
- Zero weights preserve the current score exactly.
- The new profile is accepted by compact and open benchmark runners.

## Approach B: Rescue Then Polish

Add a profile that uses Wang rescue for feasibility, then always gives the result to the existing post-polish machinery:

```text
scheduler_state_v5_wang_rescue_polish
```

The profile should combine:

- `scheduler_state_v5_wang_rescue` feasibility settings.
- `post_polish_mode=qubo_windows_v2`.
- Suffix polish and window repair settings that accept only shorter paths by default.

This keeps feasibility and path compression separate. If Wang rescue finds a long valid path, polish can attempt local compression without changing the search path ancestry rules.

Expected tests:

- The profile enables Wang rescue and Wang pair-QUBO.
- The profile enables window/suffix polish.
- Polish accept mode remains `better`, not `equal_or_better`, unless explicitly overridden.
- Open runner exposes a variant alias for the new profile.

## Approach C: Remaining Failure Autopsy

Add a focused diagnostic path for the two remaining QUBO-family failures:

- `5x5_cv1_test_4`
- `5x5_cv1_test_6`

The diagnostic should compare early search signals between QUBO-family runs and Wang BS-B:

- best `N` history and plateau length
- fixed-prefix score
- fix-pair opportunity score
- parent concentration
- early Wang BS-B moves or compact-equivalent state deltas

This should be a reproducible experiment entry, not a production behavior change.

Expected tests:

- A small synthetic case can produce a structured autopsy row.
- The runner accepts a case filter or exact case selector.
- Missing Wang diagnostic data degrades to blank fields instead of crashing.

## Benchmark Plan

Use layered evidence:

1. Unit tests for scoring and profile wiring.
2. Smoke runs on tiny internal/open subsets.
3. Full internal 30 comparison to ensure no regression from the current scheduler line.
4. Open 120 comparison against:
   - `scheduler_state_v5`
   - `scheduler_state_v5_wang_rescue`
   - `scheduler_state_v5_wang_rescue_polish`
   - `scheduler_state_v5_wang_features`
   - portfolio with Wang BS-B fallback

Report:

- valid and invalid counts
- full-quality penalty
- solved move total
- average runtime
- post-polish attempts, successes, and moves saved
- Wang feature QUBO calls and selected candidates

## Acceptance Criteria

- All new behavior is behind explicit profiles or CLI flags.
- Existing default profiles preserve current behavior.
- `tests/test_qubo.py` and classical runner tests pass.
- The two failed open cases have a documented diagnostic output path.
- Any promoted recommendation is based on full-case quality, not only solved-case quality.

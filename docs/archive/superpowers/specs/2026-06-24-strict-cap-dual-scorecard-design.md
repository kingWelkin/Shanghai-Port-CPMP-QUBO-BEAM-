# Strict-Cap Dual Scorecard Design

## Status

Approved for specification on 2026-06-24. Awaiting user review before an implementation plan is written.

## Goal

Make the active paper evaluation contract explicit: a result is a primary success only when it reaches `N=0`, obeys the original business cap, and completes within the configured 20-second budget. Retain over-cap or over-time `N=0` paths as feasibility diagnostics, but never count, select, or promote them as primary wins.

This specification applies to the current paper benchmark line with `alpha=10` and `time_limit=20`. It does not retroactively rewrite historical reports, and it does not change the Open120 benchmark contract in this milestone.

## Evidence

The compact runner records `frg_seed_cap_ok` and `frg_seed_large_layout_ok`. Some active profiles accept a seed whenever either is true, which allows large-layout `N=0` paths even when the original cap is violated.

The open benchmark runner already defines an internally valid result as `business_ok && goal_validation_ok && final_n == 0`. The paper runner is inconsistent: its summary predicate currently counts `goal_validation_ok && final_n == 0` without requiring `business_ok`. This makes the paper summary unsuitable as the strict promotion scorecard.

The residual experiments established why this distinction matters:

- `data10-6-4` and `data10-6-9` can reach `N=0` after long FRG-prefix reroot paths, but both violate the original cap.
- `data10-10-10` can reach `N=0` only after a 2071-move oracle path, also far beyond its cap.
- BF seed availability changes under eight-way contention, so a feasibility result must not be mistaken for a stable primary result.

## Design

### Explicit Result Classes

Every paper row will expose four classification fields. `business_ok` already exists; this milestone adds the other three:

| field | Definition | Intended use |
|---|---|---|
| `feasibility_ok` | goal validation succeeds, `final_n == 0`, and a move path exists | diagnostic only |
| `business_ok` | existing replayed original-cap validation | path legality component |
| `time_limit_ok` | measured solver runtime is no greater than the configured per-case limit | runtime legality component |
| `primary_ok` | `feasibility_ok && business_ok && time_limit_ok` | benchmark score, selection, promotion |

`goal_validation_ok` remains for compatibility and detailed diagnostics. It will no longer be used by itself to define a solved paper row.

The time check uses the runner's measured `runtime_sec` and the case-level configured `time_limit`. The comparison is exact: no reporting grace period is added. A caller that needs a different budget must invoke a different explicit budget; it must not rely on a hidden tolerance.

### Scorecards and Selection

The paper CSV and summary will report both scorecards:

1. Primary: case count, `primary_ok` count, primary moves, primary penalty, and average runtime.
2. Feasibility: `feasibility_ok` count, plus counts split into over-cap and over-time feasible paths.

Primary penalty and no-regret portfolio ranking will use `primary_ok`. When no candidate is primary-valid, the existing deterministic tie-break may still retain the best feasibility path in the result row for inspection, but that row remains primary-invalid and contributes the invalid penalty. This preserves traceability without treating it as a solution.

The result status will continue to expose `frg_seed_cap_ok` and `frg_seed_large_layout_ok`. The summary will make the source of every feasibility-only result visible rather than silently discarding it.

### Scope Boundaries

This first milestone changes the evaluation and selection contract only.

- It does not remove or rewrite the existing FRG seed implementation.
- It does not add the BF shared seed cache.
- It does not inject CV oracle prefixes or add a macro-clearing operator.
- It does not modify historical experiment CSVs or summaries.
- It does not promote the lazy terminal-escape portfolio.

After the strict baseline exists, BF deterministic seed sharing and CV macro-clearing are separate design and implementation projects. Each must beat the strict primary scorecard, not only feasibility coverage.

## Data Flow

1. Run the existing solver and replay its moves through the existing validators.
2. Derive `feasibility_ok`, `business_ok`, and `time_limit_ok` for the row.
3. Derive `primary_ok` from those three inputs.
4. Use `primary_ok` for paper summaries, full-quality penalty, and portfolio candidate ranking.
5. Preserve feasibility-only rows and their seed telemetry in the CSV and add their category counts to the summary.

No solver-internal state needs to be shared in this milestone. The contract is established at the runner boundary, where all variants already produce comparable rows.

## Error Handling

- An exception row sets all derived success fields to false and remains an invalid primary row.
- A missing or nonnumeric runtime is `time_limit_ok=false`.
- A missing move path or nonzero final N is `feasibility_ok=false`.
- A caller with `time_limit <= 0` must be rejected for this strict paper scorecard instead of being treated as an unlimited primary run.

## Validation

Automated tests must cover:

1. A goal-valid but over-cap row is feasibility-valid and primary-invalid.
2. A goal-valid, cap-valid but over-time row is feasibility-valid and primary-invalid.
3. A goal-valid, cap-valid, in-time row is primary-valid.
4. Paper summary solved totals use `primary_ok`, while feasibility totals retain all `feasibility_ok` rows.
5. Portfolio comparison prefers a primary-valid candidate over a faster feasibility-only candidate.
6. Existing result-field serialization contains the new derived fields for ordinary, portfolio, and exception rows.

Benchmark validation, after unit tests pass:

1. Run the current union paper sample10 set at `alpha=10`, `t=20`, with `jobs=1` and `jobs=8`.
2. Run the lazy terminal-escape portfolio under the same contract as an experiment-only comparison.
3. Compare primary totals, feasibility totals, penalty, and every feasibility-only reason. Do not compare against historical goal-only solved totals as though they were primary results.

## Promotion Rule

The strict dual scorecard is a measurement-contract change, not an algorithm promotion. It is complete when the tests pass and both paired baselines are published.

No subsequent BF or CV algorithm change may be called an improvement unless it is no-regret on the strict primary scorecard at the same `alpha=10`, `t=20`, and job configuration. Feasibility-only gains remain useful diagnosis, but they do not qualify for mainline promotion.

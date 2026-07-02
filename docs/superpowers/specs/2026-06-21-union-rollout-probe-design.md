# Union Rollout Probe Design

## Goal

Improve `scheduler_state_v5_qubo_diversity_union_polish` against the new `paper_bfcv_sample10_t20` gate by adding a small, ablation-friendly FRG rollout signal to beam/QUBO selection.

The immediate target is the gap against paper-pure BS-FRG on the same 611-case sample:

- union diversity: 572/611 solved
- paper-pure BS-FRG: 606/611 solved
- BS-FRG-only solved cases: 36
- union-only solved cases: 2

The design should try to recover some of the BS-FRG-only cases without sacrificing the accepted open120 behavior.

## Key Observation

The BS-FRG-only cases are mostly solved by `FEASIBLE_BS_FRG_ROLLOUT`, not by initial FRG completion alone. Current union diversity already has initial FRG seed plumbing, but that is too weak for these cases because the decisive information is child-level completion quality.

Therefore, the missing signal is not another local QUBO coefficient. It is a small amount of long-horizon evidence:

- whether a candidate child can be completed by FRG,
- how low its rollout `final_n` is when incomplete,
- how many moves the rollout needs when feasible,
- whether the candidate leads to a dead end quickly.

## Scope

Add a new opt-in profile:

`scheduler_state_v5_qubo_diversity_union_rollout_polish`

It extends `scheduler_state_v5_qubo_diversity_union_polish` and enables a budgeted rollout probe before union diversity QUBO selection.

The probe must remain a selector signal:

- It must not generate crane moves for the main solver.
- It must not replace the compact beam search.
- It must not return a BS-FRG solution as fallback.
- It must not run on every candidate.
- It must be fully ablation-friendly and disabled by default outside the new profile.

## Candidate Sampling

At each eligible beam layer, inspect a bounded subset of ranked child candidates.

Default candidate pool:

- use only candidates still inside the current ranked child list,
- cap probe count with `union_rollout_probe_max_candidates`,
- prefer a mix of top-ranked candidates and union-diverse candidates,
- use the existing `best_n + n_window` focus used by union diversity,
- cache by compact layout code so repeated candidates do not rerun FRG completion.

Initial defaults should be conservative:

- `union_rollout_probe_enabled = True`
- `union_rollout_probe_max_candidates = 16`
- `union_rollout_probe_time_limit = 0.003`
- `union_rollout_probe_max_steps = 512`
- `union_rollout_probe_feasible_bonus = 0.30`
- `union_rollout_probe_final_n_weight = 0.08`
- `union_rollout_probe_moves_weight = 0.002`
- `union_rollout_probe_dead_end_penalty = 0.10`

These defaults are intended for a first ablation, not as a final tuned setting.

## Scoring Integration

Extend union diversity features with optional rollout fields:

- `rollout_feasible`
- `rollout_final_n`
- `rollout_moves`
- `rollout_status`

Then modify the linear term of union diversity QUBO:

- subtract `feasible_bonus` when rollout reaches `N=0`,
- add `final_n_weight * normalized_final_n`,
- add `moves_weight * normalized_moves`,
- add `dead_end_penalty` for immediate cycle/dead-end statuses.

The pairwise diversity penalties remain unchanged for the first version. If this helps paper cases but hurts open120, tune probe eligibility before adding more pair terms.

## Telemetry

Add CSV fields:

- `union_rollout_probe_enabled`
- `union_rollout_probe_calls`
- `union_rollout_probe_candidates`
- `union_rollout_probe_feasible`
- `union_rollout_probe_time_sec`
- `union_rollout_probe_cache_hits`
- `union_rollout_probe_selected_total`

The key diagnostic is whether probe-feasible candidates are actually selected and whether selected candidates lie on solved paths.

## Tests

Focused unit tests should cover:

- rollout probe cache avoids duplicate FRG completion for the same layout code,
- feasible rollout lowers union feature energy,
- dead-end rollout increases union feature energy,
- selector telemetry increments when probe is enabled,
- the new profile is opt-in and the old union profile remains unchanged.

## Experiment Plan

1. Focused residual run:
   - the 36 BS-FRG-only cases,
   - the 2 union-only cases,
   - the 39 current union failures.

2. Paper sample gate:
   - `paper_bfcv_sample10_t20`
   - compare against both current union and paper-pure BS-FRG.

3. Open120 guard:
   - run `open120_t10`
   - reject if valid count drops below the accepted union result or if full penalty regresses materially.

4. Decision:
   - promote only if paper sample improves without losing open120.
   - keep as research profile if it helps paper sample but damages open120.

## Non-Goals

- Do not implement a portfolio between union and BS-FRG.
- Do not use BS-FRG as a terminal fallback.
- Do not add family-specific gates for BF14, BF16, BF30, BF32, or CV 10-10.
- Do not replace the existing promotion suite.

# Current Best Algorithm Note

Date: 2026-07-10
Version: v4.3

## Recommended Mainline

The promoted default variant is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
```

Both Open120 and paper BF/CV benchmark runners now use this variant when
`--variant` is omitted.

## Full Gate

The promotion decision gives Open120, BF, and CV equal solved-count priority.
All runs use `--time-limit 120 --no-business-cap --jobs 8`; full paper runs use
`--sample-per-family 9999`.

| Surface | Solved | Moves | Average runtime |
|---|---:|---:|---:|
| Open120 | 120/120 | 3,759 | 6.252s |
| BF | 721/721 | 45,906 | 6.512s |
| CV | 840/840 | 55,865 | 3.081s |
| Combined gate | 1,681/1,681 | 105,530 | - |
| Real30 regression | 30/30 | 525 | 2.446s |

## Why v4.3 Is Promoted

v4.3 preserves complete coverage and materially reduces the Open120 path total.
It selects one solver profile before search:

- small DLTS-style `5x7/25` and `7x7/35` layouts use the Wang pair-QUBO plus
  successor-cache route;
- general and paper layouts use the chain-aware QUBO beam with the bounded v4.2
  short-incumbent / BS-FRG compression profile;
- source semantics, priority duplication, layout shape, and initial pressure
  protect shape-overlapping paper CV cases from taking the wrong route.

This is a deterministic pre-solve routing policy, not a failure fallback and not
a dual-run portfolio. Validator replay remains the only final acceptance gate.

## Baselines

- v4.0 `...chainaware_polish` remains the single-profile ablation baseline.
- v4.2 `...short_incumbent_v42_polish` remains the path-compression baseline.
- `scheduler_state_v5_wang_rescue_successor_cache_polish` remains the short
  Open120 route used by v4.3 where the routing predicate permits it.
- Pure BS-FRG and paper Wang BS-B remain paper references, not defaults.

## Evidence

- `docs/experiments/v43_wang_reintegration_20260709/v43_wang_route_open120_t120_all_dlts_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_wang_route_paper_bf_cv_t120_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_real30_mainline_branch_t120_summary.md`
- [v4.3 mechanism and reproduction](v43-mainline-algorithm.md)

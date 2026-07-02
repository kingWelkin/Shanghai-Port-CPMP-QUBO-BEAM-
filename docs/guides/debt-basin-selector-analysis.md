# Debt/Basin Selector Analysis - 2026-06-29

## Decision

Do not promote:

```text
qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

The profile is useful as a diagnostic ablation, but the first full paper sample10 gate regressed against the current mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

The next algorithm pass should split route-quality guards from basin-escape mechanisms. A single broad debt/basin selector preference is too blunt.

## Evidence Sources

- Hard-case trace: `docs/experiments/paper_hard_case_trace_report_20260629.md`
- Debt/basin hard smoke: `docs/experiments/debt_basin_hard_20260629_summary.md`
- Debt/basin open smoke: `docs/experiments/debt_basin_open_smoke_20260629_summary.md`
- Debt/basin paper sample10: `docs/experiments/debt_basin_paper_sample10_t20_20260629_summary.md`
- Raw trace CSVs:
  - `docs/experiments/debt_basin_hard_20260629.csv`
  - `docs/experiments/debt_basin_hard_frontier_20260629.csv`
  - `docs/experiments/debt_basin_paper_sample10_t20_20260629.csv`

## What Was Tested

The ablation adds four selector signals inside the existing compact beam/QUBO path:

- `receiver_debt`: penalty for consuming scarce receiver slack without enough progress;
- `escape_potential`: bonus for opening future reduction space;
- `basin_label`: coarse structural route class used for pair diversity;
- `reduction_chain_depth`: bounded FRG-style reduction-chain child telemetry and cost.

The profile does not return a BS-FRG fallback path. Generated children enter the normal candidate pool and still need validator replay.

## Paper Sample10 Gate

Current accepted union mainline:

| profile | scope | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|
| `union_diversity` | BF | 401 | 383 | 336 | 1043373 | 15.703 |
| `union_diversity` | CV | 210 | 189 | 165 | 14927 | 6.577 |
| `union_diversity` | BF+CV | 611 | 572 | 501 | 1058300 | 12.567 |

Debt/basin result:

| profile | scope | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|
| `debt_basin` | BF | 401 | 379 | 179 | 1079574 | 14.218 |
| `debt_basin` | CV | 210 | 189 | 153 | 16281 | 8.108 |
| `debt_basin` | BF+CV | 611 | 568 | 332 | 1095855 | 12.118 |

Delta versus current union:

| scope | feasible solved delta | primary accepted delta | full penalty delta | avg runtime sec delta |
|---|---:|---:|---:|---:|
| BF | -4 | -157 | +36201 | -1.485 |
| CV | 0 | -12 | +1354 | +1.531 |
| BF+CV | -4 | -169 | +37555 | -0.449 |

Interpretation:

- The profile is slightly faster on aggregate, but the strict primary scorecard collapses.
- BF damage is the main regression, especially because many recovered paths are feasibility-only or over cap/time.
- CV feasible count is preserved, but primary accepted count and penalty still regress.
- Faster average runtime is not enough to compensate for primary accepted and penalty damage.

## Seven-Case Hard Smoke

Debt/basin hard smoke summary:

| scope | cases | primary solved | feasibility | over cap | over time | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 7 | 2 | 5 | 3 | 0 | 1860 | 15.507 |
| BF | 2 | 0 | 2 | 2 | 0 | 921 | 20.352 |
| CV | 5 | 2 | 3 | 1 | 0 | 939 | 13.569 |

Case-level takeaways:

- BF14 and BF32 reached `final_n=0` only as feasibility-only rows with very long paths: 946 and 747 moves.
- CV 5-8 controls stayed primary when the compact route was already good.
- CV 6-6 still showed cap risk.
- CV 10-6 and CV 10-10 remained unsolved under the debt/basin profile.

This confirms that debt/basin telemetry can expose route classes, but it does not yet solve the structural hard cases in a promotion-safe way.

## Open Smoke

The open smoke was intentionally small:

| family | cases | valid | invalid/unsolved | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `5x5_cv1_test` | 5 | 4 | 1 | 213 | 3.920 |
| `5x5_cv2_test` | 5 | 5 | 0 | 146 | 4.459 |

This is only a guard sample. It cannot replace the full open120 gate.

## Mechanism Interpretation

The hard-case trace before the profile still gives the strongest explanation:

- BF14/BF32 are not merely selector-weight failures. The compact route does not reproduce short BS-FRG recovery, and long FRG seed paths are not quality wins under the strict scorecard.
- CV 5-8 / 6-6 demonstrate path debt. A local continuation signal can consume receiver slack and create a later cleanup burden.
- CV 10-6 is a low-n dead-basin failure: the search can lower `n`, but the retained lineage does not complete.
- CV 10-10 is closer to candidate-universe failure: large candidate pools still barely lower `n`.

The first debt/basin implementation mixed these situations under one broad selector preference. That explains why it can help diagnosis while damaging the paper sample gate.

## Next Research Direction

Prefer a split design:

1. **Route-quality guard:** apply receiver-debt only as a near-tie veto when the compact route is already making clean progress.
2. **Basin-escape trigger:** activate extra basin diversity only when trace metrics show lineage collapse or no meaningful `n` reduction.
3. **Short recovery-chain supplement:** generate a tiny number of recovery children only when candidate-universe failure is detected, then keep them inside the normal selector.
4. **Strict seed accounting:** treat long FRG seed paths as feasibility evidence, not primary quality wins, unless they pass business cap and time flags.

Promotion still requires a paired open120 run and separate paper BF / paper CV sample10 rows.

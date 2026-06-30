# Union FRG Evaluated Gated Report - 2026-06-30

## Change Summary

Profile: `scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish`

This experiment keeps `scheduler_state_v5_qubo_diversity_union_polish` as the baseline and adds a guarded BS-FRG/rollout-evaluator layer:

- keeps 5-stack open layouts on the baseline QUBO width (`union_diversity_qubo_max_vars=48`);
- keeps rollout probe candidate count at `32`;
- enables rollout evaluator only when `stack_count >= 7`;
- keeps FRG frontier child injection only when `stack_count >= 7`;
- keeps BS-FRG beam seed for large/high layouts.

The key fix is structural rather than case-specific: small open layouts no longer let a feasible rollout completion replace local beam energy.

## Open120 Paired Result

Command pair:

- Baseline: `docs/experiments/union_baseline_paired_open120_t10_20260630.csv`
- New profile: `docs/experiments/union_frg_evaluated_gated_open120_t10_20260630.csv`

| Variant | Solved | Moves | Final N Sum | Avg Runtime |
| --- | ---: | ---: | ---: | ---: |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 107/120 | 3208 | 324 | 6.180s |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_evaluated_polish` | 120/120 | 3765 | 0 | 7.082s |

Rescued cases:

- `5x5_cv1_test_4`
- `5x5_cv2_test_10`
- `7x5_cv1_test_2`
- `7x5_cv1_test_9`
- `7x5_cv1_test_11`
- `7x5_cv1_test_12`
- `7x5_cv1_test_18`
- `7x5_cv2_test_2`
- `7x5_cv2_test_9`
- `7x5_cv2_test_11`
- `7x5_cv2_test_12`
- `7x5_cv2_test_18`
- `7x5_cv3_test_11`

Among the 107 cases both variants solved, the new profile adds only 32 moves total.

## Paper Smoke Result

Command output:

- `docs/experiments/union_frg_evaluated_gated_paper12_t10_smoke_20260630.csv`

Result:

| Dataset Smoke | Solved | Moves | Final N Sum | Avg Runtime |
| --- | ---: | ---: | ---: | ---: |
| BF14/BF32/CV5-5/CV10-6 sample-3 | 12/12 | 2616 | 0 | 8.671s |

This preserves the paper-set gain from BS-FRG beam seed. Full BF/CV verification is still required before using this as a final benchmark claim.

## Seed Time Check

Open120 with `--frg-seed-time-limit 5`:

- `docs/experiments/union_frg_evaluated_gated_seed5_open120_t10_20260630.csv`

| Seed Limit | Solved | Moves | Final N Sum | Avg Runtime |
| --- | ---: | ---: | ---: | ---: |
| 10s default | 120/120 | 3765 | 0 | 7.082s |
| 5s override | 120/120 | 3770 | 0 | 6.983s |

The 5s override is promising for open120, but it was not promoted because the paper runner does not currently expose `frg_seed_time_limit`, so paper12 has only been verified with the 10s default.

## Interpretation

The improvement comes from using BS-FRG as a robust large-layout recovery source while protecting small open layouts from completion-score overreach.

The failed earlier versions mixed three harmful effects:

- wide QUBO selection (`96` vars) on 5-stack layouts;
- feasible rollout completion replacing local beam energy too early;
- FRG frontier child injection touching layouts where baseline already had a stable path.

The gated version removes those harms without using per-case gates.

## Promotion Status

Promotable as a robust experimental profile after:

- open120 paired run: passed;
- focused tests: passed;
- paper smoke: passed.

Not yet a final default replacement because:

- full BF/CV benchmark is not rerun yet;
- runtime is higher than baseline union;
- seed5 is open120-validated but not paper-validated.

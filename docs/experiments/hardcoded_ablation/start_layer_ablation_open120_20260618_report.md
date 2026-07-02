# QUBO Start Layer Ablation, Open120

Date: 2026-06-18

Base variant: `qubo_scheduler_state_v5_wang_rescue_hardness_routed_polish`

Scope:

- Families: `5x5_cv1_test`, `5x5_cv2_test`, `5x5_cv3_test`, `7x5_cv1_test`, `7x5_cv2_test`, `7x5_cv3_test`
- Cases: first 20 per family, 120 total
- Time limit: 10 seconds per case
- Jobs: 8
- Parameter under test: `--qubo-start-layer`
- Wang pair-QUBO start layer was kept at the baseline value, `0`

## Commands

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 10 --jobs 8 --label start_layer_ablation_open120_20260618 --only base,qubo_start_layer_8
```

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 10 --jobs 8 --label start_layer_ablation_open120_q4_q12_20260618 --only qubo_start_layer_4,qubo_start_layer_12
```

## Combined Result

Combined summary: `docs/experiments/hardcoded_ablation/start_layer_ablation_open120_combined_20260618_summary.csv`

| ablation | valid | invalid | moves_total | full_quality_penalty_total | runtime_avg | qubo_calls | group_qubo_calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 112 | 8 | 3379 | 4513 | 5.513 | 7959 | 1090 |
| qubo_start_layer_4 | 114 | 6 | 3482 | 4255 | 5.290 | 7795 | 1027 |
| qubo_start_layer_8 | 112 | 8 | 3371 | 4520 | 5.521 | 7623 | 967 |
| qubo_start_layer_12 | 109 | 11 | 3272 | 4780 | 5.627 | 7208 | 887 |

## Case-Level Delta, `qubo_start_layer_4` vs Base

`qubo_start_layer_4` loses two cases that the base solves:

- `5x5_cv2_test_17`: base `27`, start-layer-4 unsolved
- `7x5_cv1_test_2`: base `40`, start-layer-4 unsolved

It gains four cases that the base misses:

- `7x5_cv2_test_11`: start-layer-4 `43`
- `7x5_cv2_test_18`: start-layer-4 `45`
- `7x5_cv2_test_9`: start-layer-4 `46`
- `7x5_cv3_test_11`: start-layer-4 `41`

Move improvements among cases solved by both, abs delta >= 2:

- `7x5_cv1_test_5`: `34 -> 31`
- `7x5_cv3_test_18`: `45 -> 42`
- `7x5_cv3_test_3`: `36 -> 34`

## Interpretation

The best global open120 setting in this sweep is `--qubo-start-layer 4`: it improves solved count from `112/120` to `114/120` and reduces full quality penalty from `4513` to `4255`.

The gain is not uniform. Layer 4 mostly helps hard 7x5 cases by delaying generic QUBO and group-QUBO until the early deterministic/Wang-guided beam has formed a stronger search frontier. It also hurts a small number of cases that need earlier generic QUBO guidance.

`--qubo-start-layer 8` is too conservative for coverage: it saves QUBO calls and slightly reduces solved-case moves, but does not improve `112/120`. `--qubo-start-layer 12` is too late and drops to `109/120`.

Do not delay Wang pair-QUBO by default. Sentinel tests showed delayed or disabled Wang pair-QUBO hurts coverage, so the safer direction is to keep Wang pair-QUBO active from layer 0 and route only generic QUBO/group-QUBO start timing.

## Recommendation

Promote `--qubo-start-layer 4` as a candidate routed setting, not yet as a blind global replacement. A generalized gate should keep baseline early generic QUBO for cases resembling `5x5_cv2_test_17` and `7x5_cv1_test_2`, while using layer-4 delayed generic QUBO for higher-pressure 7x5 cases.

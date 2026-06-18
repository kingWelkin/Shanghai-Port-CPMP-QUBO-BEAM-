# V5 Internal Parameter Ablation T20

Date: 2026-06-18

## Scope

This run keeps the single-route baseline `qubo_scheduler_state_v5_wang_rescue_polish`. It does not use profile fallback, portfolio selection, or failure-triggered reruns.

## Commands

Generic QUBO start-layer sweep:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --base-variant qubo_scheduler_state_v5_wang_rescue_polish --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 20 --jobs 8 --label v5_internal_qstart_open120_t20_20260618 --only base,qubo_start_layer_4,qubo_start_layer_8,qubo_start_layer_12
```

Wang pair-QUBO width/vars sweep:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --base-variant qubo_scheduler_state_v5_wang_rescue_polish --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 20 --jobs 8 --label v5_internal_widths_open120_t20_20260618 --only wang_pair_width_8,wang_pair_width_16,wang_pair_max_vars_32,rescue_qubo_width_32,rescue_qubo_width_64,rescue_max_vars_64
```

The second command stopped at the first rescue flag before CLI pass-through was added, after completing the three Wang pair-QUBO variants. Rescue variants were then run with:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --base-variant qubo_scheduler_state_v5_wang_rescue_polish --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 20 --jobs 8 --label v5_internal_rescue_open120_t20_20260618 --only rescue_qubo_width_32,rescue_qubo_width_64,rescue_max_vars_64
```

## Result Summary

Baseline for comparison is `v5_internal_qstart_open120_t20_20260618_base.csv`.

| ablation | valid | invalid | full quality penalty | delta penalty | moves total | delta moves | avg runtime | delta runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 118 | 2 | 3857 | 0 | 3676 | 0 | 5.688 | 0.0% |
| `rescue_max_vars_64` | 118 | 2 | 3857 | 0 | 3676 | 0 | 4.885 | -14.1% |
| `qubo_start_layer_12` | 118 | 2 | 3858 | +1 | 3677 | +1 | 5.108 | -10.2% |
| `rescue_qubo_width_32` | 118 | 2 | 3859 | +2 | 3678 | +2 | 5.408 | -4.9% |
| `wang_pair_max_vars_32` | 118 | 2 | 3860 | +3 | 3685 | +9 | 4.959 | -12.8% |
| `qubo_start_layer_4` | 118 | 2 | 3860 | +3 | 3679 | +3 | 5.191 | -8.7% |
| `qubo_start_layer_8` | 117 | 3 | 3908 | +51 | 3648 | -28 | 5.938 | +4.4% |
| `wang_pair_width_16` | 117 | 3 | 3926 | +69 | 3636 | -40 | 5.241 | -7.9% |
| `rescue_qubo_width_64` | 117 | 3 | 3928 | +71 | 3638 | -38 | 5.202 | -8.5% |
| `wang_pair_width_8` | 114 | 6 | 4127 | +270 | 3564 | -112 | 4.838 | -14.9% |

## Interpretation

No tested internal parameter improves solve rate beyond 118/120. The best clean change is speed-only:

- `rescue_max_vars_64` preserves valid count, full quality penalty, and solved moves.
- It reduces average runtime from `5.688s` to `4.885s`.
- It keeps Wang pair-QUBO active from layer 0 and does not shrink its supplement width.

Do not promote these as quality improvements:

- Delaying generic QUBO (`qubo_start_layer_4/12`) can save time, but it shifts moves and slightly worsens penalty.
- Shrinking Wang pair-QUBO width damages coverage.
- Increasing rescue supplement width to 64 damages coverage.

## Code Follow-Up

Added `scheduler_state_v5_wang_rescue_fast_polish` / `qubo_scheduler_state_v5_wang_rescue_fast_polish` as a named speed candidate. It is the same v5 route with only:

```text
rescue_max_qubo_vars = 64
```

Smoke check:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10,17 --variant qubo_scheduler_state_v5_wang_rescue_fast_polish --time-limit 20 --alpha 10 --jobs 4 --output docs\experiments\scratch_v5_fast_polish_residual_5x5_cases_20260618.csv
```

Result: 4/6 valid on the residual slice, with the same two residual failures as baseline (`5x5_cv1_test_4`, `5x5_cv2_test_10`).

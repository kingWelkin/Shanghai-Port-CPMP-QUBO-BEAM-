# Adaptive FRG Guarded Fallback Open120 T20

Date: 2026-06-18

## Goal

Test whether ablation should include adaptive route changes, not only fixed parameter perturbations. The candidate keeps `qubo_scheduler_state_v5_wang_rescue_polish` as the baseline and runs `qubo_scheduler_state_v5_frg_compound_guarded_polish` only when the baseline row is invalid under the open-runner validation.

## Implementation

- New open-runner variant: `qubo_scheduler_state_v5_frg_guarded_fallback_polish`
- Default ablation entry: `adaptive_frg_guarded_fallback`
- Selection rule: run v5 first; if valid, keep it; if invalid, run FRG guarded and select with `_select_open_no_regret_row`.

## Commands

Focused residual check:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10,17 --variant qubo_scheduler_state_v5_frg_guarded_fallback_polish --time-limit 20 --alpha 10 --jobs 4 --output docs\experiments\scratch_v5_frg_guarded_fallback_residual_5x5_cases_20260618.csv
```

Open120 ablation run:

```powershell
$env:PYTHONPATH='.;src;scripts'; python scripts\ablate_hardcoded_params.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --time-limit 20 --jobs 8 --label adaptive_frg_guarded_fallback_open120_t20_20260618 --only adaptive_frg_guarded_fallback
```

## Results

| run | valid | invalid | solved moves | full quality penalty | selected avg runtime | portfolio avg runtime |
|---|---:|---:|---:|---:|---:|---:|
| v5 baseline t20 rerun | 118 | 2 | 3677 | 3858 | 6.314 | 6.314 |
| adaptive FRG guarded fallback t20 | 119 | 1 | 3697 | 3793 | 4.991 | 5.043 |

Baseline reference: `docs/experiments/scratch_open_parallel_jobs16_open120_t20_20260618_summary.md`

Adaptive result:

- Summary: `docs/experiments/hardcoded_ablation/adaptive_frg_guarded_fallback_open120_t20_20260618_adaptive_frg_guarded_fallback_summary.md`
- CSV: `docs/experiments/hardcoded_ablation/adaptive_frg_guarded_fallback_open120_t20_20260618_adaptive_frg_guarded_fallback.csv`

Fallback triggered on 3/120 cases:

| case | selected profile | valid | moves | full quality penalty |
|---|---|---:|---:|---:|
| `5x5_cv1_test_4` | `scheduler_state_v5_frg_compound_guarded_polish` | 0 |  | 96 |
| `5x5_cv1_test_6` | `scheduler_state_v5_frg_compound_guarded_polish` | 1 | 38 | 38 |
| `5x5_cv2_test_10` | `scheduler_state_v5_frg_compound_guarded_polish` | 1 | 27 | 27 |

## Conclusion

This is a real solve-rate improvement over the current t20 v5 rerun: 118/120 -> 119/120, with lower full quality penalty. It is also not a per-case patch: the gate is "baseline invalid", and the specialist ran on 3 cases in the open120 run.

The remaining hard case is still `5x5_cv1_test_4`. A direct Wang BS-B final-N solution exists from earlier probing, but it exceeded the current business cap, so it cannot be counted as a valid open-runner solution without a separate cap-safe compression or different validation-aware repair.

## Verification

```powershell
$env:PYTHONPATH='.;src;scripts'; python -m pytest tests/test_qubo.py -q -p no:cacheprovider --basetemp .tmp_pytest_full_adaptive_fallback
```

Result: `231 passed in 0.79s`.

# Wang2017 Guarded Task Continuation Design

## Goal

Reduce the damage from the all-layer Wang2017 task supplement and test whether a small guarded version can improve the current `scheduler_state_v5_wang_rescue_polish` baseline.

## Evidence

The first task-guided profile triggered 1342 times and inserted 5364 candidates on open-120, solving only 19/120 versus 115/120 for the rescue-polish baseline. It did not solve any of the five baseline residual cases, so the next probe must be conservative.

## Design

Add `scheduler_state_v5_wang2017_task_guarded_polish`, inheriting from `scheduler_state_v5_wang_rescue_polish`. It keeps the task selector enabled but narrows it to:

- `wang2017_task_supplement_width=1`
- `wang2017_task_n_window=0`
- `wang2017_task_focus_n=-1`
- a 7-layer plateau gate based on recent best-N history
- a tight depth-slack gate using `rescue_cap_slack`

The existing aggressive `scheduler_state_v5_wang2017_task_polish` remains unchanged as an ablation profile. The guarded profile gets its own open benchmark variant, `qubo_scheduler_state_v5_wang2017_task_guarded_polish`.

## Measurement

Run the five baseline residual cases first:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 7x5_cv1_test 7x5_cv2_test --case-ids 4,10,9,14,12 --variant qubo_scheduler_state_v5_wang2017_task_guarded_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_task_guarded_residual_t10.csv
```

Then run full open-120 against the already recorded rescue-polish baseline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang2017_task_guarded_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_task_guarded_open120_t10.csv
```

Promotion requires at least no regression against `115/120` solved and a lower full-quality penalty, or one additional residual case solved without introducing new failures.

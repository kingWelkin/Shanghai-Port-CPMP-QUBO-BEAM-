# Wang2017 Task-Guided Continuation Design

## Goal

Add one experimental Wang2017-inspired task-guided continuation profile and run full open DLTS benchmark evidence against the current `qubo_scheduler_state_v5_wang_rescue_polish` line.

## Scope

This is not a full Feasibility-Based Heuristic reproduction. It implements a compact-search supplement that ranks legal one-move successors with a Wang2017-style task tuple:

- tier-protection indicator,
- estimated task movement pressure,
- stable-height loss,
- affected demand / future feasibility pressure,
- target stack stable height,
- negative moved group value.

The selector only adds already legal compact child states. It does not splice external Wang paths, train a model, or change the default profile.

## Integration

Add a new compact profile `scheduler_state_v5_wang2017_task_polish` extending `scheduler_state_v5_wang_rescue_polish`. Add open-runner variant `qubo_scheduler_state_v5_wang2017_task_polish`.

The selector runs after existing Wang/unified supplements and before generic QUBO supplements, like the feasibility escape selector. It records active layers, calls, added candidates, and aggregate selected task signals.

## Validation

Use TDD for:

- task feature extraction from a parent and legal child state,
- supplement ranking preference,
- profile/open-runner surface,
- CSV telemetry serialization.

Run focused tests, syntax checks, then full open-120 benchmark with:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang2017_task_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_task_open120_t10.csv
```

Run the same command with `qubo_scheduler_state_v5_wang_rescue_polish` as baseline unless a fresh same-condition baseline already exists in this turn.

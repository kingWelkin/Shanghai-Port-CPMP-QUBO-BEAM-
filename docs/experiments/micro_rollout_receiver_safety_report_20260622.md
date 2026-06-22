# Micro-Rollout Receiver-Safety Report - 2026-06-22

## Scope

Base profile remains:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Candidate profile:

```text
qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish
```

The experiment is scoring-only. QUBO still selects among legal generated child states. No rollout tail is grafted into the solution, no solver fallback is added, and the current mainline profile keeps the new flags disabled.

## Implemented Profiles

```text
qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish
qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish
qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish
qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish
```

New profile flags default off on the mainline:

```text
micro_rollout_enabled=False
micro_rollout_trace_only=False
receiver_safety_enabled=False
```

## Current Implementation Check

- Two-step legal micro probe is bounded by `micro_first_child_limit` and `micro_second_child_limit`.
- Receiver safety is a soft linear energy adjustment, not a veto.
- Deterministic core and QUBO pair penalties are unchanged.
- Candidate trace uses existing `--frontier-trace-output`.
- Trace rows now include micro and receiver fields:
  - `micro_first_children_count`
  - `micro_second_children_count`
  - `micro_best_n`
  - `micro_best_delta_n`
  - `micro_has_reduce`
  - `micro_has_safe_continuation`
  - `receiver_safe_receivers_before`
  - `receiver_safe_receivers_after`
  - `receiver_safe_delta`
  - `receiver_free_slots_before`
  - `receiver_free_slots_after`
  - `receiver_height_pressure_after`
  - `receiver_last_buffer_consumed`
  - `micro_linear_adjustment`

## Verification

Focused tests:

```text
20 passed, 306 deselected
```

Command:

```powershell
$env:PYTHONPATH='.'; $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; python -m pytest tests\test_qubo.py -k "micro or receiver or qubo_diversity_union or frontier_trace_candidate_rows" -q --basetemp .tmp_pytest_micro_trace_all
```

Compile check:

```powershell
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

Trace smoke:

```text
docs/experiments/micro_trace_feature_smoke_20260622.csv
docs/experiments/micro_trace_feature_smoke_open1_20260622.csv
docs/experiments/micro_trace_feature_smoke_open1_20260622_summary.md
```

The trace smoke produced 264 candidate rows. 10 rows carried nonzero micro/receiver feature values. In trace-only mode, `micro_linear_adjustment=0.0`, so trace output does not alter selection.

## Paired Results

### Open120, t20, jobs8

Artifacts:

```text
docs/experiments/micro_rollout_pair_baseline_open120_t20_jobs8_20260622_rerun.csv
docs/experiments/micro_rollout_pair_candidate_open120_t20_jobs8_20260622_rerun.csv
```

| profile | valid | full penalty | solved moves | gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| union baseline | 118/120 | 3868 | 3687 | 345 | 6.527 |
| micro rollout | 118/120 | 3874 | 3693 | 351 | 6.480 |

Per-case direction:

```text
better solved cases: 13
worse solved cases: 14
new solved cases: 0
lost solved cases: 0
```

Largest improvements:

```text
5x5_cv1_test_0: 33 -> 27
7x5_cv1_test_15: 40 -> 37
7x5_cv1_test_5: 34 -> 31
7x5_cv2_test_2: 44 -> 41
```

Largest regressions:

```text
7x5_cv2_test_9: 44 -> 52
7x5_cv1_test_9: 46 -> 52
7x5_cv2_test_0: 36 -> 39
7x5_cv1_test_6: 30 -> 33
```

### Paper BF/CV Sample10, t20, jobs8

Artifacts:

```text
docs/experiments/micro_rollout_pair_baseline_paper_bfcv_sample10_t20_20260622.csv
docs/experiments/micro_rollout_pair_candidate_paper_bfcv_sample10_t20_20260622.csv
```

| profile | cases | solved | full penalty | moves | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| union baseline | 611 | 568 | 1059514 | 38217 | 14.308 |
| micro rollout | 611 | 567 | 1059806 | 38497 | 14.519 |

The candidate also misses the historical promotion gate from the plan:

```text
paper sample10 target: solved >= 572, full penalty <= 1058300
candidate result: solved 567, full penalty 1059806
```

## Diagnosis

The signal is real but too weak and too blunt for promotion.

On open120, micro rollout improves several cases but hurts a nearly equal number. The net result is worse by 6 penalty and 6 moves. This means the two-step N-reduction signal sometimes identifies useful continuation, but it also rewards locally promising candidates that break the longer same-route progress chain.

On paper BF/CV sample10, the signal is more harmful. It slightly increases runtime and loses one solved case versus the paired baseline. The regressions concentrate in CV mid/large structures where a shallow two-step improvement is not enough evidence that the receiver state remains globally usable.

The current `has_safe_continuation` is still mostly an N-improvement proxy. It does not strongly prove receiver safety. Receiver scoring is soft and useful for trace, but not yet discriminative enough to protect paper-set layouts.

## Verdict

Do not promote:

```text
qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish
```

Keep it as an ablation and trace profile.

Mainline remains:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

## Next Direction

The next useful step is not a stronger weight. That would likely amplify both wins and regressions.

Prefer one of these conservative follow-ups:

1. Use micro rollout as a tie-break only among near-equal QUBO linear energies.
2. Keep receiver safety scoring, but disable `w_micro_delta_n` on paper-like high-pressure layouts.
3. Add trace-driven analysis on the large regressions before tuning:
   - `7x5_cv2_test_9`
   - `7x5_cv1_test_9`
   - `CV/5-8/data5-8-5`
   - `CV/5-9/data5-9-10`
   - `CV/6-6/data6-6-8`

Working principle:

```text
Micro rollout should break ties between comparable legal candidates.
It should not override the current union route when the baseline already has strong sustained progress.
```

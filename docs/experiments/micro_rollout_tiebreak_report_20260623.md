# Micro Rollout Tie-Break Experiment Report - 2026-06-23

## Decision

Keep `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` as an
opt-in experimental profile. Do not promote it over
`qubo_scheduler_state_v5_qubo_diversity_union_polish`.

The profile is useful as evidence that micro rollout can rescue a few paper
cases without harming open120 feasibility, but it is not a net quality win yet.

## Profile

Base:

- `qubo_scheduler_state_v5_qubo_diversity_union_polish`

Additional opt-in switches:

- `micro_rollout_enabled=True`
- `receiver_safety_enabled=True`
- `micro_rollout_tiebreak_only=True`
- `micro_rollout_tiebreak_epsilon=0.10`

Mechanism:

- Compute the micro rollout linear adjustment for union diversity QUBO
  candidates.
- Do not apply the adjustment globally.
- Apply it only to candidates whose base linear energy is within epsilon of the
  current best candidate energy.
- Keep all crane move generation and beam selection structure unchanged.

## Verification

Unit and compile checks:

- `python -m pytest tests\test_qubo.py::test_micro_rollout_tiebreak_only_applies_adjustment_only_to_near_equal_energy tests\test_qubo.py::test_solver_profile_for_micro_rollout_tiebreak_is_opt_in -q --basetemp .tmp_pytest_tiebreak_green2`
  - `2 passed`
- `python -m pytest tests\test_qubo.py -k "micro or receiver or qubo_diversity_union or frontier_trace_candidate_rows or wang_all" -q --basetemp .tmp_pytest_tiebreak_all`
  - `23 passed, 305 deselected`
- `python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py`
  - passed

## Paired Benchmarks

### open120, t20, jobs8

Baseline:

- CSV: `docs/experiments/tiebreak_open120_baseline_t20_jobs8_20260623.csv`
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_polish`

Candidate:

- CSV: `docs/experiments/tiebreak_open120_candidate_t20_jobs8_20260623.csv`
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish`

Result:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| solved | 118 / 120 | 118 / 120 | 0 |
| moves | 3681 | 3681 | 0 |
| full penalty | 3862 | 3862 | 0 |
| gap sum | 339 | 339 | 0 |
| avg runtime sec | 7.405 | 7.550 | +0.145 |

Per-case movement:

- Better solved cases: 10
- Worse solved cases: 5
- Gained solved cases: 0
- Lost solved cases: 0

Net: open120 is protected, but there is no aggregate improvement.

### paper BF+CV sample10, t20, jobs8

Baseline:

- CSV: `docs/experiments/tiebreak_paper_sample10_baseline_t20_jobs8_20260623.csv`
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_polish`

Candidate:

- CSV: `docs/experiments/tiebreak_paper_sample10_candidate_t20_jobs8_20260623.csv`
- Variant: `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish`

Result:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| solved | 569 / 611 | 571 / 611 | +2 |
| moves | 38868 | 40415 | +1547 |
| full penalty | 1059125 | 1059272 | +147 |
| avg runtime sec | 13.643 | 13.746 | +0.103 |

Main gains:

- `BF|BF14|cpmp_16_8_103_21_78_1`: unsolved to solved, 946 moves
- `BF|BF32|cpmp_20_8_128_52_96_3`: unsolved to solved, 747 moves
- `CV|6-6|data6-6-8`: 404 moves to 52 moves

Main regressions:

- `CV|5-8|data5-8-10`: 43 moves to 169 moves
- `CV|5-8|data5-8-1`: 45 moves to 106 moves
- `CV|5-9|data5-9-2`: 45 moves to 62 moves

Net: solved count improves, but path quality and full penalty get worse.

## Interpretation

The tie-break gate is conservative enough to avoid open120 damage, but it still
changes local ordering in fragile paper CV families. It can rescue deep BF cases
and one severe CV/6-6 route, yet the same signal overvalues some long feasible
paths in CV/5-8 and CV/5-9.

The likely failure mode is not infeasible move generation. It is ranking drift:
micro rollout sees a near-term safe continuation, but the selected state later
commits to a long cleanup route. This is exactly why the signal should stay as a
near-tie experiment rather than a main linear term.

## Next Direction

The next useful version should keep the current opt-in profile and add a quality
guard around the micro adjustment:

- Apply the tie-break only when the micro candidate also improves projected
  path length or final-n risk.
- Suppress the micro adjustment on CV-like high-height-pressure cases where the
  rollout path length expands sharply.
- Compare against this report with the same paired open120 and paper sample10
  commands.

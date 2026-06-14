# Wang/QUBO Full-Case Quality Trial Summary

Date: 2026-06-14

## Scope

This trial evaluates Wang-inspired modules against the current compact QUBO/scheduler line using full-case quality. Unsolved or invalid cases are penalized instead of being excluded from move totals.

Implemented trial hooks:

- `scheduler_state_v5_wang_rescue`: scheduler-state rescue with Wang pair-QUBO supplement active in rescue layers.
- `wang_all` open benchmark variant group: current `qubo`, `scheduler_state_v5`, Wang pair-QUBO, Wang rescue, and Wang hybrid.
- Portfolio full-quality fields: valid/final-N JSON, per-profile penalties, and full-case summary ranking.
- Parent-chain guard for Wang rollout supplements: do not reinsert a code already reached at an equal or shallower depth.

Penalty used in summaries:

- Valid case: actual move count.
- Invalid/unsolved case: `max(reference_or_cap13, n0) + 4 * n0`.

## Internal 30 Single-Bay Cases

Command:

```powershell
python scripts\benchmark_qubo_no_regret_portfolio_n0.py --output docs\experiments\wang_qubo_full_quality_internal30_unlimited.csv --no-portfolio-stop-on-baseline-fail
```

Full-quality summary:

| profile | valid | invalid | penalized moves | runtime sec |
|---|---:|---:|---:|---:|
| scheduler_state_v5_wang_rescue | 30/30 | 0 | 508 | 198.020 |
| scheduler_state_v5 | 30/30 | 0 | 509 | 55.974 |
| stable | 30/30 | 0 | 509 | 58.567 |
| scheduler_state_v6 | 30/30 | 0 | 509 | 59.867 |
| scheduler_state_v5_window | 30/30 | 0 | 509 | 60.521 |
| wang_incumbent | 30/30 | 0 | 509 | 92.371 |
| scheduler_state_v5_wang_pair | 30/30 | 0 | 509 | 94.085 |
| scheduler_state_v5_wang_hybrid | 30/30 | 0 | 509 | 120.357 |

Internal conclusion:

- Wang rescue improves only 3 Bay26 cases by 1 move each, for a total 508 vs 509.
- That one-move global gain costs about 3.5x runtime versus `scheduler_state_v5`.
- For internal deployment quality, keep `scheduler_state_v5`/`speed_v2` as the main line; use Wang rescue only as an optional quality rescue, not as the default.

3-second-per-variant pressure run:

```powershell
python scripts\benchmark_qubo_no_regret_portfolio_n0.py --output docs\experiments\wang_qubo_full_quality_internal30.csv --time-limit 3 --portfolio-time-limit-per-variant 3 --no-portfolio-stop-on-baseline-fail
```

Under this tight cap, `scheduler_state_v5` and `stable` solve 19/30, while Wang-heavy modules solve 15/30. Wang modules are not good tight-time defaults.

## Open DLTS 5x5/7x5 Test 120

Command:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant wang_all --time-limit 10 --output docs\experiments\open_wang_all_full_quality_5x5_7x5_test120.csv
```

Aggregate summary:

| variant | valid | invalid | full penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|
| qubo_scheduler_state_v5 | 89/120 | 31 | 6328 | 2850 | 389 | 1.856 |
| qubo_scheduler_state_v5_wang_pair | 94/120 | 26 | 5984 | 3024 | 421 | 2.086 |
| qubo_scheduler_state_v5_wang_hybrid | 93/120 | 27 | 6043 | 3002 | 420 | 2.532 |
| qubo_scheduler_state_v5_wang_rescue | 117/120 | 3 | 3944 | 3654 | 341 | 3.379 |

Wang classical BS-B reference on the same 120 cases:

```powershell
python scripts\benchmark_classical_cpmp_baselines.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --solvers wang_bs_b --time-limit 10 --max-depth 200 --beam-width 100 --max-candidates-per-layer 64 --output docs\experiments\wang_bs_b_open_5x5_7x5_test120.csv
```

| solver | valid | invalid | full penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|
| wang_bs_b | 120/120 | 0 | 4569 | 4569 | 1174 | 1.663 |

Open conclusion:

- Wang BS-B remains the strongest pure feasibility fallback: 120/120 solved.
- Wang BS-B solution quality is much worse than QUBO variants on solved cases: 4569 moves and gap sum 1174.
- Wang rescue is the best integrated QUBO improvement for full-case quality: 117/120 solved, penalty 3944, and gap sum 341.
- Pair-QUBO alone improves coverage from 89 to 94, but does not close most hard cases.
- Per-case portfolio over QUBO variants reaches 118/120, with winners: Wang rescue 74, scheduler_state_v5 34, current qubo 8, Wang pair 4.
- Per-case portfolio including Wang BS-B reaches 120/120, with 11 cases selected from Wang BS-B; total moves 3737 and gap sum 342.

## Recommendation

Use a two-stage policy:

1. Default: `scheduler_state_v5` or current `speed_v2` for internal cases and tight runtime budgets.
2. Open/hard-case rescue: trigger `scheduler_state_v5_wang_rescue` under plateau/high-risk signals.
3. Last fallback: run Wang BS-B only when QUBO variants fail, then accept it for feasibility but keep QUBO result when both are valid and QUBO is shorter.

Do not make Wang rescue unconditional on internal cases. Its solve-rate gain appears on open hard cases, while internal cases mostly pay runtime for a one-move global gain.

## Follow-Up Probes

The next Wang/QUBO probes are intentionally separated:

- `scheduler_state_v5_wang_features` moves fixed-prefix and fix-pair signals into the generic group-QUBO linear score.
- `scheduler_state_v5_wang_rescue_polish` keeps Wang rescue as a feasibility mechanism, then uses suffix/window polish to reduce long rescue paths.
- `scripts/diagnose_wang_qubo_failures.py` focuses on `5x5_cv1_test_4` and `5x5_cv1_test_6`, comparing compact plateau signals with Wang BS-B early moves.

These probes should be judged by full-case quality first: valid count, invalid count, full-quality penalty, solved moves, and runtime.

## Unified Wang-QUBO Beam Probe

Profile/variant names:

- Compact profile: `scheduler_state_v7_wang_qubo_unified`.
- Open benchmark variant: `qubo_scheduler_state_v7_wang_qubo_unified`.

This is a single-loop compact profile: Wang fixed-prefix and fix-pair terms are scored inside the compact QUBO frontier selector. It is not Wang fallback, rescue, or post-polish chaining.

Smoke commands:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 2 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 3 --output docs\experiments\scratch_unified_wang_qubo_smoke.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 10 --output docs\experiments\scratch_unified_wang_qubo_5x5_cv1_20.csv
```

Smoke outcomes:

| command | valid | invalid/unsolved | full penalty | solved moves | avg runtime sec | reference gap | QUBO calls | QUBO time sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-case, 3s | 0/2 | 2 | 188 | 0 | 3.063 | 0 | 9 | 0.010 |
| 20-case, 10s | 1/20 | 19 | 1848 | 21 | 9.993 | 2 | 452 | 0.507 |

Promotion should depend on full-case quality, not on smoke completion alone: valid count, invalid count, full-quality penalty, solved moves, reference gap, runtime, and unified QUBO counters.

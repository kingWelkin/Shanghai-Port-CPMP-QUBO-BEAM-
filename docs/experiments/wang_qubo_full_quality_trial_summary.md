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

## 2026-06-15 Current Consolidation

The latest curated direction summary is [wang_qubo_direction_summary.md](wang_qubo_direction_summary.md). Use that file as the current navigation point for what is good, what remains worth testing, and what should not be promoted.

Current open DLTS 5x5/7x5 test-120 ranking:

| algorithm | valid | full penalty | solved moves | solved gap sum | avg runtime sec | status |
|---|---:|---:|---:|---:|---:|---|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 | best single QUBO-family quality result |
| `qubo_scheduler_state_v5_wang_rescue_suffix_polish` | 118/120 | 3880 | 3699 | 357 | 3.883 | useful speed ablation, not promoted |
| `qubo_scheduler_state_v5_wang_rescue` | 118/120 | 3887 | 3706 | 364 | 3.551 | strong, slightly less polished |
| `qubo_scheduler_state_v5_wang_features` | 105/120 | 5162 | 3368 | 455 | 2.749 | useful ablation, not promoted |
| `qubo_scheduler_state_v5` | 89/120 | 6334 | 2856 | 395 | 1.941 | fast baseline, weak hard-case coverage |
| `wang_bs_b` | 120/120 | 4569 | 4569 | 1174 | 1.663 | best feasibility fallback, worse path quality |

Main reading:

- `wang_bs_b` is fastest and solves every case, but it pays for feasibility with long paths.
- `wang_rescue_polish` is the best single quality algorithm now: it is shorter than Wang on 109 of the 118 common solved cases and saves 804 moves on those common cases.
- `wang_rescue_suffix_polish` proves that window repair can be skipped as an ablation, but the clean full-120 run only saves about 0.018 seconds per case versus full polish while losing 7 full-penalty points.
- The remaining hard cases for the current best single QUBO-family line are `5x5_cv1_test_4` and `5x5_cv2_test_10`.
- Per-case portfolios can reach 120/120 and lower total moves, but they are an upper bound rather than the complete single algorithm requested for this line of work.

2026-06-15 research-all paired run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant wang_research_all --time-limit 10 --alpha 10 --output docs\experiments\wang_research_all_open120_t10.csv
```

This run should be interpreted as a same-run comparison because its paired `wang_rescue_polish` baseline solved 114/120, below the previous clean 118/120 result.

| algorithm | valid | full penalty | solved moves | solved gap sum | avg runtime sec | status |
|---|---:|---:|---:|---:|---:|---|
| `qubo_scheduler_state_v5_wang_rescue_selective_window_polish` | 114/120 | 4252 | 3501 | 305 | 3.923 | promising speed candidate, needs clean confirmation |
| `qubo_scheduler_state_v5_wang_rescue_polish` | 114/120 | 4254 | 3505 | 307 | 4.793 | paired baseline for this run |
| `qubo_scheduler_state_v5_wang_rescue_dynamic_polish` | 112/120 | 4339 | 3434 | 284 | 4.220 | faster but loses coverage |
| `qubo_scheduler_state_v5_wang_rescue_risk_polish` | 107/120 | 4986 | 3190 | 255 | 4.983 | negative on 7x5 robustness |

Key reading:

- Selective-window polish reduced window attempts from 217 to 146 and had the best same-run full-quality penalty, but it needs a clean single-variant confirmation before promotion.
- Dynamic survivor cap fired on 351 layers and saved runtime, but it added 4 baseline failures while solving only 2 baseline failures.
- Risk group-QUBO made 1111 group-QUBO calls and selected 22460 group candidates, but it produced too many 7x5 failures.

Clean selective-window confirmation:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_selective_window_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_selective_window_polish_open120_clean.csv
```

| algorithm | valid | full penalty | solved moves | solved gap sum | avg runtime sec | status |
|---|---:|---:|---:|---:|---:|---|
| `qubo_scheduler_state_v5_wang_rescue_selective_window_polish` | 117/120 | 3960 | 3655 | 345 | 4.543 | not promoted |

The clean run leaves `5x5_cv1_test_4`, `5x5_cv2_test_10`, and `7x5_cv2_test_12` unsolved. The `7x5_cv2_test_12` failure is a 10-second boundary case: a focused open-runner subset solved it in 44 moves, but substituting that optimistic rerun still gives only 118/120, 3880 full penalty, 3699 solved moves, and 357 gap sum. That matches the suffix-only ablation and remains behind `wang_rescue_polish`.

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
- `scripts/diagnose_wang_qubo_failures.py` focuses on `5x5_cv1_test_4` and `5x5_cv2_test_10`, comparing compact plateau signals with Wang BS-B early moves.

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

| command | valid | invalid/unsolved | full penalty | solved moves | avg runtime sec | solved-case reference gap | QUBO calls | QUBO time sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2-case, 3s | 0/2 | 2 | 188 | 0 | 3.063 | 0 | 9 | 0.010 |
| 20-case, 10s | 1/20 | 19 | 1848 | 21 | 9.993 | 2 | 452 | 0.507 |

The 2-case smoke has no solved-case quality signal because it solved zero cases. Overall, this smoke is negative: the profile remains experimental and must not be promoted unless full-case quality improves over the current mainline and Wang-rescue profiles.

2026-06-15 retune:

The unified profile now keeps the QUBO supplement conservative (`unified_wang_qubo_supplement_width=1`, `unified_wang_qubo_select=1`) and adds a local BS-B-style baby-lookahead rollout inside the same compact search loop. This rollout does not call the Wang solver and does not splice Wang move sequences; it evaluates local Wang giant children and inserts only the first legal baby move as a compact child.

Pressure-run commands:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 3 --output docs\experiments\scratch_unified_baby_lookahead_open120_t3.csv
```

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_unified --time-limit 3 --no-wang-rollout-enabled --output docs\experiments\scratch_unified_no_rollout_open120_t3.csv
```

| scope | rollout | valid | invalid/unsolved | full penalty | solved moves | avg runtime sec |
|---|---|---:|---:|---:|---:|---:|
| open 120, 3s | off | 47/120 | 73 | 9864 | 1272 | 2.099 |
| open 120, 3s | baby lookahead | 55/120 | 65 | 9388 | 1554 | 2.124 |
| 5x5 cv1, 3s | off | 10/20 | 10 | 1273 | 286 | 1.298 |
| 5x5 cv1, 3s | baby lookahead | 13/20 | 7 | 1082 | 381 | 1.311 |

The gain is solve-rate and full-penalty oriented, not path compression. On open 120 it adds 8 solved cases and lowers full penalty by 476 with only about 0.025 seconds more average runtime, but solved move total rises because the newly solved cases are nontrivial. It is therefore useful as integrated feasibility pressure, while suffix/window compression remains a separate future question.

Promotion should depend on full-case quality, not on smoke completion alone: valid count, invalid count, full-quality penalty, solved moves, reference gap, runtime, and unified QUBO counters.

2026-06-15 candidate-pool probe:

`scheduler_state_v7_wang_qubo_pool` keeps Wang baby-lookahead inside the ordinary candidate pool. It marks candidates preferred by the Wang baby-lookahead move and exposes that marker as `unified_wang_rollout_bonus` in the unified QUBO linear energy. It disables the post-selection rollout append, so this is the stricter "QUBO chooses the Wang-preferred candidate" variant.

Command:

```powershell
python scripts\open_cpmp_benchmarks.py --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v7_wang_qubo_pool --time-limit 3 --output docs\experiments\scratch_unified_baby_pool_open120_t3.csv
```

| scope | variant | valid | invalid/unsolved | full penalty | solved moves | avg runtime sec | qcalls |
|---|---|---:|---:|---:|---:|---:|---:|
| open 120, 3s | no rollout | 47/120 | 73 | 9864 | 1272 | 2.099 | 6069 |
| open 120, 3s | baby lookahead append | 55/120 | 65 | 9388 | 1554 | 2.124 | 5670 |
| open 120, 3s | candidate-pool QUBO bonus | 45/120 | 75 | 10044 | 1220 | 2.203 | 5905 |
| 5x5 cv1, 3s | baby lookahead append verify | 13/20 | 7 | 1083 | 382 | 1.325 | 918 |
| 5x5 cv1, 3s | candidate-pool QUBO bonus | 11/20 | 9 | 1207 | 324 | 1.387 | 1007 |

This probe is negative for promotion. The Wang-preferred one-move child is usually already present among ordinary successors, so the useful behavior of the previous baby-lookahead rollout is not candidate discovery; it is a controlled one-step continuation after the frontier selector. Turning that signal into only a QUBO linear bonus improves over a naive unmarked pool, but it underperforms both no-rollout and post-selection baby-lookahead on full-case quality. Keep `scheduler_state_v7_wang_qubo_pool` as an ablation profile only.

## Feasibility Escape And DLTS Feature Export Probe

`scheduler_state_v5_feasibility_escape_polish` extends the current `scheduler_state_v5_wang_rescue_polish` line with a bounded feasibility escape selector. It uses deterministic state-feasibility, stability, temporary-slack, and dead-end-risk signals. It does not call original DLTS, train a neural model, or splice external move paths into compact parent chains.

The first focused commands are:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test --case-ids 4,10 --variant qubo_scheduler_state_v5_feasibility_escape_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_feasibility_escape_residual.csv
python scripts\benchmark_compact_beam_qubo_n0.py --cases A1_01 --solver-profile scheduler_state_v5_feasibility_escape_polish --time-limit 1 --dlts-feature-output docs\experiments\scratch_dlts_features_A1_01.csv --output docs\experiments\scratch_feasibility_escape_A1_01.csv
```

Promotion requires a full-case quality improvement against `qubo_scheduler_state_v5_wang_rescue_polish`, or solving one residual hard case without adding new failures in a clean paired run.

The 2026-06-15 residual smoke solved 2/4 selected cases with full penalty 237 and solved moves 56; the original residual hard cases `5x5_cv1_test_4` and `5x5_cv2_test_10` remained unsolved.

Gate diagnostic follow-up:

The first residual smoke showed `feasibility_escape_calls=0`, so the selector was configured but not actually participating. The blocking gate was not depth slack; it was the parent-top-share requirement, which suppresses plateau layers when the candidate pool is already broadly spread across parents. Removing that hard parent-share requirement made the residual run produce nonzero telemetry (`5x5_cv1_test_4`: 8 active layers, 8 calls, 32 added candidates; `5x5_cv2_test_10`: 3 active layers, 3 calls, 12 added candidates), but the hard cases still ended at `final_n=17` and `final_n=15`.

Wider CLI sweeps also did not change those two hard cases:

| setting | 5x5_cv1_test_4 | 5x5_cv2_test_10 |
|---|---:|---:|
| width 12, n-window 4, no signal filter | final_n 17 | final_n 15 |
| width 24, n-window 6, no signal filter | final_n 17 | final_n 15 |
| width 24, n-window 6, slack filter | final_n 17 | final_n 15 |

This makes the current feasibility escape selector useful as telemetry and ablation scaffolding, but not as a quality improvement. Further work should not spend effort only widening this supplement; it needs a different continuation mechanism or a stronger learned/ranked branching policy.

## Wang2017 Task-Guided Continuation Probe

`scheduler_state_v5_wang2017_task_polish` extends `scheduler_state_v5_wang_rescue_polish` with a deterministic task-guided continuation supplement inspired by Wang 2017. The implemented signal infers one top move between compact parent and child states, then ranks candidate continuations by stable-tier protection, estimated task movement pressure, stable-prefix loss, feasibility pressure, target stack stable height, and moved-group value.

This is not a full FBH reproduction. It is an ablation-friendly insertion of task-priority signals into the existing compact beam frontier.

Full open-120 commands:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang2017_task_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang2017_task_open120_t10.csv
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 10 --alpha 10 --output docs\experiments\scratch_wang_rescue_polish_open120_t10.csv
```

| variant | solved | invalid/unsolved | full penalty | avg runtime sec | task calls | task added |
|---|---:|---:|---:|---:|---:|---:|
| Wang rescue polish baseline | 115/120 | 5 | 4149 | 4.770 | 0 | 0 |
| Wang2017 task polish | 19/120 | 101 | 11709 | 10.794 | 1342 | 5364 |

Family-level result:

| family | baseline solved | task solved | baseline penalty | task penalty |
|---|---:|---:|---:|---:|
| 5x5_cv1_test | 19/20 | 6/20 | 651 | 1537 |
| 5x5_cv2_test | 19/20 | 4/20 | 604 | 1576 |
| 5x5_cv3_test | 20/20 | 9/20 | 510 | 1228 |
| 7x5_cv1_test | 18/20 | 0/20 | 928 | 2504 |
| 7x5_cv2_test | 19/20 | 0/20 | 790 | 2454 |
| 7x5_cv3_test | 20/20 | 0/20 | 666 | 2410 |

The probe is strongly negative for promotion. It triggered often and inserted many candidates, but the extra feature scoring consumed runtime and displaced better baseline frontier choices. The 7x5 families are especially clear: the baseline solved 57/60, while the task-guided supplement solved 0/60.

Keep this profile as an experimental ablation only. The next useful direction is not a wider all-layer task supplement; it should be a gated or learned continuation policy that only fires on diagnosed plateaus and is measured against the same full-case quality metrics.

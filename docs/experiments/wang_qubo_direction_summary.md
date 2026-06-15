# Wang/QUBO Direction Summary

Date: 2026-06-15

This note is the current navigation page for the Wang-inspired CPMP work. It separates the strongest current result, promising next experiments, and directions that were tested but should not be promoted.

## Current Best Result

For the open DLTS 5x5/7x5 test-120 suite, the best single QUBO-family algorithm is:

| algorithm | solved | full-quality penalty | solved moves | solved gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |
| `qubo_scheduler_state_v5_wang_rescue_suffix_polish` | 118/120 | 3880 | 3699 | 357 | 3.883 |
| `qubo_scheduler_state_v5_wang_rescue` | 118/120 | 3887 | 3706 | 364 | 3.551 |
| `qubo_scheduler_state_v5_wang_features` | 105/120 | 5162 | 3368 | 455 | 2.749 |
| `qubo_scheduler_state_v5` | 89/120 | 6334 | 2856 | 395 | 1.941 |
| `wang_bs_b` | 120/120 | 4569 | 4569 | 1174 | 1.663 |

Interpretation:

- `wang_bs_b` is still the strongest pure feasibility baseline: it solves all 120 cases and is fast.
- `qubo_scheduler_state_v5_wang_rescue_polish` is the best single integrated algorithm for quality: it leaves 2 cases unsolved, but its solved paths are much shorter.
- On the 118 cases solved by both `wang_rescue_polish` and `wang_bs_b`, QUBO-rescue-polish uses 3692 moves versus Wang's 4496, saving 804 moves. QUBO-rescue-polish is shorter on 109 cases; Wang is shorter on 5; 4 are tied.
- The two remaining unsolved cases for `wang_rescue_polish` are `5x5_cv1_test_4` and `5x5_cv2_test_10`.

## Good Directions

These are worth preserving as main research directions.

1. `rescue -> polish` as the current quality mainline.
   Wang rescue is useful as a feasibility escape mechanism, and local polish recovers part of the path-length cost. This is the best current single-algorithm structure.

2. Triggered rescue, not unconditional Wang.
   Internal 30-case evidence shows unconditional Wang rescue buys only a tiny move improvement while adding substantial runtime. The rescue should be plateau/high-risk triggered.

3. Wang BS-B as fallback and teacher, not as the quality objective.
   Wang's early moves solve the hard residual cases, but its complete paths are long. The best use is to study its escape prefixes and use it as a final feasibility fallback.

4. Survivor telemetry and dynamic controls.
   Beam-survivor telemetry revealed dead-parent concentration and enabled the dynamic survivor parent-cap experiment. The first small test is not enough for promotion, but the signal is useful for diagnosing wasted beam width.

5. Path compression after rescue.
   The current best result improves from rescue to rescue-polish: 3887 -> 3873 full penalty and 3706 -> 3692 solved moves. This is a small but real quality direction.

6. Suffix-only polish as a diagnostic ablation.
   Skipping window repair keeps 118/120 coverage and lands between rescue and full polish: penalty 3880, moves 3699, avg runtime 3.883s. It is useful evidence that window repair can be made more selective, but it is not enough to replace full rescue-polish.

## Worth Trying Next

These are not proven yet, but the evidence supports further experiments.

1. Run dynamic survivor parent-cap on full open120 at the same time budget as the current best.
   The 7x5_cv1 5-case t=6 probe solved 3/5 with lower full penalty than the fixed-cap small probes, but it is too small to promote.

2. Focus only on the two residual failures.
   Compare QUBO-rescue-polish against Wang BS-B on `5x5_cv1_test_4` and `5x5_cv2_test_10`, then extract the first few Wang moves that escape the plateau.

3. Make rescue compression more local and more frequent.
   Rescue finds feasibility; compression should be a bounded suffix/window repair, not a second full solver.

4. Add risk-aware survivor triggers.
   Current dynamic cap uses dead-parent ratio. A stronger trigger should combine dead-parent ratio, on-path survivor loss, parent-top-share, and `best_n` plateau.

5. Keep QUBO as a small selector inside the beam.
   The strongest evidence still supports QUBO as candidate/suffix/window selection inside a classical search, not as the whole CPMP solver.

## Tested But Not Promoted

These directions produced useful evidence, but should not be the main line now.

| direction | best observed result | reason not promoted |
|---|---:|---|
| Pair-QUBO only | 94/120, penalty 5984 | Improves coverage over scheduler v5, but misses too many hard cases. |
| Full Wang hybrid | 93/120, penalty 6043 | Heavier than pair-QUBO and worse full quality. |
| Generic Wang fixed-prefix/fix-pair features | 105/120, penalty 5162 | Better than v5, but clearly worse than rescue/rescue-polish. |
| Unified Wang-QUBO beam | 55/120 at t=3 in the better pressure run | Integrated feasibility pressure helps, but it is far behind rescue-polish. |
| Candidate-pool QUBO bonus | 45/120 at t=3 | Wang-preferred one-step child as a QUBO bonus underperformed append-style baby lookahead and no-rollout. |
| Dynamic survivor parent cap | 3/5 on the small 7x5_cv1 t=6 probe | Promising diagnostic, but only small-sample evidence; keep as an experiment switch. |
| Suffix-only rescue polish | 118/120, penalty 3880, avg 3.883s | Skips all window repair attempts, but saves only about 0.018s average versus full polish while losing 7 full-penalty points. Keep as ablation, not default. |
| Unconditional Wang rescue on internal 30 | 508 moves vs 509 for scheduler v5, much slower | Not worth making the internal default. |

## Project Hygiene

- Keep raw CSV/log scratch outputs ignored unless they become curated evidence.
- Promote only short curated summaries into git.
- The current curated Wang/QUBO entry points are this file and `wang_qubo_full_quality_trial_summary.md`.
- Treat per-case portfolios as an upper bound, not as the user's desired "single complete algorithm."

## Evidence Files

- `scratch_wang_rescue_polish_test120_current.csv`
- `scratch_wang_rescue_suffix_polish_test120_clean.csv`
- `scratch_wang_rescue_test120_current.csv`
- `scratch_wang_features_test120_current.csv`
- `scratch_scheduler_state_v5_test120_current.csv`
- `open_wang_all_full_quality_5x5_7x5_test120.csv`
- `wang_bs_b_open_5x5_7x5_test120.csv`
- `scratch_survivor_dynamic_cap_7x5_cv1_5cases_t6.csv`

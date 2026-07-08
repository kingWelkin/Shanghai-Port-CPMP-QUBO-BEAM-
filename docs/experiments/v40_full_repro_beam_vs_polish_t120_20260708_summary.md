# v4.0 Full Reproduction: Beam vs Post-Repair, t120

Date: 2026-07-08

Variant:

`qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish`

Settings:

- `--time-limit 120`
- `--jobs 8`
- `--alpha 10`
- `--no-business-cap`
- Beam-only run: `--no-post-polish-qubo`
- Repaired run: `--post-polish-qubo`

Runner note:

- The paper-set runner was patched to accept `--post-polish-qubo` / `--no-post-polish-qubo`, matching the open120 runner, so BF/CV can now be benchmarked in beam-only and repaired modes.
- Wall-clock elapsed values below are from the run orchestrator stdout. CSV runtime values are summed per-case solver runtimes, so they are larger than wall-clock time under `--jobs 8`.

## Headline

This run reproduces the v4.0 conclusion on open120 and BF, but does **not** reproduce full CV completion under t120.

| Suite | Mode | Solved | Failed | Solved moves | Full penalty | Case runtime sum | Wall-clock |
|---|---:|---:|---:|---:|---:|---:|---:|
| open120 | beam-only | 120/120 | 0 | 8,082 | 8,082 | 5.833s | 2.465s |
| open120 | repaired | 120/120 | 0 | 6,930 | 6,930 | 107.742s | 15.525s |
| BF | beam-only | 719/721 | 2 | 47,080 | 47,694 | 2,217.260s | included in paper 460.712s |
| BF | repaired | 721/721 | 0 | 46,435 | 46,435 | 4,109.671s | included in paper 852.705s |
| CV | beam-only | 839/840 | 1 | 64,955 | 65,379 | 1,119.700s | included in paper 460.712s |
| CV | repaired | 837/840 | 3 | 59,983 | 61,106 | 2,247.144s | included in paper 852.705s |
| BF+CV | beam-only | 1558/1561 | 3 | 112,035 | 113,073 | 3,336.959s | 460.712s |
| BF+CV | repaired | 1558/1561 | 3 | 106,418 | 107,541 | 6,356.815s | 852.705s |

## Post-Repair Effect

| Suite | Comparable solved cases | Improved | Worse | Same | Net move delta |
|---|---:|---:|---:|---:|---:|
| open120 | 120 | 106 | 4 | 10 | -1,152 |
| BF | 719 | 243 | 81 | 395 | -808 |
| CV | 837 | 631 | 74 | 132 | -4,320 |

Internal post-polish telemetry:

- open120: `post_polish_moves_saved_sum=1193`, `post_polish_window_time_sum=100.652s`
- BF: `post_polish_moves_saved_sum=697`, `post_polish_window_time_sum=2003.134s`
- CV: `post_polish_moves_saved_sum=5835`, `post_polish_window_time_sum=888.257s`

The repaired run strongly reduces path length where it succeeds, especially CV, but it is not a monotone offline repair of the exact same beam output. Under the same 120s budget, it gained two BF solves and lost two CV solves relative to beam-only.

## Failure Cases

Beam-only failures:

| Dataset | Family | Case | final_n | Runtime | Message |
|---|---|---|---:|---:|---|
| BF | BF5 | `cpmp_16_5_64_13_39_13` | 39 | 121.593s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=2` |
| BF | BF29 | `cpmp_20_8_128_26_77_2` | 77 | 123.107s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=1` |
| CV | 10-10 | `data10-10-15` | 80 | 121.760s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=2` |

Repaired failures:

| Dataset | Family | Case | final_n | Runtime | Message |
|---|---|---|---:|---:|---|
| CV | 10-6 | `data10-6-13` | 50 | 120.251s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=6` |
| CV | 10-10 | `data10-10-15` | 80 | 120.603s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=2` |
| CV | 10-10 | `data10-10-28` | 82 | 120.686s | `TIME_LIMIT_OR_BEAM_EXHAUSTED; compact_lift_rejected_targets=2` |

Paired solve changes:

- Gained with repaired: BF5 `cpmp_16_5_64_13_39_13` -> 46 moves; BF29 `cpmp_20_8_128_26_77_2` -> 117 moves.
- Lost with repaired: CV/10-6 `data10-6-13` was 353 moves in beam-only; CV/10-10 `data10-10-28` was 299 moves in beam-only.
- Failed in both modes: CV/10-10 `data10-10-15`.

## Largest Move Changes

Largest repaired improvements:

| Dataset | Family | Case | Beam | Repaired | Delta |
|---|---|---|---:|---:|---:|
| DLTS | 5x5_cv2_test | `5x5_cv2_test_10` | 200 | 84 | -116 |
| DLTS | 5x5_cv3_test | `5x5_cv3_test_3` | 161 | 72 | -89 |
| CV | 5-4 | `data5-4-32` | 326 | 58 | -268 |
| CV | 5-4 | `data5-4-8` | 285 | 72 | -213 |
| CV | 10-10 | `data10-10-38` | 410 | 225 | -185 |

Largest repaired regressions among cases solved in both modes:

| Dataset | Family | Case | Beam | Repaired | Delta |
|---|---|---|---:|---:|---:|
| DLTS | 5x5_cv3_test | `5x5_cv3_test_16` | 66 | 86 | +20 |
| CV | 5-4 | `data5-4-33` | 50 | 312 | +262 |
| CV | 10-6 | `data10-6-33` | 290 | 499 | +209 |
| CV | 10-6 | `data10-6-3` | 373 | 556 | +183 |
| CV | 10-10 | `data10-10-16` | 408 | 587 | +179 |

## Reproduction Judgment

Confirmed:

- open120 remains fully solved: 120/120.
- open120 path length improves substantially with post repair: 8,082 -> 6,930 moves.
- BF is fully solved with post repair: 721/721.
- BF path length improves slightly on the repaired full run, even after including the two newly solved cases.

Not confirmed:

- CV full completion is not reproduced at t120 in this paired run. Repaired CV is 837/840, not 840/840.
- Repaired mode is not monotone under t120. It lost two CV cases that beam-only solved, while gaining two BF cases.

Current interpretation:

v4.0 is still strong, but this run shows that the "post repair" configuration should be treated as a high-quality, high-cost mode rather than a strict dominance layer. For promotion or publication-quality claims, the CV failures need either a no-regression repair guard or a two-stage run that preserves the beam incumbent before attempting post repair.

## Artifacts

- Beam open120 CSV: `docs/experiments/v40_full_repro_beam_vs_polish_t120_20260708/open120_beam_only_t120.csv`
- Repaired open120 CSV: `docs/experiments/v40_full_repro_beam_vs_polish_t120_20260708/open120_repaired_t120.csv`
- Beam BF/CV CSV: `docs/experiments/v40_full_repro_beam_vs_polish_t120_20260708/paper_bfcv_full_beam_only_t120.csv`
- Repaired BF/CV CSV: `docs/experiments/v40_full_repro_beam_vs_polish_t120_20260708/paper_bfcv_full_repaired_t120.csv`
- Computed summary JSON: `docs/experiments/v40_full_repro_beam_vs_polish_t120_20260708/computed_summary.json`

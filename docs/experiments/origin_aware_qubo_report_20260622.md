# Origin-Aware Union QUBO Experiment Report

Date: 2026-06-22

## Objective

Evaluate the first proposed integration idea: make the existing `scheduler_state_v5_qubo_diversity_union_polish` selector aware of candidate origin, so QUBO beam selection can balance quality and diversity across core, Wang rollout, and FRG frontier style candidates.

This experiment is intentionally not a fallback chain. QUBO still only selects beam supplements. It does not generate moves, replace the main search, or append a rescue solver.

## Implementation

New experimental profile:

- `scheduler_state_v5_qubo_diversity_union_origin_aware_polish`

Main code changes:

- Added `origin` to `UnionDiversityQuboFeatures`.
- Added per-layer candidate origin tracking through `args._candidate_origin_by_code`.
- Marked injected Wang rollout candidates as `wang_rollout`.
- Marked injected FRG frontier child candidates as `frg_frontier_child`.
- All unmarked candidates remain `core`.
- Added two QUBO pair penalties:
  - `union_diversity_qubo_origin_similarity_penalty`
  - `union_diversity_qubo_same_origin_parent_n_band_penalty`
- Wrote both parameters to open120 CSV output.
- Registered the profile in both open120 and paper-set runners.

Default profile safety:

- `scheduler_state_v5_qubo_diversity_union_polish` keeps both origin-aware penalties at `0.0`.
- The new profile is opt-in only.

## Test Coverage

Focused tests added or exercised:

- Origin-aware profile is opt-in.
- Same origin + same parent + same n-band pair penalty affects the objective.
- Existing union diversity selector still selects quality-diverse candidates.
- Enhanced frontier still invokes the union diversity selector.
- Runner variant list accepts the new profile.

Verification commands:

```powershell
python -m pytest tests/test_qubo.py::test_union_diversity_qubo_penalizes_same_origin_parent_and_n_band tests/test_qubo.py::test_union_diversity_qubo_origin_aware_profile_is_opt_in tests/test_qubo.py::test_union_diversity_qubo_selects_quality_diverse_candidates tests/test_qubo.py::test_union_diversity_qubo_selector_is_used_inside_enhanced_frontier tests/test_qubo.py::test_open_runner_wang_all_expands_current_and_wang_variants -q --basetemp .tmp_pytest_origin_final
python -m py_compile scripts\benchmark_compact_beam_qubo_n0.py scripts\open_cpmp_benchmarks.py scripts\benchmark_compact_paper_set.py
```

Result:

- `5 passed`
- `py_compile` passed
- Pytest emitted only a Windows `.pytest_cache` permission warning.

## Open120 Results

Command for origin-aware run:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\origin_aware_open120_t20_jobs8_20260622.csv
```

Paired baseline command:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\union_after_origin_open120_t20_jobs8_20260622.csv
```

| Variant | Valid | Full penalty | Solved moves | Gap sum | Avg runtime |
|---|---:|---:|---:|---:|---:|
| `scheduler_state_v5_qubo_diversity_union_polish` paired rerun | 117/120 | 3929 | 3669 | 350 | 6.843s |
| `scheduler_state_v5_qubo_diversity_union_origin_aware_polish` | 118/120 | 3865 | 3684 | 342 | 7.191s |

Open120 paired delta:

- Valid: `+1`
- Full penalty: `-64`
- Gap sum: `-8`
- Runtime: slower by about `0.348s` average
- Solved moves: `+15`, mainly because origin-aware solved one additional case

Important case movement:

- Origin-aware solved `5x5_cv2_test_17`, where the paired baseline failed with `final_n=14`.
- Origin-aware still failed the two persistent residuals:
  - `5x5_cv1_test_4`
  - `5x5_cv2_test_10`

Historical comparison note:

- Earlier same-day union artifact `union_repro_open120_t20_jobs8_20260622.csv` was stronger than the paired rerun:
  - `118/120`, full penalty `3854`, solved moves `3673`, gap `331`
- Therefore the origin-aware result is better than the immediate paired rerun, but not better than the best observed union artifact from the same day.

## Paper Sample10 Results

Command:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\paper_sample10_origin_aware_t20_jobs8_20260622.csv
```

| Variant | Cases | Solved | Full penalty | Solved moves | Avg runtime |
|---|---:|---:|---:|---:|---:|
| historical union `paper_sample10_union_t20_20260622.csv` | 611 | 572 | 1058435 | 42304 | 12.519s |
| origin-aware | 611 | 568 | 1059231 | 38837 | 13.872s |

Paper-set delta:

- Solved: `-4`
- Full penalty: `+796`
- Avg runtime: `+1.353s`
- Solved moves are lower, but this is not a win because fewer cases were solved.

Solved-to-unsolved flips versus historical union:

| Case | Baseline | Origin-aware |
|---|---:|---:|
| `BF/BF14/cpmp_16_8_103_21_78_5` | solved | unsolved |
| `BF/BF32/cpmp_20_8_128_52_96_1` | solved | unsolved |
| `BF/BF32/cpmp_20_8_128_52_96_3` | solved | unsolved |
| `BF/BF32/cpmp_20_8_128_52_96_9` | solved | unsolved |

Largest solved-case regressions:

| Case | Baseline penalty | Origin-aware penalty | Delta |
|---|---:|---:|---:|
| `CV/5-9/data5-9-6` | 53 | 185 | +132 |
| `CV/5-10/data5-10-6` | 41 | 169 | +128 |
| `CV/5-9/data5-9-7` | 44 | 169 | +125 |
| `CV/5-9/data5-9-1` | 45 | 169 | +124 |
| `CV/5-9/data5-9-3` | 41 | 159 | +118 |
| `CV/5-9/data5-9-8` | 45 | 159 | +114 |
| `CV/5-9/data5-9-5` | 45 | 148 | +103 |

Largest solved-case improvement:

| Case | Baseline penalty | Origin-aware penalty | Delta |
|---|---:|---:|---:|
| `BF/CV4/CV4_7` | 116 | 38 | -78 |

## Diagnosis

The origin-aware signal is real but too blunt.

On open120 it can help escape one baseline failure by preventing the beam from over-concentrating on same-origin candidates. That is why `5x5_cv2_test_17` becomes solved in the paired run.

On the paper set it damages larger and lower-slack layouts. In these cases, the search often needs repeated candidates from the same origin or same local mechanism to keep pushing along a productive route. Penalizing same-origin similarity globally interrupts those chains. The damage shows up in two forms:

- Baseline-solved large BF cases become unsolved.
- CV 5-9 and 5-10 solved cases stay solved but acquire much longer paths and much higher full penalty.

This means origin diversity should not be treated as a universal beam principle. It is useful only when same-origin concentration is genuinely crowding out distinct progress, not when the solver is already following a necessary deep line.

## Decision

Do not promote `scheduler_state_v5_qubo_diversity_union_origin_aware_polish` to mainline.

Keep it as an ablation profile because:

- It improved the immediate paired open120 rerun.
- It exposed a meaningful feature: origin concentration matters.
- It provides a useful trace point for future QUBO selector design.

Do not use it as default because:

- It loses 4 solved cases on paper sample10.
- It worsens paper full penalty by `+796`.
- It is slower on both evaluated sets.
- It does not solve the two persistent open120 residuals.

## Recommended Next Step

Try a softer successor, not a stronger gate stack:

1. Disable pure same-origin penalty:
   - Set `union_diversity_qubo_origin_similarity_penalty = 0.0`.
2. Keep only the composite penalty:
   - Same origin + same parent + same n-band.
3. Reduce composite strength:
   - Sweep `0.05`, `0.10`, `0.20`, `0.35`.
4. Add quality-margin protection:
   - Apply origin penalty only when two candidates have near-equal normalized linear energy.
5. Evaluate first on:
   - open120 t20/jobs8
   - paper sample10 t20/jobs8
   - targeted BF14, BF32, CV 5-9, CV 5-10 slices

The design principle should be:

> Diversity should break ties among comparable candidates, not override a strong repeated-progress route.

## Artifacts

- `docs/experiments/origin_aware_smoke_open6_t10_20260622.csv`
- `docs/experiments/origin_aware_smoke_open6_t10_20260622_summary.md`
- `docs/experiments/origin_aware_open120_t20_jobs8_20260622.csv`
- `docs/experiments/origin_aware_open120_t20_jobs8_20260622_summary.md`
- `docs/experiments/union_after_origin_open120_t20_jobs8_20260622.csv`
- `docs/experiments/union_after_origin_open120_t20_jobs8_20260622_summary.md`
- `docs/experiments/paper_sample10_origin_aware_t20_jobs8_20260622.csv`
- `docs/experiments/paper_sample10_origin_aware_t20_jobs8_20260622_summary.md`

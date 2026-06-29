# Current Best Algorithm Note

Date: 2026-06-29

This note resolves a recurring ambiguity: "best" depends on the benchmark surface.

## Short Answer

For the current open120 plus paper BF/CV promotion gate, the strongest default is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

For a clean paired baseline, use:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

For the historical Shanghai Port internal 30-case fast path, `speed_v2` remains useful regression context, but it is not the current open/paper promotion gate.

## Why The Mainline Changed From Wang Rescue

The older best-algorithm note treated `qubo_scheduler_state_v5_wang_rescue_polish` as the strongest single-profile quality line because it beat earlier open120 rows. The current decision now includes the paper BF/CV sample gate, and the diversity-union line has the stronger combined evidence.

Open DLTS 5x5/7x5 test-120:

| profile | valid | full penalty | solved moves | gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 118/120 | 3858 | 3677 | 335 | 4.442 |

Paper BF/CV sample10:

| profile | scope | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---|---:|---:|---:|---:|---:|
| `union_diversity` | BF | 401 | 383 | 336 | 1043373 | 15.703 |
| `union_diversity` | CV | 210 | 189 | 165 | 14927 | 6.577 |
| `union_diversity` | BF+CV | 611 | 572 | 501 | 1058300 | 12.567 |

The important part is not only the combined BF+CV total. BF and CV must both stay visible because they fail for different reasons.

## What Counts As A Promotion

A new profile should replace the mainline only after a paired comparison against `qubo_scheduler_state_v5_qubo_diversity_union_polish` shows:

- no open120 valid-count regression;
- no material open120 quality regression;
- no paper BF sample10 solved-count or penalty regression;
- no paper CV sample10 solved-count or penalty regression;
- focused tests for profile routing and telemetry;
- trace evidence explaining the improvement.

One residual-case rescue, one smoke run, or a feasibility-only long path is not enough.

## Latest Non-Promotion

The 2026-06-29 debt/basin profile:

```text
qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

is not promoted. It added useful telemetry and a concrete diagnostic profile, but its paper sample10 gate regressed:

| profile | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `union_diversity` | 611 | 572 | 501 | 1058300 | 12.567 |
| `debt_basin` | 611 | 568 | 332 | 1095855 | 12.118 |

Use it for diagnosis, especially receiver-debt and basin-lineage analysis. Do not use it as the default.

## Practical Command Choice

Current open/paper mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_open120_t10.csv
```

Paper sample gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_paper_sample10_t20.csv
```

Clean paired baseline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_wang_rescue_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\baseline_wang_rescue_open120_t10.csv
```

## See Also

- [Current mainline algorithm](current-mainline-algorithm.md)
- [Algorithm profile status](algorithm-profile-status.md)
- [Debt/Basin selector analysis](debt-basin-selector-analysis.md)

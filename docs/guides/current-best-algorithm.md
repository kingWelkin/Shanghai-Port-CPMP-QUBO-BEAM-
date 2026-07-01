# Current Best Algorithm Note

Date: 2026-07-01

This note resolves the current "best algorithm" ambiguity across benchmark surfaces.

## Short Answer

For the equal-weighted open120 plus paper BF/CV promotion gate, the current promoted mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

For open120 alone, the best full-coverage path-quality reference in the current batch is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

For a clean historical paired baseline, use:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

## Why v2 Credit Is Promoted

The previous mainline, `qubo_scheduler_state_v5_qubo_diversity_union_polish`, keeps shorter open paths on the cases it solves, but it fails two open120 cases and is much weaker on the paper BF/CV sample1 gate.

The macro-edge profile is slightly better than v2 on open120 path quality, but v2 is substantially better on the paper set. Since open120 and BF/CV are now treated as equally important, v2 is the better mainline.

Open DLTS 5x5/7x5 test-120, `t20`, 2026-07-01:

| profile | valid | full penalty | solved moves | gap sum |
|---|---:|---:|---:|---:|
| `union_polish` | 118/120 | 3858 | 3677 | 335 |
| `macro_edge_qubo_polish` | 120/120 | 4100 | 4100 | 705 |
| `macro_credit_v2_polish` | 120/120 | 4105 | 4105 | 710 |

Paper BF/CV sample1, `t20`, 2026-07-01:

| profile | primary solved | feasibility solved | moves | full penalty |
|---|---:|---:|---:|---:|
| `union_polish` | 26/62 | 56/62 | 1904 | 10699 |
| `macro_edge_qubo_polish` | 51/62 | 56/62 | 2363 | 5289 |
| `macro_credit_polish` | 50/62 | 56/62 | 2152 | 5898 |
| `macro_credit_v2_polish` | 56/62 | 56/62 | 2638 | 4437 |

## What Counts As A Future Promotion

A new profile should replace v2 only after a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

Minimum evidence:

- no open120 valid-count regression;
- no material open120 quality regression versus v2 and the macro-edge open reference;
- no paper BF solved-count or penalty regression;
- no paper CV solved-count or penalty regression;
- focused tests for profile routing, scoring, and telemetry;
- trace evidence explaining whether the gain came from generation, selection, or lineage survival.

One residual-case rescue, one smoke run, or a feasibility-only long path is not enough.

## Practical Command Choice

Current open/paper mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_macro_credit_v2_open120_t20.csv
```

Paper sample gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_macro_credit_v2_paper_sample1_t20.csv
```

Open120 path-quality reference:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\reference_macro_edge_open120_t20.csv
```

## See Also

- [Current mainline algorithm](current-mainline-algorithm.md)
- [Algorithm profile status](algorithm-profile-status.md)

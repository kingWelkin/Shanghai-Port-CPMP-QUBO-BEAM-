# Single-Frontier FRG Union Decision

## Scope

Profile: `scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish`.

The profile merges FRG descendants before ordinary beam/QUBO selection. It keeps the fixed beam width and does not append a separate FRG frontier after selection.

## Mechanism Verification

Focused `10-10` traces confirm the intended data flow.

- `selected_count` never exceeded the configured width of 1200.
- FRG children were visible as `candidate_origin=frg_frontier_child` while retaining the ordinary `selected_source` value.
- On `data10-10-1`, 356 of the top traced candidates were FRG-origin and selected by the core path.
- On `data10-10-2`, the corresponding count was 322.
- The two cases reached layer-best `N=71` under the 20-second diagnostic run, but neither reached `N=0`.

## Verification

`python -m pytest tests\\test_qubo.py -q --basetemp .tmp_pytest_single_frontier_full`:

- 317 passed.
- The only warning was the existing pytest cache permission warning for the workspace `.pytest_cache` path.

`python -m py_compile scripts\\benchmark_compact_beam_qubo_n0.py scripts\\open_cpmp_benchmarks.py` completed successfully.

## Paper CV Sample2, 20 Seconds Per Case

| Profile | Solved | Moves | Full penalty | Avg runtime sec |
|---|---:|---:|---:|---:|
| Union graft | 39/42 | 2066 | 2814 | 6.568 |
| Append adaptive frontier | 40/42 | 1898 | 2409 | 6.065 |
| Single-frontier union | 40/42 | 1980 | 2474 | 4.913 |

The single-frontier profile rescued `data5-4-1` and retained `6-6-1`, while the two `10-10` cases remained unsolved. It is faster than the adaptive append profile and has the same solved count, but it is 65 penalty worse.

Artifact: `docs/experiments/paper_cv_sample2_union_frg_single_frontier_t20_jobs2_20260622.csv`.

## Paired Open120, 10 Seconds Per Case, Jobs 8

| Profile | Valid | Moves | Full penalty | Gap sum | Avg runtime sec |
|---|---:|---:|---:|---:|---:|
| Light frontier paired baseline | 119/120 | 3821 | 3907 | 452 | 4.927 |
| Single-frontier union | 119/120 | 3839 | 3925 | 470 | 4.594 |

Both profiles failed the same case, `5x5_cv1_test_10`, at `final_n=15`. This fresh paired baseline differs from the earlier historical 120/120 artifact, so the paired comparison is the valid decision evidence for this change.

Across the 119 jointly valid cases, the single-frontier profile was 18 moves worse. It used fewer QUBO calls but merged more FRG children, which reduced runtime without improving path quality.

Artifacts:

- `docs/experiments/union_frg_frontier_light_paired_open120_t10_jobs8_20260622.csv`
- `docs/experiments/union_frg_single_frontier_open120_t10_jobs8_20260622.csv`

## Decision

Keep `scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` as an experimental ablation and do not promote it to the default or light-frontier mainline.

The experiment establishes that append-only FRG expansion was not the only issue: once FRG children compete in the same fixed-width beam, they can dominate the core selection at certain layers. The next algorithmic hypothesis, if pursued, should be an origin-aware soft diversity term inside the existing QUBO selector. It must be designed and evaluated as a separate ablation, without a quota, fallback, or case-specific gate.

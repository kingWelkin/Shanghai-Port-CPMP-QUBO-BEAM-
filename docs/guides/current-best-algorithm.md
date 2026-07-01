# Current Best Algorithm Note

Date: 2026-07-01

This note resolves the current "best algorithm" ambiguity across benchmark surfaces.

## Short Answer

For the equal-weighted open120 plus paper BF/CV promotion gate, the current promoted mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

The previous combined-evidence baseline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

For open120-only direct macro-edge ablation, keep:

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

For a clean historical paired baseline, use:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

## Why Macro-Chain Witness Is Promoted

`macro_credit_v2` proved that future evidence matters, but it mostly exposed that evidence as shallow QUBO credit. `macro_chain_witness` goes one level deeper: it gives the selector a short recoverable chain with completion-witness evidence, then lets QUBO decide whether that chain should survive in the beam.

This is the better compromise for the current gate:

- open120 remains fully solved;
- BF/CV full-set coverage is far above the previous v2 line;
- BS-FRG is used as candidate evidence, not as a fallback solver;
- selected macro paths are replayed and validated through the normal compact-path lifting boundary.

## Evidence Snapshot

Open DLTS 5x5/7x5 test-120, no business cap, `t120`, 2026-07-01:

| profile | valid | solved moves |
|---|---:|---:|
| `macro_chain_witness_polish` | 120/120 | 6886 |

Paper BF/CV full benchmark, no business cap, `t120`, 2026-07-01:

| profile or solver | primary solved | feasibility solved | primary moves | feasible moves | avg runtime |
|---|---:|---:|---:|---:|---:|
| `macro_chain_witness_polish`, patched full aggregate | 1560/1561 | 1561/1561 | 102574 | 102637 | 9.771 |
| `macro_chain_witness_polish`, original full run before lift repair | 1558/1561 | 1559/1561 | 102176 | 102239 | 9.757 |
| pure BS-FRG reproduction | 1545/1561 | 1545/1561 | 97808 | 97808 | 19.854 |
| previous `macro_credit_v2_polish` full run | 1407/1561 | 1408/1561 | 65752 | not summarized | 11.317 |

The patched full aggregate is computed from the original full BF/CV run plus targeted replacement of the two compact-lift error rows. The remaining non-primary row is BF27, which is feasibility solved but over the primary time boundary.

## What Counts As A Future Promotion

A new profile should replace macro-chain witness only after a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

Minimum evidence:

- no open120 valid-count regression;
- no paper BF solved-count regression;
- no paper CV solved-count regression;
- path-length and penalty comparison on common-solved rows;
- focused tests for profile routing, scoring, compact-path reconstruction, and telemetry;
- trace evidence explaining whether the gain came from generation, selection, lineage survival, or path lifting.

One residual-case rescue, one smoke run, or a feasibility-only long path is not enough.

## Practical Command Choice

Current open/paper mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\mainline_macro_chain_nocap_open120_t120.csv
```

Paper BF/CV full gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish --time-limit 120 --alpha 10 --jobs 8 --no-business-cap --output docs\experiments\mainline_macro_chain_nocap_paper_bf_cv_all_t120.csv
```

## See Also

- [Current mainline algorithm](current-mainline-algorithm.md)
- [Algorithm profile status](algorithm-profile-status.md)

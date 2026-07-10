# Experiment Evidence Vault

This directory preserves generated benchmark summaries and raw artifacts. It is
an evidence archive, not the source of truth for the current default solver.

## Current Release Pointer

The promoted v4.3 variant is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
```

| Surface | Solved | Moves |
|---|---:|---:|
| Open120 | 120/120 | 3,759 |
| BF | 721/721 | 45,906 |
| CV | 840/840 | 55,865 |
| Combined | 1,681/1,681 | 105,530 |

Canonical release documents:

- [Current best decision](../guides/current-best-algorithm.md)
- [v4.3 algorithm and reproduction](../guides/v43-mainline-algorithm.md)
- [Algorithm profile status](../guides/algorithm-profile-status.md)

Primary v4.3 evidence:

- [Open120 summary](v43_wang_reintegration_20260709/v43_wang_route_open120_t120_all_dlts_j8_summary.md)
- [BF/CV full summary](v43_full_gate_20260709/v43_wang_route_paper_bf_cv_t120_j8_summary.md)
- [Real30 summary](v43_full_gate_20260709/v43_real30_mainline_branch_t120_summary.md)

## Reading Rules

- A dated report describes the conclusion at that date; it does not override
  the current v4.3 decision documents.
- Do not promote from smoke, residual-only, or patched aggregate results.
- Report Open120, BF, and CV separately before using a combined total.
- Distinguish primary accepted solutions from feasibility-only `final_n=0`.
- Preserve raw CSVs and summaries together when citing a benchmark.
- Keep paper Wang, pure BS-FRG, A*, and classic beam as references, not defaults.

Older union, Wang, FRG, v4.0-v4.2, Optuna, and failed-route reports remain here
for reproducibility. Superseded narrative guides are indexed in
[the documentation archive](../archive/README.md).

# Experiment Reports

This directory stores generated benchmark summaries, diagnostic reports, plots, and CSV artifacts. Treat Markdown summaries as decision evidence; treat large CSVs as raw artifacts that need aggregation before citation.

## Current Scorecard

The current promotion gate is three-set:

1. open120 DLTS 5x5/7x5 test-120;
2. paper BF sample10;
3. paper CV sample10.

The current mainline remains:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Latest accepted scorecard:

| profile | scope | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---|---:|---:|---:|---:|---:|
| `union_diversity` | BF | 401 | 383 | 336 | 1043373 | 15.703 |
| `union_diversity` | CV | 210 | 189 | 165 | 14927 | 6.577 |
| `union_diversity` | BF+CV | 611 | 572 | 501 | 1058300 | 12.567 |

## Latest Diagnostic Result

The 2026-06-29 debt/basin profile is diagnostic only:

```text
qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

Paper sample10 result:

| scope | cases | primary solved | feasibility | over cap | over time | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALL | 611 | 332 | 568 | 73 | 163 | 1095855 | 12.118 |
| BF | 401 | 179 | 379 | 41 | 159 | 1079574 | 14.218 |
| CV | 210 | 153 | 189 | 32 | 4 | 16281 | 8.108 |

It failed the paper sample promotion gate because primary accepted count and full penalty regressed against the current union line.

## High-Value Reports

- [Debt/Basin selector analysis](../guides/debt-basin-selector-analysis.md) - latest debt/basin verdict and interpretation.
- [paper_hard_case_trace_report_20260629.md](paper_hard_case_trace_report_20260629.md) - hard-case mechanism split that motivated debt/basin.
- [debt_basin_paper_sample10_t20_20260629_summary.md](debt_basin_paper_sample10_t20_20260629_summary.md) - raw strict scorecard summary for the debt/basin paper sample gate.
- [debt_basin_hard_20260629_summary.md](debt_basin_hard_20260629_summary.md) - seven-case hard/regression smoke.
- [debt_basin_open_smoke_20260629_summary.md](debt_basin_open_smoke_20260629_summary.md) - 10-row open smoke guard.
- [strict_hard_union_sample10_t20_20260626.md](strict_hard_union_sample10_t20_20260626.md) - strict hard-family union baseline.
- [bf_seed_cache_concurrency_smoke_20260626.md](bf_seed_cache_concurrency_smoke_20260626.md) - FRG seed cache concurrency smoke.
- [qubo_diversity_union_experiment_20260621.md](qubo_diversity_union_experiment_20260621.md) - current union promotion record.

## Reading Rules

- Do not promote from a smoke run.
- Do not compare combined BF+CV totals without also checking BF and CV separately.
- Do not treat feasibility-only `final_n=0` rows as primary accepted rows.
- Do not count long FRG seed paths as quality wins unless they satisfy the same business cap and time scorecard.
- Do not describe BS-FRG as a mainline fallback unless a future explicit fallback design is approved and validated.

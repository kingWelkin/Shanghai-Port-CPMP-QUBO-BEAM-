# Documentation Index

This index points to the current decision documents first. Older experiment reports are kept for audit, but they should not override the current scorecard.

## Current Decision Path

- [Current mainline algorithm](guides/current-mainline-algorithm.md) - current default, evidence, promotion rules, and reproduction commands.
- [Algorithm profile status](guides/algorithm-profile-status.md) - compact status table for defaults, baselines, and opt-in ablations.
- [Current best algorithm note](guides/current-best-algorithm.md) - short explanation of what "best" means for open/paper versus internal 30-case runs.
- [Debt/Basin selector analysis](guides/debt-basin-selector-analysis.md) - 2026-06-29 verdict on the latest debt/basin ablation.
- [Paper hard-case trace report](experiments/paper_hard_case_trace_report_20260629.md) - mechanism split for BF/CV hard cases.

## Experiment Entrypoints

- [Experiments README](experiments/README.md) - current experiment-map and latest artifact list.
- [QUBO diversity union experiment](experiments/qubo_diversity_union_experiment_20260621.md) - promotion record for the current union mainline.
- [Strict hard-family union baseline](experiments/strict_hard_union_sample10_t20_20260626.md) - strict BF/CV hard-set baseline before debt/basin.
- [BF seed cache concurrency smoke](experiments/bf_seed_cache_concurrency_smoke_20260626.md) - parent-process seed materialization and parallel consumption smoke.

## Mechanism Guides

- [Current QUBO beam algorithm details](guides/current-qubo-beam-algorithm.md)
- [QUBO mechanism attribution](guides/qubo-mechanism-attribution.md)
- [QUBO quality profile probes](guides/qubo-quality-profile-probes.md)
- [Trace-first open residual audit](guides/trace-first-open-residual-audit.md)
- [Wang target-guided paper audit](guides/wang-target-guided-paper-audit.md)

## Project Basics

- [Usage](guides/usage.md)
- [Data format](guides/data-format.md)
- [Development](guides/development.md)
- [Project structure](guides/project-structure.md)

## Model Notes

- [Greedy model notes](models/greedy/chen_greedy_algorithm.md)
- [MIP model notes](models/mip/mip.md)
- [Hybrid MIP workflows](models/mip/hybrid_mip_workflows.md)

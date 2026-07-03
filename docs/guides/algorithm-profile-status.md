# Algorithm Profile Status

Date: 2026-07-03
Version: v4.0

This guide separates the promoted default, still-useful baselines, and opt-in
ablation profiles for the compact CPMP / QUBO beam work.

## Decision Snapshot

Current promoted mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

Previous promoted mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

Historical clean paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

Open120-only path-quality reference:

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish` | v4.0 open120 plus full BF/CV mainline | Promoted default | Same-commit no-cap t120: open120 `120/120`, moves `6928`; paper BF/CV full1561 `1561/1561`, moves `106903`, avg runtime `3.933s`. |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish` | Previous macro-chain mainline | Demoted baseline | Strong bridge from macro credit to chain candidates, but latest v4.0 direct full rerun is cleaner and stronger on full BF/CV coverage. |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | Previous combined-evidence baseline | Demoted baseline | Useful for showing that future witness credit matters, but it remains too shallow for hard BF/CV recovery. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | Open120 macro-edge reference | Reference, not default | Useful for direct macro-edge ablation and path-quality comparison. Not the current combined-surface mainline. |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | Earlier union reference | Demoted reference | Often short on solved open120 rows, but it does not meet the full-surface coverage bar. |
| `qubo_scheduler_state_v5_wang_rescue_polish` | Clean historical baseline | Keep as baseline | Interpretable Wang plus QUBO rescue baseline, important for paired historical comparisons. |
| pure BS-FRG reproduction | Paper-algorithm reference | Reference, not default | Important source of deep recovery-chain ideas; not used as a fallback in v4.0. |
| `speed_v2` internal line | Historical internal 30-case speed line | Keep for internal regression context | Not the current open120/BF/CV promotion gate. |

## v4.0 Mechanism Summary

v4.0 is still a QUBO beam selector, not a solver portfolio:

- compact beam generates legal one-step children;
- macro-chain witness generates future-proven chain prefixes;
- unified macro supply adds more chain candidates from shifted parent offsets;
- chain-aware QUBO penalizes selecting redundant macro ancestry;
- selected macro paths are expanded and repaired before move lifting;
- validator replay decides final feasibility.

## Opt-In Ablations

| Profile | Mechanism | Status |
|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_polish` | Adds unified macro supply without chain-aware ancestry penalty | Ablation below v4.0 |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_softcredit_polish` | Adds softer unified macro future credit | Ablation below v4.0 |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_stable_quality_polish` | v4.0 plus stricter post-polish settings | Experimental quality/stability branch |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_quality_polish` | Macro-chain witness with witness-move penalty | Quality ablation |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_deep_polish` | More aggressive post-polish and bridge search | Quality ablation, slower |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | Shallow macro future credit with witness/debt/cap correction | Superseded baseline |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | Direct macro-edge QUBO selection | Reference |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | Bounded rollout score as selection signal | Ablation |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | Grafts a validated rollout-complete tail | Diagnostic; avoid promotion without paired evidence |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | Appends BS-FRG child states after normal selection | Light frontier experiment |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | Merges FRG children into the same candidate pool | Experimental |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | Receiver-debt and basin-diversity selector signals | Diagnostic |

## Promotion Rule

A new profile can replace v4.0 only after a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish
```

Minimum evidence:

- open120 DLTS no-cap t120 stays at `120/120`;
- paper BF full rows stay solved;
- paper CV full rows stay solved;
- `--sample-per-family 9999` is present for full paper BF/CV;
- path length, full penalty, and runtime are reported separately for open120,
  BF, and CV;
- trace or telemetry explains whether the gain came from candidate generation,
  QUBO selection, lineage survival, compact-path lifting, or post-polish.

## Source Evidence

- `docs/experiments/chainaware_commit411da4b_open120_t120_20260703_summary.md`
- `docs/experiments/chainaware_commit411da4b_paper_bf_cv_full1561_t120_20260703_summary.md`
- [v4.0 scorecard and reproduction](v4-mainline-scorecard-and-reproduction.md)

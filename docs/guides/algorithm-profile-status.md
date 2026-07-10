# Algorithm Profile Status

Date: 2026-07-10
Version: v4.3

This guide separates the promoted default, still-useful baselines, and opt-in
ablation profiles for the compact CPMP / QUBO beam work.

## Decision Snapshot

Current promoted mainline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
```

Opt-in quality mode for long-path CV compression:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_polish
```

Faster opt-in quality mode:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_fast_polish
```

Experimental beam-native quality mode:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_beam_quality_polish
```

Experimental v4.1 beam-feedback candidate:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_feedback_v41_polish
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

## Role Positioning

Do not collapse every strong profile into one promotion race. The current
algorithm family has three separate roles:

| Role | Profile family | Decision |
|---|---|---|
| Default solver | `...wang_route_v43_polish` | Promoted v4.3 mainline: Open120 `120/120`, BF `721/721`, CV `840/840`, combined moves `105530` |
| Single-profile baseline | `...chainaware_polish` | Previous v4.0 default; retained for clean ablation of the v4.3 routing contribution |
| Quality compression | `...chainaware_quality*_polish` | Opt-in post-compress modes for shorter paths, especially long CV paths |
| Beam-native research | `...chainaware_beam_quality_polish`, `...chainaware_feedback_v41_polish`, and related witness selectors | Experimental; useful for learning, not promoted until full paired evidence beats v4.0 without solved-count loss |

Pure BS-FRG remains a reference and teacher for deep recovery chains. It is not
the default solver and should not be described as a fallback in v4.0. The
accepted engineering split is: v4.0 finds valid solutions quickly, quality modes
compress paths afterward, and beam-native witness work explores whether that
compression quality can eventually move inside the beam.

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish` | v4.3 full-surface route | Promoted default | No-cap t120: Open120 `120/120`, moves `3759`; BF `721/721`, moves `45906`; CV `840/840`, moves `55865`; combined `1681/1681`, moves `105530`. |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_polish` | v4.0 single-profile baseline | Demoted baseline | Same-commit no-cap t120 reference: Open120 `120/120`, paper BF/CV full1561 `1561/1561`; retained for ablation. |
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
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_beam_quality_polish` | v4.0 plus beam-native multi-witness macro alternatives; BS-FRG post-compress disabled | Experimental quality branch; promote only after paired open120 + BF + CV full t120 evidence |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_feedback_v41_polish` | v4.0 plus QUBO-window beam feedback: local witness-cost comparison, source scarcity, one-layer survival credit, and receiver-debt/height-risk credit reduction; only high-confidence positive feedback affects QUBO linear energy; BS-FRG post-compress disabled | Experimental v4.1 mainline candidate; full gate on 2026-07-07 failed with open120 `120/120` but paper BF/CV `1558/1561`, so do not promote |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_polish` | v4.0 plus BS-FRG post-compress witness on long paths | Opt-in quality mode; slower, use for CV long-path compression |
| `qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_quality_fast_polish` | v4.0 plus suffix-first BS-FRG post-compress with early accept | Faster opt-in quality mode; intended for long CV paths when 35s post-compress is too costly |
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

A new profile can replace v4.3 only after a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_unified_macro_supply_chainaware_wang_route_v43_polish
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

- `docs/experiments/v43_wang_reintegration_20260709/v43_wang_route_open120_t120_all_dlts_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_wang_route_paper_bf_cv_t120_j8_summary.md`
- `docs/experiments/v43_full_gate_20260709/v43_real30_mainline_branch_t120_summary.md`
- `docs/experiments/chainaware_commit411da4b_open120_t120_20260703_summary.md`
- `docs/experiments/chainaware_commit411da4b_paper_bf_cv_full1561_t120_20260703_summary.md`
- `docs/experiments/v41_feedback_open120_t120_nocap_20260707_fullgate_summary.md`
- `docs/experiments/v41_feedback_paper_bf_cv_full1561_t120_nocap_20260707_fullgate_summary.md`
- [v4.0 scorecard and reproduction](v4-mainline-scorecard-and-reproduction.md)

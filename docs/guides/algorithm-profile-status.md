# Algorithm Profile Status

Date: 2026-07-01

This guide is the compact status table for the current compact CPMP / QUBO beam work. It separates promoted mainline behavior, clean baselines, open-only references, and opt-in ablations.

## Decision Snapshot

Current promoted mainline for open120 plus paper BF/CV work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

Clean historical paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

Open120-only full-coverage path-quality reference:

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

The v2 macro-credit line is promoted because it keeps open120 at `120/120` and improves the paper BF/CV sample1 scorecard from macro-edge's `51/62` primary solved and `5289` penalty to `56/62` primary solved and `4437` penalty.

## Promotion Rule

A new profile should not replace the current mainline unless it passes a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

Minimum evidence:

- open120 DLTS 5x5/7x5 test-120 with no new failures;
- paper BF and paper CV rows reported separately, with no meaningful solved-count or penalty regression;
- focused tests for profile switches, scoring behavior, and runner wiring;
- telemetry showing whether the change helps candidate generation, QUBO selection, or selected-lineage survival.

Do not promote a profile from a smoke run, one residual-case rescue, a CSV-only probe, or a result that improves feasibility by accepting much longer paths without paper-set evidence.

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | Current open and paper benchmark mainline | Promoted default | Open120 t20: `120/120`, full penalty `4105`, solved moves `4105`, gap sum `710`. Paper BF/CV sample1 t20: primary `56/62`, feasible `56/62`, full penalty `4437`, moves `2638`. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | Open120 full-coverage path-quality reference | Reference, not default | Open120 t20: `120/120`, full penalty `4100`, solved moves `4100`, gap sum `705`. Paper sample1 t20: primary `51/62`, feasible `56/62`, full penalty `5289`. |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | Previous mainline / short-path open reference | Demoted reference | Open120 t20: `118/120`, full penalty `3858`, solved moves `3677`, gap sum `335`. Paper sample1 t20: primary `26/62`, feasible `56/62`, full penalty `10699`. |
| `qubo_scheduler_state_v5_wang_rescue_polish` | Clean historical paired baseline | Keep as baseline | Open120 historical clean run: `118/120`, full penalty `3873`, solved moves `3692`, gap sum `350`. Stable and interpretable, but no longer the combined-evidence mainline. |
| `speed_v2` / `scheduler_state_v5` internal line | Internal 30-case speed reference | Keep for deployment regression context | Useful for the historical Shanghai Port 30-case path, but not the current open/paper promotion gate. |
| `wang_bs_b` | Classical feasibility reference | Reference or teacher, not quality mainline | Can cover more open cases in older runs, but paths are much longer. Use for diagnosis, not as the default quality solver. |

## Promoted Macro Credit v2 Interpretation

The v2 selector is a QUBO beam-selection change, not a fallback solver:

- legal child states still come from the compact beam engine;
- FRG/macro candidates enter the normal candidate pool;
- QUBO linear energy gives positive macro future credit only with witness or macro-path evidence;
- receiver debt becomes a penalty pressure, so debt alone is not rewarded;
- positive macro credit is capped.

This is the current cleanest integration of BS-FRG-inspired deep recovery evidence into the QUBO beam framework.

## Opt-In Ablations

| Profile | Mechanism | Status | Decision reason |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish` | v1 macro future credit without the v2 witness/debt/cap correction. | Superseded | Open120 full coverage, but paper sample1 only `50/62`, penalty `5898`; v2 improves both paper solved count and penalty. |
| `qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish` | Adds candidate-origin pair penalties to the union QUBO selector. | Ablation only | Helped one immediate open120 paired rerun but lost paper cases and worsened paper penalty. Origin diversity is useful only as a guarded tie concept. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | Adds bounded rollout signal to union selection. | Ablation only | Rollout signal is real, but it can rank locally promising candidates that later commit to long cleanup paths. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | Grafts a validated rollout-complete tail when available. | Ablation only | Useful for diagnosis, but grafting should not become an implicit fallback path without paired evidence. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish` | Emits micro-rollout and receiver-safety telemetry without changing selection. | Diagnostic | Safe for trace collection because trace-only mode keeps the linear adjustment at zero. |
| `qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish` | Adds receiver-safety soft score without rollout. | Ablation only | Useful as a feature probe, but not yet discriminative enough as a promotion candidate. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish` | Uses the bounded micro-rollout signal without receiver safety. | Ablation only | Keep as a component-isolation profile. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish` | Combines micro rollout and receiver-safety scoring. | Ablation only | Open120 had mixed wins and regressions; paper sample evidence was worse than the paired baseline. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` | Applies micro rollout only to near-tied QUBO candidates. | Ablation only | Protected some open120 aggregate quality, but paper-set penalty regressed. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | Appends BS-FRG-derived child states after normal frontier selection. | Light frontier experiment | Candidate support only. It is not a BS-FRG fallback solver. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | Merges FRG children into the same fixed-width candidate pool before selection. | Experimental, not promoted | FRG children can dominate core selection without improving results. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | Expands FRG frontier support on high-N plateau or small-stack pressure. | Ablation only | Keep as a controlled diagnostic for FRG support strength. |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | Adds receiver-debt, escape-potential, basin-diversity, and reduction-chain selector signals. | Diagnostic, not promoted | Earlier paper sample10 gate regressed badly: combined primary `332/611`, full penalty `1095855`. |

## Current Remaining Failure Classes

The promoted v2 line still fails several paper sample1 rows:

- `BF14`
- `BF30`
- `CV/5-4`
- `CV/6-6`
- `CV/10-6`
- `CV/10-10`

Parameter sweeps around v2 did not beat the promoted defaults:

| sweep | paper sample1 result |
|---|---|
| default v2 | `56/62`, penalty `4437` |
| `receiver_debt_penalty_weight=12` | `52/62`, penalty `5452` |
| `min_witness_quality=0.25` | `54/62`, penalty `5034` |

## Recommended Next Work

The next useful round should target the remaining failure classes without broadening fallback behavior.

1. Treat BF and CV paper rows separately in every scorecard.
2. Diagnose BF14/BF30 as time-boundary and recovery-chain selection problems.
3. Diagnose CV/10-6 and CV/10-10 as candidate-universe or lineage-collapse problems.
4. Keep macro credit v2 as the paired baseline for all successor experiments.
5. Rerun open120 plus paper BF/CV before any future promotion.

## Source Evidence

- `docs/experiments/verify_open120_t20_qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish_20260701_summary.md`
- `docs/experiments/macro_credit_v2_paper_sample1_t20_20260701_summary.md`
- `docs/experiments/verify_paper_sample1_t20_qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish_20260701_summary.md`
- `docs/experiments/verify_paper_sample1_t20_qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish_20260701_summary.md`
- `docs/experiments/verify_paper_sample1_t20_qubo_scheduler_state_v5_qubo_diversity_union_polish_20260701_summary.md`

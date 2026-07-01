# Algorithm Profile Status

Date: 2026-07-01

This guide is the compact status table for the current compact CPMP / QUBO beam work. It separates promoted mainline behavior, clean baselines, open-only references, and opt-in ablations.

## Decision Snapshot

Current promoted mainline for open120 plus paper BF/CV work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

Previous combined-evidence baseline:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

Clean historical paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

Open120-only path-quality reference:

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

The macro-chain witness line is promoted because it keeps open120 at `120/120` and gives the strongest current BF/CV coverage. It turns BS-FRG value into recoverable chain candidates with completion witnesses, while QUBO remains the beam-selection layer.

## Promotion Rule

A new profile should not replace the current mainline unless it passes a paired comparison against:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish
```

Minimum evidence:

- open120 DLTS 5x5/7x5 test-120 with no new failures;
- paper BF and paper CV rows reported separately, with no meaningful solved-count regression;
- path-length and penalty deltas reported on common-solved rows;
- focused tests for profile switches, scoring behavior, reconstruction, and runner wiring;
- telemetry showing whether the gain came from candidate generation, QUBO selection, selected-lineage survival, or compact-path lifting.

Do not promote a profile from a smoke run, one residual-case rescue, a CSV-only probe, or a result that improves feasibility by accepting much longer paths without paper-set evidence.

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_chain_witness_polish` | Current open and paper benchmark mainline | Promoted default | Open120 no-cap t120: `120/120`, moves `6886`. Paper BF/CV patched full aggregate: primary `1560/1561`, feasible `1561/1561`, primary moves `102574`, feasible moves `102637`, full penalty `102881`, avg runtime `9.771`. |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | Previous combined-evidence mainline | Demoted baseline | Paper BF/CV full t120: primary `1407/1561`, feasible `1408/1561`, moves `65752`. Stronger than earlier shallow credit, but weaker coverage than macro-chain witness. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | Open120 full-coverage path-quality reference | Reference, not default | Open120 t20: `120/120`, full penalty `4100`, solved moves `4100`, gap sum `705`. Useful direct macro-edge ablation, but weaker on paper BF/CV. |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | Previous union / short-path open reference | Demoted reference | Open120 t20: `118/120`, full penalty `3858`, solved moves `3677`, gap sum `335`. Short on solved rows, but not full coverage. |
| `qubo_scheduler_state_v5_wang_rescue_polish` | Clean historical paired baseline | Keep as baseline | Open120 historical clean run: `118/120`, full penalty `3873`, solved moves `3692`, gap sum `350`. Stable and interpretable, but no longer the combined-evidence mainline. |
| pure BS-FRG reproduction | Paper-algorithm reference | Reference, not default | Paper BF/CV full: `1545/1561`, moves `97808`, avg runtime `19.854`. Strong for path ideas, but current macro-chain witness has better coverage and runtime. |
| `speed_v2` / `scheduler_state_v5` internal line | Internal 30-case speed reference | Keep for deployment regression context | Useful for the historical Shanghai Port 30-case path, but not the current open/paper promotion gate. |

## Promoted Macro-Chain Witness Interpretation

The promoted selector is a QUBO beam-selection line, not a fallback solver:

- legal one-step child states still come from the compact beam engine;
- short BS-FRG-style recovery chains can be lifted into macro candidates;
- each macro candidate must carry replayable compact-path evidence;
- completion witness quality, receiver debt, and source scarcity enter QUBO scoring;
- QUBO chooses which normal and macro candidates survive together;
- selected macro edges are expanded and repaired before crane-move lifting;
- final validator replay remains mandatory.

This is currently the cleanest integration of BS-FRG-style deep recovery evidence into the QUBO beam framework.

## Opt-In Ablations

| Profile | Mechanism | Status | Decision reason |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | shallow macro future credit with witness/debt/cap correction | Superseded baseline | It was the previous combined-evidence line, but full BF/CV coverage is much weaker than macro-chain witness. |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish` | v1 macro future credit without the v2 witness/debt/cap correction | Superseded | Useful historical ablation only. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | direct macro-edge QUBO selection | Reference | Good open120 reference, but direct macro edges are less robust on paper BF/CV than chain witnesses. |
| `qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish` | adds candidate-origin pair penalties to the union QUBO selector | Ablation only | Helped one immediate open120 paired rerun but lost paper cases and worsened paper penalty. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | adds bounded rollout signal to union selection | Ablation only | Rollout signal is real, but it can rank locally promising candidates that later commit to long cleanup paths. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | grafts a validated rollout-complete tail when available | Ablation only | Useful for diagnosis, but grafting should not become an implicit fallback path without paired evidence. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish` | emits micro-rollout and receiver-safety telemetry without changing selection | Diagnostic | Safe for trace collection because trace-only mode keeps the linear adjustment at zero. |
| `qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish` | adds receiver-safety soft score without rollout | Ablation only | Useful as a feature probe, but not yet discriminative enough as a promotion candidate. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish` | uses the bounded micro-rollout signal without receiver safety | Ablation only | Keep as a component-isolation profile. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish` | combines micro rollout and receiver-safety scoring | Ablation only | Open120 had mixed wins and regressions; paper evidence was worse than the paired baseline. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` | applies micro rollout only to near-tied candidates | Ablation only | Protected some open120 aggregate quality, but paper-set penalty regressed. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | appends BS-FRG-derived child states after normal frontier selection | Light frontier experiment | Candidate support only. It is not a BS-FRG fallback solver. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | merges FRG children into the same fixed-width candidate pool before selection | Experimental, not promoted | FRG children can dominate core selection without improving results. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | expands FRG frontier support on high-N plateau or small-stack pressure | Ablation only | Keep as a controlled diagnostic for FRG support strength. |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | adds receiver-debt, escape-potential, basin-diversity, and reduction-chain selector signals | Diagnostic, not promoted | Earlier paper sample10 gate regressed badly. |

## Current Remaining Failure Classes

After the compact-lift repair, the patched full BF/CV state is:

- BF: `720/721` primary and `721/721` feasible;
- CV: `840/840` primary and `840/840` feasible;
- ALL: `1560/1561` primary and `1561/1561` feasible.

The remaining primary miss is `BF27/cpmp_20_8_96_39_58_16`, which reached feasibility but crossed the primary time boundary in the original full run.

Quality risk remains on CV: pure BS-FRG is often shorter on common-solved CV rows, even though macro-chain witness has better coverage and runtime.

## Recommended Next Work

The next useful round should target quality without broadening fallback behavior.

1. Optionally rerun the full BF/CV `t120` command for publication hygiene.
2. Diagnose BF27 as an early-hit / primary-time-boundary problem.
3. Reduce CV common-solved path length by improving chain quality, not by adding a fallback solver.
4. Keep open120, BF, and CV equal in every promotion table.
5. Preserve compact-path lifting tests for any future macro-chain change.

## Source Evidence

- `docs/experiments/mainline_macro_chain_nocap_open120_t120_20260701_summary.md`
- `docs/experiments/mainline_macro_chain_nocap_paper_bf_cv_all_t120_20260701_summary.md`
- `docs/experiments/debug_bf32_cv10_6_parallel_lift_fixed_rerun.csv`
- `docs/experiments/macro_credit_v2_paper_bfcv_full_t120_20260701_summary.md`
- historical pure BS-FRG full paper benchmark summary, 2026-06-18

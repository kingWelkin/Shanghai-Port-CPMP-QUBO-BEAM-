# Algorithm Profile Status

Date: 2026-06-29

This guide is the compact status table for the current compact CPMP / QUBO beam work. It separates the default algorithm, clean baselines, opt-in ablations, and paper-reproduction assets so local improvements are not mistaken for promoted mainline behavior.

## Decision Snapshot

Current default for open and paper benchmark work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Clean paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

The diversity-union line remains the current mainline because it preserves the open120 `118/120` validity level while improving aggregate quality numbers, and because it gives a much stronger paper BF/CV sample10 result than the Wang rescue baseline. Paper evidence is recorded as separate BF and CV scorecard rows, not only as a combined BF+CV total.

## Promotion Rule

A new profile should not replace the current mainline unless it passes a paired comparison against `qubo_scheduler_state_v5_qubo_diversity_union_polish`.

Minimum evidence:

- open120 DLTS 5x5/7x5 test-120 with no new failures and a better aggregate quality signal;
- paper BF and paper CV sample10 rows, each with no meaningful solved-count or penalty regression;
- focused tests for profile switches and runner wiring;
- clear mechanism telemetry showing why the change helps.

Do not promote a profile from a smoke run, one residual-case rescue, a CSV-only probe, or a result that improves feasibility by accepting much longer paths.

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | Current open and paper benchmark mainline | Default | Open120: `118/120`, full penalty `3858`, solved moves `3677`, gap sum `335`. Paper sample10 split scorecard: BF `383/401` feasible, `336/401` primary, full penalty `1043373`; CV `189/210` feasible, `165/210` primary, full penalty `14927`; combined feasible `572/611`, primary `501/611`, full penalty `1058300`. |
| `qubo_scheduler_state_v5_wang_rescue_polish` | Clean paired baseline | Keep as baseline | Open120: `118/120`, full penalty `3873`, solved moves `3692`, gap sum `350`. Stable and interpretable, but no longer the combined-evidence mainline. |
| `speed_v2` / `scheduler_state_v5` internal line | Internal 30-case speed reference | Keep for deployment regression context | Useful for the historical Shanghai Port 30-case path, but not the current promotion gate. |
| `wang_bs_b` | Classical feasibility reference | Reference or teacher, not quality mainline | Can cover more open cases in older runs, but paths are much longer. Use for diagnosis, not as the default quality solver. |

## Opt-In Ablations

| Profile | Mechanism | Status | Decision reason |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish` | Adds candidate-origin pair penalties to the union QUBO selector. | Ablation only | Helped one immediate open120 paired rerun but lost paper sample10 cases and worsened paper penalty. Origin diversity is useful only as a guarded tie concept. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | Adds bounded rollout signal to union selection. | Ablation only | Rollout signal is real, but it can rank locally promising candidates that later commit to long cleanup paths. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | Grafts a validated rollout-complete tail when available. | Ablation only | Useful for diagnosis, but grafting should not become an implicit fallback path without paired evidence. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish` | Emits micro-rollout and receiver-safety telemetry without changing selection. | Diagnostic | Safe for trace collection because trace-only mode keeps the linear adjustment at zero. |
| `qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish` | Adds receiver-safety soft score without rollout. | Ablation only | Useful as a feature probe, but not yet discriminative enough as a promotion candidate. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish` | Uses the bounded micro-rollout signal without receiver safety. | Ablation only | Keep as a component-isolation profile. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish` | Combines micro rollout and receiver-safety scoring. | Ablation only | Open120 had mixed wins and regressions; paper sample10 was worse than the paired baseline. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` | Applies micro rollout only to near-tied QUBO candidates. | Ablation only | Protected open120 aggregate quality, but paper sample10 gained solved cases while worsening moves and full penalty. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | Appends BS-FRG-derived child states after normal frontier selection. | Light frontier experiment | Candidate support only. It is not a BS-FRG fallback solver. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | Merges FRG children into the same fixed-width candidate pool before selection. | Experimental, not promoted | Same solved count as a light frontier paired run, but worse path quality. FRG children can dominate core selection without improving results. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | Expands FRG frontier support on high-N plateau or small-stack pressure. | Ablation only | Keep as a controlled diagnostic for FRG support strength. |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | Adds receiver-debt, escape-potential, basin-diversity, and reduction-chain selector signals. | Diagnostic, not promoted | 2026-06-29 paper sample10: `568/611` feasible, `332/611` primary, full penalty `1095855`, worse than current union `572/611` feasible, `501/611` primary, `1058300` penalty. |

## 2026-06-29 Debt/Basin Interpretation

The debt/basin profile clarified mechanisms but did not improve the gate:

- BF14/BF32 still need short recovery-basin access; long FRG seed paths are feasibility-only under strict scorecards.
- CV 5-8 / 6-6 show that local micro/debt signals can introduce path debt.
- CV 10-6 reaches low `n` but falls into a dead retained lineage.
- CV 10-10 barely lowers `n`, suggesting missing useful recovery candidates.

The next successor should separate route-quality guards from basin-escape activation instead of using debt as a broad linear preference.

## Reproduction And Reference Assets

| Asset | Role | Status | Evidence summary |
|---|---|---|---|
| `paper_pure=True` BS-FRG via `scripts/benchmark_bs_frg_paper.py` | Paper-style BS-FRG reproduction | Reproduction asset | Full BF/CV run solved `1545/1561`: BF `719/721`, CV `826/840`, total moves `97808`, average runtime about `19.854s`. |
| `scripts/run_bs_frg_paper_parallel.ps1` | Family-level BS-FRG runner | Keep | Useful for paper-faithful corpus runs and failed-case reruns. |
| `scripts/run_bs_frg_bf14_failed_unlimited.py` | No-deadline failed-case rerun helper | Keep | Useful for separating quick `NO_SOLUTION` cases from deep-search outliers such as BF14. |
| `docs/experiments/bf_seed_cache_smoke_20260626.jsonl` | Materialized FRG seed cache smoke artifact | Experimental | Jobs1 and jobs2 consumed artifact seeds for BF14/BF32 and reproduced the same feasibility-only move counts. |

BS-FRG is a strong reproduction and diagnostic line, not the promoted compact QUBO-beam default. Do not describe the mainline as "run BS-FRG if the beam fails" unless a future explicit fallback design is approved and validated.

## Rejected Or Diagnostic Conclusions

- Bigger QUBO influence is not automatically better.
- Origin diversity is not a universal beam principle.
- Micro rollout should not override the main union score.
- FRG child injection can help diagnose missing candidates, but raw injection can consume frontier budget or dominate the fixed beam.
- Debt/basin features are informative telemetry, but the first broad selector integration damaged the strict paper scorecard.
- A profile that rescues a few cases but introduces long paths or paper-set regressions remains an ablation, not a mainline.

## Recommended Next Work

The next useful round should be trace-first and mode-split.

1. Keep BF and CV paper sample rows separate throughout analysis.
2. Add a guarded route-quality mode that uses receiver-debt as a near-tie veto only when the compact route is already making clean progress.
3. Add a separate basin-escape trigger for low-n lineage collapse or no-n-drop candidate-universe failure.
4. Re-run the seven hard/regression cases before any full gate.
5. Evaluate every successor with paired open120 and paper sample10 commands before promotion.

## Source Evidence

- `README.md`
- `docs/guides/current-mainline-algorithm.md`
- `docs/experiments/paper_hard_case_trace_report_20260629.md`
- `docs/guides/debt-basin-selector-analysis.md`
- `docs/experiments/debt_basin_paper_sample10_t20_20260629_summary.md`
- `docs/experiments/debt_basin_hard_20260629_summary.md`
- `docs/experiments/debt_basin_open_smoke_20260629_summary.md`
- `docs/experiments/strict_hard_union_sample10_t20_20260626.md`
- `docs/experiments/bf_seed_cache_concurrency_smoke_20260626.md`

# Algorithm Profile Status

Date: 2026-06-24

This guide is the compact status table for the current compact CPMP / QUBO beam
work. It separates the default algorithm, clean baselines, opt-in ablations, and
paper-reproduction assets so local improvements are not mistaken for promoted
mainline behavior.

## Decision Snapshot

Current default for open and paper benchmark work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Clean paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

The diversity-union line remains the current mainline because it preserves the
open120 `118/120` validity level while improving the aggregate quality numbers,
and because it gives a much stronger paper BF/CV sample10 result than the Wang
rescue baseline.

## Promotion Rule

A new profile should not replace the current mainline unless it passes a paired
comparison against `qubo_scheduler_state_v5_qubo_diversity_union_polish`.

Minimum evidence:

- open120 DLTS 5x5/7x5 test-120 with no new failures and a better aggregate
  quality signal;
- paper BF/CV sample10 with no meaningful solved-count or penalty regression;
- focused tests for the profile switches and runner wiring;
- clear mechanism telemetry showing why the change helps.

Do not promote a profile from a smoke run, one residual-case rescue, a CSV-only
probe, or a result that improves solve count by accepting much longer paths.

## Stable Defaults And Baselines

| Profile or solver | Role | Current status | Evidence summary |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | Current open and paper benchmark mainline | Default | Open120: `118/120`, full penalty `3858`, solved moves `3677`, gap sum `335` in the published mainline table. Paper sample10: `572/611`, full penalty about `1058300`, with `+348` solved cases versus Wang rescue. |
| `qubo_scheduler_state_v5_wang_rescue_polish` | Clean paired baseline | Keep as baseline | Open120: `118/120`, full penalty `3873`, solved moves `3692`, gap sum `350`. It is stable and interpretable, but no longer the combined-evidence mainline. |
| `speed_v2` / `scheduler_state_v5` internal line | Internal 30-case speed reference | Keep for deployment regression context | Useful for the historical Shanghai Port 30-case path, but not the current promotion gate. Current promotion is open120 plus paper BF/CV sample10. |
| `wang_bs_b` | Classical feasibility reference | Reference or teacher, not quality mainline | Can cover more open cases in older runs, but the paths are much longer. Use for diagnosis, not as the default quality solver. |

## Opt-In Ablations

| Profile | Mechanism | Status | Decision reason |
|---|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_origin_aware_polish` | Adds candidate-origin pair penalties to the union QUBO selector. | Ablation only | Helped one immediate open120 paired rerun but lost paper sample10 cases and worsened paper penalty. Origin diversity is useful only as a guarded tie concept, not a global rule. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | Adds bounded rollout signal to union selection. | Ablation only | Rollout signal is real, but it can rank locally promising candidates that later commit to long cleanup paths. |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | Grafts a validated rollout-complete tail when available. | Ablation only | Useful for diagnosis, but grafting should not become an implicit fallback path without paired evidence. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_trace_polish` | Emits micro-rollout and receiver-safety telemetry without changing selection. | Diagnostic | Safe for trace collection because trace-only mode keeps the linear adjustment at zero. |
| `qubo_scheduler_state_v5_qubo_diversity_union_receiver_safety_polish` | Adds receiver-safety soft score without rollout. | Ablation only | Useful as a feature probe, but not yet discriminative enough as a promotion candidate. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_only_polish` | Uses the bounded micro-rollout signal without receiver safety. | Ablation only | Keep as a component-isolation profile. Do not use as the default. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_rollout_polish` | Combines micro rollout and receiver-safety scoring. | Ablation only | Open120 had mixed wins and regressions; paper sample10 was worse than the paired baseline. |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` | Applies micro rollout only to near-tied QUBO candidates. | Ablation only | Protected open120 aggregate quality, but paper sample10 gained 2 solved cases while worsening moves and full penalty. Not a net quality win. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | Appends BS-FRG-derived child states after normal frontier selection. | Light frontier experiment | Candidate support only. It is not a BS-FRG fallback solver. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | Merges FRG children into the same fixed-width candidate pool before selection. | Experimental, not promoted | Same solved count as a light frontier paired run, but worse path quality. FRG children can dominate core selection without improving results. |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | Expands FRG frontier support on high-N plateau or small-stack pressure. | Ablation only | Keep as a controlled diagnostic for FRG support strength. |

## Reproduction And Reference Assets

| Asset | Role | Status | Evidence summary |
|---|---|---|---|
| `paper_pure=True` BS-FRG via `scripts/benchmark_bs_frg_paper.py` | Paper-style BS-FRG reproduction | Reproduction asset | Full BF/CV run solved `1545/1561`: BF `719/721`, CV `826/840`, total moves `97808`, average runtime about `19.854s`. |
| `scripts/run_bs_frg_paper_parallel.ps1` | Family-level BS-FRG runner | Keep | Useful for paper-faithful corpus runs and failed-case reruns. |
| `scripts/run_bs_frg_bf14_failed_unlimited.py` | No-deadline failed-case rerun helper | Keep | Useful for separating quick `NO_SOLUTION` cases from deep-search outliers such as BF14. |

BS-FRG is a strong reproduction and diagnostic line, not the promoted compact
QUBO-beam default. Do not describe the mainline as "run BS-FRG if the beam
fails" unless a future explicit fallback design is approved and validated.

## Rejected Or Diagnostic Conclusions

- Bigger QUBO influence is not automatically better. Several variants changed
  selection behavior without improving the candidate universe.
- Origin diversity is not a universal beam principle. It can break a crowded
  frontier, but it can also interrupt a repeated-progress route needed by paper
  CV and large BF cases.
- Micro rollout should not override the main union score. It is currently most
  plausible as a near-tie signal guarded by path-quality risk.
- FRG child injection can help diagnose missing candidates, but raw injection
  can consume frontier budget or dominate the fixed beam without improving path
  quality.
- A profile that rescues a few cases but introduces long paths or paper-set
  regressions remains an ablation, not a mainline.

## Recommended Next Work

The next useful round should be trace-first rather than weight-first.

1. Run frontier survivor and ancestry traces on residual and regression cases:
   `5x5_cv1_test_4`, `5x5_cv2_test_10`, BF14/BF32 gains, and CV 5-8 / 5-9 /
   6-6 regressions.
2. Classify each failure as one of three mechanisms:
   useful child absent, useful child generated but not selected, or selected
   child trapped in a dead parent region.
3. Keep micro rollout, receiver safety, and origin diversity as guarded
   near-tie adjustments unless trace evidence proves a stronger rule.
4. Evaluate every successor with paired open120 and paper sample10 commands
   before promotion.

## Source Evidence

- `README.md`
- `docs/guides/current-mainline-algorithm.md`
- `docs/experiments/origin_aware_qubo_report_20260622.md`
- `docs/experiments/micro_rollout_receiver_safety_report_20260622.md`
- `docs/experiments/micro_rollout_tiebreak_report_20260623.md`
- `docs/experiments/single_frontier_frg_union_decision_20260622.md`
- `docs/experiments/bs_frg_pure_paper_full_20260618_summary.md`

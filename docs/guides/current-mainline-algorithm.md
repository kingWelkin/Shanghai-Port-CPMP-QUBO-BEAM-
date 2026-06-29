# Current Mainline Algorithm

Date: 2026-06-29

## Decision

The current open and paper benchmark mainline candidate is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

The previous strong line remains the paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

This distinction matters. `wang_rescue_polish` is still the clean comparison point, but the diversity-union profile is the stronger current candidate on the combined open120 and paper BF/CV evidence.

## Evidence

Open DLTS 5x5/7x5 test-120 evidence:

| profile | valid | full penalty | solved moves | gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 118/120 | 3858 | 3677 | 335 | 4.442 |

Paper sample10 evidence, split by dataset:

| profile | scope | cases | feasible solved | primary accepted | full penalty | feasible moves | avg runtime sec |
|---|---|---:|---:|---:|---:|---:|---:|
| `union_diversity` | BF | 401 | 383 | 336 | 1043373 | 32499 | 15.703 |
| `union_diversity` | CV | 210 | 189 | 165 | 14927 | 9756 | 6.577 |
| `union_diversity` | BF+CV | 611 | 572 | 501 | 1058300 | 42255 | 12.567 |

In this table, feasible solved means replay reaches `final_n=0`; primary accepted additionally passes the stricter benchmark acceptance flags.

## Latest Ablation Verdict

The 2026-06-29 debt/basin profile is implemented:

```text
qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish
```

It adds receiver-debt, escape-potential, basin-diversity, and bounded reduction-chain selector signals. It remains diagnostic only because the paper sample10 gate regressed:

| profile | scope | cases | feasible solved | primary accepted | full penalty | avg runtime sec |
|---|---|---:|---:|---:|---:|---:|
| `union_diversity` | BF+CV | 611 | 572 | 501 | 1058300 | 12.567 |
| `debt_basin` | BF+CV | 611 | 568 | 332 | 1095855 | 12.118 |
| `debt_basin` | BF | 401 | 379 | 179 | 1079574 | 14.218 |
| `debt_basin` | CV | 210 | 189 | 153 | 16281 | 8.108 |

The hard-case trace still supports the mechanism split that motivated debt/basin, but the selector preference is too blunt for promotion. See `docs/guides/debt-basin-selector-analysis.md`.

## Mechanism

`scheduler_state_v5_qubo_diversity_union_polish` extends the FRG-seeded v5 line and adds a small QUBO exploration selector:

- candidate generation remains legal compact beam generation;
- FRG seed struct-skip remains enabled;
- direct candidate-ledger rank perturbation remains disabled;
- union QUBO selects a small supplement from already generated legal candidates;
- pair penalties discourage same parent, same source stack, same transition signature, same reduce signature, and same n-band.

QUBO remains a beam-selection mechanism. It does not generate crane moves and does not bypass validator replay.

## Operational Rule

Use `qubo_scheduler_state_v5_qubo_diversity_union_polish` for current open/paper benchmark runs unless a command is explicitly asking for a baseline.

Use `qubo_scheduler_state_v5_wang_rescue_polish` when a paired baseline is needed.

Use `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` only for trace, telemetry, and ablation work.

Do not promote `scheduler_state_v5_pareto_union_polish`: the light version was neutral on samples, and the wide version damaged open sample quality.

## Three-Set Scorecard Rule

The strongest-algorithm label is based on three recorded sets:

1. open120 DLTS 5x5/7x5, where valid count, full penalty, solved moves, gap sum, and runtime are tracked.
2. paper BF sample10, where feasible solved, primary accepted, full penalty, feasible moves, and runtime are tracked.
3. paper CV sample10, using the same paper-set metrics as BF.

A candidate that only improves the combined BF+CV total, or only improves one paper dataset while damaging the other, remains an ablation until this scorecard is clean.

## Current Ablation Branches

The current codebase contains several opt-in ablation profiles derived from the diversity-union line:

| profile | purpose | promotion status |
|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | add a bounded FRG-style rollout probe as a selection signal | ablation only |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | splice a validated rollout-complete tail when the selected child reaches one | ablation only |
| `qubo_scheduler_state_v5_qubo_diversity_union_micro_tiebreak_polish` | use micro rollout only for near-tied QUBO candidates | ablation only; paper-set penalty regressed |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | append BS-FRG-generated child states after normal frontier selection | light frontier experiment; not default |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | merge FRG-generated children into the same fixed-width candidate pool before selection | experimental; not promoted |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | receiver-debt, escape-potential, basin-diversity, and bounded reduction-chain selector signals | diagnostic; failed paper sample10 promotion gate |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | start with a smaller FRG frontier and expand on high-n plateau or small-stack pressure | ablation only |

These profiles are useful for paper-set and residual-case diagnosis, but they should not replace `qubo_scheduler_state_v5_qubo_diversity_union_polish` without paired full open120 and paper-set improvement.

## Latest Engineering State

The debt/basin implementation series through commit `bf01ff2` added:

- opt-in debt/basin profile and open-runner mapping;
- receiver-debt and escape-potential feature scoring;
- basin similarity pair penalty in the union selector;
- reduction-chain depth tracking for FRG frontier children;
- result-row and frontier-trace telemetry;
- on-path telemetry tightening for debt/basin selected states.

The 2026-06-29 generated artifacts are diagnostic:

- `docs/experiments/paper_hard_case_trace_report_20260629.md`
- `docs/experiments/debt_basin_hard_20260629_summary.md`
- `docs/experiments/debt_basin_open_smoke_20260629_summary.md`
- `docs/experiments/debt_basin_paper_sample10_t20_20260629_summary.md`
- `docs/guides/debt-basin-selector-analysis.md`

## Reproduction

Open test-120 mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_open120_t10.csv
```

Paper sample10 mainline:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_paper_sample10_t20.csv
```

Debt/basin diagnostic paper gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\debt_basin_paper_sample10_t20_YYYYMMDD.csv
```

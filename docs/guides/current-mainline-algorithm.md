# Current Mainline Algorithm

Date: 2026-06-22

## Decision

The current open and paper benchmark mainline candidate is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

The previous strong line remains the paired baseline:

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

This distinction matters. `wang_rescue_polish` is still the clean comparison point, but the diversity-union profile is the stronger current candidate on the combined evidence.

## Evidence

Open DLTS 5x5/7x5 test-120 evidence:

| profile | valid | full penalty | solved moves | gap sum | avg runtime sec |
|---|---:|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_wang_rescue_polish` | 118/120 | 3873 | 3692 | 350 | 3.901 |
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 118/120 | 3858 | 3677 | 335 | 4.442 |

Paper sample10 paired evidence:

| profile | cases | solved | full penalty | avg runtime sec |
|---|---:|---:|---:|---:|
| `wang_rescue_polish` | 611 | 224 | 1114273 | 14.995 |
| `union_diversity` | 611 | 572 | 1058300 | 12.567 |

The paper sample run had `+348` solved cases for union, no lost solved cases, and `-55973` full penalty.

## Mechanism

`scheduler_state_v5_qubo_diversity_union_polish` extends the FRG-seeded v5 line and adds a small QUBO exploration selector:

- candidate generation remains legal compact beam generation;
- FRG seed struct-skip remains enabled;
- direct candidate-ledger rank perturbation remains disabled;
- union QUBO selects at most 4 extra candidates per active layer from the top 48 remaining candidates;
- pair penalties discourage same parent, same source stack, same transition signature, same reduce signature, and same n-band.

QUBO remains a beam-selection mechanism. It does not generate crane moves and does not bypass validator replay.

## Operational Rule

Use `qubo_scheduler_state_v5_qubo_diversity_union_polish` for current open/paper benchmark runs unless a command is explicitly asking for a baseline.

Use `qubo_scheduler_state_v5_wang_rescue_polish` when a paired baseline is needed.

Do not promote `scheduler_state_v5_pareto_union_polish`: the light version was neutral on samples, and the wide version damaged open sample quality.

## Current Ablation Branches

The current codebase also contains several opt-in ablation profiles derived from the diversity-union line:

| profile | purpose | promotion status |
|---|---|---|
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_polish` | add a bounded FRG-style rollout probe as a selection signal | ablation only |
| `qubo_scheduler_state_v5_qubo_diversity_union_rollout_graft_polish` | splice a validated rollout-complete tail when the selected child reaches one | ablation only |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_polish` | append BS-FRG-generated child states after normal frontier selection | light frontier experiment; not default |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_single_frontier_polish` | merge FRG-generated children into the same candidate pool before selection | experimental; latest paired run did not beat light frontier |
| `qubo_scheduler_state_v5_qubo_diversity_union_debt_basin_polish` | adds receiver-debt, escape-potential, basin-diversity, and bounded reduction-chain selector signals | experimental; not default |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_frontier_adaptive_polish` | start with a smaller FRG frontier and expand on high-n plateau / small-stack pressure | ablation only |

These profiles are useful for paper-set and residual-case diagnosis, but they should not replace `qubo_scheduler_state_v5_qubo_diversity_union_polish` without a paired full open120 and paper-set improvement. In particular, FRG frontier injection is candidate generation support, not a BS-FRG fallback solver.

For the consolidated status of current defaults, baselines, opt-in ablations,
and reproduction assets, see `docs/guides/algorithm-profile-status.md`.

## Latest Engineering Commit

Commit `c5062ef` records the current implementation state:

- union rollout terminal graft telemetry and cap controls;
- FRG frontier child generation, append-only mode, single-frontier mode, and adaptive expansion parameters;
- `candidate_origin` frontier trace support;
- open120 and paper-set runner wiring for the new profiles;
- standalone BS-FRG benchmark runners for open CPMP and paper BF/CV reproduction;
- focused tests for union rollout, FRG frontier child injection, BS-FRG QUBO selection, and runner variant mapping.

Validation for that commit used:

- `python -m py_compile` on the modified benchmark scripts;
- `git diff --cached --check`;
- `python -m pytest tests\test_qubo.py -k "union_rollout or frg_frontier_child or qubo_diversity_union or bs_frg_qubo" -q --basetemp .tmp_pytest_commit_union`, which reported `31 passed, 286 deselected`.

## Reproduction

Open test-120 mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_open120_t10.csv
```

Paper sample10 mainline:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_paper_sample10_t20.csv
```

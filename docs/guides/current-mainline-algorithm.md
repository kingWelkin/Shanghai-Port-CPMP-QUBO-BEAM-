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

## Reproduction

Open test-120 mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 10 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_open120_t10.csv
```

Paper sample10 mainline:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 10 --variant qubo_scheduler_state_v5_qubo_diversity_union_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_union_paper_sample10_t20.csv
```

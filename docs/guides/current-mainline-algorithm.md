# Current Mainline Algorithm

Date: 2026-07-01

## Decision

The current promoted open120 plus BF/CV paper-set mainline is:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

This promotion treats open120 and the paper BF/CV set as equally important. The reason is not that v2 is the shortest open120 profile; it is that it keeps open120 full coverage while materially improving the paper BF/CV scorecard.

Reference roles:

```text
qubo_scheduler_state_v5_qubo_diversity_union_polish
```

Previous mainline and short-path open reference. It has better solved-move totals on open120, but only solves `118/120` open cases and is much weaker on the paper sample1 gate.

```text
qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish
```

Best open120 single-scope reference in this batch. It solves `120/120` with 5 fewer open120 moves than v2, but is clearly weaker on paper BF/CV.

```text
qubo_scheduler_state_v5_wang_rescue_polish
```

Clean historical paired baseline.

## Promotion Evidence

Open DLTS 5x5/7x5 test-120, `t20`, 2026-07-01:

| profile | valid | full penalty | solved moves | gap sum |
|---|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 118/120 | 3858 | 3677 | 335 |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | 120/120 | 4100 | 4100 | 705 |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish` | 120/120 | 4105 | 4105 | 710 |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | 120/120 | 4105 | 4105 | 710 |

Paper BF/CV sample1, `t20`, 2026-07-01:

| profile | primary solved | feasibility solved | moves | full penalty |
|---|---:|---:|---:|---:|
| `qubo_scheduler_state_v5_qubo_diversity_union_polish` | 26/62 | 56/62 | 1904 | 10699 |
| `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` | 51/62 | 56/62 | 2363 | 5289 |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_polish` | 50/62 | 56/62 | 2152 | 5898 |
| `qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish` | 56/62 | 56/62 | 2638 | 4437 |

On the equal-weighted open120 plus BF/CV gate, v2 is the best current compromise: it preserves open120 full solve coverage and gives the strongest paper-set primary solved count and full penalty.

## Mechanism

`macro_credit_v2` keeps the same compact beam and legal child generation boundary as the union line. It changes only QUBO beam selection scoring for macro candidates:

- QUBO still selects among already legal child states.
- It does not generate crane moves.
- It does not run BS-FRG as a fallback solver.
- It rewards macro/recovery candidates only when they carry completion-witness or macro-path evidence.
- It subtracts receiver-debt pressure so "high debt without witness" is not treated as future value.
- It caps positive macro future credit to prevent one macro family from overwhelming the selector.

The practical interpretation is: QUBO now selects recovery chains with future proof, not merely diverse states or raw FRG macro edges.

## Operational Rule

Use this profile for current open120 and paper BF/CV benchmark work:

```text
qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish
```

Use `qubo_scheduler_state_v5_qubo_diversity_union_frg_macro_edge_qubo_polish` when the question is specifically "best open120 path quality among full-coverage profiles".

Use `qubo_scheduler_state_v5_qubo_diversity_union_polish` only as the previous mainline / short-path open reference.

Use `qubo_scheduler_state_v5_wang_rescue_polish` as the clean historical paired baseline.

## Remaining Failure Classes

The promoted v2 line still leaves these paper sample1 primary failures:

- `BF14`
- `BF30`
- `CV/5-4`
- `CV/6-6`
- `CV/10-6`
- `CV/10-10`

Small parameter sweeps did not beat the promoted v2 defaults:

| sweep | result |
|---|---|
| `macro_future_credit_cap=32` | helped some BF edges but damaged CV/5-9 on the edge set |
| `receiver_debt_penalty_weight=12` | looked good on a 9-case edge set, but dropped sample1 to `52/62`, penalty `5452` |
| `min_witness_quality=0.25` | improved some edge behavior, but sample1 was only `54/62`, penalty `5034` |

So the current defaults remain:

```text
min_witness_quality = 0.35
receiver_debt_penalty_weight = 18.0
macro_future_credit_cap = 24.0
```

## Reproduction

Open test-120 mainline:

```powershell
python scripts\open_cpmp_benchmarks.py --source external\DLTS --no-auto-clone --families 5x5_cv1_test 5x5_cv2_test 5x5_cv3_test 7x5_cv1_test 7x5_cv2_test 7x5_cv3_test --limit-per-family 20 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_macro_credit_v2_open120_t20.csv
```

Paper BF/CV sample1 mainline gate:

```powershell
python scripts\benchmark_compact_paper_set.py --source external\paper_cpmp --datasets BF CV --sample-per-family 1 --variant qubo_scheduler_state_v5_qubo_diversity_union_macro_credit_v2_polish --time-limit 20 --alpha 10 --jobs 8 --output docs\experiments\mainline_macro_credit_v2_paper_sample1_t20.csv
```

For a heavier future stress gate, rerun paper sample10 with the same profile and keep BF and CV rows separate.

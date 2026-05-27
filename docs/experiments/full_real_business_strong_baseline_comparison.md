# Full Real Business Strong Baseline Comparison

This table compares stronger non-MIP baselines against the current beam and QUBO-beam search under the 1.3*N0 business cap.

CSV: `docs\experiments\full_real_business_strong_baseline_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 15 |

## Per Case Final N

| case | qubo_beam | beam | wide_beam_baseline | multi_start_greedy_repair | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|
| A1_01_26_paired | 18 | 18 | 18 | 16 | 22 |
| A2_01_26_paired | 5 | 4 | 4 | 6 | 16 |
| A3_01_26_paired | 10 | 10 | 12 | 9 | 17 |
| A4_01_26_paired | 13 | 13 | 13 | 17 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 16 | 22 |
| B1_01_26_paired | 6 | 6 | 6 | 12 | 14 |
| B2_01_26_paired | 9 | 9 | 10 | 10 | 15 |
| B3_01_26_paired | 12 | 12 | 13 | 7 | 11 |
| B4_01_26_paired | 10 | 10 | 10 | 10 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 7 | 17 |
| C1_01_26_paired | 7 | 7 | 6 | 6 | 12 |
| C2_01_26_paired | 7 | 9 | 10 | 12 | 17 |
| C3_01_26_paired | 17 | 17 | 17 | 14 | 24 |
| C4_01_26_paired | 11 | 11 | 11 | 11 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 20 |

## Notes

- `multi_start_greedy_repair` is a stronger greedy baseline: six move-ranking rules, stochastic top-k tie breaking, and per-rollout state de-duplication.
- `wide_beam_baseline` is plain beam with wider width and more candidate expansion, without QUBO/API.
- Both stronger baselines improve substantially over `chen_greedy_algorithm_1p3`, but current beam/QUBO-beam still has the best average final N on this full real set.

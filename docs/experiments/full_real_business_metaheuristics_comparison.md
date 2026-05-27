# Full Real Business Metaheuristics Comparison

This report adds GRASP, constructive simulated annealing, and VNS baselines. All methods only apply legal relocations and filter every intermediate state by the 1.3*N0 business G cap.

CSV: `docs\experiments\full_real_business_metaheuristics_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 15 |
| tabu_search_baseline | 15 | 12.27 | 16.80 | 22.93 | 4.787 | 15 |
| vns_baseline | 15 | 12.60 | 16.47 | 22.53 | 4.566 | 15 |
| grasp_baseline | 15 | 12.73 | 16.33 | 21.87 | 5.006 | 15 |
| simulated_annealing_baseline | 15 | 14.53 | 14.53 | 22.27 | 3.446 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 15 |

## Per Case Final N

| case | qubo_beam | beam | wide_beam_baseline | multi_start_greedy_repair | tabu_search_baseline | vns_baseline | grasp_baseline | simulated_annealing_baseline | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 18 | 18 | 18 | 16 | 16 | 16 | 17 | 19 | 22 |
| A2_01_26_paired | 5 | 4 | 4 | 6 | 10 | 10 | 8 | 13 | 16 |
| A3_01_26_paired | 10 | 10 | 12 | 9 | 13 | 13 | 13 | 12 | 17 |
| A4_01_26_paired | 13 | 13 | 13 | 17 | 13 | 18 | 14 | 18 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 16 | 16 | 18 | 19 | 22 | 22 |
| B1_01_26_paired | 6 | 6 | 6 | 12 | 12 | 13 | 14 | 12 | 14 |
| B2_01_26_paired | 9 | 9 | 10 | 10 | 10 | 11 | 11 | 13 | 15 |
| B3_01_26_paired | 12 | 12 | 13 | 7 | 14 | 8 | 11 | 13 | 11 |
| B4_01_26_paired | 10 | 10 | 10 | 10 | 10 | 11 | 10 | 10 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 7 | 6 | 8 | 10 | 10 | 17 |
| C1_01_26_paired | 7 | 7 | 6 | 6 | 6 | 6 | 8 | 10 | 12 |
| C2_01_26_paired | 7 | 9 | 10 | 12 | 14 | 16 | 14 | 16 | 17 |
| C3_01_26_paired | 17 | 17 | 17 | 14 | 18 | 17 | 17 | 19 | 24 |
| C4_01_26_paired | 11 | 11 | 11 | 11 | 10 | 10 | 9 | 15 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 14 | 16 | 16 | 20 |

## Notes

- GRASP and VNS are competitive with the earlier multi-start greedy repair and tabu baselines on several individual cases, but do not beat beam/QUBO-beam on average.
- The constructive simulated annealing baseline is the weakest of the three because accepted uphill moves consume the finite G budget quickly.
- These results support using beam/QUBO-beam as the main non-MIP algorithm and keeping metaheuristics as sanity-check baselines.

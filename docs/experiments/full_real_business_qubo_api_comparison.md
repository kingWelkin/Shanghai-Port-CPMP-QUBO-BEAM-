# Full Real Cases: Local QUBO vs API QUBO

Scope: 15 paired real cases. All current rows use `G<=floor(1.3*N0)`, except the dense/sparse API rows are only QUBO-beam variants and do not include greedy.

CSV: `docs/experiments/full_real_business_qubo_api_comparison.csv`

## Aggregate

| Solver | Avg final N | Avg reduction | Avg moves | Avg runtime(s) | Avg QUBO calls | Avg QUBO time(s) | API/fallback notes |
|---|---:|---:|---:|---:|---:|---:|---|
| chen_greedy_algorithm_1p3 | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 0.000 | - |
| beam | 10.67 | 18.40 | 24.13 | 3.689 | 0.00 | 0.000 | - |
| qubo_beam | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | 0.099 | - |
| qubo_beam_api_dense | 14.93 | 14.13 | 14.73 | 10.335 | 4.00 | 8.580 | BEST_EFFORT; samplers=ising_matrix_api; BEST_EFFORT; samplers=ising_matrix_api,ising_matrix_api_fallback_simulated_annealing |
| qubo_beam_api_sparse | 10.73 | 18.33 | 24.20 | 7.087 | 2.13 | 4.544 | BEST_EFFORT; samplers=ising_matrix_api; BEST_EFFORT; samplers=ising_matrix_api,ising_matrix_api_fallback_simulated_annealing |

## Per Case API Sparse

| Case | Local QUBO N | API sparse N | API dense N | Beam N |
|---|---:|---:|---:|---:|
| A1_01_26_paired | 18 | 18 | 18 | 18 |
| A2_01_26_paired | 4 | 4 | 12 | 4 |
| A3_01_26_paired | 10 | 10 | 14 | 10 |
| A4_01_26_paired | 13 | 13 | 21 | 13 |
| A5_01_26_paired | 15 | 15 | 18 | 15 |
| B1_01_26_paired | 5 | 6 | 14 | 6 |
| B2_01_26_paired | 9 | 9 | 11 | 9 |
| B3_01_26_paired | 12 | 12 | 12 | 12 |
| B4_01_26_paired | 10 | 10 | 12 | 10 |
| B5_01_26_paired | 3 | 3 | 14 | 3 |
| C1_01_26_paired | 7 | 7 | 9 | 7 |
| C2_01_26_paired | 7 | 10 | 14 | 9 |
| C3_01_26_paired | 17 | 17 | 20 | 17 |
| C4_01_26_paired | 11 | 11 | 16 | 11 |
| C5_01_26_paired | 16 | 16 | 19 | 16 |

## Notes

- Dense API (`call_interval=5`) spent most of each case budget inside API calls and produced weaker trajectories.
- Sparse API (`call_interval=16`) preserves search depth and matches the tuned local QUBO quality on most cases, but is still slower than local SA.
- API calls were real on most sparse rows; B3 had one fallback mixed with API.

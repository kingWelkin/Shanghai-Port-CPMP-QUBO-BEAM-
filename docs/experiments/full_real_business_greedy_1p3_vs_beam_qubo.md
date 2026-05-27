# Full Real Cases: 1.3 Greedy vs Tuned Beam vs QUBO Beam

Scope: 15 paired `data/input` cases (`01+26`, `W=20,H=6,N about 60`).
All rows are checked against the current `G=M+N(Future)<=floor(1.3*N0)` cap.
Greedy rows were rerun into `data/results_greedy_1p3` with `--g-ratio 1.3`.
Beam/QUBO beam use the tuned business configuration: current `N(Future)` first, narrow beam (`beam_width=10`, `candidate_limit_per_state=3`), and QUBO elite retention (`80%`) with low diversity (`0.1`).

CSV: `docs/experiments/full_real_business_greedy_1p3_vs_beam_qubo.csv`

## Aggregate

| Solver | OK13 cases | Avg final N | Avg reduction | Avg moves | Avg runtime(s) | Avg QUBO calls | Best/tied cases |
|---|---:|---:|---:|---:|---:|---:|---:|
| chen_greedy_algorithm_1p3 | 15/15 | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 1/15 |
| beam | 15/15 | 10.67 | 18.40 | 24.13 | 3.689 | 0.00 | 12/15 |
| qubo_beam | 15/15 | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | 14/15 |

## Detail

| Case | Solver | N0 | Cap13 | Moves | Final N | Reduction | G_end | Max G | OK13 | Runtime(s) | QUBO calls | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1_01_26_paired | chen_greedy_algorithm_1p3 | 29 | 37 | 15 | 22 | 7 | 37 | 37 | True | 0.106 | 0 | rerun; cap=1.3*N0 |
| A1_01_26_paired | beam | 29 | 37 | 15 | 18 | 11 | 33 | 33 | True | 2.967 | 0 | BEST_EFFORT |
| A1_01_26_paired | qubo_beam | 29 | 37 | 15 | 18 | 11 | 33 | 33 | True | 3.061 | 4 | BEST_EFFORT |
| A2_01_26_paired | chen_greedy_algorithm_1p3 | 27 | 35 | 19 | 16 | 11 | 35 | 35 | True | 0.095 | 0 | rerun; cap=1.3*N0 |
| A2_01_26_paired | beam | 27 | 35 | 27 | 4 | 23 | 31 | 31 | True | 3.613 | 0 | BEST_EFFORT |
| A2_01_26_paired | qubo_beam | 27 | 35 | 30 | 4 | 23 | 34 | 34 | True | 3.592 | 7 | BEST_EFFORT |
| A3_01_26_paired | chen_greedy_algorithm_1p3 | 30 | 39 | 22 | 17 | 13 | 39 | 39 | True | 0.136 | 0 | rerun; cap=1.3*N0 |
| A3_01_26_paired | beam | 30 | 39 | 26 | 10 | 20 | 36 | 36 | True | 4.424 | 0 | BEST_EFFORT |
| A3_01_26_paired | qubo_beam | 30 | 39 | 26 | 10 | 20 | 36 | 36 | True | 4.611 | 6 | BEST_EFFORT |
| A4_01_26_paired | chen_greedy_algorithm_1p3 | 33 | 42 | 26 | 16 | 17 | 42 | 42 | True | 0.110 | 0 | rerun; cap=1.3*N0 |
| A4_01_26_paired | beam | 33 | 42 | 27 | 13 | 20 | 40 | 40 | True | 4.601 | 0 | BEST_EFFORT |
| A4_01_26_paired | qubo_beam | 33 | 42 | 29 | 13 | 20 | 42 | 42 | True | 4.365 | 6 | BEST_EFFORT |
| A5_01_26_paired | chen_greedy_algorithm_1p3 | 32 | 41 | 19 | 22 | 10 | 41 | 41 | True | 0.125 | 0 | rerun; cap=1.3*N0 |
| A5_01_26_paired | beam | 32 | 41 | 21 | 15 | 17 | 36 | 36 | True | 3.711 | 0 | BEST_EFFORT |
| A5_01_26_paired | qubo_beam | 32 | 41 | 21 | 15 | 17 | 36 | 36 | True | 3.991 | 6 | BEST_EFFORT |
| B1_01_26_paired | chen_greedy_algorithm_1p3 | 30 | 39 | 25 | 14 | 16 | 39 | 39 | True | 0.144 | 0 | rerun; cap=1.3*N0 |
| B1_01_26_paired | beam | 30 | 39 | 33 | 6 | 24 | 39 | 39 | True | 3.472 | 0 | BEST_EFFORT |
| B1_01_26_paired | qubo_beam | 30 | 39 | 33 | 5 | 25 | 38 | 38 | True | 5.109 | 7 | BEST_EFFORT |
| B2_01_26_paired | chen_greedy_algorithm_1p3 | 26 | 33 | 18 | 15 | 11 | 33 | 33 | True | 0.111 | 0 | rerun; cap=1.3*N0 |
| B2_01_26_paired | beam | 26 | 33 | 22 | 9 | 17 | 31 | 31 | True | 3.733 | 0 | BEST_EFFORT |
| B2_01_26_paired | qubo_beam | 26 | 33 | 22 | 9 | 17 | 31 | 31 | True | 3.637 | 5 | BEST_EFFORT |
| B3_01_26_paired | chen_greedy_algorithm_1p3 | 27 | 35 | 24 | 11 | 16 | 35 | 35 | True | 0.128 | 0 | rerun; cap=1.3*N0 |
| B3_01_26_paired | beam | 27 | 35 | 16 | 12 | 15 | 28 | 28 | True | 3.811 | 0 | BEST_EFFORT |
| B3_01_26_paired | qubo_beam | 27 | 35 | 16 | 12 | 15 | 28 | 28 | True | 3.526 | 5 | BEST_EFFORT |
| B4_01_26_paired | chen_greedy_algorithm_1p3 | 27 | 35 | 21 | 14 | 13 | 35 | 35 | True | 0.116 | 0 | rerun; cap=1.3*N0 |
| B4_01_26_paired | beam | 27 | 35 | 20 | 10 | 17 | 30 | 30 | True | 3.661 | 0 | BEST_EFFORT |
| B4_01_26_paired | qubo_beam | 27 | 35 | 20 | 10 | 17 | 30 | 30 | True | 4.265 | 5 | BEST_EFFORT |
| B5_01_26_paired | chen_greedy_algorithm_1p3 | 30 | 39 | 22 | 17 | 13 | 39 | 39 | True | 0.119 | 0 | rerun; cap=1.3*N0 |
| B5_01_26_paired | beam | 30 | 39 | 36 | 3 | 27 | 39 | 39 | True | 4.171 | 0 | BEST_EFFORT |
| B5_01_26_paired | qubo_beam | 30 | 39 | 36 | 3 | 27 | 39 | 39 | True | 5.213 | 8 | BEST_EFFORT |
| C1_01_26_paired | chen_greedy_algorithm_1p3 | 23 | 29 | 17 | 12 | 11 | 29 | 29 | True | 0.091 | 0 | rerun; cap=1.3*N0 |
| C1_01_26_paired | beam | 23 | 29 | 18 | 7 | 16 | 25 | 25 | True | 3.398 | 0 | BEST_EFFORT |
| C1_01_26_paired | qubo_beam | 23 | 29 | 18 | 7 | 16 | 25 | 25 | True | 3.653 | 5 | BEST_EFFORT |
| C2_01_26_paired | chen_greedy_algorithm_1p3 | 29 | 37 | 20 | 17 | 12 | 37 | 37 | True | 0.103 | 0 | rerun; cap=1.3*N0 |
| C2_01_26_paired | beam | 29 | 37 | 28 | 9 | 20 | 37 | 37 | True | 3.951 | 0 | BEST_EFFORT |
| C2_01_26_paired | qubo_beam | 29 | 37 | 30 | 7 | 22 | 37 | 37 | True | 4.062 | 6 | BEST_EFFORT |
| C3_01_26_paired | chen_greedy_algorithm_1p3 | 33 | 42 | 18 | 24 | 9 | 42 | 42 | True | 0.120 | 0 | rerun; cap=1.3*N0 |
| C3_01_26_paired | beam | 33 | 42 | 23 | 17 | 16 | 40 | 40 | True | 3.620 | 0 | BEST_EFFORT |
| C3_01_26_paired | qubo_beam | 33 | 42 | 23 | 17 | 16 | 40 | 40 | True | 3.358 | 5 | BEST_EFFORT |
| C4_01_26_paired | chen_greedy_algorithm_1p3 | 31 | 40 | 20 | 20 | 11 | 40 | 40 | True | 0.120 | 0 | rerun; cap=1.3*N0 |
| C4_01_26_paired | beam | 31 | 40 | 29 | 11 | 20 | 40 | 40 | True | 3.227 | 0 | BEST_EFFORT |
| C4_01_26_paired | qubo_beam | 31 | 40 | 29 | 11 | 20 | 40 | 40 | True | 4.067 | 6 | BEST_EFFORT |
| C5_01_26_paired | chen_greedy_algorithm_1p3 | 29 | 37 | 17 | 20 | 9 | 37 | 37 | True | 0.102 | 0 | rerun; cap=1.3*N0 |
| C5_01_26_paired | beam | 29 | 37 | 21 | 16 | 13 | 37 | 37 | True | 2.968 | 0 | BEST_EFFORT |
| C5_01_26_paired | qubo_beam | 29 | 37 | 21 | 16 | 13 | 37 | 37 | True | 3.071 | 5 | BEST_EFFORT |

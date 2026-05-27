# Full Real Cases: Greedy vs Beam vs QUBO Beam

Scope: 15 paired `data/input` cases (`01+26`, `W=20,H=6,N=60`).
Metric: best business trajectory under `G=M+N(Future)<=floor(1.3*N0)` for beam/QUBO beam.
The historical greedy rows are parsed from `data/results` and were generated under the old `1.5*N0` cap.

CSV: `docs/experiments/full_real_business_beam_qubo_vs_greedy.csv`

Run config for beam/QUBO beam: `10s` per solver per case, `beam_width=10`, `max_candidates_per_layer=100`, `candidate_limit_per_state=3`, `entropy_alpha=1.3`; QUBO uses `diversity_weight=0.1` with elite retention. Duplicate original sequence IDs are tie-broken by position to satisfy the unique-container state engine.

## Aggregate

| Solver | OK13 cases | Avg final N | Avg reduction | Avg moves | Avg runtime(s) | Avg QUBO calls | Best/tied among OK13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| chen_greedy_algorithm_legacy | 0/15 | 16.13 | 12.93 | 27.13 | 0.371 | 0.00 | 0/15 |
| beam | 15/15 | 10.67 | 18.40 | 24.13 | 3.689 | 0.00 | 13/15 |
| qubo_beam | 15/15 | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | 15/15 |

## Detail

| Case | Solver | N0 | Cap13 | Moves | Final N | Reduction | G_end | Max G | OK13 | Runtime(s) | Expanded | Generated | QUBO calls | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1_01_26_paired | chen_greedy_algorithm_legacy | 29 | 37 | 23 | 20 | 9 | 43 | 43 | False | 0.367 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| A1_01_26_paired | beam | 29 | 37 | 15 | 18 | 11 | 33 | 33 | True | 2.967 | 183 | 503 | 0 | BEST_EFFORT |
| A1_01_26_paired | qubo_beam | 29 | 37 | 15 | 18 | 11 | 33 | 33 | True | 3.061 | 183 | 503 | 4 | BEST_EFFORT |
| A2_01_26_paired | chen_greedy_algorithm_legacy | 27 | 35 | 24 | 16 | 11 | 40 | 40 | False | 0.307 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| A2_01_26_paired | beam | 27 | 35 | 27 | 4 | 23 | 31 | 31 | True | 3.613 | 303 | 829 | 0 | BEST_EFFORT |
| A2_01_26_paired | qubo_beam | 27 | 35 | 30 | 4 | 23 | 34 | 34 | True | 3.592 | 296 | 796 | 7 | BEST_EFFORT |
| A3_01_26_paired | chen_greedy_algorithm_legacy | 30 | 39 | 29 | 16 | 14 | 45 | 45 | False | 0.429 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| A3_01_26_paired | beam | 30 | 39 | 26 | 10 | 20 | 36 | 36 | True | 4.424 | 282 | 783 | 0 | BEST_EFFORT |
| A3_01_26_paired | qubo_beam | 30 | 39 | 26 | 10 | 20 | 36 | 36 | True | 4.611 | 282 | 781 | 6 | BEST_EFFORT |
| A4_01_26_paired | chen_greedy_algorithm_legacy | 33 | 42 | 35 | 14 | 19 | 49 | 49 | False | 0.358 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| A4_01_26_paired | beam | 33 | 42 | 27 | 13 | 20 | 40 | 40 | True | 4.601 | 283 | 806 | 0 | BEST_EFFORT |
| A4_01_26_paired | qubo_beam | 33 | 42 | 29 | 13 | 20 | 42 | 42 | True | 4.365 | 283 | 775 | 6 | BEST_EFFORT |
| A5_01_26_paired | chen_greedy_algorithm_legacy | 32 | 41 | 26 | 22 | 10 | 48 | 48 | False | 0.410 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| A5_01_26_paired | beam | 32 | 41 | 21 | 15 | 17 | 36 | 36 | True | 3.711 | 251 | 705 | 0 | BEST_EFFORT |
| A5_01_26_paired | qubo_beam | 32 | 41 | 21 | 15 | 17 | 36 | 36 | True | 3.991 | 251 | 703 | 6 | BEST_EFFORT |
| B1_01_26_paired | chen_greedy_algorithm_legacy | 30 | 39 | 32 | 13 | 17 | 45 | 45 | False | 0.437 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| B1_01_26_paired | beam | 30 | 39 | 33 | 6 | 24 | 39 | 39 | True | 3.472 | 318 | 812 | 0 | BEST_EFFORT |
| B1_01_26_paired | qubo_beam | 30 | 39 | 33 | 5 | 25 | 38 | 38 | True | 5.109 | 333 | 895 | 7 | BEST_EFFORT |
| B2_01_26_paired | chen_greedy_algorithm_legacy | 26 | 33 | 24 | 15 | 11 | 39 | 39 | False | 0.374 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| B2_01_26_paired | beam | 26 | 33 | 22 | 9 | 17 | 31 | 31 | True | 3.733 | 233 | 660 | 0 | BEST_EFFORT |
| B2_01_26_paired | qubo_beam | 26 | 33 | 22 | 9 | 17 | 31 | 31 | True | 3.637 | 233 | 657 | 5 | BEST_EFFORT |
| B3_01_26_paired | chen_greedy_algorithm_legacy | 27 | 35 | 29 | 11 | 16 | 40 | 40 | False | 0.355 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| B3_01_26_paired | beam | 27 | 35 | 16 | 12 | 15 | 28 | 28 | True | 3.811 | 223 | 610 | 0 | BEST_EFFORT |
| B3_01_26_paired | qubo_beam | 27 | 35 | 16 | 12 | 15 | 28 | 28 | True | 3.526 | 223 | 607 | 5 | BEST_EFFORT |
| B4_01_26_paired | chen_greedy_algorithm_legacy | 27 | 35 | 30 | 10 | 17 | 40 | 40 | False | 0.346 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| B4_01_26_paired | beam | 27 | 35 | 20 | 10 | 17 | 30 | 30 | True | 3.661 | 243 | 685 | 0 | BEST_EFFORT |
| B4_01_26_paired | qubo_beam | 27 | 35 | 20 | 10 | 17 | 30 | 30 | True | 4.265 | 243 | 684 | 5 | BEST_EFFORT |
| B5_01_26_paired | chen_greedy_algorithm_legacy | 30 | 39 | 28 | 17 | 13 | 45 | 45 | False | 0.372 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| B5_01_26_paired | beam | 30 | 39 | 36 | 3 | 27 | 39 | 39 | True | 4.171 | 348 | 939 | 0 | BEST_EFFORT |
| B5_01_26_paired | qubo_beam | 30 | 39 | 36 | 3 | 27 | 39 | 39 | True | 5.213 | 343 | 935 | 8 | BEST_EFFORT |
| C1_01_26_paired | chen_greedy_algorithm_legacy | 23 | 29 | 22 | 12 | 11 | 34 | 34 | False | 0.288 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| C1_01_26_paired | beam | 23 | 29 | 18 | 7 | 16 | 25 | 25 | True | 3.398 | 212 | 581 | 0 | BEST_EFFORT |
| C1_01_26_paired | qubo_beam | 23 | 29 | 18 | 7 | 16 | 25 | 25 | True | 3.653 | 212 | 581 | 5 | BEST_EFFORT |
| C2_01_26_paired | chen_greedy_algorithm_legacy | 29 | 37 | 27 | 16 | 13 | 43 | 43 | False | 0.333 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| C2_01_26_paired | beam | 29 | 37 | 28 | 9 | 20 | 37 | 37 | True | 3.951 | 272 | 739 | 0 | BEST_EFFORT |
| C2_01_26_paired | qubo_beam | 29 | 37 | 30 | 7 | 22 | 37 | 37 | True | 4.062 | 287 | 768 | 6 | BEST_EFFORT |
| C3_01_26_paired | chen_greedy_algorithm_legacy | 33 | 42 | 27 | 22 | 11 | 49 | 49 | False | 0.453 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| C3_01_26_paired | beam | 33 | 42 | 23 | 17 | 16 | 40 | 40 | True | 3.620 | 243 | 678 | 0 | BEST_EFFORT |
| C3_01_26_paired | qubo_beam | 33 | 42 | 23 | 17 | 16 | 40 | 40 | True | 3.358 | 243 | 675 | 5 | BEST_EFFORT |
| C4_01_26_paired | chen_greedy_algorithm_legacy | 31 | 40 | 26 | 20 | 11 | 46 | 46 | False | 0.376 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| C4_01_26_paired | beam | 31 | 40 | 29 | 11 | 20 | 40 | 40 | True | 3.227 | 275 | 759 | 0 | BEST_EFFORT |
| C4_01_26_paired | qubo_beam | 31 | 40 | 29 | 11 | 20 | 40 | 40 | True | 4.067 | 275 | 759 | 6 | BEST_EFFORT |
| C5_01_26_paired | chen_greedy_algorithm_legacy | 29 | 37 | 25 | 18 | 11 | 43 | 43 | False | 0.364 | 0 | 0 | 0 | historical; cap=1.5*N0 |
| C5_01_26_paired | beam | 29 | 37 | 21 | 16 | 13 | 37 | 37 | True | 2.968 | 201 | 519 | 0 | BEST_EFFORT |
| C5_01_26_paired | qubo_beam | 29 | 37 | 21 | 16 | 13 | 37 | 37 | True | 3.071 | 202 | 532 | 5 | BEST_EFFORT |

## Reading Notes

- Lower `Final N` is better; `Reduction=N0-Final N` is the direct business improvement.
- `OK13` checks the current stricter `1.3*N0` cap. Historical greedy used `1.5*N0`, so it can fail `OK13` even when its original report said compliant.
- `BEST_EFFORT` means the search returned the best trajectory seen within the time/depth budget, not a proof of optimality.

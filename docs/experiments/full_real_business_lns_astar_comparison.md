# Full Real Business LNS / Weighted-A* Benchmark

This report tests exact-inspired non-MIP benchmarks that are intended to be stronger than simple heuristics but faster than the full 4D MIP.

CSV: `docs\experiments\full_real_business_lns_astar_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| beam_focused_with_qubo_portfolio | 15 | 8.53 | 20.53 | 26.80 | 37.313 | 46.47 | 15 |
| beam_focused_with_tuned_qubo | 15 | 8.80 | 20.27 | 26.47 | 21.491 | 32.93 | 15 |
| beam_focused_with_dense_qubo | 15 | 8.93 | 20.13 | 26.33 | 18.559 | 27.27 | 15 |
| qbo_beam_standalone_portfolio | 15 | 9.00 | 20.07 | 26.87 | 26.027 | 46.47 | 15 |
| beam_portfolio_with_qubo | 15 | 9.33 | 19.73 | 25.47 | 41.332 | 5.67 | 15 |
| beam_focused_with_qubo | 15 | 9.33 | 19.73 | 25.53 | 14.217 | 5.67 | 15 |
| beam_portfolio_plain | 15 | 9.47 | 19.60 | 25.40 | 38.400 | 0.00 | 15 |
| beam_focused_plain | 15 | 9.47 | 19.60 | 25.47 | 11.286 | 0.00 | 15 |
| qbo_beam_dense | 15 | 10.07 | 19.00 | 25.20 | 7.273 | 27.27 | 15 |
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 5.67 | 15 |
| qbo_beam_base_rerun | 15 | 10.60 | 18.47 | 24.27 | 4.205 | 5.67 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 0.00 | 15 |
| qbo_beam_low_div | 15 | 10.67 | 18.40 | 24.73 | 5.628 | 7.00 | 15 |
| enhanced_beam_fast | 15 | 10.87 | 18.20 | 24.27 | 8.826 | 0.00 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 0.00 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 0.00 | 15 |
| weighted_astar_business | 15 | 11.60 | 17.47 | 23.93 | 10.004 | 0.00 | 15 |
| genetic_tabu_baseline | 15 | 11.73 | 17.33 | 23.73 | 6.175 | 0.00 | 15 |
| qbo_beam_wide | 15 | 11.87 | 17.20 | 21.67 | 8.921 | 6.53 | 15 |
| hybrid_enhanced_beam | 15 | 12.13 | 16.93 | 19.27 | 10.028 | 0.00 | 15 |
| tabu_search_baseline | 15 | 12.27 | 16.80 | 22.93 | 4.787 | 0.00 | 15 |
| vns_baseline | 15 | 12.60 | 16.47 | 22.53 | 4.566 | 0.00 | 15 |
| grasp_baseline | 15 | 12.73 | 16.33 | 21.87 | 5.006 | 0.00 | 15 |
| lns_beam_repair | 15 | 13.87 | 15.20 | 16.80 | 24.015 | 17.60 | 15 |
| lns_qubo_repair | 15 | 14.13 | 14.93 | 16.20 | 30.283 | 59.73 | 15 |
| simulated_annealing_baseline | 15 | 14.53 | 14.53 | 22.27 | 3.446 | 0.00 | 15 |
| enhanced_beam_wide | 15 | 15.20 | 13.87 | 14.33 | 10.119 | 0.00 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 15 |

## Per Case Final N

| case | beam_focused_with_qubo_portfolio | beam_focused_with_tuned_qubo | beam_focused_with_dense_qubo | qbo_beam_standalone_portfolio | beam_portfolio_with_qubo | beam_focused_with_qubo | beam_portfolio_plain | beam_focused_plain | qbo_beam_dense | qubo_beam | qbo_beam_base_rerun | beam | qbo_beam_low_div | enhanced_beam_fast | wide_beam_baseline | multi_start_greedy_repair | weighted_astar_business | genetic_tabu_baseline | qbo_beam_wide | hybrid_enhanced_beam | tabu_search_baseline | vns_baseline | grasp_baseline | lns_beam_repair | lns_qubo_repair | simulated_annealing_baseline | enhanced_beam_wide | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 16 | 16 | 16 | 18 | 16 | 16 | 16 | 16 | 18 | 18 | 18 | 18 | 21 | 16 | 18 | 16 | 16 | 16 | 24 | 16 | 16 | 16 | 17 | 17 | 17 | 19 | 18 | 22 |
| A2_01_26_paired | 1 | 1 | 1 | 1 | 4 | 4 | 4 | 4 | 1 | 5 | 5 | 4 | 4 | 7 | 4 | 6 | 10 | 10 | 9 | 8 | 10 | 10 | 8 | 9 | 9 | 13 | 12 | 16 |
| A3_01_26_paired | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 12 | 13 | 12 | 9 | 13 | 10 | 14 | 14 | 13 | 13 | 13 | 15 | 15 | 12 | 15 | 17 |
| A4_01_26_paired | 8 | 9 | 9 | 8 | 9 | 9 | 9 | 9 | 14 | 13 | 13 | 13 | 8 | 9 | 13 | 17 | 12 | 17 | 13 | 15 | 13 | 18 | 14 | 19 | 18 | 18 | 19 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 18 | 19 | 15 | 15 | 16 | 18 | 19 | 17 | 18 | 22 | 18 | 22 |
| B1_01_26_paired | 3 | 6 | 6 | 3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 3 | 10 | 6 | 12 | 12 | 12 | 5 | 12 | 12 | 13 | 14 | 13 | 13 | 12 | 17 | 14 |
| B2_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 10 | 9 | 9 | 9 | 12 | 10 | 10 | 10 | 11 | 10 | 9 | 11 | 10 | 11 | 11 | 12 | 13 | 13 | 12 | 15 |
| B3_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 12 | 12 | 12 | 12 | 9 | 9 | 13 | 7 | 10 | 8 | 13 | 10 | 14 | 8 | 11 | 11 | 10 | 13 | 12 | 11 |
| B4_01_26_paired | 1 | 1 | 1 | 1 | 6 | 6 | 6 | 6 | 1 | 10 | 10 | 10 | 10 | 6 | 10 | 10 | 5 | 11 | 10 | 8 | 10 | 11 | 10 | 11 | 11 | 10 | 13 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 8 | 3 | 7 | 9 | 3 | 5 | 8 | 6 | 8 | 10 | 13 | 13 | 10 | 14 | 17 |
| C1_01_26_paired | 5 | 5 | 5 | 7 | 5 | 5 | 5 | 5 | 7 | 7 | 7 | 7 | 7 | 5 | 6 | 6 | 6 | 6 | 7 | 7 | 6 | 6 | 8 | 7 | 8 | 10 | 10 | 12 |
| C2_01_26_paired | 7 | 7 | 9 | 7 | 7 | 7 | 9 | 9 | 10 | 7 | 7 | 9 | 10 | 14 | 10 | 12 | 14 | 16 | 10 | 12 | 14 | 16 | 14 | 14 | 16 | 16 | 15 | 17 |
| C3_01_26_paired | 15 | 15 | 15 | 17 | 15 | 15 | 15 | 15 | 17 | 17 | 17 | 17 | 17 | 15 | 17 | 14 | 15 | 17 | 17 | 17 | 18 | 17 | 17 | 18 | 18 | 19 | 20 | 24 |
| C4_01_26_paired | 10 | 10 | 10 | 11 | 10 | 10 | 10 | 10 | 11 | 11 | 11 | 11 | 11 | 10 | 11 | 11 | 14 | 5 | 11 | 13 | 10 | 10 | 9 | 15 | 15 | 15 | 15 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 18 | 16 | 16 | 16 | 9 | 16 | 16 | 16 | 16 | 14 | 16 | 17 | 18 | 16 | 18 | 20 |

## Notes

- `lns_beam_repair` starts from `qbo_beam_dense`, rolls back selected suffix windows, and replans the suffix with beam search.
- `lns_qubo_repair` uses the same prefix rollback scheme but replans each suffix with QUBO beam.
- `weighted_astar_business` is a best-first search over legal business-feasible states with a weighted disorder/inversion heuristic.
- These are benchmark routes, not optimality proofs.

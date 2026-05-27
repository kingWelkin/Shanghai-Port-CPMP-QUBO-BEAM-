# Full Real Business Enhanced QUBO Beam Comparison

This report tests QUBO selection after the improved enhanced-beam candidate generation and business ranking.
The QUBO samples are only layer-selection hints; selections are repaired to exact beam width and all moves remain classical legal moves.

CSV: `docs\experiments\full_real_business_enhanced_qubo_beam_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| beam_portfolio_with_qubo | 15 | 9.33 | 19.73 | 25.47 | 41.332 | 5.67 | 15 |
| beam_focused_with_qubo | 15 | 9.33 | 19.73 | 25.53 | 14.217 | 5.67 | 15 |
| beam_portfolio_plain | 15 | 9.47 | 19.60 | 25.40 | 38.400 | 0.00 | 15 |
| beam_focused_plain | 15 | 9.47 | 19.60 | 25.47 | 11.286 | 0.00 | 15 |
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 5.67 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 0.00 | 15 |
| enhanced_beam_fast | 15 | 10.87 | 18.20 | 24.27 | 8.826 | 0.00 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 0.00 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 0.00 | 15 |
| genetic_tabu_baseline | 15 | 11.73 | 17.33 | 23.73 | 6.175 | 0.00 | 15 |
| hybrid_enhanced_beam | 15 | 12.13 | 16.93 | 19.27 | 10.028 | 0.00 | 15 |
| tabu_search_baseline | 15 | 12.27 | 16.80 | 22.93 | 4.787 | 0.00 | 15 |
| enhanced_qubo_beam_elite | 15 | 12.33 | 16.73 | 20.13 | 9.838 | 3.93 | 15 |
| vns_baseline | 15 | 12.60 | 16.47 | 22.53 | 4.566 | 0.00 | 15 |
| grasp_baseline | 15 | 12.73 | 16.33 | 21.87 | 5.006 | 0.00 | 15 |
| enhanced_qubo_beam_ranked | 15 | 13.33 | 15.73 | 17.33 | 10.301 | 5.60 | 15 |
| simulated_annealing_baseline | 15 | 14.53 | 14.53 | 22.27 | 3.446 | 0.00 | 15 |
| enhanced_beam_wide | 15 | 15.20 | 13.87 | 14.33 | 10.119 | 0.00 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 15 |

## Per Case Final N

| case | beam_portfolio_with_qubo | beam_focused_with_qubo | beam_portfolio_plain | beam_focused_plain | qubo_beam | beam | enhanced_beam_fast | wide_beam_baseline | multi_start_greedy_repair | genetic_tabu_baseline | hybrid_enhanced_beam | tabu_search_baseline | enhanced_qubo_beam_elite | vns_baseline | grasp_baseline | enhanced_qubo_beam_ranked | simulated_annealing_baseline | enhanced_beam_wide | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 16 | 16 | 16 | 16 | 18 | 18 | 16 | 18 | 16 | 16 | 16 | 16 | 16 | 16 | 17 | 16 | 19 | 18 | 22 |
| A2_01_26_paired | 4 | 4 | 4 | 4 | 5 | 4 | 7 | 4 | 6 | 10 | 8 | 10 | 8 | 10 | 8 | 9 | 13 | 12 | 16 |
| A3_01_26_paired | 10 | 10 | 10 | 10 | 10 | 10 | 13 | 12 | 9 | 10 | 14 | 13 | 13 | 13 | 13 | 15 | 12 | 15 | 17 |
| A4_01_26_paired | 9 | 9 | 9 | 9 | 13 | 13 | 9 | 13 | 17 | 17 | 15 | 13 | 14 | 18 | 14 | 18 | 18 | 19 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 19 | 15 | 16 | 18 | 18 | 19 | 18 | 22 | 18 | 22 |
| B1_01_26_paired | 6 | 6 | 6 | 6 | 6 | 6 | 10 | 6 | 12 | 12 | 12 | 12 | 12 | 13 | 14 | 12 | 12 | 17 | 14 |
| B2_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 10 | 10 | 10 | 10 | 11 | 10 | 10 | 11 | 11 | 12 | 13 | 12 | 15 |
| B3_01_26_paired | 9 | 9 | 9 | 9 | 12 | 12 | 9 | 13 | 7 | 8 | 10 | 14 | 9 | 8 | 11 | 9 | 13 | 12 | 11 |
| B4_01_26_paired | 6 | 6 | 6 | 6 | 10 | 10 | 6 | 10 | 10 | 11 | 8 | 10 | 10 | 11 | 10 | 11 | 10 | 13 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 3 | 3 | 3 | 8 | 3 | 7 | 3 | 8 | 6 | 11 | 8 | 10 | 11 | 10 | 14 | 17 |
| C1_01_26_paired | 5 | 5 | 5 | 5 | 7 | 7 | 5 | 6 | 6 | 6 | 7 | 6 | 5 | 6 | 8 | 7 | 10 | 10 | 12 |
| C2_01_26_paired | 7 | 7 | 9 | 9 | 7 | 9 | 14 | 10 | 12 | 16 | 12 | 14 | 14 | 16 | 14 | 16 | 16 | 15 | 17 |
| C3_01_26_paired | 15 | 15 | 15 | 15 | 17 | 17 | 15 | 17 | 14 | 17 | 17 | 18 | 17 | 17 | 17 | 17 | 19 | 20 | 24 |
| C4_01_26_paired | 10 | 10 | 10 | 10 | 11 | 11 | 10 | 11 | 11 | 5 | 13 | 10 | 12 | 10 | 9 | 13 | 15 | 15 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 14 | 16 | 16 | 16 | 18 | 20 |

## Notes

- `enhanced_qubo_beam_ranked` calls QUBO more often and uses a moderate elite lock.
- `enhanced_qubo_beam_elite` is conservative: most elite beam states are fixed before QUBO repair.
- Compare these rows against `beam_focused_plain` and `beam_focused_with_qubo` to decide whether QUBO should be integrated into the best beam itself or kept as a separate portfolio component.

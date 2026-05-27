# Full Real Business Focused Beam Comparison

This report adds focused beam portfolios selected from the completed beam experiments. They are cheaper than the full beam portfolio and avoid the short-probe selection failure observed in adaptive racing.

CSV: `docs\experiments\full_real_business_focused_beam_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|
| beam_focused_with_qubo_portfolio | 15 | 8.53 | 20.53 | 26.80 | 37.313 | 15 |
| beam_focused_with_tuned_qubo | 15 | 8.80 | 20.27 | 26.47 | 21.491 | 15 |
| beam_focused_with_dense_qubo | 15 | 8.93 | 20.13 | 26.33 | 18.559 | 15 |
| qbo_beam_standalone_portfolio | 15 | 9.00 | 20.07 | 26.87 | 26.027 | 15 |
| beam_portfolio_with_qubo | 15 | 9.33 | 19.73 | 25.47 | 41.332 | 15 |
| beam_focused_with_qubo | 15 | 9.33 | 19.73 | 25.53 | 14.217 | 15 |
| beam_portfolio_plain | 15 | 9.47 | 19.60 | 25.40 | 38.400 | 15 |
| beam_focused_plain | 15 | 9.47 | 19.60 | 25.47 | 11.286 | 15 |
| qbo_beam_dense | 15 | 10.07 | 19.00 | 25.20 | 7.273 | 15 |
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 15 |
| qbo_beam_base_rerun | 15 | 10.60 | 18.47 | 24.27 | 4.205 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 15 |
| qbo_beam_low_div | 15 | 10.67 | 18.40 | 24.73 | 5.628 | 15 |
| enhanced_beam_fast | 15 | 10.87 | 18.20 | 24.27 | 8.826 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 15 |
| genetic_tabu_baseline | 15 | 11.73 | 17.33 | 23.73 | 6.175 | 15 |
| qbo_beam_wide | 15 | 11.87 | 17.20 | 21.67 | 8.921 | 15 |
| hybrid_enhanced_beam | 15 | 12.13 | 16.93 | 19.27 | 10.028 | 15 |
| tabu_search_baseline | 15 | 12.27 | 16.80 | 22.93 | 4.787 | 15 |
| vns_baseline | 15 | 12.60 | 16.47 | 22.53 | 4.566 | 15 |
| grasp_baseline | 15 | 12.73 | 16.33 | 21.87 | 5.006 | 15 |
| simulated_annealing_baseline | 15 | 14.53 | 14.53 | 22.27 | 3.446 | 15 |
| enhanced_beam_wide | 15 | 15.20 | 13.87 | 14.33 | 10.119 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 15 |

## Per Case Final N

| case | beam_focused_with_qubo_portfolio | beam_focused_with_tuned_qubo | beam_focused_with_dense_qubo | qbo_beam_standalone_portfolio | beam_portfolio_with_qubo | beam_focused_with_qubo | beam_portfolio_plain | beam_focused_plain | qbo_beam_dense | qubo_beam | qbo_beam_base_rerun | beam | qbo_beam_low_div | enhanced_beam_fast | wide_beam_baseline | multi_start_greedy_repair | genetic_tabu_baseline | qbo_beam_wide | hybrid_enhanced_beam | tabu_search_baseline | vns_baseline | grasp_baseline | simulated_annealing_baseline | enhanced_beam_wide | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 16 | 16 | 16 | 18 | 16 | 16 | 16 | 16 | 18 | 18 | 18 | 18 | 21 | 16 | 18 | 16 | 16 | 24 | 16 | 16 | 16 | 17 | 19 | 18 | 22 |
| A2_01_26_paired | 1 | 1 | 1 | 1 | 4 | 4 | 4 | 4 | 1 | 5 | 5 | 4 | 4 | 7 | 4 | 6 | 10 | 9 | 8 | 10 | 10 | 8 | 13 | 12 | 16 |
| A3_01_26_paired | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 12 | 13 | 12 | 9 | 10 | 14 | 14 | 13 | 13 | 13 | 12 | 15 | 17 |
| A4_01_26_paired | 8 | 9 | 9 | 8 | 9 | 9 | 9 | 9 | 14 | 13 | 13 | 13 | 8 | 9 | 13 | 17 | 17 | 13 | 15 | 13 | 18 | 14 | 18 | 19 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 16 | 19 | 15 | 15 | 16 | 18 | 19 | 22 | 18 | 22 |
| B1_01_26_paired | 3 | 6 | 6 | 3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 3 | 10 | 6 | 12 | 12 | 5 | 12 | 12 | 13 | 14 | 12 | 17 | 14 |
| B2_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 10 | 9 | 9 | 9 | 12 | 10 | 10 | 10 | 10 | 9 | 11 | 10 | 11 | 11 | 13 | 12 | 15 |
| B3_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 12 | 12 | 12 | 12 | 9 | 9 | 13 | 7 | 8 | 13 | 10 | 14 | 8 | 11 | 13 | 12 | 11 |
| B4_01_26_paired | 1 | 1 | 1 | 1 | 6 | 6 | 6 | 6 | 1 | 10 | 10 | 10 | 10 | 6 | 10 | 10 | 11 | 10 | 8 | 10 | 11 | 10 | 10 | 13 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 8 | 3 | 7 | 3 | 5 | 8 | 6 | 8 | 10 | 10 | 14 | 17 |
| C1_01_26_paired | 5 | 5 | 5 | 7 | 5 | 5 | 5 | 5 | 7 | 7 | 7 | 7 | 7 | 5 | 6 | 6 | 6 | 7 | 7 | 6 | 6 | 8 | 10 | 10 | 12 |
| C2_01_26_paired | 7 | 7 | 9 | 7 | 7 | 7 | 9 | 9 | 10 | 7 | 7 | 9 | 10 | 14 | 10 | 12 | 16 | 10 | 12 | 14 | 16 | 14 | 16 | 15 | 17 |
| C3_01_26_paired | 15 | 15 | 15 | 17 | 15 | 15 | 15 | 15 | 17 | 17 | 17 | 17 | 17 | 15 | 17 | 14 | 17 | 17 | 17 | 18 | 17 | 17 | 19 | 20 | 24 |
| C4_01_26_paired | 10 | 10 | 10 | 11 | 10 | 10 | 10 | 10 | 11 | 11 | 11 | 11 | 11 | 10 | 11 | 11 | 5 | 11 | 13 | 10 | 10 | 9 | 15 | 15 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 18 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 14 | 16 | 16 | 18 | 20 |

## Notes

- `beam_focused_plain` runs only `beam` and `enhanced_beam_fast`.
- `beam_focused_with_qubo` adds `qubo_beam` to recover cases where QUBO improves the standard beam.
- `beam_focused_with_dense_qubo` replaces the original QUBO branch with the denser standalone QUBO configuration.
- `beam_focused_with_tuned_qubo` keeps both original `qubo_beam` and `qbo_beam_dense`.
- `beam_focused_with_qubo_portfolio` plugs the QUBO-only mini-portfolio into the original focused algorithm.
- Runtime is conservatively summed across the selected components.

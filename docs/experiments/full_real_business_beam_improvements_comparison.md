# Full Real Business Beam Improvements Comparison

This report tests three additional beam improvement ideas: fused single-search beam, feature-rule adaptive beam, and post-repair suffix beam.

CSV: `docs\experiments\full_real_business_beam_improvements_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|
| beam_portfolio_with_qubo | 15 | 9.33 | 19.73 | 25.47 | 41.332 | 15 |
| beam_focused_with_qubo | 15 | 9.33 | 19.73 | 25.53 | 14.217 | 15 |
| beam_portfolio_plain | 15 | 9.47 | 19.60 | 25.40 | 38.400 | 15 |
| beam_focused_plain | 15 | 9.47 | 19.60 | 25.47 | 11.286 | 15 |
| feature_adaptive_beam | 15 | 9.93 | 19.13 | 24.80 | 5.568 | 15 |
| qubo_beam | 15 | 10.60 | 18.47 | 24.27 | 2.932 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 2.460 | 15 |
| beam_post_repair | 15 | 10.73 | 18.33 | 25.27 | 3.332 | 15 |
| enhanced_beam_fast | 15 | 10.87 | 18.20 | 24.27 | 8.826 | 15 |
| wide_beam_baseline | 15 | 10.93 | 18.13 | 24.13 | 6.968 | 15 |
| multi_start_greedy_repair | 15 | 11.27 | 17.80 | 24.00 | 10.175 | 15 |
| fused_beam | 15 | 11.73 | 17.33 | 20.47 | 10.042 | 15 |
| genetic_tabu_baseline | 15 | 11.73 | 17.33 | 23.73 | 6.175 | 15 |
| hybrid_enhanced_beam | 15 | 12.13 | 16.93 | 19.27 | 10.028 | 15 |
| tabu_search_baseline | 15 | 12.27 | 16.80 | 22.93 | 4.787 | 15 |
| vns_baseline | 15 | 12.60 | 16.47 | 22.53 | 4.566 | 15 |
| grasp_baseline | 15 | 12.73 | 16.33 | 21.87 | 5.006 | 15 |
| simulated_annealing_baseline | 15 | 14.53 | 14.53 | 22.27 | 3.446 | 15 |
| enhanced_beam_wide | 15 | 15.20 | 13.87 | 14.33 | 10.119 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 15 |

## Per Case Final N

| case | beam_portfolio_with_qubo | beam_focused_with_qubo | beam_portfolio_plain | beam_focused_plain | feature_adaptive_beam | qubo_beam | beam | beam_post_repair | enhanced_beam_fast | wide_beam_baseline | multi_start_greedy_repair | fused_beam | genetic_tabu_baseline | hybrid_enhanced_beam | tabu_search_baseline | vns_baseline | grasp_baseline | simulated_annealing_baseline | enhanced_beam_wide | chen_greedy_algorithm_1p3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 16 | 16 | 16 | 16 | 16 | 18 | 18 | 16 | 16 | 18 | 16 | 16 | 16 | 16 | 16 | 16 | 17 | 19 | 18 | 22 |
| A2_01_26_paired | 4 | 4 | 4 | 4 | 4 | 5 | 4 | 4 | 7 | 4 | 6 | 9 | 10 | 8 | 10 | 10 | 8 | 13 | 12 | 16 |
| A3_01_26_paired | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 13 | 13 | 12 | 9 | 11 | 10 | 14 | 13 | 13 | 13 | 12 | 15 | 17 |
| A4_01_26_paired | 9 | 9 | 9 | 9 | 9 | 13 | 13 | 13 | 9 | 13 | 17 | 17 | 17 | 15 | 13 | 18 | 14 | 18 | 19 | 16 |
| A5_01_26_paired | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 14 | 15 | 15 | 16 | 14 | 19 | 15 | 16 | 18 | 19 | 22 | 18 | 22 |
| B1_01_26_paired | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 4 | 10 | 6 | 12 | 11 | 12 | 12 | 12 | 13 | 14 | 12 | 17 | 14 |
| B2_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 11 | 10 | 10 | 10 | 11 | 10 | 11 | 10 | 11 | 11 | 13 | 12 | 15 |
| B3_01_26_paired | 9 | 9 | 9 | 9 | 9 | 12 | 12 | 11 | 9 | 13 | 7 | 10 | 8 | 10 | 14 | 8 | 11 | 13 | 12 | 11 |
| B4_01_26_paired | 6 | 6 | 6 | 6 | 10 | 10 | 10 | 9 | 6 | 10 | 10 | 10 | 11 | 8 | 10 | 11 | 10 | 10 | 13 | 14 |
| B5_01_26_paired | 3 | 3 | 3 | 3 | 8 | 3 | 3 | 3 | 8 | 3 | 7 | 8 | 3 | 8 | 6 | 8 | 10 | 10 | 14 | 17 |
| C1_01_26_paired | 5 | 5 | 5 | 5 | 5 | 7 | 7 | 6 | 5 | 6 | 6 | 2 | 6 | 7 | 6 | 6 | 8 | 10 | 10 | 12 |
| C2_01_26_paired | 7 | 7 | 9 | 9 | 7 | 7 | 9 | 14 | 14 | 10 | 12 | 11 | 16 | 12 | 14 | 16 | 14 | 16 | 15 | 17 |
| C3_01_26_paired | 15 | 15 | 15 | 15 | 15 | 17 | 17 | 16 | 15 | 17 | 14 | 18 | 17 | 17 | 18 | 17 | 17 | 19 | 20 | 24 |
| C4_01_26_paired | 10 | 10 | 10 | 10 | 10 | 11 | 11 | 11 | 10 | 11 | 11 | 12 | 5 | 13 | 10 | 10 | 9 | 15 | 15 | 20 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 14 | 16 | 16 | 18 | 20 |

## Notes

- `fused_beam` shares one visited set and one beam while combining plain, enhanced, and QUBO selection signals.
- `feature_adaptive_beam` uses initial layout features to choose one beam variant.
- `beam_post_repair` first gets an incumbent from standard beam, then runs suffix beam repair from that state.

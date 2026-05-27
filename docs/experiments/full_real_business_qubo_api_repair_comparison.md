# Full Real Business QUBO API Repair Comparison

The new repair uses all returned samples, projects each sample to exact beam cardinality, then applies objective-aware swap repair before selecting the next beam. Both local SA QUBO and API QUBO now use this decoder.

CSV: `docs\experiments\full_real_business_qubo_api_repair_comparison.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | avg QUBO time s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qubo_beam_old_decode | 15 | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | 0.099 | 15 |
| qubo_beam_repair | 15 | 10.60 | 18.47 | 24.07 | 3.511 | 5.73 | 0.095 | 15 |
| beam | 15 | 10.67 | 18.40 | 24.13 | 3.689 | 0.00 | 0.000 | 15 |
| qubo_beam_api_sparse_repair | 15 | 10.73 | 18.33 | 24.00 | 7.083 | 2.13 | 4.716 | 15 |
| qubo_beam_api_sparse_old_decode | 15 | 10.73 | 18.33 | 24.20 | 7.087 | 2.13 | 4.544 | 15 |
| chen_greedy_algorithm_1p3 | 15 | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 0.000 | 15 |

## Per Case Final N

| case | greedy 1.3 | beam | qbo local old | qbo local repair | api sparse old | api sparse repair |
|---|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 22 | 18 | 18 | 18 | 18 | 18 |
| A2_01_26_paired | 16 | 4 | 4 | 4 | 4 | 4 |
| A3_01_26_paired | 17 | 10 | 10 | 10 | 10 | 10 |
| A4_01_26_paired | 16 | 13 | 13 | 13 | 13 | 13 |
| A5_01_26_paired | 22 | 15 | 15 | 15 | 15 | 15 |
| B1_01_26_paired | 14 | 6 | 5 | 5 | 6 | 6 |
| B2_01_26_paired | 15 | 9 | 9 | 9 | 9 | 9 |
| B3_01_26_paired | 11 | 12 | 12 | 12 | 12 | 12 |
| B4_01_26_paired | 14 | 10 | 10 | 10 | 10 | 10 |
| B5_01_26_paired | 17 | 3 | 3 | 3 | 3 | 3 |
| C1_01_26_paired | 12 | 7 | 7 | 7 | 7 | 7 |
| C2_01_26_paired | 17 | 9 | 7 | 9 | 10 | 10 |
| C3_01_26_paired | 24 | 17 | 17 | 17 | 17 | 17 |
| C4_01_26_paired | 20 | 11 | 11 | 11 | 11 | 11 |
| C5_01_26_paired | 20 | 16 | 16 | 16 | 16 | 16 |

## Notes

- Current local QUBO repair reduced average moves from 24.60 to 24.07, but average final N changed from 10.47 to 10.60 on this run. The one-step repaired beam objective can still alter later search quality, so the repair slack is now configurable.
- API sparse repair preserved the previous sparse API average final N=10.73, while improving A2 move count/G-end from 30/34 to 27/31 at the same final N=4.
- API call overhead is still the dominant runtime component; local SA QUBO remains faster for this single-instance beam-selection use case.

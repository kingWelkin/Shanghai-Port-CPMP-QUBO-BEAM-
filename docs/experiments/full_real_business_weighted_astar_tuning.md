# Full Real Business Weighted-A* Tuning

This report compares the previous single-queue weighted-A* benchmark with the enhanced portfolio weighted-A* route.

CSV: `docs\experiments\full_real_business_weighted_astar_tuning.csv`

## Aggregate

| solver | cases | avg final N | avg reduction | avg moves | avg runtime s | ok 1.3N0 |
|---|---:|---:|---:|---:|---:|---:|
| weighted_astar_business_portfolio | 15 | 10.13 | 18.93 | 25.80 | 30.053 | 15 |
| weighted_astar_business_previous | 15 | 11.60 | 17.47 | 23.93 | 10.004 | 15 |

## Per Case Final N

| case | weighted_astar_business_portfolio | weighted_astar_business_previous |
|---|---:|---:|
| A1_01_26_paired | 16 | 16 |
| A2_01_26_paired | 4 | 10 |
| A3_01_26_paired | 13 | 13 |
| A4_01_26_paired | 12 | 12 |
| A5_01_26_paired | 15 | 18 |
| B1_01_26_paired | 6 | 12 |
| B2_01_26_paired | 11 | 11 |
| B3_01_26_paired | 7 | 10 |
| B4_01_26_paired | 5 | 5 |
| B5_01_26_paired | 6 | 9 |
| C1_01_26_paired | 6 | 6 |
| C2_01_26_paired | 13 | 14 |
| C3_01_26_paired | 15 | 15 |
| C4_01_26_paired | 14 | 14 |
| C5_01_26_paired | 9 | 9 |

## Notes

- `weighted_astar_business_previous` is the earlier single-queue weighted-A* result from the LNS/A* benchmark.
- `weighted_astar_business_portfolio` runs legacy, aggressive multi-queue, and hybrid multi-queue A* branches, then keeps the best validated route.
- The portfolio is still a best-effort heuristic benchmark; it is not an optimality proof.

# Open CPMP Time-Matched No-QUBO Ablation

result_csv: `docs\experiments\open_cpmp_time_matched_5x5_cv123.csv`
breakdown_csv: `docs\experiments\open_cpmp_time_matched_5x5_cv123_breakdown.csv`

| family | variant | cases | N=0 solved | moves total | avg runtime sec | avg budget sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 5x5_cv1_test | classic_time_matched_best | 20 | 12 | 360 | 0.697 | 1.067 | 4.500 |
| 5x5_cv1_test | classic_tm_w1200 | 20 | 12 | 362 | 0.640 | 1.067 | 4.667 |
| 5x5_cv1_test | classic_tm_w2400 | 20 | 2 | 51 | 1.065 | 1.067 | 2.500 |
| 5x5_cv1_test | classic_tm_w4800 | 20 | 0 | 0 | 1.087 | 1.067 | 0.000 |
| 5x5_cv1_test | qubo_reference | 20 | 10 | 281 | 1.067 | 1.067 | 2.600 |
| 5x5_cv2_test | classic_time_matched_best | 20 | 14 | 385 | 0.797 | 1.205 | 3.071 |
| 5x5_cv2_test | classic_tm_w1200 | 20 | 12 | 335 | 0.674 | 1.205 | 3.333 |
| 5x5_cv2_test | classic_tm_w2400 | 20 | 5 | 129 | 1.021 | 1.205 | 2.200 |
| 5x5_cv2_test | classic_tm_w4800 | 20 | 1 | 26 | 1.193 | 1.205 | 1.000 |
| 5x5_cv2_test | qubo_adaptive_reference | 20 | 14 | 368 | 1.161 | 1.161 | 2.000 |
| 5x5_cv2_test | qubo_reference | 20 | 14 | 368 | 1.205 | 1.205 | 2.000 |
| 5x5_cv3_test | classic_time_matched_best | 20 | 20 | 518 | 0.729 | 1.153 | 2.000 |
| 5x5_cv3_test | classic_tm_w1200 | 20 | 20 | 522 | 0.650 | 1.153 | 2.200 |
| 5x5_cv3_test | classic_tm_w2400 | 20 | 6 | 142 | 1.081 | 1.153 | 1.333 |
| 5x5_cv3_test | classic_tm_w4800 | 20 | 0 | 0 | 1.168 | 1.153 | 0.000 |
| 5x5_cv3_test | qubo_adaptive_reference | 20 | 18 | 460 | 1.170 | 1.170 | 2.056 |
| 5x5_cv3_test | qubo_reference | 20 | 18 | 460 | 1.153 | 1.153 | 2.056 |

## Case Breakdown

| family | category | cases |
|---|---|---:|
| 5x5_cv1_test | both_classic_fewer | 2 |
| 5x5_cv1_test | both_qubo_fewer | 2 |
| 5x5_cv1_test | both_tie | 4 |
| 5x5_cv1_test | classic_only | 4 |
| 5x5_cv1_test | neither_solved | 6 |
| 5x5_cv1_test | qubo_only | 2 |
| 5x5_cv2_test | both_classic_fewer | 2 |
| 5x5_cv2_test | both_qubo_fewer | 3 |
| 5x5_cv2_test | both_tie | 7 |
| 5x5_cv2_test | classic_only | 2 |
| 5x5_cv2_test | neither_solved | 4 |
| 5x5_cv2_test | qubo_only | 2 |
| 5x5_cv3_test | both_classic_fewer | 3 |
| 5x5_cv3_test | both_qubo_fewer | 2 |
| 5x5_cv3_test | both_tie | 13 |
| 5x5_cv3_test | classic_only | 2 |

# Open CPMP Time-Matched No-QUBO Ablation

result_csv: `docs\experiments\open_cpmp_time_matched_7x5_cv123.csv`
breakdown_csv: `docs\experiments\open_cpmp_time_matched_7x5_cv123_breakdown.csv`

| family | variant | cases | N=0 solved | moves total | avg runtime sec | avg budget sec | avg ref gap |
|---|---|---:|---:|---:|---:|---:|---:|
| 7x5_cv1_test | classic_time_matched_best | 10 | 6 | 249 | 1.981 | 2.728 | 10.500 |
| 7x5_cv1_test | classic_tm_w1200 | 10 | 6 | 249 | 1.981 | 2.728 | 10.500 |
| 7x5_cv1_test | classic_tm_w2400 | 10 | 0 | 0 | 2.743 | 2.728 | 0.000 |
| 7x5_cv1_test | classic_tm_w4800 | 10 | 0 | 0 | 2.762 | 2.728 | 0.000 |
| 7x5_cv1_test | qubo_reference | 10 | 4 | 149 | 2.728 | 2.728 | 5.750 |
| 7x5_cv2_test | classic_time_matched_best | 10 | 9 | 349 | 2.185 | 3.252 | 8.333 |
| 7x5_cv2_test | classic_tm_w1200 | 10 | 9 | 353 | 2.011 | 3.252 | 8.778 |
| 7x5_cv2_test | classic_tm_w2400 | 10 | 2 | 68 | 2.993 | 3.252 | 6.500 |
| 7x5_cv2_test | classic_tm_w4800 | 10 | 0 | 0 | 3.280 | 3.252 | 0.000 |
| 7x5_cv2_test | qubo_adaptive_reference | 10 | 8 | 273 | 2.052 | 2.052 | 4.500 |
| 7x5_cv2_test | qubo_reference | 10 | 9 | 330 | 3.252 | 3.252 | 6.222 |
| 7x5_cv3_test | classic_time_matched_best | 10 | 10 | 366 | 1.853 | 3.214 | 6.500 |
| 7x5_cv3_test | classic_tm_w1200 | 10 | 10 | 366 | 1.853 | 3.214 | 6.500 |
| 7x5_cv3_test | classic_tm_w2400 | 10 | 1 | 33 | 3.086 | 3.214 | 4.000 |
| 7x5_cv3_test | classic_tm_w4800 | 10 | 0 | 0 | 3.256 | 3.214 | 0.000 |
| 7x5_cv3_test | qubo_adaptive_reference | 10 | 10 | 359 | 2.783 | 2.783 | 5.800 |
| 7x5_cv3_test | qubo_reference | 10 | 10 | 356 | 3.214 | 3.214 | 5.500 |

## Case Breakdown

| family | category | cases |
|---|---|---:|
| 7x5_cv1_test | both_qubo_fewer | 4 |
| 7x5_cv1_test | classic_only | 2 |
| 7x5_cv1_test | neither_solved | 4 |
| 7x5_cv2_test | both_classic_fewer | 1 |
| 7x5_cv2_test | both_qubo_fewer | 5 |
| 7x5_cv2_test | both_tie | 3 |
| 7x5_cv2_test | neither_solved | 1 |
| 7x5_cv3_test | both_classic_fewer | 1 |
| 7x5_cv3_test | both_qubo_fewer | 6 |
| 7x5_cv3_test | both_tie | 3 |

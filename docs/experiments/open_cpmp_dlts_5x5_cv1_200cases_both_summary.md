# Open CPMP Benchmark Summary

source: `external\DLTS`
result_csv: `docs\experiments\open_cpmp_dlts_5x5_cv1_200cases_both.csv`

| variant | family | cases | N=0 solved | moves total | avg runtime sec | avg ref gap | qcalls | qtime sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| classic_noqubo | 5x5_cv1_test | 100 | 41 | 1149 | 0.709 | 3.561 | 0 | 0.000 |
| classic_noqubo | 5x5_cv1_validation | 100 | 43 | 1190 | 0.823 | 2.465 | 0 | 0.000 |
| qubo | 5x5_cv1_test | 100 | 51 | 1395 | 1.216 | 2.392 | 1931 | 2.123 |
| qubo | 5x5_cv1_validation | 100 | 50 | 1378 | 1.208 | 1.900 | 1693 | 1.864 |

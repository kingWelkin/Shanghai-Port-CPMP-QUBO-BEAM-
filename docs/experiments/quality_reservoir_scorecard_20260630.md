# Quality Reservoir Equal-Weight Scorecard

Date: 2026-06-30

## Scope

- variant: `qubo_scheduler_state_v5_qubo_diversity_union_quality_reservoir_polish`
- time limit: `10s` per case
- decision groups: `open120`, `BF`, `CV`; each group is treated as equally important.
- promotion rule: do not promote if any group materially regresses solved count or penalty versus the current same-day Union baseline.

## Results

| group | cases | solved | feasible | full quality penalty | solved moves | avg runtime sec | avg ref gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| open120 | 120 | 107 | 107 | 5075 | 3709 | 5.195 | 6.467 |
| BF | 401 | 318 | 318 | 75416 | 16412 | 7.633 | 0.000 |
| CV | 210 | 163 | 163 | 15482 | 4434 | 3.966 | 0.000 |

## Reservoir Telemetry

| group | generated | kept | selected | witness found | pair penalty |
|---|---:|---:|---:|---:|---:|
| open120 | 15505512 | 1863269 | 1416350 | 109 | 673.340 |
| BF | 172150395 | 2203387 | 1680031 | 12 | 100.860 |
| CV | 27343329 | 1995915 | 1519559 | 134 | 704.840 |

## Worst Family Slices

| group | family | cases | solved | penalty | solved moves |
|---|---|---:|---:|---:|---:|
| open120 | 5x5_cv1_test | 20 | 16 | 902 | 494 |
| open120 | 5x5_cv2_test | 20 | 16 | 855 | 455 |
| open120 | 5x5_cv3_test | 20 | 17 | 742 | 443 |
| open120 | 7x5_cv1_test | 20 | 19 | 920 | 774 |
| open120 | 7x5_cv2_test | 20 | 19 | 876 | 763 |
| open120 | 7x5_cv3_test | 20 | 20 | 780 | 780 |
| BF | BF30 | 10 | 0 | 5080 | 0 |
| BF | BF26 | 10 | 0 | 3810 | 0 |
| BF | BF28 | 10 | 0 | 3810 | 0 |
| BF | BF32 | 10 | 1 | 5080 | 118 |
| BF | BF31 | 10 | 3 | 4080 | 320 |
| BF | BF18 | 10 | 3 | 2380 | 135 |
| BF | BF29 | 10 | 4 | 4080 | 401 |
| BF | BF20 | 10 | 4 | 2380 | 180 |
| BF | BF14 | 10 | 5 | 4130 | 600 |
| BF | BF17 | 10 | 6 | 1900 | 220 |
| BF | BF16 | 10 | 7 | 3836 | 820 |
| BF | BF19 | 10 | 7 | 1900 | 259 |
| CV | 10-10 | 10 | 0 | 4396 | 0 |
| CV | 10-6 | 10 | 0 | 2675 | 0 |
| CV | 6-6 | 10 | 2 | 1117 | 87 |
| CV | 6-10 | 10 | 5 | 1654 | 390 |
| CV | 5-4 | 10 | 5 | 444 | 123 |
| CV | 5-10 | 10 | 8 | 879 | 509 |
| CV | 5-9 | 10 | 8 | 729 | 423 |
| CV | 5-5 | 10 | 8 | 458 | 217 |
| CV | 5-6 | 10 | 9 | 505 | 320 |
| CV | 4-4 | 10 | 9 | 159 | 138 |
| CV | 3-3 | 10 | 9 | 85 | 70 |
| CV | 5-7 | 10 | 10 | 538 | 382 |

## CSV Completeness Check

- no blank `final_n` or `runtime_sec`; no solved row has blank `moves`.

## Failure Sample

- `open120/5x5_cv1_test/2`: final_n=18 moves= runtime=9.149s feasible=False
- `open120/5x5_cv1_test/8`: final_n=19 moves= runtime=10.133s feasible=False
- `open120/5x5_cv1_test/12`: final_n=18 moves= runtime=9.922s feasible=False
- `open120/5x5_cv1_test/17`: final_n=18 moves= runtime=10.152s feasible=False
- `open120/5x5_cv2_test/1`: final_n=17 moves= runtime=6.385s feasible=False
- `open120/5x5_cv2_test/2`: final_n=18 moves= runtime=10.214s feasible=False
- `open120/5x5_cv2_test/12`: final_n=18 moves= runtime=10.247s feasible=False
- `open120/5x5_cv2_test/16`: final_n=19 moves= runtime=8.234s feasible=False
- `open120/5x5_cv3_test/1`: final_n=17 moves= runtime=6.323s feasible=False
- `open120/5x5_cv3_test/2`: final_n=18 moves= runtime=10.207s feasible=False
- `open120/5x5_cv3_test/16`: final_n=19 moves= runtime=10.030s feasible=False
- `open120/7x5_cv1_test/11`: final_n=27 moves= runtime=10.353s feasible=False
- `open120/7x5_cv2_test/10`: final_n=21 moves= runtime=10.137s feasible=False
- `BF/BF13/cpmp_16_8_103_21_62_6`: final_n=62 moves= runtime=10.780s feasible=False
- `BF/BF13/cpmp_16_8_103_21_62_9`: final_n=62 moves= runtime=10.710s feasible=False
- `BF/BF14/cpmp_16_8_103_21_78_1`: final_n=78 moves= runtime=10.422s feasible=False
- `BF/BF14/cpmp_16_8_103_21_78_2`: final_n=78 moves= runtime=10.714s feasible=False
- `BF/BF14/cpmp_16_8_103_21_78_4`: final_n=78 moves= runtime=10.469s feasible=False
- `BF/BF14/cpmp_16_8_103_21_78_6`: final_n=78 moves= runtime=10.640s feasible=False
- `BF/BF14/cpmp_16_8_103_21_78_9`: final_n=78 moves= runtime=10.820s feasible=False
- `BF/BF15/cpmp_16_8_103_42_62_5`: final_n=62 moves= runtime=10.416s feasible=False
- `BF/BF16/cpmp_16_8_103_42_78_2`: final_n=78 moves= runtime=10.731s feasible=False
- `BF/BF16/cpmp_16_8_103_42_78_5`: final_n=78 moves= runtime=10.857s feasible=False
- `BF/BF16/cpmp_16_8_103_42_78_8`: final_n=78 moves= runtime=10.384s feasible=False
- `BF/BF17/cpmp_20_5_60_12_36_1`: final_n=36 moves= runtime=10.756s feasible=False
- `BF/BF17/cpmp_20_5_60_12_36_6`: final_n=36 moves= runtime=11.002s feasible=False
- `BF/BF17/cpmp_20_5_60_12_36_9`: final_n=36 moves= runtime=10.580s feasible=False
- `BF/BF17/cpmp_20_5_60_12_36_10`: final_n=36 moves= runtime=10.794s feasible=False
- `BF/BF18/cpmp_20_5_60_12_45_1`: final_n=45 moves= runtime=10.101s feasible=False
- `BF/BF18/cpmp_20_5_60_12_45_2`: final_n=45 moves= runtime=10.430s feasible=False
- `BF/BF18/cpmp_20_5_60_12_45_4`: final_n=45 moves= runtime=10.633s feasible=False
- `BF/BF18/cpmp_20_5_60_12_45_5`: final_n=45 moves= runtime=10.755s feasible=False
- `BF/BF18/cpmp_20_5_60_12_45_8`: final_n=45 moves= runtime=10.565s feasible=False
- `CV/3-3/data3-3-4`: final_n=3 moves= runtime=0.005s feasible=False
- `CV/4-4/data4-4-6`: final_n=4 moves= runtime=0.150s feasible=False
- `CV/5-4/data5-4-1`: final_n=12 moves= runtime=6.045s feasible=False
- `CV/5-4/data5-4-2`: final_n=15 moves= runtime=5.197s feasible=False
- `CV/5-4/data5-4-6`: final_n=11 moves= runtime=5.862s feasible=False
- `CV/5-4/data5-4-8`: final_n=11 moves= runtime=4.656s feasible=False
- `CV/5-4/data5-4-9`: final_n=12 moves= runtime=4.982s feasible=False
- `CV/5-5/data5-5-1`: final_n=18 moves= runtime=6.492s feasible=False
- `CV/5-5/data5-5-4`: final_n=17 moves= runtime=6.291s feasible=False
- `CV/5-6/data5-6-9`: final_n=21 moves= runtime=10.066s feasible=False
- `CV/5-9/data5-9-1`: final_n=32 moves= runtime=10.029s feasible=False
- `CV/5-9/data5-9-10`: final_n=26 moves= runtime=5.045s feasible=False
- `CV/5-10/data5-10-5`: final_n=33 moves= runtime=10.318s feasible=False
- `CV/5-10/data5-10-10`: final_n=37 moves= runtime=10.088s feasible=False
- `CV/6-6/data6-6-1`: final_n=21 moves= runtime=9.873s feasible=False
- `CV/6-6/data6-6-2`: final_n=25 moves= runtime=10.402s feasible=False
- `CV/6-6/data6-6-3`: final_n=27 moves= runtime=10.026s feasible=False
- `CV/6-6/data6-6-4`: final_n=25 moves= runtime=10.057s feasible=False
- `CV/6-6/data6-6-5`: final_n=23 moves= runtime=10.252s feasible=False
- `CV/6-6/data6-6-6`: final_n=27 moves= runtime=10.654s feasible=False

## Decision

Do not promote this profile as mainline from these runs. The reservoir is active and measurable, but it reduces open120 solved count and leaves BF/CV hard-cap families unresolved. Keep it as an experimental ablation input for feature analysis, not as the strongest Union route.

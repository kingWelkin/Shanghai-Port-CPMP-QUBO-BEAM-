# Wang BS-B vs Current Algorithms: Global Comparison

This report compares the adapted Wang-Jin-Lim BS-B / BS-G baselines with the current compact QUBO beam family. The comparison is split by benchmark scope because the internal single-bay cases and open DLTS cases use different instance sets and reporting goals.

## Method

Primary metric order:

1. `final_n=0` solved count and validation status.
2. Business feasibility on internal cases, especially `g_ok_13`.
3. Move count, only after comparing solved coverage.
4. Runtime, reported as observed on this machine; time budgets differ by experiment family and are not a proof of asymptotic speed.

Wang settings used for the fresh comparison:

```text
solver = wang_bs_b / wang_bs_g
beam_width = 100
max_depth = 260
max_candidates_per_layer = 2000
time_limit = 10s
entropy_alpha = 10
```

Current algorithm references:

- Internal mainline: `speed_v2`, result file `docs/experiments/micro_optimized_speed_v2_full30.csv`.
- Internal experimental post-polish: `window_polish_v2`, result file `docs/experiments/scratch_window_polish_v2_full30.csv`.
- Open DLTS current raw comparison: best available per case among `qubo_reference` and `qubo_adaptive_reference` from the time-matched CSVs.
- Open DLTS scheduler line: `scheduler_state_v5` from `docs/guides/scheduler-state-v5-tuning-summary.md`.

## Internal 30 Single-Bay Cases

Fresh Wang result file: `docs/experiments/scratch_wang_internal30_w100.csv`.

| algorithm | solved | moves total | avg runtime sec | max runtime sec | `g_ok_13` among solved |
|---|---:|---:|---:|---:|---:|
| `speed_v2` | 30/30 | 509 | 2.237 | 5.597 | 30/30 |
| `window_polish_v2` | 30/30 | 509 | 2.395 | 4.897 | 30/30 |
| `wang_bs_b` | 30/30 | 757 | 1.635 | 4.967 | 9/30 |
| `wang_bs_g` | 30/30 | 926 | 0.281 | 0.528 | 2/30 |

Pairwise against `speed_v2` on the same 30 cases:

| Wang variant | common solved | `speed_v2` fewer moves | Wang fewer moves | ties | Wang minus `speed_v2` moves |
|---|---:|---:|---:|---:|---:|
| `wang_bs_b` | 30 | 24 | 0 | 6 | +248 |
| `wang_bs_g` | 30 | 28 | 0 | 2 | +417 |

Internal conclusion: Wang is a useful feasibility baseline, but it is not competitive with the current internal mainline. `speed_v2` has the same 30/30 solve count, 248 fewer moves than BS-B, and satisfies the 1.3 business cap on all solved cases. Wang BS-B is faster on average in this run, but the cap failures make it unsuitable as the internal deployment mainline.

## Open DLTS Test Subset

Common test subset:

- `5x5_cv1/cv2/cv3_test`: 20 cases per family, 60 total.
- `7x5_cv1/cv2/cv3_test`: 10 cases per family, 30 total.
- 90 cases total.

Wang sources:

- `docs/experiments/wang_bs_b_reproduction_5x5_cv123_20_w100.csv`
- `docs/experiments/wang_bs_b_reproduction_7x5_cv123_10_w100.csv`

Current raw sources:

- `docs/experiments/open_cpmp_time_matched_5x5_cv123.csv`
- `docs/experiments/open_cpmp_time_matched_7x5_cv123.csv`

Per-case current result selects the better available row among `qubo_reference` and `qubo_adaptive_reference`.

| scope | algorithm | solved | solved moves total | avg gap among solved | avg runtime sec |
|---|---|---:|---:|---:|---:|
| 5x5 test 60 | Wang BS-B | 60/60 | 1923 | 6.800 | 0.858 |
| 5x5 test 60 | current QUBO best raw | 42/60 | 1109 | 2.167 | 1.102 |
| 7x5 test 30 | Wang BS-B | 30/30 | 1300 | 12.200 | 2.560 |
| 7x5 test 30 | current QUBO best raw | 23/30 | 825 | 5.391 | 2.500 |
| all test 90 | Wang BS-B | 90/90 | 3223 | 8.600 | 1.425 |
| all test 90 | current QUBO best raw | 65/90 | 1934 | 3.308 | 1.568 |

Common-solved move comparison:

| scope | common solved | current fewer | Wang fewer | ties | Wang minus current moves | Wang-only solved |
|---|---:|---:|---:|---:|---:|---:|
| 5x5 test 60 | 42 | 35 | 3 | 4 | +158 | 18 |
| 7x5 test 30 | 23 | 19 | 4 | 0 | +133 | 7 |
| all test 90 | 65 | 54 | 7 | 4 | +291 | 25 |

Open-test conclusion: Wang BS-B is clearly stronger for global feasibility on the current DLTS test subset: it solves all 90 cases. The current QUBO family is clearly stronger on path quality when it solves: among the 65 cases solved by both, the current result is shorter on 54 cases and saves 291 moves total versus Wang.

## Open DLTS cv1 Test + Validation

Fresh Wang sources:

- `docs/experiments/scratch_wang_bs_b_repro_7x5_cv1_test20_w100.csv`
- `docs/experiments/scratch_wang_bs_b_repro_cv1_validation_w100.csv`

| scope | algorithm | solved | solved moves total | avg gap among solved | avg runtime sec |
|---|---|---:|---:|---:|---:|
| 5x5 cv1 test+validation 40 | Wang BS-B | 40/40 | 1410 | 8.625 | 0.645 |
| 5x5 cv1 test+validation 40 | `scheduler_state_v5` | 20/40 | 575 | not reported | 0.949 |
| 7x5 cv1 test+validation 40 | Wang BS-B | 40/40 | 1915 | 15.600 | 1.884 |
| 7x5 cv1 test+validation 40 | `scheduler_state_v5` | 23/40 | 878 | not reported | 2.135 |
| 7x5 cv1 test+validation 40 | `qubo_adaptive` | 22/40 | 836 | 6.955 | 2.014 |

cv1 conclusion: Wang BS-B is the stronger feasibility baseline on both 5x5 and 7x5 cv1 test+validation scopes. The current QUBO/scheduler variants have better reported gap among the cases they solve, but they do not yet match Wang's solve coverage on cv1.

## Open DLTS 7x5 cv2/cv3

This is the strongest open-DLTS slice for the current scheduler line.

| scope | algorithm | solved | solved moves total | avg runtime sec |
|---|---|---:|---:|---:|
| 7x5 cv2+cv3 test 20 | Wang BS-B | 20/20 | 820 | 2.865 |
| 7x5 cv2+cv3 test 20 | `scheduler_state_v5` | 20/20 | 738 | 2.286 |

7x5 cv2/cv3 conclusion: when the scheduler line also reaches full solve coverage, it beats Wang on both moves and runtime. This is the best evidence that the current approach can surpass Wang, but it is not yet global because cv1 remains weaker.

## Final Judgment

There is no honest single global winner yet.

Use this ranking instead:

| role | current winner | reason |
|---|---|---|
| Internal Shanggang 30 deployment/mainline | `speed_v2` | 30/30, 509 moves, 30/30 `g_ok_13`; Wang BS-B is 757 moves and only 9/30 `g_ok_13`. |
| Open DLTS global feasibility baseline | Wang BS-B | 90/90 on the common test subset and 40/40 on cv1 validation scopes. |
| Open DLTS solved-case path quality | current QUBO/scheduler family | On common solved test cases, current is shorter on 54/65 and saves 291 moves. |
| Open 7x5 repeated-priority cv2/cv3 | `scheduler_state_v5` | 20/20 with 738 moves versus Wang BS-B 20/20 with 820 moves. |

The correct research statement is:

> Wang BS-B should be the mandatory classical feasibility baseline. The current compact QUBO/scheduler line is already superior on the internal Shanggang 30 cases and on solved-case path quality, and it beats Wang outright on 7x5 cv2/cv3. It is not yet globally superior on open DLTS because Wang still has stronger full-coverage feasibility on cv1 and the broader test set.

## Next Rigorous Step

The next experiment should combine both strengths:

1. Run Wang BS-B as a guaranteed incumbent.
2. Run `speed_v2` / `scheduler_state_v5` as quality challengers.
3. Select the shortest validator-passing path per case under a fixed total time budget.
4. Report both "single-profile" and "portfolio total time" metrics, not just the selected-path quality.

This would answer whether Wang should be a fallback baseline only, or part of a no-regret production portfolio.

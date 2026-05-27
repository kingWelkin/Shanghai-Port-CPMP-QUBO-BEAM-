# 完整真实配对算例扩展算法对比

本文只比较真实 `01+26` 配对算例的业务目标，即最终 `N(Future)`。所有进入主表的行都来自完整 15 个真实配对算例，并按平均最终 `N(Future)` 升序排序。

业务约束仍为：`G = M(Now) + N(Future) <= floor(1.3 * N(0))`。

- 汇总 CSV：`docs\experiments\full_real_business_all_algorithms_extended_summary.csv`
- 明细 CSV：`docs\experiments\full_real_business_all_algorithms_extended_detail.csv`

## 汇总排序

| rank | solver | avg final N | avg reduction | avg moves | avg runtime s | avg QUBO calls | ok 1.3N0 | wins |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `beam_focused_with_qubo_portfolio` | 8.53 | 20.53 | 26.80 | 37.313 | 46.47 | 15/15 | 8 |
| 2 | `beam_focused_with_tuned_qubo` | 8.80 | 20.27 | 26.47 | 21.491 | 32.93 | 15/15 | 6 |
| 3 | `beam_focused_with_dense_qubo` | 8.93 | 20.13 | 26.33 | 18.559 | 27.27 | 15/15 | 5 |
| 4 | `qbo_beam_standalone_portfolio` | 9.00 | 20.07 | 26.87 | 26.027 | 46.47 | 15/15 | 7 |
| 5 | `beam_portfolio_with_qubo` | 9.33 | 19.73 | 25.47 | 41.332 | 5.67 | 15/15 | 4 |
| 6 | `beam_focused_with_qubo` | 9.33 | 19.73 | 25.53 | 14.217 | 5.67 | 15/15 | 4 |
| 7 | `beam_portfolio_plain` | 9.47 | 19.60 | 25.40 | 38.400 | 0.00 | 15/15 | 3 |
| 8 | `beam_focused_plain` | 9.47 | 19.60 | 25.47 | 11.286 | 0.00 | 15/15 | 3 |
| 9 | `feature_adaptive_beam` | 9.93 | 19.13 | 24.80 | 5.568 | 0.80 | 15/15 | 3 |
| 10 | `qbo_beam_dense` | 10.07 | 19.00 | 25.20 | 7.273 | 27.27 | 15/15 | 3 |
| 11 | `weighted_astar_business_portfolio` | 10.13 | 18.93 | 25.80 | 30.053 | 0.00 | 15/15 | 3 |
| 12 | `qubo_beam_old_decode` | 10.47 | 18.60 | 24.60 | 3.972 | 5.73 | 15/15 | 3 |
| 13 | `adaptive_beam_top2` | 10.53 | 18.53 | 24.13 | 9.485 | 7.67 | 15/15 | 3 |
| 14 | `qubo_beam_repair` | 10.60 | 18.47 | 24.07 | 3.511 | 5.73 | 15/15 | 2 |
| 15 | `qubo_beam` | 10.60 | 18.47 | 24.27 | 2.932 | 5.67 | 15/15 | 3 |
| 16 | `qbo_beam_base_rerun` | 10.60 | 18.47 | 24.27 | 4.205 | 5.67 | 15/15 | 3 |
| 17 | `beam` | 10.67 | 18.40 | 24.13 | 2.460 | 0.00 | 15/15 | 2 |
| 18 | `adaptive_beam_top1` | 10.67 | 18.40 | 24.13 | 6.534 | 2.00 | 15/15 | 2 |
| 19 | `qbo_beam_low_div` | 10.67 | 18.40 | 24.73 | 5.628 | 7.00 | 15/15 | 3 |
| 20 | `qubo_beam_api_sparse_repair` | 10.73 | 18.33 | 24.00 | 7.083 | 2.13 | 15/15 | 2 |
| 21 | `qubo_beam_api_sparse_old_decode` | 10.73 | 18.33 | 24.20 | 7.087 | 2.13 | 15/15 | 2 |
| 22 | `qubo_beam_api_sparse` | 10.73 | 18.33 | 24.20 | 7.087 | 2.13 | 15/15 | 2 |
| 23 | `beam_post_repair` | 10.73 | 18.33 | 25.27 | 3.332 | 0.00 | 15/15 | 3 |
| 24 | `enhanced_beam_fast` | 10.87 | 18.20 | 24.27 | 8.826 | 0.00 | 15/15 | 1 |
| 25 | `wide_beam_baseline` | 10.93 | 18.13 | 24.13 | 6.968 | 0.00 | 15/15 | 1 |
| 26 | `multi_start_greedy_repair` | 11.27 | 17.80 | 24.00 | 10.175 | 0.00 | 15/15 | 4 |
| 27 | `weighted_astar_business_previous` | 11.60 | 17.47 | 23.93 | 10.004 | 0.00 | 15/15 | 2 |
| 28 | `weighted_astar_business` | 11.60 | 17.47 | 23.93 | 10.004 | 0.00 | 15/15 | 2 |
| 29 | `fused_beam` | 11.73 | 17.33 | 20.47 | 10.042 | 5.00 | 15/15 | 3 |
| 30 | `genetic_tabu_baseline` | 11.73 | 17.33 | 23.73 | 6.175 | 0.00 | 15/15 | 3 |
| 31 | `qbo_beam_wide` | 11.87 | 17.20 | 21.67 | 8.921 | 6.53 | 15/15 | 1 |
| 32 | `hybrid_enhanced_beam` | 12.13 | 16.93 | 19.27 | 10.028 | 0.00 | 15/15 | 1 |
| 33 | `tabu_search_baseline` | 12.27 | 16.80 | 22.93 | 4.787 | 0.00 | 15/15 | 1 |
| 34 | `enhanced_qubo_beam_elite` | 12.33 | 16.73 | 20.13 | 9.838 | 3.93 | 15/15 | 1 |
| 35 | `vns_baseline` | 12.60 | 16.47 | 22.53 | 4.566 | 0.00 | 15/15 | 1 |
| 36 | `grasp_baseline` | 12.73 | 16.33 | 21.87 | 5.006 | 0.00 | 15/15 | 0 |
| 37 | `enhanced_qubo_beam_ranked` | 13.33 | 15.73 | 17.33 | 10.301 | 5.60 | 15/15 | 1 |
| 38 | `lns_beam_repair` | 13.87 | 15.20 | 16.80 | 24.015 | 17.60 | 15/15 | 0 |
| 39 | `lns_qubo_repair` | 14.13 | 14.93 | 16.20 | 30.283 | 59.73 | 15/15 | 0 |
| 40 | `simulated_annealing_baseline` | 14.53 | 14.53 | 22.27 | 3.446 | 0.00 | 15/15 | 0 |
| 41 | `qubo_beam_api_dense` | 14.93 | 14.13 | 14.73 | 10.335 | 4.00 | 15/15 | 0 |
| 42 | `enhanced_beam_wide` | 15.20 | 13.87 | 14.33 | 10.119 | 0.00 | 15/15 | 0 |
| 43 | `chen_greedy_algorithm_legacy` | 16.13 | 12.93 | 27.13 | 0.371 | 0.00 | 0/15 | 0 |
| 44 | `chen_greedy_algorithm_1p3` | 17.13 | 11.93 | 20.20 | 0.115 | 0.00 | 15/15 | 0 |

## 前 20 条路线逐例最终 N(Future)

| case | `beam_focused_with_qubo_portfolio` | `beam_focused_with_tuned_qubo` | `beam_focused_with_dense_qubo` | `qbo_beam_standalone_portfolio` | `beam_portfolio_with_qubo` | `beam_focused_with_qubo` | `beam_portfolio_plain` | `beam_focused_plain` | `feature_adaptive_beam` | `qbo_beam_dense` | `weighted_astar_business_portfolio` | `qubo_beam_old_decode` | `adaptive_beam_top2` | `qubo_beam_repair` | `qubo_beam` | `qbo_beam_base_rerun` | `beam` | `adaptive_beam_top1` | `qbo_beam_low_div` | `qubo_beam_api_sparse_repair` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1_01_26_paired | 16 | 16 | 16 | 18 | 16 | 16 | 16 | 16 | 16 | 18 | 16 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 21 | 18 |
| A2_01_26_paired | 1 | 1 | 1 | 1 | 4 | 4 | 4 | 4 | 4 | 1 | 4 | 4 | 4 | 4 | 5 | 5 | 4 | 4 | 4 | 4 |
| A3_01_26_paired | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 13 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 12 | 10 |
| A4_01_26_paired | 8 | 9 | 9 | 8 | 9 | 9 | 9 | 9 | 9 | 14 | 12 | 13 | 13 | 13 | 13 | 13 | 13 | 13 | 8 | 13 |
| A5_01_26_paired | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 | 15 |
| B1_01_26_paired | 3 | 6 | 6 | 3 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 5 | 6 | 6 | 6 | 6 | 3 | 6 |
| B2_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 10 | 11 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 12 | 9 |
| B3_01_26_paired | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 12 | 7 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 9 | 12 |
| B4_01_26_paired | 1 | 1 | 1 | 1 | 6 | 6 | 6 | 6 | 10 | 1 | 5 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| B5_01_26_paired | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 8 | 3 | 6 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| C1_01_26_paired | 5 | 5 | 5 | 7 | 5 | 5 | 5 | 5 | 5 | 7 | 6 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 |
| C2_01_26_paired | 7 | 7 | 9 | 7 | 7 | 7 | 9 | 9 | 7 | 10 | 13 | 7 | 7 | 9 | 7 | 7 | 9 | 9 | 10 | 10 |
| C3_01_26_paired | 15 | 15 | 15 | 17 | 15 | 15 | 15 | 15 | 15 | 17 | 15 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 | 17 |
| C4_01_26_paired | 10 | 10 | 10 | 11 | 10 | 10 | 10 | 10 | 10 | 11 | 14 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 | 11 |
| C5_01_26_paired | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 9 | 16 | 16 | 16 | 16 | 16 | 16 | 16 | 18 | 16 |

## 算法族结论

- 当前最强仍是 `beam_focused_with_qubo_portfolio`，平均最终 `N(Future)=8.53`。
- QUBO-only portfolio 可达到 `9.00`，说明 QUBO 分支本身有实质收益。
- 无 QUBO 的 focused beam 为 `9.47`，强于多数 tabu / GRASP / VNS / SA / genetic baseline。
- `weighted_astar_business_portfolio` 增强后达到 `10.13`，是较强 best-first 对照，但仍弱于 focused beam。
- tabu、GRASP、VNS、simulated annealing、genetic+tabu 均可作为 sanity-check baseline，但平均质量没有超过 beam/QUBO 系列。
- `chen_greedy_algorithm_1p3` 最快，但平均最终 `N(Future)=17.13`，主要作为生产快速 baseline。

## 关于 MIP

本表的主目标是完整真实配对算例的最终 `N(Future)`。当前没有完整 15 个真实配对算例的 MIP 可比最终 `N(Future)` 结果；已有 MIP 主要用于 small/medium/larger CPMP 参考基准。

正在后台运行的长时 4D MIP 尚未产生可验证的真实算例结果，因此不进入本次排序。若后续 MIP 输出 validated incumbent，可再追加为单独行，但应标注 time limit、是否有 incumbent、是否满足 `1.3N0`。

## 数据来源

- `full_real_business_focused_beam_comparison.csv`
- `full_real_business_weighted_astar_tuning.csv`
- `full_real_business_lns_astar_comparison.csv`
- `full_real_business_qubo_standalone_tuning_comparison.csv`
- `full_real_business_enhanced_qubo_beam_comparison.csv`
- `full_real_business_adaptive_beam_comparison.csv`
- `full_real_business_beam_improvements_comparison.csv`
- `full_real_business_genetic_tabu_comparison.csv`
- `full_real_business_metaheuristics_comparison.csv`
- `full_real_business_tabu_baseline_comparison.csv`
- `full_real_business_strong_baseline_comparison.csv`
- `full_real_business_qubo_api_repair_comparison.csv`
- `full_real_business_qubo_api_comparison.csv`
- `full_real_business_greedy_1p3_vs_beam_qubo.csv`
- `full_real_business_beam_qubo_vs_greedy.csv`

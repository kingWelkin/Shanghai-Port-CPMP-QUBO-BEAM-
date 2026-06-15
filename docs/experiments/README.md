# 实验结论

本目录存放当前项目的实验报告、历史结果和 CSV 明细。当前主结论只保留 30 个单贝位 `N=0` 分支上的高效算法簇。

## 当前主结果

| 算法 | 定位 | 解出 | 总步数 | 平均时间 | 主要文件 |
|---|---|---:|---:|---:|---|
| QUBO Beam `speed_v2` 微优化版 | 当前主推 | 30/30 | 509 | 2.237 秒 | `micro_optimized_speed_v2_full30.csv` |
| enhanced no-QUBO | 强工程消融 | 30/30 | 510 | 2.258 秒 | `micro_optimized_strict_noqubo_enhanced_w1200_full30.csv` |
| classic no-QUBO time-matched | 传统 Beam 严格消融 | 25/30 | 364* | 2.409 秒 | `micro_optimized_strict_ablation_full30.csv` |
| classic no-QUBO budget-best | 传统 Beam 严格消融 | 25/30 | 364* | 2.409 秒 | `micro_optimized_strict_ablation_full30.csv` |
| dynamic QUBO quality | 质量增强变体 | 30/30 | 507 | 4.902 秒 | `astar_qubo_baseline_beam_comparison.md` |
| A* no-proof | 速度强基线，不证明最优 | 30/30 | 514 | 2.829 秒 | `astar_qubo_baseline_beam_comparison.md` |

`*` classic no-QUBO 的总步数只统计成功算例，不能直接与 30/30 算法比较总步数。

26 号贝位：

| 算法 | 解出 | 总步数 | 平均时间 |
|---|---:|---:|---:|
| QUBO Beam `speed_v2` 微优化版 | 15/15 | 414 | 4.031 秒 |
| enhanced no-QUBO | 15/15 | 415 | 4.004 秒 |
| classic no-QUBO time-matched | 10/15 | 269* | 4.396 秒 |

## 当前推荐结论

推荐主线写法：

> QUBO Beam `speed_v2` 微优化版在 30 个单贝位算例上全部达到 `N=0`，总步数 509，平均 2.237 秒。严格时间匹配下，classic no-QUBO 只解出 25/30，26 号贝位为 10/15。enhanced no-QUBO 作为强工程消融也能 30/30，但总步数为 510，说明 deterministic parent-diversity 机制本身很强；QUBO 当前的独立可见贡献主要是 1 步路径压缩和少量 frontier 路径祖先影响。

## Wang/QUBO 方向收束

最新 Wang/QUBO 方向总览见 [wang_qubo_direction_summary.md](wang_qubo_direction_summary.md)。当前 open DLTS 5x5/7x5 test-120 口径下，单一 QUBO-family 最好结果是 `qubo_scheduler_state_v5_wang_rescue_polish`：118/120 解出，full-quality penalty 3873，solved moves 3692，平均 3.901 秒。纯 Wang `wang_bs_b` 为 120/120 解出、4569 步、平均 1.663 秒；它更适合作为 feasibility fallback 和失败样例教师，不适合作为质量主算法。

不再把旧版 `30/30 vs 21/30` 作为当前主结论。该结果属于早期代码/参数口径下的历史对照。

## 开源 CPMP 外部算例

当前新增 DLTS 外部 benchmark 初测，作为上港 30 例之外的泛化证据，不替代上面的当前主结论。

| scope | variant | cases | N=0 solved | avg runtime sec | avg ref gap among solved |
|---|---|---:|---:|---:|---:|
| 5x5_cv1 | QUBO Beam | 200 | 101 | 1.212 | 2.149 |
| 5x5_cv1 | classic no-QUBO | 200 | 84 | 0.766 | 3.000 |
| 7x5_cv1 | QUBO Beam | 40 | 13 | 2.431 | 4.231 |
| 7x5_cv1 | classic no-QUBO | 40 | 15 | 2.236 | 6.733 |

cv1 唯一优先级分支的详细说明见 [open_cpmp_dlts_initial_external_benchmark_summary.md](open_cpmp_dlts_initial_external_benchmark_summary.md)。

新增 adaptive frontier 变体见 [open_cpmp_adaptive_frontier_experiment_summary.md](open_cpmp_adaptive_frontier_experiment_summary.md)。该变体主要改善 7x5 外部算例成功率，不替代上港 30 例主线。

当前已补充 DLTS cv2/cv3 的 priority-group 语义支持。同一优先级的箱子不可区分，最终目标是每个栈自底向顶非递增；合法性仍要求只移动栈顶箱、不能移出堆场。扩展结果如下：

| scope | variant | cases | N=0 solved | moves total | avg runtime sec | avg ref gap among solved |
|---|---|---:|---:|---:|---:|---:|
| 5x5_cv2_test | QUBO Beam | 20 | 14 | 368 | 1.205 | 2.000 |
| 5x5_cv2_test | classic no-QUBO | 20 | 13 | 357* | 0.720 | 3.077 |
| 5x5_cv3_test | QUBO Beam | 20 | 18 | 460 | 1.153 | 2.056 |
| 5x5_cv3_test | classic no-QUBO | 20 | 20 | 522 | 0.763 | 2.200 |
| 7x5_cv2_test | QUBO Beam | 10 | 9 | 330 | 3.252 | 6.222 |
| 7x5_cv2_test | classic no-QUBO | 10 | 9 | 353 | 2.407 | 8.778 |
| 7x5_cv3_test | QUBO Beam | 10 | 10 | 356 | 3.214 | 5.500 |
| 7x5_cv3_test | classic no-QUBO | 10 | 10 | 366 | 2.380 | 6.500 |

`*` 只统计成功算例步数。重复优先级分支上，QUBO 的更稳妥结论是改善成功解质量，而不是稳定提高成功率。详细说明见 [open_cpmp_priority_group_benchmark_summary.md](open_cpmp_priority_group_benchmark_summary.md)。

进一步的严格时间匹配消融见 [open_cpmp_time_matched_ablation_summary.md](open_cpmp_time_matched_ablation_summary.md)。该实验用同一 case 的 QUBO 实际 runtime 作为 classic no-QUBO 的预算，并扫 `1200/2400/4800` 三个 classic beam width。结论是：7x5 `cv2/cv3` 上 QUBO 在同成功率下有更清楚的质量优势；5x5 分支则是成功率和质量的混合信号，不能写成 QUBO 单向全胜。

## 文件说明

当前主线：

- [micro_optimized_speed_v2_full30.md](micro_optimized_speed_v2_full30.md)：微优化后 QUBO Beam 主线结果。
- [micro_optimizations_task07_summary.md](micro_optimizations_task07_summary.md)：Task 07 微优化记录。
- [micro_optimized_strict_noqubo_enhanced_w1200_full30.md](micro_optimized_strict_noqubo_enhanced_w1200_full30.md)：当前 enhanced no-QUBO 强工程消融。

消融与贡献拆解：

- [frontier_qubo_contribution_task04_summary.md](frontier_qubo_contribution_task04_summary.md)：Task 04 QUBO 搜索/后处理贡献拆解。
- [n0_efficient_algorithm_matrix_task08_summary.md](n0_efficient_algorithm_matrix_task08_summary.md)：Task 08 当前 `N=0` 高效算法矩阵。
- [time_aligned_ablation_task09_summary.md](time_aligned_ablation_task09_summary.md)：Task 09 严格时间匹配消融。
- [open_cpmp_time_matched_ablation_summary.md](open_cpmp_time_matched_ablation_summary.md)：DLTS 外部算例严格时间匹配 no-QUBO 消融与逐 case 拆解。
- [v3_task_closure_summary.md](v3_task_closure_summary.md)：v3 任务最终收口记录。
- [micro_optimized_strict_ablation_full30.md](micro_optimized_strict_ablation_full30.md)：严格消融自动生成报告。
- [group_qubo_task03_summary.md](group_qubo_task03_summary.md)：Task 03 分组 QUBO 实验记录，结论是不进主线。

历史和附加对照：

- [astar_qubo_baseline_beam_comparison.md](astar_qubo_baseline_beam_comparison.md)：A* no-proof、dynamic QUBO、早期 Beam 对照。
- [qubo_astar_speed_alignment.md](qubo_astar_speed_alignment.md)：QUBO Beam 与 A* 速度对齐旧版记录。
- [compact_beam_qubo_speed_optimized_v2_full30_repeat.md](compact_beam_qubo_speed_optimized_v2_full30_repeat.md)：历史 2.274 秒档结果。
- [open_cpmp_priority_group_benchmark_summary.md](open_cpmp_priority_group_benchmark_summary.md)：DLTS cv2/cv3 重复优先级扩展基准。
- [results.md](results.md)：输出字段解释。
- [wang_qubo_direction_summary.md](wang_qubo_direction_summary.md)：Wang/QUBO 当前最好结果、值得继续试的方向、以及实验后不应推进的方向。

历史业务改善、Greedy/MIP 小规模 benchmark 和更早的配对算例报告仍保留在本目录，但不进入当前主结论。

## 复现命令

当前主线：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --output docs\experiments\micro_optimized_speed_v2_full30.csv
```

enhanced no-QUBO：

```powershell
python scripts\benchmark_compact_beam_qubo_n0.py --profile speed_v2 --no-use-qubo --no-post-polish-qubo --selection-policy enhanced --beam-width 1200 --core-width 1080 --max-qubo-calls 0 --output docs\experiments\micro_optimized_strict_noqubo_enhanced_w1200_full30.csv
```

strict classic no-QUBO 时间匹配：

```powershell
python scripts\benchmark_compact_qubo_ablation_time_aligned.py --solver-profile speed_v2 --sweep-all-no-qubo --output docs\experiments\micro_optimized_strict_ablation_full30.csv --no-qubo-widths 1300 1400 1600 2000 2400 3200 4800 6400
```

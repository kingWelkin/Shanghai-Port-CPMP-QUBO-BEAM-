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

不再把旧版 `30/30 vs 21/30` 作为当前主结论。该结果属于早期代码/参数口径下的历史对照。

## 文件说明

当前主线：

- [micro_optimized_speed_v2_full30.md](micro_optimized_speed_v2_full30.md)：微优化后 QUBO Beam 主线结果。
- [micro_optimizations_task07_summary.md](micro_optimizations_task07_summary.md)：Task 07 微优化记录。
- [micro_optimized_strict_noqubo_enhanced_w1200_full30.md](micro_optimized_strict_noqubo_enhanced_w1200_full30.md)：当前 enhanced no-QUBO 强工程消融。

消融与贡献拆解：

- [frontier_qubo_contribution_task04_summary.md](frontier_qubo_contribution_task04_summary.md)：Task 04 QUBO 搜索/后处理贡献拆解。
- [n0_efficient_algorithm_matrix_task08_summary.md](n0_efficient_algorithm_matrix_task08_summary.md)：Task 08 当前 `N=0` 高效算法矩阵。
- [time_aligned_ablation_task09_summary.md](time_aligned_ablation_task09_summary.md)：Task 09 严格时间匹配消融。
- [v3_task_closure_summary.md](v3_task_closure_summary.md)：v3 任务最终收口记录。
- [micro_optimized_strict_ablation_full30.md](micro_optimized_strict_ablation_full30.md)：严格消融自动生成报告。
- [group_qubo_task03_summary.md](group_qubo_task03_summary.md)：Task 03 分组 QUBO 实验记录，结论是不进主线。

历史和附加对照：

- [astar_qubo_baseline_beam_comparison.md](astar_qubo_baseline_beam_comparison.md)：A* no-proof、dynamic QUBO、早期 Beam 对照。
- [qubo_astar_speed_alignment.md](qubo_astar_speed_alignment.md)：QUBO Beam 与 A* 速度对齐旧版记录。
- [compact_beam_qubo_speed_optimized_v2_full30_repeat.md](compact_beam_qubo_speed_optimized_v2_full30_repeat.md)：历史 2.274 秒档结果。
- [results.md](results.md)：输出字段解释。

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

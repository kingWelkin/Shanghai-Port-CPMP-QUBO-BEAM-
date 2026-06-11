# Task 09：严格时间匹配消融记录

本轮使用 Task 07 微优化后的 `speed_v2` 主线，重跑 QUBO / no-QUBO 严格消融。

## 运行设置

命令：

```powershell
python scripts\benchmark_compact_qubo_ablation_time_aligned.py --solver-profile speed_v2 --sweep-all-no-qubo --output docs\experiments\micro_optimized_strict_ablation_full30.csv --no-qubo-widths 1300 1400 1600 2000 2400 3200 4800 6400
```

设置含义：

- QUBO 主线：`speed_v2`，beam width = 1200。
- no-QUBO 基线：`classic` selection。
- no-QUBO 禁用 frontier QUBO 和 post-polish QUBO。
- 每个 no-QUBO 宽度的时间预算限制为该算例 QUBO 主线时间。
- 使用 `--sweep-all-no-qubo`，完整扫描宽度集合，不提前停止。
- 默认要求 `final_n=0`，选 time-matched / budget-best 时优先从已归零记录中选择。

## 核心结果

| 方案 | 解出 | 总步数(仅成功算例) | 平均时间(秒) | 平均宽度 |
|---|---:|---:|---:|---:|
| QUBO `speed_v2` | 30/30 | 509 | 2.459 | 1200 |
| no-QUBO time-matched | 25/30 | 364 | 2.409 | 1820 |
| no-QUBO budget-best | 25/30 | 364 | 2.409 | 1820 |

26 号贝位：

| 方案 | 解出 | 总步数(仅成功算例) | 平均时间(秒) | 平均宽度 |
|---|---:|---:|---:|---:|
| QUBO `speed_v2` | 15/15 | 414 | 4.494 | 1200 |
| no-QUBO time-matched | 10/15 | 269 | 4.396 | 1640 |
| no-QUBO budget-best | 10/15 | 269 | 4.396 | 1640 |

## no-QUBO 失败算例

严格时间预算内，classic no-QUBO 的失败全部出现在 26 号贝位：

| 算例 | 选中宽度 | 最终 N | 时间(秒) |
|---|---:|---:|---:|
| `A1_26` | 2000 | 24 | 3.257 |
| `B1_26` | 1400 | 25 | 3.850 |
| `B3_26` | 3200 | 20 | 3.805 |
| `B5_26` | 1300 | 26 | 4.826 |
| `C2_26` | 2400 | 25 | 4.781 |

这些算例在扫过的宽度集合内都没有 `final_n=0` 记录。

## time-matched 与 budget-best

本轮 `time-matched` 和 `budget-best` 完全一致。

原因：

- 对已归零算例，5% 时间容差内没有更少步数的 no-QUBO 记录。
- 对失败算例，所有宽度都未归零，只能从失败记录中选择离 QUBO 时间最近的版本。

## 结论

当前最适合写入主线结论的是：

- 同样代码框架、同样状态加速、同样时间预算下，QUBO `speed_v2` 为 30/30。
- 严格 classic no-QUBO 为 25/30。
- 26 号贝位上，QUBO 为 15/15，classic no-QUBO 为 10/15。
- 这轮消融比早期 “30/30 vs 21/30” 更接近当前代码，因为它基于微优化后的同一版本重跑。

## 输出文件

- `docs/experiments/micro_optimized_strict_ablation_full30.csv`
- `docs/experiments/micro_optimized_strict_ablation_full30_sweep.csv`
- `docs/experiments/micro_optimized_strict_ablation_full30.md`

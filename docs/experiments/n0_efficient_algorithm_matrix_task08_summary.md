# Task 08：`N=0` 高效算法簇实验矩阵

本矩阵只保留当前主线相关的 `N=0` 高效算法簇。早期 `01+26` 配对业务改善、MIP 小规模 benchmark、失败型调参记录不进入主结论。

## 1. 当前主线与强基线

| 算法 | 定位 | 解出 | 总步数 | 平均时间(秒) | 说明 |
|---|---|---:|---:|---:|---|
| QUBO Beam `speed_v2` 微优化版 | 当前主推 | 30/30 | 509 | 2.237 | `micro_optimized_speed_v2_full30.csv` |
| enhanced no-QUBO | 强工程消融 | 30/30 | 510 | 2.258 | 保留 core / parent cap，关闭 frontier QUBO 和 post QUBO |
| classic no-QUBO time-matched | 严格传统 Beam 消融 | 25/30 | 364* | 2.409 | 只统计成功算例步数 |
| classic no-QUBO budget-best | 严格传统 Beam 消融 | 25/30 | 364* | 2.409 | 本轮与 time-matched 完全一致 |

`*` classic no-QUBO 的总步数只统计 25 个成功算例，不能直接与 30/30 算法的 509 对比。

## 2. 26 号贝位

| 算法 | 解出 | 总步数 | 平均时间(秒) |
|---|---:|---:|---:|
| QUBO Beam `speed_v2` 微优化版 | 15/15 | 414 | 4.031 |
| enhanced no-QUBO | 15/15 | 415 | 4.004 |
| classic no-QUBO time-matched | 10/15 | 269* | 4.396 |
| classic no-QUBO budget-best | 10/15 | 269* | 4.396 |

`*` 只统计成功算例。

## 3. 质量增强和外部速度基线

以下结果来自此前对照报告，作为附加参考，不替代当前主线：

| 算法 | 定位 | 解出 | 总步数 | 平均时间(秒) |
|---|---|---:|---:|---:|
| dynamic QUBO quality | 质量增强变体 | 30/30 | 507 | 4.902 |
| dynamic QUBO `balanced2200` | 质量/速度中间档 | 30/30 | 508 | 4.137 |
| A* no-proof | 速度强基线，不证明最优 | 30/30 | 514 | 2.829 |

## 4. 结论

当前推荐主线是 QUBO Beam `speed_v2` 微优化版：

- 达到 30/30 `N=0`。
- 总步数 509。
- 平均时间 2.237 秒。
- 比 A* no-proof 少 5 步。
- 相比 enhanced no-QUBO 少 1 步。
- 相比 classic no-QUBO time-matched，成功率从 25/30 提升到 30/30。

dynamic QUBO 保留为质量增强变体，不作为速度主线。classic no-QUBO 是传统 Beam 消融基线，不应与 enhanced no-QUBO 混用。

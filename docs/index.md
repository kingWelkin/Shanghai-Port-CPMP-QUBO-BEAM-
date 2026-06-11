# 文档导航

## 读者入口

- 想了解当前主算法：看 [当前 QUBO Beam 主算法完整描述](guides/current-qubo-beam-algorithm.md)
- 想看最新实验结论：看 [实验结论汇总](experiments/README.md)
- 想理解 QUBO 到底贡献了什么：看 [Task 04：QUBO 贡献拆解](experiments/frontier_qubo_contribution_task04_summary.md)
- 想看当前 `N=0` 高效算法矩阵：看 [Task 08：实验矩阵](experiments/n0_efficient_algorithm_matrix_task08_summary.md)
- 想看严格 QUBO / no-QUBO 消融：看 [Task 09：时间匹配消融](experiments/time_aligned_ablation_task09_summary.md)
- 想看 v3 任务是否全部收口：看 [v3 任务收口记录](experiments/v3_task_closure_summary.md)
- 想运行程序：看 [运行说明](guides/usage.md)
- 想准备输入：看 [数据格式](guides/data-format.md)
- 想改代码：看 [开发说明](guides/development.md)

## 当前主结论

当前项目主线只看 30 个单贝位 `N=0` 任务。约束是只能移动栈顶箱，不能把箱子移出堆场，验收条件是 `final_n=0` 且 validator 通过。

| 算法 | 定位 | 解出 | 总步数 | 平均时间 |
|---|---|---:|---:|---:|
| QUBO Beam `speed_v2` 微优化版 | 当前主推 | 30/30 | 509 | 2.237 秒 |
| enhanced no-QUBO | 强工程消融 | 30/30 | 510 | 2.258 秒 |
| classic no-QUBO time-matched | 传统 Beam 严格消融 | 25/30 | 364* | 2.409 秒 |
| dynamic QUBO quality | 质量增强变体 | 30/30 | 507 | 4.902 秒 |
| A* no-proof | 速度强基线 | 30/30 | 514 | 2.829 秒 |

`*` classic no-QUBO 只统计成功算例步数。

## 文档分层

- `guides/`：运行、数据格式、开发、当前算法说明。
- `experiments/`：当前实验结论、消融、历史结果和 CSV 明细。
- `models/greedy/`：`chen_greedy_algorithm` 说明和旧版 PDF。
- `models/mip/`：MIP 求解器说明、4D 模型和 archived 3D 对照模型。
- `assets/`：文档图片素材。
- `legacy/`：迁移前旧说明，仅作对照。

## 推荐写法

当前主文档建议采用以下层级：

1. 先写 QUBO Beam `speed_v2` 微优化版是当前主推：30/30、509 步、平均 2.237 秒。
2. 再写 strict classic no-QUBO 是传统 Beam 消融：同时间预算下 25/30，26 号贝位 10/15。
3. 单独说明 enhanced no-QUBO 是强工程消融：也能 30/30，但总步数 510；它不是传统 baseline。
4. 写清 QUBO 的边界：Post-polish 只做路径压缩，不负责把 `N>0` 变成 `N=0`；frontier QUBO 是稀疏前沿纠偏，不是全程主导路径。

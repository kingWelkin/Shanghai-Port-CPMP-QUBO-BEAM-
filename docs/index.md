# 文档导航

## 读者入口

- 想跑程序：看 [运行说明](guides/usage.md)
- 想了解目录：看 [项目结构](guides/project-structure.md)
- 想准备输入：看 [数据格式](guides/data-format.md)
- 想理解 `chen_greedy_algorithm`：看 [chen_greedy_algorithm 说明](models/greedy/chen_greedy_algorithm.md)
- 想求小规模精确最优：看 [精确 MIP 求解](models/mip/mip.md)
- 想看精确模型：优先看 [4D/legacy 3D MIP 说明](models/mip/mip.md)；3D 已封存为历史对照。
- 想看试验结论：看 [试验结论汇总](experiments/README.md)
- 想看当前 QUBO 主结果：看 [QUBO Beam 与 A* 速度对齐结论](experiments/qubo_astar_speed_alignment.md)
- 想看稳定版结果明细：看 [稳定版 QUBO Beam speed 结果](experiments/compact_beam_qubo_speed_optimized_v2_full30_repeat.md)
- 想看历史配对业务改善口径：看 [完整真实配对算例当前推荐口径](experiments/full_real_business_current_recommendation.md)
- 想看输出含义：看 [结果解读](experiments/results.md)
- 想改代码：看 [开发说明](guides/development.md)

## 目录分层

- `guides/`：运行、数据格式、开发和工程结构说明。
- `models/greedy/`：`chen_greedy_algorithm` 说明和旧版 PDF。
- `models/mip/`：MIP 求解器说明、推荐 4D 模型和 archived 3D 对照模型。
- `experiments/`：历史结果解读、贪心与 MIP 对比报告、CSV 明细。
- `assets/`：文档图片素材。
- `legacy/`：迁移前旧说明，仅作对照。

## 当前主结论

当前项目的推荐主线是稳定版非 dynamic QUBO Beam speed。它解决的是 30 个单贝
CPMP 完全良序任务，约束为只能搬移栈顶箱、不能移出堆场，接受条件为 `N=0`。

主要结果：

- 稳定版 QUBO Beam speed：30/30 达到 `N=0`，总步数 509，平均时间 2.274 秒。
- A* no-proof：30/30 达到 `N=0`，总步数 514，平均时间 2.829 秒。
- classic Beam 时间匹配消融：21/30 达到 `N=0`。
- dynamic QUBO 质量变体：可降到 508 或 507 步，但运行时间高于稳定速度版。

因此主文档应优先写 QUBO Beam 主算法，再写 A* no-proof 和 classic Beam 作为对照。
早期 `01+26` 配对业务改善结果仍保留，但它们比较的是
`G = 搬箱步数 + N(Future) <= floor(1.3 * N(0))` 预算内的改善轨迹，
不是当前 `N=0` 单贝完全良序主结果。

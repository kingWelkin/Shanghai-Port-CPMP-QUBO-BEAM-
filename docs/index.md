# 文档导航

## 读者入口

- 想跑程序：看 [运行说明](guides/usage.md)
- 想了解目录：看 [项目结构](guides/project-structure.md)
- 想准备输入：看 [数据格式](guides/data-format.md)
- 想理解 `chen_greedy_algorithm`：看 [chen_greedy_algorithm 说明](models/greedy/chen_greedy_algorithm.md)
- 想求小规模精确最优：看 [精确 MIP 求解](models/mip/mip.md)
- 想看精确模型：优先看 [4D/legacy 3D MIP 说明](models/mip/mip.md)；3D 已封存为历史对照。
- 想看试验结论：看 [试验结论汇总](experiments/README.md)
- 想看当前推荐算法口径：看 [完整真实配对算例当前推荐口径](experiments/full_real_business_current_recommendation.md)
- 想看输出含义：看 [结果解读](experiments/results.md)
- 想改代码：看 [开发说明](guides/development.md)

## 目录分层

- `guides/`：运行、数据格式、开发和工程结构说明。
- `models/greedy/`：`chen_greedy_algorithm` 说明和旧版 PDF。
- `models/mip/`：MIP 求解器说明、推荐 4D 模型和 archived 3D 对照模型。
- `experiments/`：历史结果解读、贪心与 MIP 对比报告、CSV 明细。
- `assets/`：文档图片素材。
- `legacy/`：迁移前旧说明，仅作对照。

## 关键结论

当前项目的稳定快速主流程是 `chen_greedy_algorithm`：

1. 读取 `data/input` 中同一箱区的 `{ZONE}_01_input.txt` 与 `{ZONE}_26_input.txt`。
2. 合并为一个 `W=20, H=6` 的堆场。
3. 反复选择一个合法跨堆/跨贝顶箱搬移动作。
4. 全程约束 `G = 搬箱步数 + N(Future) <= floor(1.3 * N(0))`。
5. 输出每步搬箱轨迹、末态堆表和验收报告。

如果目标是进一步降低完整真实配对案例的 `N(Future)`，当前推荐
`beam_focused_with_qubo` 作为质量优先路线，`feature_adaptive_beam` 作为速度/质量折中路线。
QUBO 精炼代码保留在候选选择层使用，不直接声明可行或最优。
